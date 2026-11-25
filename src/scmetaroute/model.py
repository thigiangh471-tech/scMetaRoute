"""Two-phase continuous VAE training with fixed tree-smoothed targets."""

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from scipy.sparse import issparse
from torch import nn
from torch.nn.functional import mse_loss

from .data import blocks, empty_output, export_metacells, read_panel, write_json
from .vae import CountVAE, negative_binomial_loss
from .hierarchy import partition


@dataclass(frozen=True)
class Config:
    seed: int = 1101
    latent_dim: int = 32
    microcells: int = 750
    metacells: int = 500
    min_support: int = 8
    tree_k: int = 8
    base_epochs: int = 160
    graph_epochs: int = 40
    epoch_max: int = 300
    batch_size: int = 0
    device: str = "auto"
    deterministic: bool = False
    anchor: str = "graph"
    final_merger: str = "ward"


def _training_batches(n, size, epoch):
    offset = (epoch * size) % n
    yield from (slice(offset + s.start, offset + s.stop) for s in blocks(n - offset, size))
    yield from blocks(offset, size)


def _tensor(a, device):
    return torch.from_numpy(np.array(a, dtype=np.float32, copy=True)).to(device)


@torch.inference_mode()
def _encode(net, x, batch_size, device, latent_dim, path):
    z = np.lib.format.open_memmap(path, "w+", np.float32, (x.shape[0], latent_dim))
    net.eval()
    for sl in blocks(x.shape[0], batch_size):
        z[sl] = net.encoders[0](_tensor(x[sl], device)).float().cpu().numpy()
    z.flush()
    return z


def _epochs(n, cfg):
    base, graph = cfg.base_epochs, cfg.graph_epochs
    if n >= 500_000:
        base, graph = min(base, 40), min(graph, 10)
    if base + graph > cfg.epoch_max:
        graph = max(1, min(graph, cfg.epoch_max - base))
        base = min(base, max(1, cfg.epoch_max - graph))
    return base, graph


def fit(panel, output, cfg=Config(), *, raw_csr=None, encoder_csr=None, save_checkpoint=False, progress=None):
    """Fit every row of a frozen panel and write indexed model products."""
    panel, output = Path(panel), Path(output)
    x, raw, library, ids, genes = read_panel(panel, raw_csr, encoder_csr)
    n, g = x.shape
    if not (2 <= cfg.metacells <= cfg.microcells <= n and cfg.min_support >= 1
            and cfg.tree_k >= 1 and cfg.latent_dim >= 1):
        raise ValueError("require 2 <= metacells <= microcells <= cells; support, k and latent dim must be positive")
    if min(cfg.base_epochs, cfg.graph_epochs) < 1 or cfg.epoch_max < 2 or cfg.batch_size < 0:
        raise ValueError("need positive epoch counts, epoch-max >= 2 and a nonnegative batch size")
    if cfg.anchor not in {"graph", "none", "center"} or cfg.final_merger not in {"ward", "kmeans"}:
        raise ValueError("unknown anchoring or final merger mode")
    size = cfg.batch_size or (4096 if n < 100_000 else 16384 if n < 500_000 else 32768)
    if size == 1:
        raise ValueError("BatchNorm requires batch size of at least two")
    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device)
    if cfg.deterministic:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    start = monotonic()
    empty_output(output)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    net = CountVAE(g, cfg.latent_dim).to(device)
    logvar = nn.Linear(cfg.latent_dim, cfg.latent_dim).to(device)
    nn.init.zeros_(logvar.weight)
    nn.init.constant_(logvar.bias, -2.0)
    params = list(net.parameters()) + list(logvar.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
    base_epochs, graph_epochs = _epochs(n, cfg)
    history = []

    def epoch_step(epoch, target=None, membership=None):
        net.train()
        logvar.train()
        loss_sums = dict.fromkeys(("loss", "reconstruction", "kl", "graph"), 0.0)
        for sl in _training_batches(n, size, epoch):
            count = sl.stop - sl.start
            # A singleton tail is duplicated only for BatchNorm; the logical
            # weight and exported membership still refer to the original row.
            rows = sl if count > 1 else np.array([sl.start, sl.start])
            enc = _tensor(x[rows], device)
            values = raw[rows]
            counts = _tensor(values.toarray() if issparse(values) else values, device)
            exposure = _tensor(library[rows], device).reshape(-1, 1) / 10000.0
            h = net.encoders[0](enc)
            v = torch.clamp(logvar(h), -8.0, 8.0)
            z = h + torch.exp(0.5 * v) * torch.randn_like(h)
            mean, disp = net.decoders[0](z)
            rec = negative_binomial_loss(mean, disp, exposure, counts).mean()
            kl = -0.5 * torch.mean(1.0 + v - h.square() - v.exp())
            anchor = torch.zeros((), device=device)
            if target is not None:
                anchor = mse_loss(h, torch.from_numpy(target[membership[rows]]).to(device))
            loss = rec + 1e-3 * kl + (0.05 * anchor if target is not None else 0.0)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 5.0)
            optimizer.step()
            for key, value in zip(loss_sums, (loss, rec, kl, anchor)):
                loss_sums[key] += float(value.detach().cpu()) * count
        row = dict(epoch=epoch + 1, phase="base" if epoch < base_epochs else "graph",
                   **{k: v / n for k, v in loss_sums.items()})
        history.append(row)
        if progress:
            progress(row)

    for epoch in range(base_epochs):
        epoch_step(epoch)
    latent = _encode(net, x, size, device, cfg.latent_dim, output / "STAGE1_LATENT.npy")
    init, centers, graph, stage1 = partition(latent, output, "STAGE1", cfg)
    target = None
    if cfg.anchor == "graph":
        degree = np.asarray(graph.sum(axis=1)).reshape(-1)
        target = (0.75 * centers + 0.25 * (graph @ centers) / np.maximum(degree[:, None], 1e-12)).astype(np.float32)
    elif cfg.anchor == "center":
        target = centers
    if target is not None:
        np.save(output / "GRAPH_TARGET_CENTERS.npy", target)
    for epoch in range(base_epochs, base_epochs + graph_epochs):
        epoch_step(epoch, target, init)
    if save_checkpoint:
        torch.save({
            "model_kind": "scMetaRoute-Stage2", "seed": cfg.seed,
            "continuous_encoder": net.encoders.state_dict(), "continuous_decoder": net.decoders.state_dict(),
            "logvar_head": logvar.state_dict(), "optimizer": optimizer.state_dict(), "epochs": len(history),
        }, output / "STAGE2_CHECKPOINT.pt")
    latent = _encode(net, x, size, device, cfg.latent_dim, output / "GRAPH_REGULARIZED_LATENT.npy")
    assignment, centers, graph, stage2 = partition(latent, output, "STAGE2", cfg, flat=cfg.final_merger == "kmeans")
    export_metacells(raw, ids, genes, assignment, centers, graph, output, cfg.tree_k)
    with (output / "TRAINING_HISTORY.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, history[0], delimiter="\t")
        writer.writeheader()
        writer.writerows(history)
    run = {
        **asdict(cfg), "status": "completed", "fit_scope": "full_data", "cell_count": n,
        "fit_rows": n, "hvg_count": g, "stage2_prior": cfg.anchor, "micro_count": cfg.microcells,
        "support_floor": cfg.min_support, "tree_smoothing": 0.25, "graph_lambda": 0.05,
        "requested_base_epochs": cfg.base_epochs, "requested_graph_epochs": cfg.graph_epochs,
        "base_epochs": base_epochs, "graph_epochs": graph_epochs, "batch_size": size,
        "device": str(device), "stage1": stage1, "stage2": stage2, "runtime_sec": monotonic() - start,
    }
    write_json(output / "RUN_REPORT.json", run)
    return run

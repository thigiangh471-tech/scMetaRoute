"""Prepare count inputs, read frozen panels, and export indexed metacells."""

import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem, sparse_dataset
from scipy import sparse


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def blocks(n, size):
    for start in range(0, n, size):
        yield slice(start, min(n, start + size))


class BackgroundCSR:
    """Legacy sparse encoder cache: explicit entries replace gene background."""

    def __init__(self, path):
        self.values = sparse.load_npz(path).tocsr()
        self.background = np.load(path.with_suffix(".background.npy"))
        self.shape = self.values.shape
        if self.background.shape != (self.shape[1],):
            raise ValueError("encoder background shape does not match its cache")

    def __getitem__(self, index):
        selected = self.values[index].tocoo()
        result = np.broadcast_to(self.background, selected.shape).copy()
        result[selected.row, selected.col] = selected.data
        return result


def empty_output(path):
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def read_panel(folder, raw_csr=None, encoder_csr=None):
    raw = np.load(folder / "X_panel_raw.npy", mmap_mode="r")
    x = np.load(folder / "X_encoder.npy", mmap_mode="r")
    library = np.load(folder / "panel_library_size.npy", mmap_mode="r")
    cells = pd.read_csv(folder / "CELL_IDS.tsv", sep="\t", dtype=str, keep_default_na=False)
    genes = pd.read_csv(folder / "PANEL_GENES.tsv", sep="\t", dtype=str, keep_default_na=False)
    cell_ids, gene_ids = cells["cell_id"], genes["gene_id"]
    if (x.ndim != 2 or x.shape != raw.shape or library.shape != (len(x),)
            or (len(cell_ids), len(gene_ids)) != x.shape
            or not cell_ids.is_unique or not gene_ids.is_unique):
        raise ValueError("panel arrays and unique cell/gene IDs must be aligned")
    if "cell_index" in cells and not np.array_equal(cells.cell_index.to_numpy(dtype=np.int64), np.arange(len(cells))):
        raise ValueError("cell_index must match array row order")
    # Historical filtered panels kept pre-filter gene positions (with gaps).
    if "panel_index" in genes and np.any(np.diff(genes.panel_index.to_numpy(dtype=np.int64)) <= 0):
        raise ValueError("PANEL_GENES.tsv must be ordered as the array columns")
    if not np.isfinite(library).all() or np.any(library <= 0):
        raise ValueError("library exposures must be finite and positive")
    if raw_csr is not None:
        cached = sparse.load_npz(raw_csr).tocsr()
        if cached.shape != raw.shape or cached.dtype != raw.dtype:
            raise ValueError("raw cache must match frozen panel shape and dtype")
        raw = cached
    if encoder_csr is not None:
        cached = BackgroundCSR(encoder_csr)
        if cached.shape != x.shape:
            raise ValueError("encoder cache must match frozen panel shape")
        x = cached
    return x, raw, library, cell_ids.to_numpy(), gene_ids.to_numpy()


def _hvg(matrix, ids, size, n_top):
    """Blockwise Seurat-style dispersion with the historical tie-break."""
    sums = np.zeros(len(ids), dtype=np.float64)
    sumsq = np.zeros_like(sums)
    for sl in blocks(matrix.shape[0], size):
        x = matrix[sl]
        if sparse.issparse(x):
            x = x.tocsr()
            values = x.data.astype(np.float64)
        else:
            values = np.asarray(x, dtype=np.float64)
        if (not np.isfinite(values).all() or (values < 0).any()
                or not np.array_equal(values, np.rint(values))):
            raise ValueError("input must be nonnegative integer RNA counts")
        exposure = np.asarray(x.sum(axis=1, dtype=np.float64)).ravel()
        if (exposure <= 0).any():
            raise ValueError("input contains an empty cell")
        if sparse.issparse(x):
            values *= np.repeat(10000.0 / exposure, np.diff(x.indptr))
            sums += np.bincount(x.indices, weights=values, minlength=len(ids))
            sumsq += np.bincount(x.indices, weights=values * values, minlength=len(ids))
        else:
            norm = values / exposure[:, None] * 10000.0
            sums += norm.sum(axis=0)
            sumsq += np.square(norm).sum(axis=0)
    n = matrix.shape[0]
    mean = sums / n
    var = np.maximum((sumsq - sums * sums / n) / (n - 1), 0)
    disp = var / np.maximum(mean, 1e-12)
    disp[disp == 0] = np.nan
    log_disp = np.log(disp)
    norm = np.full(len(ids), -np.inf)
    categories = pd.cut(np.log1p(mean), bins=20)
    for category in categories.categories:
        mask = (categories == category) & np.isfinite(log_disp)
        v = log_disp[mask]
        if len(v) == 1:
            norm[mask] = 1.0
        elif len(v) > 1:
            sd = v.std(ddof=1)
            norm[mask] = (v - v.mean()) / sd if sd else 0.0
    valid = np.flatnonzero(np.isfinite(norm))
    if not valid.size:
        raise ValueError("no variable genes found")
    order = np.lexsort((ids[valid], -var[valid], -norm[valid]))
    return valid[order[:n_top]]


def prepare_panel(source, output, *, counts="auto", n_hvg=2000, genes=None, chunk_size=2048):
    """Prepare a portable frozen panel; only load row blocks of the count matrix."""
    if n_hvg < 1 or chunk_size < 1:
        raise ValueError("n-hvg and chunk-size must be positive")
    with h5py.File(source, "r") as handle:
        if counts == "auto":
            counts = "counts" if "layers/counts" in handle else "raw" if "raw" in handle else "X"
        if counts == "raw":
            if "raw/X" not in handle:
                raise ValueError("AnnData has no raw/X")
            key, var = "raw/X", handle["raw/var"]
        elif counts == "X":
            key, var = "X", handle["var"]
        else:
            key, var = f"layers/{counts}", handle["var"]
        matrix = handle[key]
        if isinstance(matrix, h5py.Group):
            matrix = sparse_dataset(matrix)
        obs = handle["obs"]
        cell_ids = pd.Index(read_elem(obs[obs.attrs["_index"]])).astype(str)
        ids = pd.Index(read_elem(var[var.attrs["_index"]])).astype(str)
        n = len(cell_ids)
        if not ids.is_unique or not cell_ids.is_unique or n < 2 or matrix.shape != (n, len(ids)):
            raise ValueError("need at least two cells and unique cell and gene identifiers")
        if genes is None:
            cols = _hvg(matrix, ids.to_numpy(), chunk_size, n_hvg)
        else:
            wanted = pd.Index(genes.read_text(encoding="utf-8").splitlines())
            cols = ids.get_indexer(wanted)
            if wanted.empty or not wanted.is_unique or (cols < 0).any():
                raise ValueError("the supplied gene list must contain unique gene IDs found in the input")
        empty_output(output)
        shape = (n, len(cols))
        raw = np.lib.format.open_memmap(output / "X_panel_raw.npy", "w+", np.int32, shape)
        x = np.lib.format.open_memmap(output / "X_encoder.npy", "w+", np.float32, shape)
        library = np.empty(n, dtype=np.int64)
        sums = np.zeros(len(cols), dtype=np.float64)
        sumsq = np.zeros_like(sums)
        for sl in blocks(n, chunk_size):
            counts_block = matrix[sl][:, cols]
            if sparse.issparse(counts_block):
                counts_block = counts_block.toarray()
            values = np.asarray(counts_block, dtype=np.float64)
            if (not np.isfinite(values).all() or (values < 0).any()
                    or not np.array_equal(values, np.rint(values))
                    or (values > np.iinfo(np.int32).max).any()):
                raise ValueError("selected values must be finite nonnegative int32 counts")
            raw[sl] = values
            exposure = values.sum(axis=1)
            if (exposure <= 0).any():
                raise ValueError("a cell has zero counts in the selected panel; revise input or gene list")
            library[sl] = exposure
            log = np.log1p(values / exposure[:, None] * 10000.0)
            sums += log.sum(axis=0)
            sumsq += np.square(log).sum(axis=0)
        mean = (sums / n).astype(np.float32)
        std = np.sqrt(np.maximum((sumsq - sums * sums / n) / (n - 1), 0)).astype(np.float32)
        std[std == 0] = 1
        for sl in blocks(n, chunk_size):
            log = np.log1p(raw[sl].astype(np.float32) / library[sl, None].astype(np.float32) * np.float32(10000))
            x[sl] = np.clip((log - mean) / std, -10, 10)
        raw.flush()
        x.flush()
        np.save(output / "panel_library_size.npy", library)
        np.savez_compressed(output / "SCALE_STATS.npz", feature_mean=mean, feature_std=std)
        pd.DataFrame({"cell_index": np.arange(n), "cell_id": cell_ids}).to_csv(
            output / "CELL_IDS.tsv", sep="\t", index=False,
        )
        pd.DataFrame({"panel_index": np.arange(len(cols)), "source_column": cols,
                      "gene_id": ids[cols]}).to_csv(output / "PANEL_GENES.tsv", sep="\t", index=False)
        write_json(output / "FULL_COHORT_MANIFEST.json", {
            "fit_scope": "full_data", "source": Path(source).name, "count_location": counts,
            "source_n_obs": n, "panel_genes": len(cols), "uses_metadata": False,
        })


def normalized_log(counts):
    counts = np.asarray(counts, dtype=np.float64)
    library = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    return np.log1p(counts / library * 10000.0)


def export_metacells(raw, ids, genes, assignment, centers, graph, output, tree_k):
    support = np.bincount(assignment, minlength=len(centers))
    sums = np.zeros((len(centers), len(genes)), dtype=np.float64)
    totals = np.zeros(len(genes), dtype=np.float64)
    for sl in blocks(len(ids), 4096):
        rows = raw[sl]
        values = np.asarray(rows.toarray() if sparse.issparse(rows) else rows, dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any() or not np.array_equal(values, np.rint(values)):
            raise ValueError("the frozen raw panel must contain nonnegative integer counts")
        np.add.at(sums, assignment[sl], values)
        totals += values.sum(axis=0)
    if not np.array_equal(sums.sum(axis=0), totals):
        raise RuntimeError("raw gene count totals were not conserved")
    means = sums / support[:, None]
    np.save(output / "METACELL_RAW_SUM.npy", sums)
    np.save(output / "METACELL_RAW_MEAN.npy", means)
    meta = ad.AnnData(
        X=normalized_log(sums).astype(np.float32),
        obs=pd.DataFrame({"support": support}, index=[str(i) for i in range(len(centers))]),
        var=pd.DataFrame(index=genes),
        layers={"raw_sum": sums, "raw_mean": means},
        obsm={"X_latent": centers},
        uns={"method": "scMetaRoute-Stage2", "tree_k": tree_k},
    )
    if graph is not None:
        meta.obsp["relation"] = graph
    meta.write_h5ad(output / "METACELL_EXPRESSION.h5ad", compression="gzip")
    pd.DataFrame({"cell_index": np.arange(len(ids)), "cell_id": ids, "metacell": assignment}).to_csv(
        output / "CELL_TO_METACELL.tsv", sep="\t", index=False,
    )
    write_json(output / "PRODUCT_AUDIT.json", {
        "status": "passed", "input_cells": len(ids), "assigned_cells": int(support.sum()),
        "expression_shape": list(sums.shape), "all_gene_totals_preserved": True,
        "relation_shape": list(graph.shape) if graph is not None else None,
    })

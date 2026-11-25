"""Portable command-line entry point."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="scmetaroute")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="prepare a frozen panel from raw counts in H5AD")
    prepare.add_argument("input", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--counts", default="auto", help="auto, X, raw, or a layer name")
    prepare.add_argument("--n-hvg", type=int, default=2000)
    prepare.add_argument("--genes", type=Path, help="one gene ID per line, in panel order")
    prepare.add_argument("--chunk-size", type=int, default=2048)
    run = commands.add_parser("fit", help="train both phases and export the metacells")
    run.add_argument("panel", type=Path)
    run.add_argument("output", type=Path)
    from .model import Config

    defaults = Config()
    for key in ("seed", "latent_dim", "microcells", "metacells", "min_support", "tree_k",
                "base_epochs", "graph_epochs", "epoch_max", "batch_size"):
        run.add_argument("--" + key.replace("_", "-"), type=int, default=getattr(defaults, key))
    run.add_argument("--device", default="auto", help="auto, cpu, or cuda[:index]")
    run.add_argument("--deterministic", action="store_true", help="PyTorch deterministic mode used for the paper ablations")
    run.add_argument("--anchor", choices=("graph", "none", "center"), default="graph")
    run.add_argument("--final-merger", choices=("ward", "kmeans"), default="ward")
    run.add_argument("--raw-csr", type=Path, help="optional frozen raw-count CSR cache")
    run.add_argument("--encoder-csr", type=Path, help="optional frozen replacement-CSR encoder cache")
    run.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from .data import prepare_panel

        prepare_panel(args.input, args.output, counts=args.counts, n_hvg=args.n_hvg,
                      genes=args.genes, chunk_size=args.chunk_size)
    else:
        from .model import fit

        cfg = Config(**{key: getattr(args, key) for key in Config.__dataclass_fields__})
        report = fit(args.panel, args.output, cfg, raw_csr=args.raw_csr, encoder_csr=args.encoder_csr,
                     save_checkpoint=args.save_checkpoint, progress=lambda row: print(json.dumps(row), flush=True))
        print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the complete bundled SLE example from counts to downstream tables."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("runs/sle_demo"))
    p.add_argument("--device", default="cpu", help="cpu, cuda:0, or auto")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists():
        p.error("output already exists; choose a new directory")
    output.mkdir(parents=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    panel, model, metadata = output / "panel", output / "model", output / "metadata.tsv"
    assignment = model / "CELL_TO_METACELL.tsv"
    tasks = root / "downstream_tasks"
    commands = [
        ["-m", "scmetaroute.cli", "prepare", root / "data/sle_5k.h5ad", panel, "--counts", "X", "--n-hvg", "2000"],
        [tasks / "export_metadata.py", root / "data/sle_5k.h5ad", metadata],
        ["-m", "scmetaroute.cli", "fit", panel, model, "--device", args.device,
         "--seed", "1101", "--microcells", "150", "--metacells", "100", "--min-support", "8",
         "--batch-size", "512", "--base-epochs", "160", "--graph-epochs", "40", "--save-checkpoint"],
        [tasks / "metacell_quality.py", panel, assignment, metadata, output / "quality"],
        [tasks / "differential_expression.py", panel, assignment, metadata, output / "differential_expression",
         "--case", "systemic lupus erythematosus"],
        [tasks / "downstream_metrics.py", panel, assignment, metadata, output / "downstream",
         "--de-genes", output / "differential_expression/DE_GENES.tsv.gz"],
    ]
    started = time.monotonic()
    for i, command in enumerate(commands, 1):
        argv = [sys.executable, *map(str, command)]
        print(f"[{i}/{len(commands)}] {' '.join(map(str, command[:3]))}", flush=True)
        with (output / f"step_{i:02d}.log").open("w", encoding="utf-8") as log:
            try:
                subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError:
                print(f"Step {i} failed; inspect {log.name}", file=sys.stderr)
                raise
    print(f"Completed in {time.monotonic() - started:.1f} seconds. Results: {output}")
    print((output / "quality/METRICS_RAW.tsv").read_text(), end="")
    print((output / "downstream/DOWNSTREAM_METRICS.tsv").read_text(), end="")


if __name__ == "__main__":
    main()

"""Rebuild the bundled SLE count subset from the linked source atlas."""

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
from anndata.io import read_elem, sparse_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path, help="original SLE atlas H5AD")
    p.add_argument("output", type=Path)
    args = p.parse_args()
    provenance = args.output.with_name(args.output.stem + "_provenance.json")
    if args.output.exists() or provenance.exists():
        p.error("output already exists; choose a new path")
    rng = np.random.default_rng(1101)
    with h5py.File(args.source, "r") as handle:
        obs = read_elem(handle["obs"])
        donors = obs.groupby("donor_id", observed=True).agg(
            disease=("disease", "first"), cells=("disease", "size"),
            disease_count=("disease", "nunique"),
        )
        if (donors.disease_count != 1).any():
            raise ValueError("each donor must have exactly one disease label")
        rows = []
        for disease in ("normal", "systemic lupus erythematosus"):
            eligible = donors.index[donors.disease.eq(disease) & donors.cells.ge(500)].to_numpy()
            for donor in np.sort(rng.choice(eligible, 10, replace=False)):
                rows.extend(rng.choice(np.flatnonzero(obs.donor_id.to_numpy() == donor), 250, replace=False))
        rows = np.sort(rows)
        matrix = sparse_dataset(handle["raw/X"])[rows].tocsr()
        if not np.isfinite(matrix.data).all() or not np.array_equal(matrix.data, np.rint(matrix.data)):
            raise ValueError("raw/X must contain integer counts")
        if (matrix.data < 0).any() or (matrix.data > np.iinfo(np.int32).max).any():
            raise ValueError("raw/X must contain nonnegative int32 counts")
        matrix = matrix.astype(np.int32)
        var = read_elem(handle["raw/var"])
        var = var[[c for c in ("feature_name", "gene_symbols", "feature_reference") if c in var]].copy()
        metadata = obs.iloc[rows][["cell_type", "donor_id", "disease", "tissue"]].copy()
        for key in metadata:
            metadata[key] = metadata[key].astype(str).astype("category")
        citation = str(read_elem(handle["uns/citation"]))
    a = ad.AnnData(matrix, obs=metadata, var=var)
    a.uns["citation"] = citation
    a.uns["example"] = "SLE; seed 1101; 10 donors per condition; 250 random cells per donor; all source genes"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(args.output, compression="gzip")
    info = {
        "seed": 1101, "cells": a.n_obs, "genes": a.n_vars, "donors": 20,
        "donors_per_condition": 10, "cells_per_donor": 250,
        "minimum_source_cells_per_donor": 500,
        "sampling": "Within each condition, sample 10 eligible donors, sort their IDs, then sample 250 cells per donor; retain source row order.",
        "count_location": "X (copied from source raw/X)",
        "gene_selection": "All source raw genes; no feature selection in the bundled file.",
        "cell_types_used_for_sampling": False, "source_citation": citation,
        "cell_order_sha256": hashlib.sha256("\n".join(a.obs_names).encode()).hexdigest(),
        "gene_order_sha256": hashlib.sha256("\n".join(a.var_names).encode()).hexdigest(),
        "h5ad_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    provenance.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {a.n_obs:,} cells × {a.n_vars:,} genes to {args.output}")


if __name__ == "__main__":
    main()

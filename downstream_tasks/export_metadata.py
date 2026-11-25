"""Export evaluation metadata from H5AD in the original cell order."""

import argparse
from pathlib import Path

import h5py
from anndata.io import read_elem


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--cell-type-key", default="cell_type")
    p.add_argument("--donor-key", default="donor_id")
    p.add_argument("--disease-key", default="disease")
    p.add_argument("--tissue-key", default="tissue")
    args = p.parse_args()
    if args.output.exists():
        p.error("output already exists; choose a new path")
    with h5py.File(args.input, "r") as handle:
        obs = read_elem(handle["obs"])
    if args.cell_type_key not in obs or not obs.index.is_unique:
        p.error("need unique cell IDs and the specified cell-type column")
    fields = {args.cell_type_key: "cell_type"}
    for key, name in [(args.donor_key, "donor_id"), (args.disease_key, "disease"), (args.tissue_key, "tissue")]:
        if key in obs:
            fields[key] = name
    frame = obs[list(fields)].rename(columns=fields).rename_axis("cell_id")
    if frame.cell_type.isna().any():
        p.error("cell_type must not contain missing annotations")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, sep="\t")
    print(f"Exported {len(frame):,} cells; fields: {', '.join(frame.columns)}")


if __name__ == "__main__":
    main()

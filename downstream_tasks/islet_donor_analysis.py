"""Replay frozen islet donor-level abundance estimates and paired contrasts."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def abundance(regions, parent):
    wide = regions.pivot(index="donor_id", columns="method", values="fraction").sort_index()
    if wide.isna().any().any():
        raise ValueError("abundance comparisons must have the same set of donors")
    meta = regions.drop_duplicates("donor_id").set_index("donor_id").loc[wide.index]
    if set(meta.status) != {"ND", "T2D"}:
        raise ValueError("the abundance model expects ND and T2D only")
    design = np.column_stack([np.ones(len(meta)), meta.status.eq("T2D").astype(float),
                              (meta.age - meta.age.mean()) / 10, meta.male])
    if np.linalg.matrix_rank(design) < 4:
        raise ValueError("age/sex-adjusted donor design is rank deficient")
    parent = parent.set_index("donor_id").loc[wide.index]
    wide["parent_state_within_beta"] = parent.state4_cells / parent.beta_cells
    rows = []
    for name, f in wide.items():
        model = sm.OLS(np.arcsin(np.sqrt(f.to_numpy())), design).fit(cov_type="HC3", use_t=True)
        lo, hi = model.conf_int()[1]
        rows.append({
            "comparison": name, "effect": model.params[1], "CI025": lo, "CI975": hi,
            "P": model.pvalues[1], "ND": meta.status.eq("ND").sum(), "T2D": meta.status.eq("T2D").sum(),
            "ND_fraction_mean": f[meta.status.eq("ND")].mean(), "T2D_fraction_mean": f[meta.status.eq("T2D")].mean(),
        })
    return pd.DataFrame(rows)


def contrasts(frame, keys, delta):
    rows = []
    for key, group in frame.groupby(keys):
        if not group.donor_id.is_unique:
            raise ValueError(f"duplicate donor in {key}")
        value = delta(group)
        ci = stats.t.ppf(.975, len(value) - 1) * stats.sem(value)
        rows.append(dict(zip(keys, key if isinstance(key, tuple) else (key,)),
                         donors=len(value), effect=value.mean(), CI025=value.mean() - ci,
                         CI975=value.mean() + ci, P=stats.ttest_1samp(value, 0).pvalue))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source_data", type=Path, help="frozen source tables from the MC124 figure handoff")
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def read(name):
        return pd.read_csv(args.source_data / f"{name}.tsv", sep="\t")

    region = read("DISEASE_REGION_DONORS")
    np.testing.assert_allclose(region.fraction, region.selected_cells / region.parent_cells)
    abundance(region, read("PARENT_STATE_DONORS")).to_csv(
        args.output / "DISEASE_ABUNDANCE.tsv", sep="\t", index=False,
    )
    stress = read("STRESS_DONOR_PAIRS")
    stress = stress.loc[stress.version.eq("selection_genes_removed")]
    results = []
    for threshold in (5, 20):
        eligible = stress.local_cells.ge(threshold) & stress.rest_cells.ge(threshold)
        results.append(contrasts(stress.loc[eligible], ["stimulus"], lambda g: g.difference).assign(min_side_cells=threshold))
    pd.concat(results).to_csv(args.output / "STRESS_RESPONSE.tsv", sep="\t", index=False)
    exact = read("EXACT_SIZE_DONOR_PAIRS")
    contrasts(exact, ["raw_method", "stimulus"], lambda g: g.MC_contrast - g.raw_contrast).to_csv(
        args.output / "SIZE_MATCHED_RESPONSE.tsv", sep="\t", index=False,
    )


if __name__ == "__main__":
    main()

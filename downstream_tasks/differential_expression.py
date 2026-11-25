"""Figure 3a–c: matched donor-by-cell-type disease effect recovery."""

import numpy as np
import pandas as pd
from scipy import sparse, stats

from metacell_quality import aligned_inputs, parse


def logcpm(x):
    return np.log2(1 + x / np.maximum(x.sum(1, keepdims=True), 1e-12) * 1e6)


def bh(p):
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty(len(p))
    result[order] = np.minimum(q, 1)
    return result


def disease_de(raw, assignment, meta, genes, cases, control, min_cells):
    key = pd.MultiIndex.from_frame(meta[["donor_id", "disease", "cell_type"]])
    group, levels = pd.factorize(key, sort=True)
    info = levels.to_frame(index=False)
    info.columns = ["donor_id", "disease", "cell_type"]
    info["cells"] = np.bincount(group)
    if info.groupby("donor_id")["disease"].nunique().max() != 1:
        raise ValueError("a donor cannot have multiple disease labels in this protocol")
    counts = np.zeros((len(levels), len(genes)), dtype=np.float64)
    mc_counts = np.zeros((assignment.max() + 1, len(genes)), dtype=np.float64)
    for start in range(0, len(raw), 4096):
        sl = slice(start, start + 4096)
        values = np.asarray(raw[sl], dtype=np.float64)
        np.add.at(counts, group[sl], values)
        np.add.at(mc_counts, assignment[sl], values)
    cross = sparse.csr_matrix((np.ones(len(raw)), (group, assignment)), shape=(len(levels), len(mc_counts)))
    truth = logcpm(counts)
    estimate = logcpm(cross @ (mc_counts / np.bincount(assignment)[:, None]))
    summaries, tables = [], []
    for case in cases:
        for label in sorted(meta.cell_type.unique()):
            eligible = info.cells.ge(min_cells) & info.cell_type.eq(label)
            left = np.flatnonzero(eligible & info.disease.eq(case))
            right = np.flatnonzero(eligible & info.disease.eq(control))
            if min(len(left), len(right)) < 3:
                continue
            ref = truth[left].mean(0) - truth[right].mean(0)
            eff = estimate[left].mean(0) - estimate[right].mean(0)
            q = bh(np.nan_to_num(stats.ttest_ind(truth[left], truth[right], axis=0, equal_var=False).pvalue, nan=1))
            sig = q < .05
            match = np.sign(ref) == np.sign(eff)
            order = np.argsort(-np.abs(eff), kind="stable")
            summaries.append({
                "case": case, "cell_type": label, "case_donors": len(left), "control_donors": len(right),
                "reference_DE_genes": int(sig.sum()), "spearman": stats.spearmanr(ref, eff).statistic,
                "rmse": np.sqrt(np.mean((ref - eff)**2)), "sign_agreement": match.mean(),
                "significant_sign_agreement": match[sig].mean() if sig.any() else np.nan,
                **{f"recovery_top_{n}": (sig & match)[order[:n]].sum() / sig.sum() if sig.any() else np.nan
                   for n in (100, 400)},
            })
            tables.append(pd.DataFrame({
                "case": case, "cell_type": label, "gene": genes, "reference_log2CPM_difference": ref,
                "compressed_log2CPM_difference": eff, "reference_Welch_BH": q,
            }))
    if not tables:
        raise ValueError("no disease contrasts meet the minimum donor/cell requirements")
    return pd.DataFrame(summaries), pd.concat(tables, ignore_index=True)


def main():
    p = parse(__doc__)
    p.add_argument("--case", action="append", required=True)
    p.add_argument("--control", default="normal")
    p.add_argument("--min-cells", type=int, default=20)
    args = p.parse_args()
    raw, assignment, meta, genes = aligned_inputs(args.panel, args.assignment, args.metadata)
    summary, detail = disease_de(raw, assignment, meta, genes, args.case, args.control, args.min_cells)
    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "DE_SUMMARY.tsv", sep="\t", index=False)
    detail.to_csv(args.output / "DE_GENES.tsv.gz", sep="\t", index=False)


if __name__ == "__main__":
    main()

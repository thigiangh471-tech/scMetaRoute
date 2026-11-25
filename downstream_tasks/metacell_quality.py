"""Figure 2: score a single scMetaRoute assignment on aligned frozen counts."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score, adjusted_mutual_info_score, adjusted_rand_score,
    balanced_accuracy_score, homogeneity_score,
)

from scmetaroute.data import normalized_log, read_panel


def quality(values, assignment):
    centered = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    valid = norms > 0
    z = centered[valid] / norms[valid, None]
    a = assignment[valid]
    corr = np.clip(z @ z.T, -1, 1)
    np.fill_diagonal(corr, np.nan)
    compact, separate = [], []
    for mc in np.unique(a):
        own, other = np.flatnonzero(a == mc), np.flatnonzero(a != mc)
        if len(own) >= 2:
            compact.append(np.nanmean(corr[np.ix_(own, own)]))
        if len(other):
            separate.append(np.nanmean(1 - np.nanmax(corr[np.ix_(own, other)], axis=1)))
    return np.mean(compact) if compact else np.nan, np.mean(separate) if separate else np.nan


def type_de(raw, a, labels, predicted, columns, gene_ids):
    counts = Counter(labels[~np.isin(labels, ["", "nan", "None"])])
    eligible = sorted((x for x in counts if counts[x] >= 10), key=lambda x: (-counts[x], x))
    if len(eligible) < 2:
        return None, None
    left, right = eligible[:2]
    rows = np.sort(np.concatenate([np.flatnonzero(labels == c)[:10000] for c in (left, right)]))
    if min(np.count_nonzero(predicted[rows] == c) for c in (left, right)) < 2:
        return None, None
    values = np.asarray(raw[np.ix_(rows, columns)], dtype=np.float64)
    leaf = np.zeros((a.max() + 1, len(columns)), dtype=np.float64)
    np.add.at(leaf, a[rows], values)
    majority = {int(m): predicted[np.flatnonzero(a == m)[0]] for m in np.unique(a)}
    reference = normalized_log(values[labels[rows] == left].sum(0, keepdims=True))[0] - normalized_log(
        values[labels[rows] == right].sum(0, keepdims=True)
    )[0]
    # Index order matches the historical terminal-count reducer.
    groups = [[m for m in sorted(majority) if majority[m] == c and m <= a[rows].max()] for c in (left, right)]
    estimate = normalized_log(leaf[groups[0]].sum(0, keepdims=True))[0] - normalized_log(
        leaf[groups[1]].sum(0, keepdims=True)
    )[0]
    order = np.argsort(-np.abs(estimate), kind="stable")
    ref_order = np.argsort(-np.abs(reference), kind="stable")
    seen = np.zeros(len(order), dtype=bool)
    curve = []
    for i, gene in enumerate(order[:2000]):
        seen[gene] = True
        curve.append(seen[ref_order[:i + 1]].mean() * (np.sign(estimate[gene]) == np.sign(reference[gene])))
    return pd.DataFrame({
        "gene_id": gene_ids[columns], "reference": reference, "estimate": estimate,
        "left_type": left, "right_type": right,
    }), pd.DataFrame({"top_genes": np.arange(1, len(curve) + 1), "signed_concordance": curve})


def score(raw, a, labels, gene_ids):
    sample = np.arange(min(2500, len(a)))
    columns = np.arange(raw.shape[1])
    if len(columns) > 3000:
        columns = np.sort(np.argsort(-np.var(raw[sample].astype(np.float64), axis=0), kind="stable")[:3000])
    compact, separate = quality(normalized_log(raw[np.ix_(sample, columns)]), a[sample])
    valid = ~np.isin(labels, ["", "nan", "None"])
    predicted = np.full(len(a), "__unknown__", dtype=object)
    purity = []
    for m in np.unique(a[valid]):
        mask = a == m
        values = labels[mask & valid]
        majority, count = Counter(values).most_common(1)[0]
        predicted[mask] = majority
        purity.append(count / len(values))
    result = {"purity": np.mean(purity) if purity else np.nan, "compactness": compact, "separation": separate}
    if len(np.unique(labels[valid])) > 1:
        result.update({
            "Accuracy": accuracy_score(labels[valid], predicted[valid]),
            "Balanced Accuracy": balanced_accuracy_score(labels[valid], predicted[valid]),
            "AMI": adjusted_mutual_info_score(labels[valid], a[valid]),
            "ARI": adjusted_rand_score(labels[valid], a[valid]),
            "Homogeneity": homogeneity_score(labels[valid], a[valid]),
        })
    de, curve = type_de(raw, a, labels, predicted, columns, gene_ids)
    if curve is not None:
        result["AUCC"] = curve.signed_concordance.mean()
        result["signed logFC Spearman"] = spearmanr(de.reference, de.estimate).statistic
    return result, de, curve


def aligned_inputs(panel, assignment, metadata):
    _, raw, _, ids, genes = read_panel(panel)
    frame = pd.read_csv(metadata, sep="\t", dtype=str, keep_default_na=False).set_index("cell_id", verify_integrity=True)
    frame = frame.loc[ids].reset_index()
    if assignment.suffix == ".npy":
        a = np.load(assignment)
    else:
        members = pd.read_csv(assignment, sep="\t", dtype={"cell_id": str}, keep_default_na=False)
        np.testing.assert_array_equal(members.cell_id.to_numpy(), ids)
        a = members.metacell.to_numpy()
    if a.shape != (len(ids),) or not np.issubdtype(a.dtype, np.integer) or (a < 0).any():
        raise ValueError("assignment must provide one nonnegative integer per panel cell")
    return raw, np.unique(a, return_inverse=True)[1], frame, genes


def parse(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("panel", type=Path)
    p.add_argument("assignment", type=Path, help="CELL_TO_METACELL.tsv or panel-ordered assignment NPY")
    p.add_argument("metadata", type=Path, help="TSV containing cell_id and cell_type")
    p.add_argument("output", type=Path)
    return p


def main():
    args = parse(__doc__).parse_args()
    raw, a, frame, genes = aligned_inputs(args.panel, args.assignment, args.metadata)
    args.output.mkdir(parents=True, exist_ok=True)
    scores, de, curve = score(raw, a, frame.cell_type.to_numpy(), genes)
    pd.Series(scores, name="value").rename_axis("metric").to_csv(args.output / "METRICS_RAW.tsv", sep="\t")
    if de is not None:
        de.to_csv(args.output / "DE_GENES.tsv", sep="\t", index=False)
        curve.to_csv(args.output / "DE_CURVE.tsv", sep="\t", index=False)
    type_rows = []
    totals = np.bincount(a)
    for label in np.unique(frame.cell_type):
        c = np.bincount(a[frame.cell_type.to_numpy() == label], minlength=len(totals))
        type_rows.append({
            "cell_type": label, "cells": int(c.sum()),
            "member_weighted_type_purity": c @ (c / totals) / c.sum(),
            "majority_coverage": c[c / totals > .5].sum() / c.sum(),
        })
    pd.DataFrame(type_rows).to_csv(args.output / "CELL_TYPE_RETENTION.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()

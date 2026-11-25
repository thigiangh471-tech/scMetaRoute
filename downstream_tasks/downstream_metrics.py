"""Figure 3d: downstream metrics on one frozen scMetaRoute run."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (average_precision_score, confusion_matrix, fowlkes_mallows_score,
                             normalized_mutual_info_score, roc_auc_score)
from sklearn.model_selection import GroupKFold

from metacell_quality import aligned_inputs, parse

SEED, K = 1101, 15


def expr(x):
    return np.log1p(x * (1e4 / np.maximum(1, x.sum(axis=1, keepdims=True))))


def ranks(z):
    d = cdist(z, z, "sqeuclidean")
    np.fill_diagonal(d, np.inf)
    order = np.argsort(d, axis=1, kind="stable")
    return d, order[:, :K], np.argsort(order, axis=1)


def geometry(ref, z, triplets):
    d0, i0, r0 = ref
    d1, i1, r1 = ranks(z)
    row = np.arange(len(z))[:, None]
    norm = 2 / (len(z) * K * (2 * len(z) - 3 * K - 1))
    a, b, c = triplets.T
    delta0, delta1 = d0[a, b] - d0[a, c], d1[a, b] - d1[a, c]
    ok = delta0 != 0
    return {
        "Trustworthiness": 1 - norm * np.maximum(r0[row, i1] + 1 - K, 0).sum(),
        "Continuity": 1 - norm * np.maximum(r1[row, i0] + 1 - K, 0).sum(),
        "Neighbor overlap": (r0[row, i1] < K).mean(),
        "Triplet agreement": np.mean((delta0[ok] * delta1[ok] > 0) + .5 * (delta1[ok] == 0)),
    }


def clustering(mcz, index, labels):
    known, counts = np.unique(labels, return_counts=True)
    known = [x for x, n in zip(known, counts) if n >= 20 and x.lower() not in ("", "unknown", "unassigned")]
    if len(known) < 2 or len(known) > len(mcz):
        return {}
    keep = np.isin(labels, known)
    fit = KMeans(n_clusters=len(known), n_init=10, random_state=SEED).fit_predict(mcz)
    ref, pred = np.searchsorted(known, labels[keep]), fit[index[keep]]
    tab = confusion_matrix(ref, pred, labels=np.arange(len(known)))
    recall = tab / tab.sum(axis=1, keepdims=True)
    a, b = linear_sum_assignment(recall, maximize=True)
    precision = tab / np.maximum(1, tab.sum(axis=0, keepdims=True))
    r, p = (tab * recall).sum() / tab.sum(), (tab * precision).sum() / tab.sum()
    return {
        "NMI": normalized_mutual_info_score(ref, pred), "Fowlkes–Mallows": fowlkes_mallows_score(ref, pred),
        "Matched macro recall": recall[a, b].mean(), "B³ F1": 2 * r * p / (r + p),
    }


def transfer(mcz, index, meta, labels, rng):
    if not {"donor_id", "tissue"} <= set(meta):
        return {}
    meta = meta.assign(label=labels)
    meta = meta[meta.donor_id.notna() & meta.donor_id.ne("") & meta.tissue.notna()]
    if meta.empty:
        return {}
    tissue = meta.groupby("tissue").donor_id.nunique().idxmax()
    meta = meta[meta.tissue.eq(tissue)]
    eligible = meta.groupby("label").donor_id.nunique()
    classes = sorted(c for c in eligible[eligible >= 10].index if c.lower() not in ("", "unknown", "unassigned"))
    if len(classes) < 3:
        return {}
    sel = np.concatenate([rng.choice(p.index, min(25, len(p)), replace=False)
                          for _, p in meta[meta.label.isin(classes)].groupby(["donor_id", "label"])])
    y, mc, n = np.searchsorted(classes, labels[sel]), index[sel], len(classes)
    donors = meta.loc[sel, "donor_id"].to_numpy()
    response = np.zeros((len(sel), n))
    for train, test in GroupKFold(n_splits=5).split(sel, groups=donors):
        if len(np.unique(y[train])) != n:
            raise ValueError("a training fold is missing a reference cell type")
        counts = np.zeros((len(mcz), n))
        np.add.at(counts, (mc[train], y[train]), 1)
        available = np.flatnonzero(counts.sum(axis=1))
        vote = counts[available] / counts[available].sum(axis=1, keepdims=True)
        dis = cdist(mcz, mcz[available])
        nearest = np.argsort(dis, axis=1, kind="stable")[:, :10]
        weight = 1 / (1 + np.take_along_axis(dis, nearest, axis=1))
        smoothed = (vote[nearest] * weight[:, :, None]).sum(axis=1) / weight.sum(axis=1, keepdims=True)
        response[test] = smoothed[mc[test]]
    tab = confusion_matrix(y, response.argmax(axis=1), labels=np.arange(n))
    p, r = tab.diagonal() / np.maximum(1, tab.sum(0)), tab.diagonal() / tab.sum(1)
    in_top = (np.argsort(-response, axis=1, kind="stable")[:, :3] == y[:, None]).any(axis=1)
    return {
        "Macro F1": np.mean(2 * p * r / np.maximum(1e-12, p + r)),
        "Macro AP": average_precision_score(np.eye(n)[y], response, average="macro"),
        "Macro AUROC": roc_auc_score(np.eye(n)[y], response, average="macro"),
        "Top-3 macro recall": np.mean([in_top[y == c].mean() for c in range(n)]),
    }


def signed_ranking(y, x):
    y, x = np.r_[y > 0, y < 0], np.r_[x, -x]
    n = y.sum()
    order = np.argsort(-x, kind="stable")
    cut = x[order[n - 1]]
    above, tied = x > cut, x == cut
    precision = (y[above].sum() + (n - above.sum()) * y[tied].mean()) / n
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(x[order]))]
    size = np.diff(np.r_[starts, len(order)])
    avg = np.repeat(np.add.reduceat(y[order].astype(float), starts) / size, size)[:n]
    discount = 1 / np.log2(2 + np.arange(n))
    return {
        "Signed AUROC": roc_auc_score(y, x), "Signed AP": average_precision_score(y, x),
        "Signed precision@R": precision, "Signed nDCG@R": avg @ discount / discount.sum(),
    }


def coexpression(x, decoded):
    genes = np.argsort(x.var(axis=0))[-200:]
    c0 = np.nan_to_num(np.corrcoef(x[:, genes].T), nan=0.)
    c1 = np.nan_to_num(np.corrcoef(decoded[:, genes].T), nan=0.)
    tri = np.triu_indices(len(genes), 1)
    edges = np.argsort(np.abs(c0[tri]), kind="stable")[-int(.05 * len(tri[0])):]
    truth = np.zeros(len(tri[0]))
    truth[edges] = np.sign(c0[tri][edges])
    def eig(x):
        return eigh(x, subset_by_index=[len(x) - 5, len(x) - 1])[1]
    return {
        "Gene-pair Spearman": spearmanr(c0[tri], c1[tri]).statistic,
        "Signed edge AP": signed_ranking(truth, c1[tri])["Signed AP"],
        "Gene centrality Spearman": spearmanr(np.abs(c0).sum(0) - np.diag(c0),
                                             np.abs(c1).sum(0) - np.diag(c1)).statistic,
        "Module subspace overlap": np.linalg.norm(eig(c0).T @ eig(c1), "fro")**2 / 5,
    }


def score(raw, index, meta):
    n, g = raw.shape
    if min(n, g) < 30:
        raise ValueError("the paper protocol requires at least 30 cells and genes")
    rng = np.random.default_rng(SEED)
    sample = rng.choice(n, min(1800, n), replace=False)
    x = expr(np.asarray(raw[sample], dtype=np.float32))
    pca = PCA(n_components=30, svd_solver="randomized", random_state=SEED).fit(x)
    triplets = rng.integers(len(sample), size=(50000, 3))
    triplets = triplets[np.sum([triplets[:, a] == triplets[:, b] for a, b in [(0, 1), (0, 2), (1, 2)]], axis=0) == 0]
    sums = np.zeros((index.max() + 1, g), dtype=np.float64)
    for start in range(0, n, 8192):
        sl = slice(start, min(start + 8192, n))
        rows = np.arange(sl.stop - sl.start)
        group = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (index[sl], rows)),
                                  shape=(len(sums), len(rows)))
        sums += group @ raw[sl].astype(np.float32)
    norm = expr(sums).astype(np.float32)
    mcz = pca.transform(norm)
    labels = meta.cell_type.to_numpy()
    return {
        "Neighborhood preservation": geometry(ranks(pca.transform(x)), mcz[index[sample]], triplets),
        "Downstream clustering": clustering(mcz, index, labels),
        "Gene co-expression": coexpression(x, norm[index[sample]]),
        "Cross-donor annotation": transfer(mcz, index, meta, labels, rng),
    }


def score_de(path):
    records = []
    tab = pd.read_csv(path, sep="\t").sort_values(["case", "cell_type", "gene"])
    for _, f in tab.groupby(["case", "cell_type"]):
        y = np.where(f.reference_Welch_BH.to_numpy() < .05, np.sign(f.reference_log2CPM_difference), 0)
        if np.any(y):
            records.append(signed_ranking(y, f.compressed_log2CPM_difference.to_numpy()))
    return pd.DataFrame(records).mean().to_dict()


def main():
    p = parse(__doc__)
    p.add_argument("--de-genes", type=Path)
    args = p.parse_args()
    raw, index, meta, _ = aligned_inputs(args.panel, args.assignment, args.metadata)
    suites = score(raw, index, meta)
    if args.de_genes:
        suites["Disease differential expression"] = score_de(args.de_genes)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [(task, metric, value) for task, values in suites.items() for metric, value in values.items()]
    pd.DataFrame(rows, columns=["task", "metric", "value"]).to_csv(
        args.output / "DOWNSTREAM_METRICS.tsv", sep="\t", index=False,
    )
    reasons = {
        "Downstream clustering": "Need at least two annotated cell types with at least 20 cells each, and no more types than metacells.",
        "Cross-donor annotation": "Need donor_id and tissue, with at least three cell types observed in at least ten donors in one tissue.",
        "Disease differential expression": "No eligible contrast has reference genes with Welch BH < 0.05.",
    }
    pd.DataFrame([
        {"task": task, "status": "completed" if values else "not_applicable",
         "metrics": len(values), "reason": "" if values else reasons.get(task, "Insufficient input.")}
        for task, values in suites.items()
    ]).to_csv(args.output / "TASK_STATUS.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()

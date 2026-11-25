"""Microcluster repair, Ward hierarchy and metacell relation construction."""

import numpy as np
from scipy.cluster.hierarchy import cophenet, cut_tree, linkage
from scipy.sparse import csr_matrix, save_npz
from sklearn.cluster import KMeans, MiniBatchKMeans


def compact(labels):
    return np.unique(np.asarray(labels, dtype=np.int64), return_inverse=True)[1].astype(np.int64)


def aggregate(values, labels):
    values = np.asarray(values, dtype=np.float32)
    labels = compact(labels)
    support = np.bincount(labels).astype(np.int64)
    sums = np.zeros((len(support), values.shape[1]), dtype=np.float64)
    np.add.at(sums, labels, values)
    return (sums / support[:, None]).astype(np.float32), support


def repair_support(labels, centers, minimum):
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=len(centers))
    valid = np.flatnonzero(counts >= minimum)
    if not valid.size:
        raise ValueError("no microcluster meets the support floor; reduce --microcells or --min-support")
    mapping = np.arange(len(counts), dtype=np.int64)
    for source in np.flatnonzero(counts < minimum):
        delta = centers[valid].astype(np.float64) - centers[source].astype(np.float64)
        mapping[source] = valid[np.argmin(np.einsum("ij,ij->i", delta, delta))]
    return compact(mapping[labels])


def tree_graph(centers, k):
    tree = linkage(centers, method="ward", metric="euclidean", optimal_ordering=False)
    m = len(centers)
    distance = np.zeros((m, m), dtype=np.float64)
    distance[np.triu_indices(m, 1)] = cophenet(tree)
    distance += distance.T
    positive = distance[distance > 0]
    tau = float(np.median(positive)) if positive.size else 1e-8
    scale = max(tau, 1e-8)
    rows, cols, data = [], [], []
    for i in range(m):
        order = np.argsort(distance[i], kind="stable")
        for j in order[order != i][:k]:
            rows.append(i)
            cols.append(j)
            data.append(float(np.exp(-distance[i, j] / scale)))
    local = csr_matrix((data, (rows, cols)), shape=distance.shape)
    local = local.maximum(local.T).tocsr()
    representatives = list(range(m))
    rows, cols, data = [], [], []
    for left, right, height, _ in tree:
        i, j = representatives[int(left)], representatives[int(right)]
        weight = 0.1 * float(np.exp(-float(height) / scale))
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([weight, weight])
        representatives.append(i)
    backbone = csr_matrix((data, (rows, cols)), shape=distance.shape)
    graph = local.maximum(backbone).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph, tree, {
        "cophenetic_tau": tau, "k": k, "edge_count": graph.nnz // 2,
        "mean_degree": graph.nnz / m, "backbone_edge_count": len(tree),
        "backbone_scale": 0.1, "weight_min": float(graph.data.min()),
        "weight_max": float(graph.data.max()),
    }


def partition(latent, output, stage, cfg, *, flat=False):
    km_options = dict(n_clusters=cfg.microcells, random_state=cfg.seed, n_init=3, max_iter=150)
    if len(latent) <= 200_000 or flat:
        model = KMeans(**km_options, algorithm="lloyd")
        micro = model.fit_predict(latent)
        method = "full_kmeans_ninit3"
    else:
        model = MiniBatchKMeans(**km_options, batch_size=16384).fit(latent)
        micro = np.empty(len(latent), dtype=np.int32)
        for start in range(0, len(latent), 16384):
            sl = slice(start, start + 16384)
            micro[sl] = model.predict(latent[sl])
        method = "minibatch_kmeans_ninit3"
    micro = repair_support(micro, model.cluster_centers_, cfg.min_support)
    centers, _ = aggregate(latent, micro)
    if len(centers) < cfg.metacells:
        raise ValueError(f"only {len(centers)} supported microclusters for {cfg.metacells} metacells")
    if flat:
        merge = KMeans(
            n_clusters=cfg.metacells, random_state=cfg.seed, n_init=3, max_iter=150, algorithm="lloyd"
        ).fit_predict(centers)
    else:
        ward = linkage(centers, method="ward", metric="euclidean", optimal_ordering=False)
        merge = cut_tree(ward, n_clusters=[cfg.metacells]).reshape(-1)
        np.save(output / f"{stage}_WARD_LINKAGE.npy", ward)
    assignment = compact(merge[micro])
    final_centers, support = aggregate(latent, assignment)
    stem = f"{stage}_{'KMEANS' if flat else 'WARD'}{cfg.metacells}"
    for name, values in (("ASSIGNMENT", assignment), ("CENTERS", final_centers), ("SUPPORT", support)):
        np.save(output / f"{stem}_{name}.npy", values)
    if flat:
        return assignment, final_centers, None, {"kmeans_method": method, "merger": "kmeans"}
    graph, terminal_tree, audit = tree_graph(final_centers, cfg.tree_k)
    save_npz(output / f"{stage}_TREE_GRAPH.npz", graph)
    np.save(output / f"{stage}_METACELL_WARD_LINKAGE.npy", terminal_tree)
    return assignment, final_centers, graph, {
        "kmeans_method": method, "micro_count_after_repair": len(centers), "tree": audit,
    }

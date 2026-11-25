import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from scmetaroute.data import prepare_panel, read_panel
from scmetaroute.hierarchy import repair_support, tree_graph
from scmetaroute.model import Config, fit


@pytest.fixture
def counts():
    rng = np.random.default_rng(32)
    return rng.poisson(2, (65, 16)).astype(np.int32)


@pytest.mark.parametrize("kind", ["dense", "csr", "csc", "raw", "layer"])
def test_prepare_preserves_order_and_counts(tmp_path, counts, kind):
    x = {"csr": sparse.csr_matrix, "csc": sparse.csc_matrix}.get(kind, np.array)(counts)
    a = ad.AnnData(x)
    if kind == "layer":
        a.layers["counts"] = sparse.csr_matrix(counts)
        a.X = np.log1p(counts)
    if kind == "raw":
        a.raw = a.copy()
        a.X = np.log1p(counts)
    source = tmp_path / "counts.h5ad"
    a.write_h5ad(source)
    panel = tmp_path / "panel"
    prepare_panel(source, panel, n_hvg=12, chunk_size=7)
    encoder, raw, exposure, cells, genes = read_panel(panel)
    np.testing.assert_array_equal(cells, a.obs_names)
    np.testing.assert_array_equal(raw, counts[:, genes.astype(int)])
    np.testing.assert_array_equal(exposure, raw.sum(axis=1))
    assert encoder.shape == raw.shape and np.isfinite(encoder).all()


@pytest.mark.parametrize("anchor,merger", [("graph", "ward"), ("none", "kmeans")])
def test_cpu_fit(tmp_path, counts, anchor, merger):
    torch.set_num_threads(2)
    source = tmp_path / "counts.h5ad"
    ad.AnnData(sparse.csr_matrix(counts)).write_h5ad(source)
    panel, dest = tmp_path / "panel", tmp_path / "result"
    prepare_panel(source, panel, n_hvg=16)
    fit(panel, dest, Config(
        microcells=7, metacells=4, min_support=2, batch_size=16, base_epochs=1, graph_epochs=1,
        device="cpu", anchor=anchor, final_merger=merger,
    ), save_checkpoint=True)
    a = ad.read_h5ad(dest / "METACELL_EXPRESSION.h5ad")
    cells = pd.read_csv(dest / "CELL_TO_METACELL.tsv", sep="\t")
    np.testing.assert_array_equal(a.layers["raw_sum"].sum(0), counts[:, a.var_names.astype(int)].sum(0))
    np.testing.assert_array_equal(a.obs.support, np.bincount(cells.metacell))
    assert len(cells) == len(counts) and a.shape == (4, 16)
    if merger == "ward":
        graph = a.obsp["relation"]
        assert connected_components(graph, directed=False)[0] == 1 and (graph != graph.T).nnz == 0
        assert np.load(dest / "STAGE2_METACELL_WARD_LINKAGE.npy").shape == (3, 4)
    else:
        assert "relation" not in a.obsp


def test_repair_and_backbone():
    labels = [0, 0, 0, 1, 2, 2, 2]
    np.testing.assert_array_equal(repair_support(labels, np.array([[0], [1], [3]]), 2), [0, 0, 0, 0, 1, 1, 1])
    graph, linkage, _ = tree_graph(np.array([[0], [1], [4], [6]]), 1)
    assert linkage.shape == (3, 4)
    assert (graph != graph.T).nnz == 0 and connected_components(graph, directed=False)[0] == 1

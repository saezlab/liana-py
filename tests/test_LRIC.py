import anndata as ad
import numpy as np
import pandas as pd
from pytest import raises
from scipy.sparse import csr_matrix

from liana.method.sp._LRIC import (
    _circle_bbox_fractions,
    _filter_by_min_cells,
    _index_resource,
    _linear_transform,
    _make_radii,
    _pair_weights,
    _spatial_pairs,
    _to_dense,
    cross_pcf,
    lric,
)
from liana.testing._sample_anndata import generate_toy_spatial
from liana.testing._sample_resource import sample_resource

# ── module-level fixtures ──────────────────────────────────────────────────────

adata = generate_toy_spatial()
adata.obs["cell_type"] = adata.obs["bulk_labels"]

resource = sample_resource(adata, n_lrs=5, seed=42)

_KWARGS = {
    "max_radius": 100,
    "radius_step": 20,
    "annulus_width": 20,
    "n_angle_samples": 36,
    "verbose": False,
}

_cross_pcf_result = cross_pcf(adata, groupby="cell_type", inplace=False, **_KWARGS)
_cross_pcf_pair = cross_pcf(
    adata, groupby="cell_type",
    cell_types=["CD14+ Monocyte", "CD19+ B"],
    inplace=False, **_KWARGS,
)

_lric_agnostic = lric(adata, resource=resource, inplace=False, **_KWARGS)
_lric_pairwise = lric(
    adata, resource=resource, groupby="cell_type",
    cell_types=["CD14+ Monocyte", "CD19+ B"],
    min_cells=5, inplace=False, **_KWARGS,
)

# ── helpers ────────────────────────────────────────────────────────────────────


def test_linear_transform():
    np.testing.assert_almost_equal(_linear_transform(np.array([[2.0, 4.0], [2.0, 4.0]])), np.ones((2, 2)))
    assert (_linear_transform(np.array([[0.0, 1.0], [1.0, 1.0]])) >= 0).all()
    assert (_linear_transform(np.zeros((3, 2))) == 0).all()


def test_to_dense():
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = _to_dense(csr_matrix(data))
    assert isinstance(out, np.ndarray)
    np.testing.assert_almost_equal(out, data)
    assert _to_dense(data.astype(np.float64)).dtype == np.float32


def test_make_radii():
    ri, ro = _make_radii(max_radius=100, radius_step=20, annulus_width=20)
    np.testing.assert_almost_equal(ri, [20, 40, 60, 80, 100])
    np.testing.assert_almost_equal(ro, [40, 60, 80, 100, 120])
    ri2, ro2 = _make_radii(60, 20, 10)
    np.testing.assert_almost_equal(ro2 - ri2, 10)


def test_circle_bbox_fractions():
    radii = np.array([10.0])
    fracs_in = _circle_bbox_fractions(np.array([[50.0, 50.0]]), radii, 0, 100, 0, 100, n_samples=360)
    np.testing.assert_almost_equal(fracs_in, [[1.0]], decimal=2)

    fracs_edge = _circle_bbox_fractions(np.array([[0.0, 50.0]]), radii, 0, 100, 0, 100, n_samples=360)
    np.testing.assert_almost_equal(fracs_edge, [[0.5]], decimal=2)

    centers = np.random.default_rng(0).uniform(10, 90, size=(5, 2))
    fracs = _circle_bbox_fractions(centers, np.array([5.0, 10.0]), 0, 100, 0, 100)
    assert fracs.shape == (5, 2) and (fracs > 0).all() and (fracs <= 1.0).all()


def test_filter_by_min_cells():
    types = np.array(["A"] * 10 + ["B"] * 3 + ["C"] * 20)
    assert _filter_by_min_cells(types, ["A", "B", "C"], min_cells=5, verbose=False) == ["A", "C"]
    assert set(_filter_by_min_cells(np.array(["A"] * 10 + ["B"] * 10), ["A", "B"], 5, False)) == {"A", "B"}
    assert _filter_by_min_cells(np.array(["A"] * 2), ["A"], min_cells=5, verbose=False) == []


def test_spatial_pairs():
    recv = np.array([[0.0, 0.0], [10.0, 0.0]])
    send = np.array([[5.0, 0.0], [100.0, 0.0]])
    rows, cols, dists = _spatial_pairs(recv, send, max_r=20.0)
    assert rows.tolist() == [0, 1] and cols.tolist() == [0, 0]
    np.testing.assert_almost_equal(dists, [5.0, 5.0])

    coords = np.array([[0.0, 0.0], [1.0, 0.0], [100.0, 0.0]])
    rows_ex, cols_ex, _ = _spatial_pairs(coords, coords, max_r=50.0, exclude_self=True)
    assert len(rows_ex[rows_ex == cols_ex]) == 0

    rows_e, cols_e, dists_e = _spatial_pairs(np.array([[0.0, 0.0]]), np.array([[999.0, 999.0]]), max_r=1.0)
    assert rows_e.size == cols_e.size == dists_e.size == 0


def test_pair_weights_transform_sees_unique_genes():
    # transform is applied to the unique-gene matrix, then gathered to per-pair columns
    a = ad.AnnData(np.arange(12, dtype=np.float32).reshape(3, 4))
    a.var_names = ["g0", "g1", "g2", "g3"]
    idx = np.array([0, 0, 1])  # 3 pairs drawn from 2 unique genes (g0 shared)

    seen = {}
    def tf(x):
        seen["n_cols"] = x.shape[1]
        return x

    out = _pair_weights(a, np.ones(3, bool), ["g0", "g1"], idx, tf)
    assert seen["n_cols"] == 2          # transform saw the 2 unique genes, not 3 gathered cols
    assert out.shape == (3, 3)          # gathered to one column per pair
    np.testing.assert_array_equal(out[:, 0], out[:, 1])  # shared gene duplicated post-transform


def test_index_resource():
    a = ad.AnnData(np.ones((5, 4)))
    a.var_names = ["GeneA", "GeneB", "GeneC", "GeneD"]
    lr_pairs, _, _, names = _index_resource(a, pd.DataFrame({"ligand": ["GeneA", "MISSING"], "receptor": ["GeneB", "GeneD"]}))
    assert lr_pairs.shape == (1, 2) and names == ["GeneA^GeneB"]

    a2 = ad.AnnData(np.ones((3, 4)))
    a2.var_names = ["L1", "L2", "R1", "R2"]
    _, _, _, names2 = _index_resource(a2, pd.DataFrame({"ligand": ["L1", "L2"], "receptor": ["R1", "R2"]}))
    assert all("^" in n for n in names2)

    a3 = ad.AnnData(np.ones((3, 2)))
    a3.var_names = ["GeneX", "GeneY"]
    lr_pairs3, _, _, _ = _index_resource(a3, pd.DataFrame({"ligand": ["L1"], "receptor": ["R1"]}))
    assert lr_pairs3.size == 0


# ── CrossPCF ──────────────────────────────────────────────────────────────────


def test_cross_pcf():
    assert set(_cross_pcf_result.keys()) == {"cell_types", "radii", "results"}
    np.testing.assert_almost_equal(_cross_pcf_result["radii"], [20, 40, 60, 80, 100])
    n_ct = len(_cross_pcf_result["cell_types"])
    assert len(_cross_pcf_result["results"]) == n_ct * (n_ct - 1)
    for arr in _cross_pcf_result["results"].values():
        assert arr.shape == (5,) and np.all(np.isnan(arr) | (arr >= 0))


def test_cross_pcf_pair_values():
    assert _cross_pcf_pair["cell_types"] == ["CD14+ Monocyte", "CD19+ B"]
    assert len(_cross_pcf_pair["results"]) == 2
    np.testing.assert_almost_equal(
        _cross_pcf_pair["results"][("CD14+ Monocyte", "CD19+ B")].sum(), 6.461110, decimal=3
    )


def test_cross_pcf_inplace():
    cross_pcf(adata, groupby="cell_type", key_added="cross_pcf_test", inplace=True, **_KWARGS)
    assert "cross_pcf_test" in adata.uns
    assert set(adata.uns["cross_pcf_test"].keys()) == {"cell_types", "radii", "results"}


def test_cross_pcf_min_cells_filter():
    result = cross_pcf(adata, groupby="cell_type", min_cells=200, inplace=False, **_KWARGS)
    assert len(result["cell_types"]) < 10


# ── LRIC — agnostic mode ──────────────────────────────────────────────────────


def test_lric_agnostic():
    assert set(_lric_agnostic.keys()) == {"pair_names", "radii", "lric"}
    assert _lric_agnostic["lric"].shape == (5, 5)
    assert np.all(_lric_agnostic["lric"] >= 0)
    np.testing.assert_almost_equal(_lric_agnostic["lric"].sum(), 22.316525, decimal=3)
    assert _lric_agnostic["pair_names"] == [
        "C1QB^PPA1", "DHRS4L2^GNG7", "NDUFA11^SUPT4H1", "SFPQ^C20orf27", "PGAM1^WBP11"
    ]


def test_lric_agnostic_min_expressing():
    # min_expressing now applies in agnostic mode too (counts over all cells)
    result = lric(adata, resource=resource, min_expressing=9999, inplace=False, **_KWARGS)
    assert np.all(np.isnan(result["lric"])), "all pairs NaN when threshold exceeds cell count"

    result0 = lric(adata, resource=resource, min_expressing=0, inplace=False, **_KWARGS)
    np.testing.assert_array_equal(result0["lric"], _lric_agnostic["lric"])  # default = no-op


def test_lric_agnostic_inplace_and_transform():
    lric(adata, resource=resource, key_added="lric_test", inplace=True, **_KWARGS)
    assert "lric_test" in adata.uns and "lric" in adata.uns["lric_test"]

    result_id = lric(adata, resource=resource, transform_fn=lambda x: x, inplace=False, **_KWARGS)
    result_default = lric(adata, resource=resource, transform_fn=None, inplace=False, **_KWARGS)
    assert not np.allclose(result_id["lric"], result_default["lric"])


# ── LRIC — pairwise mode ──────────────────────────────────────────────────────


def test_lric_pairwise():
    assert set(_lric_pairwise.keys()) == {"cell_types", "pair_names", "radii", "results"}
    n_ct = len(_lric_pairwise["cell_types"])
    assert len(_lric_pairwise["results"]) == n_ct * (n_ct - 1)
    for arr in _lric_pairwise["results"].values():
        assert arr.shape == (5, 5) and np.all(arr >= 0)
    assert _lric_pairwise["pair_names"] == [
        "C1QB^PPA1", "DHRS4L2^GNG7", "NDUFA11^SUPT4H1", "SFPQ^C20orf27", "PGAM1^WBP11"
    ]
    np.testing.assert_almost_equal(
        _lric_pairwise["results"][("CD14+ Monocyte", "CD19+ B")].sum(), 15.36005, decimal=3
    )


def test_lric_pairwise_extras():
    result_sub = lric(
        adata, resource=resource, groupby="cell_type",
        cell_types=["CD14+ Monocyte", "CD19+ B", "CD56+ NK"],
        min_cells=5, inplace=False, **_KWARGS,
    )
    assert set(result_sub["cell_types"]) == {"CD14+ Monocyte", "CD19+ B", "CD56+ NK"}

    result_filter = lric(adata, resource=resource, groupby="cell_type", min_cells=200, inplace=False, **_KWARGS)
    assert len(result_filter["cell_types"]) < 10

    lric(adata, resource=resource, groupby="cell_type", min_cells=5,
         key_added="lric_pairwise_test", inplace=True, **_KWARGS)
    assert "lric_pairwise_test" in adata.uns and "results" in adata.uns["lric_pairwise_test"]


def test_lric_min_expressing():
    result = lric(
        adata, resource=resource, groupby="cell_type",
        cell_types=["CD14+ Monocyte", "CD19+ B"],
        min_cells=5, min_expressing=9999, inplace=False, **_KWARGS,
    )
    for arr in result["results"].values():
        assert np.all(np.isnan(arr)), "All pairs should be NaN when min_expressing exceeds all cell counts"

    result_partial = lric(
        adata, resource=resource, groupby="cell_type",
        cell_types=["CD14+ Monocyte", "CD19+ B"],
        min_cells=5, min_expressing=1, inplace=False, **_KWARGS,
    )
    for arr in result_partial["results"].values():
        assert arr.shape == (5, 5)


# ── error cases ───────────────────────────────────────────────────────────────


def test_lric_no_lr_pairs_raises():
    bad = pd.DataFrame({"ligand": ["NOTEXIST1"], "receptor": ["NOTEXIST2"]})
    with raises(ValueError, match="No LR pairs"):
        lric(adata, resource=bad, inplace=False, verbose=False)
    with raises(ValueError, match="No LR pairs"):
        lric(adata, resource=bad, groupby="cell_type", inplace=False, verbose=False)

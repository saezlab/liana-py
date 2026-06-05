"""
Unit tests for LRwPCF_pairwise.py (CrossPCF and WeightedPCF).

Style follows liana-py conventions (tests/test_bivar.py):
  - Module-level fixtures computed once per session
  - Plain ``test_*`` functions, no classes
  - ``np.testing.assert_almost_equal`` for floats
  - ``pytest.raises`` for expected exceptions

Import path will move to ``liana.method.sp`` when merged into LIANA+.
"""

import sys
import os

import numpy as np
import pandas as pd
import anndata as ad
from pytest import raises
from scipy.sparse import csr_matrix

# ── resolve local module ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from notebooks.LRwPCF_pairwise import (
    _circle_bbox_fractions,
    _filter_by_min_cells,
    _index_resource,
    _linear_transform,
    _make_radii,
    _spatial_pairs,
    _to_dense,
    cross_pcf,
    lr_wpcf,
)
from liana.testing._sample_anndata import generate_toy_spatial
from liana.testing._sample_resource import sample_resource

# ── module-level fixtures ──────────────────────────────────────────────────────

adata = generate_toy_spatial()
adata.obs["cell_type"] = adata.obs["bulk_labels"]

resource = sample_resource(adata, n_lrs=5, seed=42)

# ── helper: _linear_transform ─────────────────────────────────────────────────


def test_linear_transform_uniform():
    x = np.array([[2.0, 4.0], [2.0, 4.0]])
    out = _linear_transform(x)
    np.testing.assert_almost_equal(out, np.ones((2, 2)))


def test_linear_transform_clips_negative():
    x = np.array([[-1.0, 1.0], [1.0, 1.0]])
    out = _linear_transform(x)
    assert (out >= 0).all()


def test_linear_transform_zero_mean():
    x = np.zeros((3, 2))
    out = _linear_transform(x)
    assert (out == 0).all()


# ── helper: _to_dense ─────────────────────────────────────────────────────────


def test_to_dense_sparse():
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    sp = csr_matrix(data)
    out = _to_dense(sp)
    assert isinstance(out, np.ndarray)
    np.testing.assert_almost_equal(out, data)


def test_to_dense_already_dense():
    data = np.array([[1.0, 2.0]], dtype=np.float64)
    out = _to_dense(data)
    assert out.dtype == np.float32
    np.testing.assert_almost_equal(out, data.astype(np.float32))


# ── helper: _make_radii ───────────────────────────────────────────────────────


def test_make_radii_values():
    ri, ro = _make_radii(max_radius=100, radius_step=20, annulus_width=20)
    np.testing.assert_almost_equal(ri, [20, 40, 60, 80, 100])
    np.testing.assert_almost_equal(ro, [40, 60, 80, 100, 120])


def test_make_radii_width():
    ri, ro = _make_radii(60, 20, 10)
    np.testing.assert_almost_equal(ro - ri, 10)


# ── helper: _circle_bbox_fractions ────────────────────────────────────────────


def test_circle_bbox_fractions_interior():
    centers = np.array([[50.0, 50.0]])
    radii = np.array([10.0])
    fracs = _circle_bbox_fractions(centers, radii, 0, 100, 0, 100, n_samples=360)
    np.testing.assert_almost_equal(fracs, [[1.0]], decimal=2)


def test_circle_bbox_fractions_edge():
    centers = np.array([[0.0, 50.0]])
    radii = np.array([10.0])
    fracs = _circle_bbox_fractions(centers, radii, 0, 100, 0, 100, n_samples=360)
    np.testing.assert_almost_equal(fracs, [[0.5]], decimal=2)


def test_circle_bbox_fractions_in_range():
    centers = np.random.default_rng(0).uniform(10, 90, size=(5, 2))
    radii = np.array([5.0, 10.0])
    fracs = _circle_bbox_fractions(centers, radii, 0, 100, 0, 100)
    assert fracs.shape == (5, 2)
    assert (fracs > 0).all() and (fracs <= 1.0).all()


# ── helper: _filter_by_min_cells ─────────────────────────────────────────────


def test_filter_by_min_cells_drops_small():
    types = np.array(["A"] * 10 + ["B"] * 3 + ["C"] * 20)
    kept = _filter_by_min_cells(types, ["A", "B", "C"], min_cells=5, verbose=False)
    assert kept == ["A", "C"]


def test_filter_by_min_cells_keeps_all():
    types = np.array(["A"] * 10 + ["B"] * 10)
    kept = _filter_by_min_cells(types, ["A", "B"], min_cells=5, verbose=False)
    assert set(kept) == {"A", "B"}


def test_filter_by_min_cells_empty_result():
    types = np.array(["A"] * 2)
    kept = _filter_by_min_cells(types, ["A"], min_cells=5, verbose=False)
    assert kept == []


# ── helper: _spatial_pairs ────────────────────────────────────────────────────


def test_spatial_pairs_within_radius():
    recv = np.array([[0.0, 0.0], [10.0, 0.0]])
    send = np.array([[5.0, 0.0], [100.0, 0.0]])
    rows, cols, dists = _spatial_pairs(recv, send, max_r=20.0)
    assert rows.tolist() == [0, 1]
    assert cols.tolist() == [0, 0]
    np.testing.assert_almost_equal(dists, [5.0, 5.0])


def test_spatial_pairs_exclude_self():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [100.0, 0.0]])
    rows, cols, dists = _spatial_pairs(coords, coords, max_r=50.0, exclude_self=True)
    self_pairs = rows[rows == cols]
    assert len(self_pairs) == 0


def test_spatial_pairs_no_neighbors():
    recv = np.array([[0.0, 0.0]])
    send = np.array([[999.0, 999.0]])
    rows, cols, dists = _spatial_pairs(recv, send, max_r=1.0)
    assert rows.size == 0
    assert cols.size == 0
    assert dists.size == 0


# ── helper: _index_resource ───────────────────────────────────────────────────


def test_index_resource_filters_missing_genes():
    adata_mini = ad.AnnData(np.ones((5, 4)))
    adata_mini.var_names = ["GeneA", "GeneB", "GeneC", "GeneD"]
    res = pd.DataFrame({"ligand": ["GeneA", "MISSING"], "receptor": ["GeneB", "GeneD"]})
    lr_pairs, ligs, recs, names = _index_resource(adata_mini, res)
    assert lr_pairs.shape == (1, 2)
    assert names == ["GeneA^GeneB"]


def test_index_resource_pair_names_format():
    adata_mini = ad.AnnData(np.ones((3, 4)))
    adata_mini.var_names = ["L1", "L2", "R1", "R2"]
    res = pd.DataFrame({"ligand": ["L1", "L2"], "receptor": ["R1", "R2"]})
    _, _, _, names = _index_resource(adata_mini, res)
    assert all("^" in n for n in names)


def test_index_resource_empty_when_no_overlap():
    adata_mini = ad.AnnData(np.ones((3, 2)))
    adata_mini.var_names = ["GeneX", "GeneY"]
    res = pd.DataFrame({"ligand": ["L1"], "receptor": ["R1"]})
    lr_pairs, ligs, recs, names = _index_resource(adata_mini, res)
    assert lr_pairs.size == 0


# ── CrossPCF ──────────────────────────────────────────────────────────────────


def test_cross_pcf_output_keys():
    result = cross_pcf(
        adata, groupby="cell_type",
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, inplace=False, verbose=False,
    )
    assert set(result.keys()) == {"cell_types", "radii", "results"}


def test_cross_pcf_radii():
    result = cross_pcf(
        adata, groupby="cell_type",
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, inplace=False, verbose=False,
    )
    np.testing.assert_almost_equal(result["radii"], [20, 40, 60, 80, 100])


def test_cross_pcf_results_n_pairs():
    result = cross_pcf(
        adata, groupby="cell_type",
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, inplace=False, verbose=False,
    )
    n_ct = len(result["cell_types"])
    assert len(result["results"]) == n_ct * (n_ct - 1)


def test_cross_pcf_result_shape():
    result = cross_pcf(
        adata, groupby="cell_type",
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, inplace=False, verbose=False,
    )
    for arr in result["results"].values():
        assert arr.shape == (5,)


def test_cross_pcf_nonnegative():
    result = cross_pcf(
        adata, groupby="cell_type",
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, inplace=False, verbose=False,
    )
    for arr in result["results"].values():
        assert np.all(np.isnan(arr) | (arr >= 0))


def test_cross_pcf_inplace():
    cross_pcf(
        adata, groupby="cell_type",
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, key_added="cross_pcf_test",
        inplace=True, verbose=False,
    )
    assert "cross_pcf_test" in adata.uns
    assert set(adata.uns["cross_pcf_test"].keys()) == {"cell_types", "radii", "results"}


def test_cross_pcf_subset_cell_types():
    result = cross_pcf(
        adata, groupby="cell_type",
        cell_types=["CD14+ Monocyte", "CD19+ B"],
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, inplace=False, verbose=False,
    )
    assert result["cell_types"] == ["CD14+ Monocyte", "CD19+ B"]
    assert len(result["results"]) == 2


def test_cross_pcf_specific_pair_values():
    result = cross_pcf(
        adata, groupby="cell_type",
        cell_types=["CD14+ Monocyte", "CD19+ B"],
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, inplace=False, verbose=False,
    )
    pcf = result["results"][("CD14+ Monocyte", "CD19+ B")]
    np.testing.assert_almost_equal(pcf.sum(), 6.461110, decimal=3)


def test_cross_pcf_min_cells_filter():
    result = cross_pcf(
        adata, groupby="cell_type",
        min_cells=200,  # very high threshold — most types dropped
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, inplace=False, verbose=False,
    )
    assert len(result["cell_types"]) < 10


# ── WeightedPCF — agnostic mode ───────────────────────────────────────────────


def test_wpcf_agnostic_output_keys():
    result = lr_wpcf(
        adata, resource=resource,
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, inplace=False, verbose=False,
    )
    assert set(result.keys()) == {"pair_names", "radii", "wpcf"}


def test_wpcf_agnostic_wpcf_shape():
    result = lr_wpcf(
        adata, resource=resource,
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, inplace=False, verbose=False,
    )
    assert result["wpcf"].shape == (5, 5)


def test_wpcf_agnostic_pair_names():
    result = lr_wpcf(
        adata, resource=resource,
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, inplace=False, verbose=False,
    )
    assert result["pair_names"] == [
        "C1QB^PPA1", "DHRS4L2^GNG7", "NDUFA11^SUPT4H1", "SFPQ^C20orf27", "PGAM1^WBP11"
    ]


def test_wpcf_agnostic_nonnegative():
    result = lr_wpcf(
        adata, resource=resource,
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, inplace=False, verbose=False,
    )
    assert np.all(result["wpcf"] >= 0)


def test_wpcf_agnostic_specific_sum():
    result = lr_wpcf(
        adata, resource=resource,
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, inplace=False, verbose=False,
    )
    np.testing.assert_almost_equal(result["wpcf"].sum(), 22.316526, decimal=3)


def test_wpcf_agnostic_inplace():
    lr_wpcf(
        adata, resource=resource,
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, key_added="lr_wpcf_test",
        inplace=True, verbose=False,
    )
    assert "lr_wpcf_test" in adata.uns
    assert "wpcf" in adata.uns["lr_wpcf_test"]


def test_wpcf_agnostic_custom_transform():
    result_identity = lr_wpcf(
        adata, resource=resource,
        transform_fn=lambda x: x,
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, inplace=False, verbose=False,
    )
    result_default = lr_wpcf(
        adata, resource=resource,
        transform_fn=None,
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, inplace=False, verbose=False,
    )
    assert not np.allclose(result_identity["wpcf"], result_default["wpcf"])


# ── WeightedPCF — pairwise mode ───────────────────────────────────────────────


def test_wpcf_pairwise_output_keys():
    result = lr_wpcf(
        adata, resource=resource, groupby="cell_type",
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, min_cells=5, inplace=False, verbose=False,
    )
    assert set(result.keys()) == {"cell_types", "pair_names", "radii", "results"}


def test_wpcf_pairwise_result_shape():
    result = lr_wpcf(
        adata, resource=resource, groupby="cell_type",
        cell_types=["CD14+ Monocyte", "CD19+ B"],
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, min_cells=5, inplace=False, verbose=False,
    )
    for arr in result["results"].values():
        assert arr.shape == (5, 5)


def test_wpcf_pairwise_n_ct_pairs():
    result = lr_wpcf(
        adata, resource=resource, groupby="cell_type",
        cell_types=["CD14+ Monocyte", "CD19+ B"],
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, min_cells=5, inplace=False, verbose=False,
    )
    n_ct = len(result["cell_types"])
    assert len(result["results"]) == n_ct * (n_ct - 1)


def test_wpcf_pairwise_pair_names():
    result = lr_wpcf(
        adata, resource=resource, groupby="cell_type",
        cell_types=["CD14+ Monocyte", "CD19+ B"],
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, min_cells=5, inplace=False, verbose=False,
    )
    assert result["pair_names"] == [
        "C1QB^PPA1", "DHRS4L2^GNG7", "NDUFA11^SUPT4H1", "SFPQ^C20orf27", "PGAM1^WBP11"
    ]


def test_wpcf_pairwise_specific_sum():
    result = lr_wpcf(
        adata, resource=resource, groupby="cell_type",
        cell_types=["CD14+ Monocyte", "CD19+ B"],
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, min_cells=5, inplace=False, verbose=False,
    )
    wpcf = result["results"][("CD14+ Monocyte", "CD19+ B")]
    np.testing.assert_almost_equal(wpcf.sum(), 15.36005, decimal=3)


def test_wpcf_pairwise_nonnegative():
    result = lr_wpcf(
        adata, resource=resource, groupby="cell_type",
        cell_types=["CD14+ Monocyte", "CD19+ B"],
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, min_cells=5, inplace=False, verbose=False,
    )
    for arr in result["results"].values():
        assert np.all(arr >= 0)


def test_wpcf_pairwise_inplace():
    lr_wpcf(
        adata, resource=resource, groupby="cell_type",
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, min_cells=5,
        key_added="lr_wpcf_pairwise_test",
        inplace=True, verbose=False,
    )
    assert "lr_wpcf_pairwise_test" in adata.uns
    assert "results" in adata.uns["lr_wpcf_pairwise_test"]


def test_wpcf_pairwise_subset_cell_types():
    result = lr_wpcf(
        adata, resource=resource, groupby="cell_type",
        cell_types=["CD14+ Monocyte", "CD19+ B", "CD56+ NK"],
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, min_cells=5, inplace=False, verbose=False,
    )
    assert set(result["cell_types"]) == {"CD14+ Monocyte", "CD19+ B", "CD56+ NK"}


def test_wpcf_pairwise_min_cells_filters():
    result = lr_wpcf(
        adata, resource=resource, groupby="cell_type",
        max_radius=100, radius_step=20, annulus_width=20,
        n_angle_samples=36, min_cells=200,
        inplace=False, verbose=False,
    )
    assert len(result["cell_types"]) < 10


# ── error cases ───────────────────────────────────────────────────────────────


def test_wpcf_no_lr_pairs_raises():
    bad_resource = pd.DataFrame({"ligand": ["NOTEXIST1"], "receptor": ["NOTEXIST2"]})
    with raises(ValueError, match="No LR pairs"):
        lr_wpcf(adata, resource=bad_resource, inplace=False, verbose=False)


def test_wpcf_pairwise_no_lr_pairs_raises():
    bad_resource = pd.DataFrame({"ligand": ["NOTEXIST1"], "receptor": ["NOTEXIST2"]})
    with raises(ValueError, match="No LR pairs"):
        lr_wpcf(
            adata, resource=bad_resource, groupby="cell_type",
            inplace=False, verbose=False,
        )

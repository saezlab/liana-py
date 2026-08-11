import anndata as ad
import numpy as np
import pandas as pd
from pytest import raises
from scipy.sparse import csr_matrix

from liana.method.sp._LRIC import (
    _circle_bbox_fractions,
    _index_resource,
    _linear_transform,
    _make_radii,
    _pair_weights,
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
    # default extend_first_annulus=True merges the [0, radius_step) band into the first bin
    ri, ro = _make_radii(max_radius=100, radius_step=20, annulus_width=20)
    np.testing.assert_almost_equal(ri, [0, 40, 60, 80, 100])
    np.testing.assert_almost_equal(ro, [40, 60, 80, 100, 120])
    # extend_first_annulus=False keeps the first annulus at [radius_step, ...)
    ri_f, ro_f = _make_radii(max_radius=100, radius_step=20, annulus_width=20, extend_first_annulus=False)
    np.testing.assert_almost_equal(ri_f, [20, 40, 60, 80, 100])
    np.testing.assert_almost_equal(ro_f, [40, 60, 80, 100, 120])
    # annulus_width is respected on every bin when the first one is not extended
    ri2, ro2 = _make_radii(60, 20, 10, extend_first_annulus=False)
    np.testing.assert_almost_equal(ro2 - ri2, 10)
    # with the default merge, only the first bin is widened (to radius_step + annulus_width)
    ri3, ro3 = _make_radii(60, 20, 10)
    np.testing.assert_almost_equal((ro3 - ri3)[0], 30)
    np.testing.assert_almost_equal((ro3 - ri3)[1:], 10)


def test_circle_bbox_fractions():
    radii = np.array([10.0])
    fracs_in = _circle_bbox_fractions(np.array([[50.0, 50.0]]), radii, 0, 100, 0, 100, n_samples=360)
    np.testing.assert_almost_equal(fracs_in, [[1.0]], decimal=2)

    fracs_edge = _circle_bbox_fractions(np.array([[0.0, 50.0]]), radii, 0, 100, 0, 100, n_samples=360)
    np.testing.assert_almost_equal(fracs_edge, [[0.5]], decimal=2)

    centers = np.random.default_rng(0).uniform(10, 90, size=(5, 2))
    fracs = _circle_bbox_fractions(centers, np.array([5.0, 10.0]), 0, 100, 0, 100)
    assert fracs.shape == (5, 2) and (fracs > 0).all() and (fracs <= 1.0).all()


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
    np.testing.assert_almost_equal(_cross_pcf_result["radii"], [0, 40, 60, 80, 100])
    n_ct = len(_cross_pcf_result["cell_types"])
    assert len(_cross_pcf_result["results"]) == n_ct * (n_ct - 1)
    for arr in _cross_pcf_result["results"].values():
        assert arr.shape == (5,) and np.all(np.isnan(arr) | (arr >= 0))


def test_cross_pcf_pair_values():
    assert _cross_pcf_pair["cell_types"] == ["CD14+ Monocyte", "CD19+ B"]
    assert len(_cross_pcf_pair["results"]) == 2
    np.testing.assert_almost_equal(
        _cross_pcf_pair["results"][("CD14+ Monocyte", "CD19+ B")].sum(), 5.932712, decimal=3
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
    # NOTE: this reference value is not 22.490379 (the value under the old
    # implementation). `use_raw=True` is the default, and the old `_get_expr`
    # sliced `.raw.X` from a `gene_names`-subset view -- but `.raw` does not
    # follow the parent's var subsetting, so it silently gathered whichever
    # genes sat at the requested *positions* in the full, unsliced `.raw.var`
    # instead of the intended ligands/receptors. `prep_check_adata` resolves
    # `use_raw`/`layer` into `.X` once up front, so this LRIC only ever reads
    # the genes it actually asked for.
    np.testing.assert_almost_equal(_lric_agnostic["lric"].sum(), 26.89414, decimal=3)
    assert _lric_agnostic["pair_names"] == [
        "C1QB^PPA1", "DHRS4L2^GNG7", "NDUFA11^SUPT4H1", "SFPQ^C20orf27", "PGAM1^WBP11"
    ]


def test_lric_agnostic_min_expressing():
    # min_expressing now applies in agnostic mode too (counts over all cells)
    result = lric(adata, resource=resource, min_expressing=9999, inplace=False, **_KWARGS)
    assert np.all(np.isnan(result["lric"])), "all pairs NaN when threshold exceeds cell count"

    result0 = lric(adata, resource=resource, min_expressing=0, inplace=False, **_KWARGS)
    np.testing.assert_array_equal(result0["lric"], _lric_agnostic["lric"])  # default = no-op

    # partial threshold: masking is per-pair and leaves kept pairs untouched
    partial = lric(adata, resource=resource, min_expressing=100, inplace=False, **_KWARGS)["lric"]
    masked = np.isnan(partial).all(axis=0)
    assert masked.any() and not masked.all(), "expected a mix of masked and kept pairs"
    np.testing.assert_array_equal(partial[:, ~masked], _lric_agnostic["lric"][:, ~masked])


def test_extend_first_annulus_flag():
    # extend_first_annulus=False restores the pre-merge behaviour: first annulus
    # starts at radius_step and the [0, radius_step) contact band is excluded.
    ag_f = lric(adata, resource=resource, extend_first_annulus=False, inplace=False, **_KWARGS)
    np.testing.assert_almost_equal(ag_f["radii"], [20, 40, 60, 80, 100])
    np.testing.assert_almost_equal(ag_f["lric"].sum(), 27.211477, decimal=3)

    cp_f = cross_pcf(
        adata, groupby="cell_type",
        cell_types=["CD14+ Monocyte", "CD19+ B"],
        extend_first_annulus=False, inplace=False, **_KWARGS,
    )
    np.testing.assert_almost_equal(cp_f["radii"], [20, 40, 60, 80, 100])
    np.testing.assert_almost_equal(
        cp_f["results"][("CD14+ Monocyte", "CD19+ B")].sum(), 6.461110, decimal=3
    )

    # merging only changes the first bin; bins beyond the first are identical
    ag_t = lric(adata, resource=resource, inplace=False, **_KWARGS)
    np.testing.assert_array_equal(ag_t["lric"][1:], ag_f["lric"][1:])


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
    # see test_lric_agnostic for why this differs from the old implementation's value
    np.testing.assert_almost_equal(
        _lric_pairwise["results"][("CD14+ Monocyte", "CD19+ B")].sum(), 25.468023, decimal=3
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
    # a resource with none of its genes in `adata.var_names` is now caught by the
    # shared `assert_covered` check in `prep_check_adata`'s call site (same as
    # `_inflow`/`_spatial_bivariate`), before `_index_resource` is ever reached.
    bad = pd.DataFrame({"ligand": ["NOTEXIST1"], "receptor": ["NOTEXIST2"]})
    with raises(ValueError,
                match="Please check if appropriate organism/ID type"):
        lric(adata, resource=bad, inplace=False, verbose=False)
    with raises(ValueError,
                match="Please check if appropriate organism/ID type"):
        lric(adata, resource=bad, groupby="cell_type", inplace=False, verbose=False)


# ── regression: adata passed by the caller must never be mutated ──────────────


def test_lric_does_not_mutate_input_view_on_complex_resource():
    """`_add_complexes_to_var` used to be called on the caller's `adata` directly,
    mutating its `.var` in place (new rows for each complex) without a matching
    `.X`/`.layers` update. When `adata` was a view (e.g. a random cell subsample),
    this corrupted it silently -- `.var` grew while `.X`/`.layers` did not -- and
    raised a shape-mismatch `ValueError` the next time the view was touched (e.g.
    by `inplace=True`, which writes to `.uns` and forces the view to materialise).

    `prep_check_adata` now isolates a fresh copy up front, so the caller's object
    (view or not) must come out unchanged.
    """
    rng = np.random.default_rng(0)
    idx = rng.choice(adata.n_obs, adata.n_obs // 2, replace=False)
    adata_sub = adata[idx, :]
    assert adata_sub.is_view

    complex_receptor = f"{adata.var_names[0]}_{adata.var_names[1]}"
    complex_resource = pd.DataFrame({
        "ligand": [adata.var_names[2]],
        "receptor": [complex_receptor],
    })

    n_vars_before = adata_sub.n_vars
    lric(
        adata_sub, resource=complex_resource, groupby="cell_type",
        key_added="lric_view_test", inplace=True, **_KWARGS,
    )

    assert adata_sub.n_vars == n_vars_before
    assert "lric_view_test" in adata_sub.uns
    assert "results" in adata_sub.uns["lric_view_test"]

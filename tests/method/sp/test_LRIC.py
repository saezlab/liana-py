import anndata as ad
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from pytest import raises
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from liana.method.sp._LRIC import (
    _default_min_cells,
    _edge_group_bounds,
    _index_resource,
    _linear_transform,
    _make_radii,
    _pair_weights,
    _support_edge_list,
    _to_dense,
    _type_mean_weights,
    cross_pcf,
    lric,
)
from liana.testing import generate_toy_spatial
from liana.testing._sample_resource import sample_resource

_KWARGS = {"max_radius": 100, "radius_step": 20, "verbose": False}

# NOTE: the fixtures below are module-scoped, as running `cross_pcf`/`lric` is
# comparatively expensive and the tests only read from their results.


@pytest.fixture(scope="module")
def adata():
    """Toy spatial data with `cell_type` labels.

    Shared, so tests must only read from it. Anything that writes -- i.e. any
    call with `inplace=True` -- takes `adata_copy` instead.
    """
    adata = generate_toy_spatial()
    adata.obs["cell_type"] = adata.obs["bulk_labels"]
    return adata


@pytest.fixture(scope="module")
def resource(adata):
    """Five ligand-receptor pairs, all drawn from `adata`'s own genes."""
    return sample_resource(adata, n_lrs=5, seed=42)


@pytest.fixture
def adata_copy(adata):
    """A throwaway copy, for the tests that write to `.uns` with `inplace=True`.

    The module-scoped `adata` is shared, so mutating it would leak state into
    whichever test happens to run next (the suite is order-randomised in CI).
    """
    return adata.copy()


@pytest.fixture(scope="module")
def cross_pcf_result(adata):
    return cross_pcf(adata, groupby="cell_type", inplace=False, **_KWARGS)


@pytest.fixture(scope="module")
def cross_pcf_pair(adata):
    return cross_pcf(
        adata, groupby="cell_type",
        cell_types=["CD14+ Monocyte", "CD19+ B"],
        inplace=False, **_KWARGS,
    )


@pytest.fixture(scope="module")
def lric_agnostic(adata, resource):
    """Default agnostic LRIC, computed once and shared as the reference the
    lr_sep / expr_prop / transform_fn variants are each compared against."""
    return lric(adata, resource=resource, inplace=False, **_KWARGS)


@pytest.fixture(scope="module")
def lric_pairwise(adata, resource):
    """Default pairwise LRIC (all directed cell-type pairs), computed once and
    shared as the reference for the decomposition and min_cells variants."""
    return lric(adata, resource=resource, groupby="cell_type", inplace=False, **_KWARGS)


# ── helpers: pure math / small synthetic inputs ────────────────────────────


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
    # annulus_steps=1 (default): disjoint annuli one radius_step wide; the default
    # extend_first_annulus=True merges the [0, radius_step) band into the first bin
    ri, ro = _make_radii(max_radius=100, radius_step=20)
    np.testing.assert_almost_equal(ri, [0, 40, 60, 80, 100])
    np.testing.assert_almost_equal(ro, [40, 60, 80, 100, 120])
    # extend_first_annulus=False keeps the first annulus at [radius_step, ...)
    ri_f, ro_f = _make_radii(max_radius=100, radius_step=20, extend_first_annulus=False)
    np.testing.assert_almost_equal(ri_f, [20, 40, 60, 80, 100])
    np.testing.assert_almost_equal(ro_f, [40, 60, 80, 100, 120])
    # annulus_steps=2: same inner edges (so the same `radii`), annuli twice as wide
    # and therefore overlapping
    ri2, ro2 = _make_radii(max_radius=100, radius_step=20, annulus_steps=2)
    np.testing.assert_almost_equal(ri2, ri)
    np.testing.assert_almost_equal(ro2, [60, 80, 100, 120, 140])


def test_annulus_steps_validation(adata):
    for bad in (0, 1.5):
        with raises(ValueError, match="annulus_steps"):
            cross_pcf(adata, groupby="cell_type", annulus_steps=bad, inplace=False, verbose=False)


def test_index_resource():
    a = ad.AnnData(np.ones((5, 4)))
    a.var_names = ["GeneA", "GeneB", "GeneC", "GeneD"]
    lr_pairs, _, _, names = _index_resource(
        a, pd.DataFrame({"ligand": ["GeneA", "MISSING"], "receptor": ["GeneB", "GeneD"]}), "^"
    )
    assert lr_pairs.shape == (1, 2) and names == ["GeneA^GeneB"]

    a2 = ad.AnnData(np.ones((3, 4)))
    a2.var_names = ["L1", "L2", "R1", "R2"]
    _, _, _, names2 = _index_resource(
        a2, pd.DataFrame({"ligand": ["L1", "L2"], "receptor": ["R1", "R2"]}), "^"
    )
    assert all("^" in n for n in names2)

    a3 = ad.AnnData(np.ones((3, 2)))
    a3.var_names = ["GeneX", "GeneY"]
    lr_pairs3, _, _, _ = _index_resource(a3, pd.DataFrame({"ligand": ["L1"], "receptor": ["R1"]}), "^")
    assert lr_pairs3.size == 0

    # custom separator
    _, _, _, names_custom = _index_resource(
        a2, pd.DataFrame({"ligand": ["L1", "L2"], "receptor": ["R1", "R2"]}), "|"
    )
    assert all("|" in n and "^" not in n for n in names_custom)


def test_pair_weights_transform_sees_unique_genes():
    # transform is applied to the unique-gene matrix, then gathered to per-pair columns
    a = ad.AnnData(np.arange(12, dtype=np.float32).reshape(3, 4))
    a.var_names = ["g0", "g1", "g2", "g3"]
    idx = np.array([0, 0, 1])  # 3 pairs drawn from 2 unique genes (g0 shared)

    seen = {}
    def tf(x):
        seen["n_cols"] = x.shape[1]
        return x

    out = _pair_weights(a, ["g0", "g1"], idx, tf)
    assert seen["n_cols"] == 2          # transform saw the 2 unique genes, not 3 gathered cols
    assert out.shape == (3, 3)          # gathered to one column per pair
    np.testing.assert_array_equal(out[:, 0], out[:, 1])  # shared gene duplicated post-transform


def test_default_min_cells():
    a = ad.AnnData(np.ones((1000, 2)))
    # an explicit int passes through untouched
    assert _default_min_cells(a, 7, verbose=False) == 7
    # None derives floor(_MIN_CELLS_FRAC * n_obs) + 1 == floor(0.01 * 1000) + 1
    assert _default_min_cells(a, None, verbose=False) == 11


def test_support_edge_list():
    coords = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    tree = cKDTree(coords)
    radii_inner = np.array([0.0, 15.0])
    radii_outer = np.array([15.0, 25.0])
    I, J, bin_idx = _support_edge_list(tree, radii_inner, radii_outer)
    assert (I != J).all()  # self-pairs excluded
    pairs = set(zip(I.tolist(), J.tolist()))
    assert pairs == {(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)}
    bin_of = dict(zip(zip(I.tolist(), J.tolist()), bin_idx.tolist()))
    assert bin_of[(0, 1)] == 0 and bin_of[(1, 2)] == 0
    assert bin_of[(0, 2)] == 1


def test_edge_group_bounds():
    group_key_sorted = np.array([0, 0, 1, 1, 1, 3])
    bounds = _edge_group_bounds(group_key_sorted, n_groups=4)
    np.testing.assert_array_equal(bounds, [0, 2, 5, 5, 6])


def test_type_mean_weights():
    W = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    obs_types = np.array(["A", "A", "B", "B"])
    m = _type_mean_weights(W, obs_types, ["A", "B"])
    np.testing.assert_almost_equal(m, [[2.0, 3.0], [6.0, 7.0]])


# ── CrossPCF ──────────────────────────────────────────────────────────────────


def test_cross_pcf(cross_pcf_result):
    result = cross_pcf_result
    assert set(result.keys()) == {"cell_types", "radii", "results"}
    np.testing.assert_almost_equal(result["radii"], [0, 40, 60, 80, 100])
    n_ct = len(result["cell_types"])
    assert len(result["results"]) == n_ct * (n_ct - 1)
    for arr in result["results"].values():
        assert arr.shape == (5,) and np.all(np.isnan(arr) | (arr >= 0))


def test_cross_pcf_pair_values(cross_pcf_pair):
    result = cross_pcf_pair
    assert result["cell_types"] == ["CD14+ Monocyte", "CD19+ B"]
    assert len(result["results"]) == 2
    np.testing.assert_almost_equal(
        result["results"][("CD14+ Monocyte", "CD19+ B")].sum(), 5.541360, decimal=3
    )


def test_cross_pcf_inplace(adata_copy):
    cross_pcf(adata_copy, groupby="cell_type", key_added="cross_pcf_test", inplace=True, **_KWARGS)
    assert "cross_pcf_test" in adata_copy.uns
    assert set(adata_copy.uns["cross_pcf_test"].keys()) == {"cell_types", "radii", "results"}


def test_cross_pcf_min_cells(adata):
    # min_cells=None derives an abundance-relative threshold from slide composition
    default = cross_pcf(adata, groupby="cell_type", min_cells=None, inplace=False, **_KWARGS)
    # a high explicit min_cells drops most/all cell types
    strict = cross_pcf(adata, groupby="cell_type", min_cells=200, inplace=False, **_KWARGS)
    assert len(strict["cell_types"]) < len(default["cell_types"])


def test_extend_first_annulus_integration(adata):
    cp_t = cross_pcf(
        adata, groupby="cell_type",
        cell_types=["CD14+ Monocyte", "CD19+ B"], inplace=False, **_KWARGS,
    )
    cp_f = cross_pcf(
        adata, groupby="cell_type",
        cell_types=["CD14+ Monocyte", "CD19+ B"],
        extend_first_annulus=False, inplace=False, **_KWARGS,
    )
    np.testing.assert_almost_equal(cp_f["radii"], [20, 40, 60, 80, 100])
    # merging only changes the first bin; bins beyond the first are identical
    pair = ("CD14+ Monocyte", "CD19+ B")
    np.testing.assert_array_equal(cp_t["results"][pair][1:], cp_f["results"][pair][1:])


# ── LRIC — agnostic mode ──────────────────────────────────────────────────────


def test_lric_agnostic(lric_agnostic):
    assert set(lric_agnostic.keys()) == {"pair_names", "radii", "lric"}
    assert lric_agnostic["lric"].shape == (5, 5)
    assert np.all(lric_agnostic["lric"] >= 0)
    assert lric_agnostic["pair_names"] == [
        "C1QB^PPA1", "DHRS4L2^GNG7", "NDUFA11^SUPT4H1", "SFPQ^C20orf27", "PGAM1^WBP11"
    ]


def test_lric_agnostic_reduces_to_cross_pcf(adata):
    """Docstring claim: agnostic LRIC reduces exactly to CrossPCF's directed
    curve when ligand/receptor weights are one-hot cell-type indicators.

    Both bin on the same half-open `[inner, outer)` tiles, so this holds even on
    the integer toy lattice, where distances land exactly on bin edges.
    """
    sender_type, receiver_type = "CD14+ Monocyte", "CD19+ B"
    ind = pd.DataFrame(
        {
            "ind_send": (adata.obs["cell_type"] == sender_type).astype(float).values,
            "ind_recv": (adata.obs["cell_type"] == receiver_type).astype(float).values,
        },
        index=adata.obs_names,
    )
    indicator_adata = AnnData(ind.values, obs=adata.obs, var=pd.DataFrame(index=ind.columns))
    indicator_adata.obsm["spatial"] = adata.obsm["spatial"]

    ind_resource = pd.DataFrame({"ligand": ["ind_send"], "receptor": ["ind_recv"]})
    agnostic = lric(
        indicator_adata, resource=ind_resource, transform_fn=lambda x: x,
        complex_sep=None, use_raw=False, inplace=False, **_KWARGS,
    )
    cp = cross_pcf(adata, groupby="cell_type", min_cells=1, inplace=False, **_KWARGS)

    np.testing.assert_array_almost_equal(
        agnostic["lric"][:, 0], cp["results"][(sender_type, receiver_type)], decimal=6
    )


def test_lric_agnostic_expr_prop(adata, resource, lric_agnostic):
    n = adata.n_obs
    base = lric_agnostic

    result = lric(adata, resource=resource, expr_prop=1.1, inplace=False, **_KWARGS)
    assert np.all(np.isnan(result["lric"])), "all pairs NaN when threshold exceeds any possible proportion"

    result0 = lric(adata, resource=resource, expr_prop=0, inplace=False, **_KWARGS)
    np.testing.assert_array_equal(result0["lric"], base["lric"])  # default = no-op

    # partial threshold (equivalent to the old min_expressing=100 out of `n` cells):
    # masking is per-pair and leaves kept pairs untouched
    partial = lric(adata, resource=resource, expr_prop=100 / n, inplace=False, **_KWARGS)["lric"]
    masked = np.isnan(partial).all(axis=0)
    assert masked.any() and not masked.all(), "expected a mix of masked and kept pairs"
    np.testing.assert_array_equal(partial[:, ~masked], base["lric"][:, ~masked])


def test_lric_lr_sep(adata, resource, lric_agnostic):
    default = lric_agnostic
    custom = lric(adata, resource=resource, lr_sep="|", inplace=False, **_KWARGS)

    assert all("^" in n for n in default["pair_names"])
    assert all("|" in n and "^" not in n for n in custom["pair_names"])
    np.testing.assert_array_equal(custom["lric"], default["lric"])


def test_lric_agnostic_transform_fn(adata, resource, lric_agnostic):
    base = lric_agnostic["lric"]

    # a genuinely nonlinear transform changes the result
    nonlinear = lric(adata, resource=resource, transform_fn=np.sqrt, inplace=False, **_KWARGS)["lric"]
    assert not np.allclose(nonlinear, base, equal_nan=True)

    # a pure per-gene rescaling (identity, skipping the default mean-normalisation)
    # cancels exactly in the closed-form ratio, so the result is scale-invariant
    identity = lric(adata, resource=resource, transform_fn=lambda x: x, inplace=False, **_KWARGS)["lric"]
    np.testing.assert_array_almost_equal(identity, base, decimal=4)


def test_lric_agnostic_inplace(adata_copy, resource):
    lric(adata_copy, resource=resource, key_added="lric_test", inplace=True, **_KWARGS)
    assert "lric_test" in adata_copy.uns
    assert set(adata_copy.uns["lric_test"].keys()) == {"pair_names", "radii", "lric"}


# ── LRIC — pairwise mode ──────────────────────────────────────────────────────


def test_lric_pairwise(lric_pairwise):
    result = lric_pairwise
    assert set(result.keys()) == {"cell_types", "pair_names", "radii", "results", "g_expr", "g_pcf"}
    n_ct = len(result["cell_types"])
    assert len(result["results"]) == n_ct * (n_ct - 1)
    for arr in result["results"].values():
        assert arr.shape == (5, 5) and np.all(np.isnan(arr) | (arr >= 0))


def test_lric_pairwise_g_pcf_matches_cross_pcf(adata, resource):
    """`g_pcf` is architecture-alone and should equal CrossPCF exactly for the
    same directed pair (no dependence on ligand/receptor expression weights)."""
    pair = ("CD14+ Monocyte", "CD19+ B")
    lric_pw = lric(
        adata, resource=resource, groupby="cell_type",
        cell_types=list(pair), inplace=False, **_KWARGS,
    )
    cp = cross_pcf(adata, groupby="cell_type", cell_types=list(pair), inplace=False, **_KWARGS)
    np.testing.assert_array_almost_equal(lric_pw["g_pcf"][pair], cp["results"][pair], decimal=12)


def test_lric_pairwise_results_equals_g_pcf_times_g_expr(lric_pairwise):
    """`results` (architecture x expression coupling) decomposes exactly into
    `g_pcf` (architecture alone) times `g_expr` (expression coupling alone),
    since `expected == exp_T * pair_prod` in `LRIC._pairwise`.

    One floating-point wrinkle: when a bin has zero observed sender->receiver
    edges (`T_SR == 0`) but a nonzero null expectation, `results` is a
    well-defined 0 (`0 / expected`), while `g_expr` is a literal `0/0` NaN
    (`Num_SR / (T_SR * pair_prod)`) -- so `g_pcf * g_expr` is `0 * nan == nan`
    there instead of 0. That is a removable singularity in the decomposition,
    not a disagreement, so it is excluded from the elementwise comparison.
    """
    result = lric_pairwise
    for pair, mat in result["results"].items():
        g_pcf, g_expr = result["g_pcf"][pair], result["g_expr"][pair]
        recon = g_pcf[:, None] * g_expr
        removable_singularity = np.isnan(g_expr) & ~np.isnan(mat)
        assert np.all(mat[removable_singularity] == 0.0)
        keep = ~removable_singularity
        np.testing.assert_array_almost_equal(mat[keep], recon[keep], decimal=3)


def test_lric_pairwise_cell_types_and_min_cells(adata, resource, lric_pairwise):
    result_sub = lric(
        adata, resource=resource, groupby="cell_type",
        cell_types=["CD14+ Monocyte", "CD19+ B", "CD56+ NK"],
        min_cells=5, inplace=False, **_KWARGS,
    )
    assert set(result_sub["cell_types"]) == {"CD14+ Monocyte", "CD19+ B", "CD56+ NK"}

    # lric_pairwise omits min_cells, so it is the min_cells=None (default-threshold) baseline
    default = lric_pairwise
    strict = lric(adata, resource=resource, groupby="cell_type", min_cells=200, inplace=False, **_KWARGS)
    assert len(strict["cell_types"]) < len(default["cell_types"])


def test_lric_pairwise_groupby_pairs(adata, resource):
    sender, receiver = "CD14+ Monocyte", "CD19+ B"
    groupby_pairs = pd.DataFrame({"source": [sender], "target": [receiver]})

    result = lric(
        adata, resource=resource, groupby="cell_type",
        groupby_pairs=groupby_pairs, min_cells=5, inplace=False, **_KWARGS,
    )
    # only the requested directed pair is computed, not the reverse
    assert set(result["results"].keys()) == {(sender, receiver)}

    # cell types referenced by `groupby_pairs` are folded into the retained population
    # even though they were not explicitly passed via `cell_types`
    assert {sender, receiver}.issubset(set(result["cell_types"]))

    # same population scope (same normalisation baseline) via explicit `cell_types`,
    # but without `groupby_pairs`, computes both directed pairs among the two types
    both_dirs = lric(
        adata, resource=resource, groupby="cell_type",
        cell_types=[sender, receiver], min_cells=5, inplace=False, **_KWARGS,
    )
    assert set(both_dirs["results"].keys()) == {(sender, receiver), (receiver, sender)}
    np.testing.assert_array_almost_equal(
        result["results"][(sender, receiver)], both_dirs["results"][(sender, receiver)], decimal=4
    )


def test_lric_pairwise_inplace(adata_copy, resource):
    lric(
        adata_copy, resource=resource, groupby="cell_type", min_cells=5,
        key_added="lric_pairwise_test", inplace=True, **_KWARGS,
    )
    assert "lric_pairwise_test" in adata_copy.uns
    assert set(adata_copy.uns["lric_pairwise_test"].keys()) == {
        "cell_types", "pair_names", "radii", "results", "g_expr", "g_pcf"
    }


def test_lric_pairwise_expr_prop(adata, resource):
    cell_types = ["CD14+ Monocyte", "CD19+ B"]
    result = lric(
        adata, resource=resource, groupby="cell_type",
        cell_types=cell_types, min_cells=5, expr_prop=1.1, inplace=False, **_KWARGS,
    )
    for arr in result["results"].values():
        assert np.all(np.isnan(arr)), "all pairs should be NaN when expr_prop exceeds any possible proportion"

    result_partial = lric(
        adata, resource=resource, groupby="cell_type",
        cell_types=cell_types, min_cells=5, expr_prop=0.01, inplace=False, **_KWARGS,
    )
    for arr in result_partial["results"].values():
        assert arr.shape == (5, 5)


# ── regression: numerator and denominator must bin pairs identically ──────────


def _constant_expression_adata(coords, cell_types=None):
    """An AnnData whose every cell expresses the single L and R at exactly 1.

    With position-independent weights the expression term of `g(r)` is
    identically 1, so LRIC must return exactly 1 in every bin: the numerator and
    the denominator are then counting the very same pairs. Any deviation is a
    binning inconsistency between the two sides.
    """
    a = AnnData(
        np.ones((len(coords), 2), dtype=np.float32), var=pd.DataFrame(index=["L", "R"])
    )
    a.obsm["spatial"] = np.asarray(coords, dtype=float)
    if cell_types is not None:
        a.obs["ct"] = pd.Categorical(cell_types)
    return a


_CONST_RESOURCE = pd.DataFrame({"ligand": ["L"], "receptor": ["R"]})
_CONST_KWARGS = {
    "resource": _CONST_RESOURCE, "max_radius": 100, "radius_step": 20,
    "use_raw": False, "verbose": False, "inplace": False,
}
_RNG = np.random.default_rng(0)
_CONT_COORDS = _RNG.uniform(0, 500, (400, 2))
# a 20-unit lattice, i.e. Visium-like: whole distance shells land exactly on bin
# edges, which used to swing `g` between ~0.67 and ~1.25
_GRID_COORDS = np.stack(
    np.meshgrid(np.arange(20) * 20.0, np.arange(20) * 20.0), axis=-1
).reshape(-1, 2)
# coincident cells: distance-0 pairs between *distinct* cells, which used to
# enter the numerator but not the denominator and so inflated the contact bin
_DUP_COORDS = np.repeat(_RNG.uniform(0, 500, (200, 2)), 2, axis=0)


@pytest.mark.parametrize(
    ("coords", "annulus_steps", "extend_first"),
    [
        (_CONT_COORDS, 1, True),   # continuous coordinates, disjoint annuli
        (_CONT_COORDS, 2, True),   # overlapping annuli used to bias `g` by ~1/k
        (_CONT_COORDS, 1, False),  # unmerged first annulus (the other `_roll_tiles` window)
        (_GRID_COORDS, 1, True),   # gridded coordinates: distances exactly on bin edges
        (_GRID_COORDS, 2, True),
        (_DUP_COORDS, 1, True),    # duplicated coordinates (distance-0 distinct pairs)
    ],
    ids=["continuous-k1", "continuous-k2", "continuous-k1-nomerge",
         "grid-k1", "grid-k2", "duplicated-k1"],
)
def test_lric_agnostic_constant_expression_is_one(coords, annulus_steps, extend_first):
    result = lric(
        _constant_expression_adata(coords), annulus_steps=annulus_steps,
        extend_first_annulus=extend_first, **_CONST_KWARGS
    )["lric"][:, 0]
    keep = ~np.isnan(result)
    assert keep.any()
    np.testing.assert_allclose(result[keep], 1.0, atol=1e-9)


def test_lric_pairwise_constant_expression_is_one():
    """Pairwise counterpart: with constant weights the expression term `g_expr` is
    exactly 1 and `results` collapses onto the pure-architecture `g_pcf`.

    `g_pcf` itself is a finite-sample statistic of the (random) labelling rather
    than an identity, so it only sits *near* 1 -- it is the exactness of `g_expr`
    that pins the numerator and denominator to the same binning.
    """
    cell_types = np.array(["A", "B"] * (len(_GRID_COORDS) // 2))
    _RNG.shuffle(cell_types)
    result = lric(
        _constant_expression_adata(_GRID_COORDS, cell_types), groupby="ct", **_CONST_KWARGS
    )
    assert len(result["results"]) == 2
    for pair, mat in result["results"].items():
        keep = ~np.isnan(mat[:, 0])
        assert keep.any()
        np.testing.assert_allclose(result["g_expr"][pair][keep, 0], 1.0, atol=1e-9)
        np.testing.assert_allclose(mat[keep, 0], result["g_pcf"][pair][keep], rtol=1e-5)
        np.testing.assert_allclose(result["g_pcf"][pair][keep], 1.0, atol=0.1)


# ── error cases ───────────────────────────────────────────────────────────────


def test_lric_no_lr_pairs_raises(adata):
    # a resource with none of its genes in `adata.var_names` is caught by the
    # shared `assert_covered` check (same as `_inflow`/`_spatial_bivariate`),
    # before the LRIC-specific "no LR pairs" check is ever reached.
    bad = pd.DataFrame({"ligand": ["NOTEXIST1"], "receptor": ["NOTEXIST2"]})
    with raises(ValueError, match="Please check if appropriate organism/ID type"):
        lric(adata, resource=bad, inplace=False, verbose=False)
    with raises(ValueError, match="Please check if appropriate organism/ID type"):
        lric(adata, resource=bad, groupby="cell_type", inplace=False, verbose=False)


# ── regression: adata passed by the caller must never be mutated ──────────────


def test_lric_does_not_mutate_input_view_on_complex_resource(adata):
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

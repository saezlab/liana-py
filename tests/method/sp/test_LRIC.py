import anndata as ad
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from pytest import raises
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

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

_CROSS_PCF_COLS = ["source", "target", "interaction", "radius", "g"]
_LRIC_AG_COLS = ["ligand_complex", "receptor_complex", "interaction", "radius", "g"]
_LRIC_CT_COLS = [
    "source", "target", "ligand_complex", "receptor_complex",
    "interaction", "radius", "g", "g_expr", "g_pcf",
]


def _curve(df, col="g", **sel):
    """The `col` values of the rows matching `sel`, ordered by radius."""
    for k, v in sel.items():
        df = df[df[k] == v]
    return df.sort_values("radius")[col].to_numpy()


def _cell_types(df):
    """The retained cell types, in the order the method sorted them."""
    return list(df["source"].cat.categories)


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
    assert list(result.columns) == _CROSS_PCF_COLS
    np.testing.assert_almost_equal(np.unique(result["radius"]), [0, 40, 60, 80, 100])
    n_ct = len(_cell_types(result))
    # symmetric g(r) -> one row per *unordered* pair x radius bin
    assert len(result) == n_ct * (n_ct - 1) / 2 * 5
    assert np.all(np.isnan(result["g"]) | (result["g"] >= 0))


def test_cross_pcf_pair_values(cross_pcf_pair):
    result = cross_pcf_pair
    assert _cell_types(result) == ["CD14+ Monocyte", "CD19+ B"]
    assert len(result) == 5  # the single unordered pair, once
    np.testing.assert_almost_equal(
        _curve(result, source="CD14+ Monocyte", target="CD19+ B").sum(), 5.541360, decimal=3
    )


@pytest.mark.parametrize("annulus_steps", [1, 2])
def test_cross_pcf_matches_brute_force(adata, annulus_steps):
    """Every `g` against an O(n^2) reference that shares no code with the method.

    The other tests pin `g` to liana's own code paths, so a wrong null (say `N**2`
    instead of `N*(N-1)`) would agree with itself. Here the counts and the null are
    recomputed from the distance matrix: annulus `b` spans
    ``[inner, radius_step * (b + 1 + annulus_steps))``, the first reaching back to 0.
    """
    step = _KWARGS["radius_step"]
    result = cross_pcf(adata, groupby="cell_type", annulus_steps=annulus_steps,
                       inplace=False, **_KWARGS)
    dist = squareform(pdist(np.asarray(adata.obsm["spatial"], dtype=float)))
    np.fill_diagonal(dist, np.inf)  # ordered pairs, self excluded
    types = adata.obs["cell_type"].to_numpy()
    n_cells = len(types)
    outer = {r: step * (b + 1 + annulus_steps)
             for b, r in enumerate(np.sort(result["radius"].unique()))}

    for source, target, radius, g in zip(result["source"], result["target"],
                                         result["radius"], result["g"]):
        in_annulus = (dist >= radius) & (dist < outer[radius])
        is_source, is_target = types == source, types == target
        expected = (is_source.sum() * is_target.sum() * in_annulus.sum()
                    / (n_cells * (n_cells - 1)))
        observed = in_annulus[np.ix_(is_source, is_target)].sum()
        np.testing.assert_allclose(g, observed / expected, rtol=1e-5, atol=1e-6)


def test_cross_pcf_inplace(adata_copy):
    cross_pcf(adata_copy, groupby="cell_type", key_added="cross_pcf_test", inplace=True, **_KWARGS)
    assert "cross_pcf_test" in adata_copy.uns
    assert list(adata_copy.uns["cross_pcf_test"].columns) == _CROSS_PCF_COLS


def test_cross_pcf_min_cells(adata):
    # min_cells=None derives an abundance-relative threshold from slide composition
    default = cross_pcf(adata, groupby="cell_type", min_cells=None, inplace=False, **_KWARGS)
    # a high explicit min_cells drops most/all cell types
    strict = cross_pcf(adata, groupby="cell_type", min_cells=200, inplace=False, **_KWARGS)
    assert len(_cell_types(strict)) < len(_cell_types(default))


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
    np.testing.assert_almost_equal(np.unique(cp_f["radius"]), [20, 40, 60, 80, 100])
    # merging only changes the first bin; bins beyond the first are identical
    np.testing.assert_array_equal(_curve(cp_t)[1:], _curve(cp_f)[1:])


# ── LRIC — agnostic mode ──────────────────────────────────────────────────────


def test_lric_agnostic(lric_agnostic):
    assert list(lric_agnostic.columns) == _LRIC_AG_COLS
    assert len(lric_agnostic) == 5 * 5  # 5 LR pairs x 5 radius bins
    assert np.all(lric_agnostic["g"] >= 0)
    assert lric_agnostic["interaction"].drop_duplicates().tolist() == [
        "C1QB^PPA1", "DHRS4L2^GNG7", "NDUFA11^SUPT4H1", "SFPQ^C20orf27", "PGAM1^WBP11"
    ]
    # the complexes are the split of `interaction`, matching liana's sc results
    assert lric_agnostic["ligand_complex"].astype(str).tolist() == [
        i.split("^")[0] for i in lric_agnostic["interaction"].astype(str)
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
        _curve(agnostic, interaction="ind_send^ind_recv"),
        _curve(cp, source=sender_type, target=receiver_type),
        decimal=6,
    )


def test_lric_agnostic_expr_prop(adata, resource, lric_agnostic):
    n = adata.n_obs
    base = lric_agnostic

    result = lric(adata, resource=resource, expr_prop=1.1, inplace=False, **_KWARGS)
    assert result["g"].isna().all(), "all pairs NaN when threshold exceeds any possible proportion"

    result0 = lric(adata, resource=resource, expr_prop=0, inplace=False, **_KWARGS)
    np.testing.assert_array_equal(result0["g"], base["g"])  # default = no-op

    # partial threshold (equivalent to the old min_expressing=100 out of `n` cells):
    # masking is per-pair and leaves kept pairs untouched
    partial = lric(adata, resource=resource, expr_prop=100 / n, inplace=False, **_KWARGS)
    masked = partial.groupby("interaction", observed=True)["g"].apply(lambda s: s.isna().all())
    assert masked.any() and not masked.all(), "expected a mix of masked and kept pairs"
    keep = ~partial["interaction"].isin(masked.index[masked])
    np.testing.assert_array_equal(partial["g"][keep], base["g"][keep])


def test_lric_lr_sep(adata, resource, lric_agnostic):
    default = lric_agnostic
    custom = lric(adata, resource=resource, lr_sep="|", inplace=False, **_KWARGS)

    assert all("^" in n for n in default["interaction"].astype(str))
    assert all("|" in n and "^" not in n for n in custom["interaction"].astype(str))
    np.testing.assert_array_equal(custom["g"], default["g"])
    # the complex columns are unaffected by the separator
    np.testing.assert_array_equal(custom["ligand_complex"], default["ligand_complex"])


def test_lric_agnostic_transform_fn(adata, resource, lric_agnostic):
    base = lric_agnostic["g"].to_numpy()

    # a genuinely nonlinear transform changes the result
    nonlinear = lric(adata, resource=resource, transform_fn=np.sqrt, inplace=False, **_KWARGS)["g"]
    assert not np.allclose(nonlinear, base, equal_nan=True)

    # a pure per-gene rescaling (identity, skipping the default mean-normalisation)
    # cancels exactly in the closed-form ratio, so the result is scale-invariant
    identity = lric(adata, resource=resource, transform_fn=lambda x: x, inplace=False, **_KWARGS)["g"]
    np.testing.assert_array_almost_equal(identity, base, decimal=4)


def test_lric_agnostic_inplace(adata_copy, resource):
    lric(adata_copy, resource=resource, key_added="lric_test", inplace=True, **_KWARGS)
    assert "lric_test" in adata_copy.uns
    assert list(adata_copy.uns["lric_test"].columns) == _LRIC_AG_COLS


# ── LRIC — pairwise mode ──────────────────────────────────────────────────────
def test_lric_pairwise(lric_pairwise):
    result = lric_pairwise
    assert list(result.columns) == _LRIC_CT_COLS
    n_ct = len(_cell_types(result))
    # all directed cell-type pairs x 5 LR pairs x 5 radius bins
    assert len(result) == n_ct * (n_ct - 1) * 5 * 5
    assert np.all(np.isnan(result["g"]) | (result["g"] >= 0))


def test_lric_pairwise_g_pcf_matches_cross_pcf(adata, resource):
    """`g_pcf` is architecture-alone and should equal CrossPCF exactly for the
    same directed pair (no dependence on ligand/receptor expression weights)."""
    source, target = "CD14+ Monocyte", "CD19+ B"
    lric_pw = lric(
        adata, resource=resource, groupby="cell_type",
        cell_types=[source, target], inplace=False, **_KWARGS,
    )
    cp = cross_pcf(
        adata, groupby="cell_type", cell_types=[source, target], inplace=False, **_KWARGS
    )
    # g_pcf is shared across LR pairs -> one value per radius bin
    g_pcf = lric_pw[(lric_pw["source"] == source) & (lric_pw["target"] == target)]
    g_pcf = g_pcf.drop_duplicates("radius").sort_values("radius")["g_pcf"].to_numpy()
    np.testing.assert_array_almost_equal(
        g_pcf, _curve(cp, source=source, target=target), decimal=6
    )


def test_lric_pairwise_results_equals_g_pcf_times_g_expr(lric_pairwise):
    result = lric_pairwise
    mat, g_pcf, g_expr = (result[c].to_numpy() for c in ("g", "g_pcf", "g_expr"))
    removable_singularity = np.isnan(g_expr) & ~np.isnan(mat)
    assert np.all(mat[removable_singularity] == 0.0)
    keep = ~removable_singularity
    # relative, not absolute: the columns are stored float32 while the identity holds
    # in float64, so the error tracks |g| (~1e-7 relative) rather than any fixed decimal
    np.testing.assert_allclose(mat[keep], (g_pcf * g_expr)[keep], rtol=1e-5, atol=1e-6)


def test_lric_pairwise_cell_types_and_min_cells(adata, resource, lric_pairwise):
    result_sub = lric(
        adata, resource=resource, groupby="cell_type",
        cell_types=["CD14+ Monocyte", "CD19+ B", "CD56+ NK"],
        min_cells=5, inplace=False, **_KWARGS,
    )
    assert set(_cell_types(result_sub)) == {"CD14+ Monocyte", "CD19+ B", "CD56+ NK"}

    # lric_pairwise omits min_cells, so it is the min_cells=None (default-threshold) baseline
    default = lric_pairwise
    strict = lric(adata, resource=resource, groupby="cell_type", min_cells=200, inplace=False, **_KWARGS)
    assert len(_cell_types(strict)) < len(_cell_types(default))


def _directed_pairs(df):
    return set(map(tuple, df[["source", "target"]].astype(str).drop_duplicates().to_numpy()))


def test_lric_pairwise_groupby_pairs(adata, resource):
    sender, receiver = "CD14+ Monocyte", "CD19+ B"
    groupby_pairs = pd.DataFrame({"source": [sender], "target": [receiver]})

    result = lric(
        adata, resource=resource, groupby="cell_type",
        groupby_pairs=groupby_pairs, min_cells=5, inplace=False, **_KWARGS,
    )
    # only the requested directed pair is computed, not the reverse
    assert _directed_pairs(result) == {(sender, receiver)}

    # cell types referenced by `groupby_pairs` are folded into the retained population
    # even though they were not explicitly passed via `cell_types`
    assert {sender, receiver}.issubset(set(_cell_types(result)))

    # same population scope (same normalisation baseline) via explicit `cell_types`,
    # but without `groupby_pairs`, computes both directed pairs among the two types
    both_dirs = lric(
        adata, resource=resource, groupby="cell_type",
        cell_types=[sender, receiver], min_cells=5, inplace=False, **_KWARGS,
    )
    assert _directed_pairs(both_dirs) == {(sender, receiver), (receiver, sender)}
    np.testing.assert_array_almost_equal(
        result["g"],
        both_dirs[(both_dirs["source"] == sender) & (both_dirs["target"] == receiver)]["g"],
        decimal=4,
    )


def test_lric_pairwise_inplace(adata_copy, resource):
    lric(
        adata_copy, resource=resource, groupby="cell_type", min_cells=5,
        key_added="lric_pairwise_test", inplace=True, **_KWARGS,
    )
    assert "lric_pairwise_test" in adata_copy.uns
    assert list(adata_copy.uns["lric_pairwise_test"].columns) == _LRIC_CT_COLS


def test_lric_pairwise_expr_prop(adata, resource):
    cell_types = ["CD14+ Monocyte", "CD19+ B"]
    result = lric(
        adata, resource=resource, groupby="cell_type",
        cell_types=cell_types, min_cells=5, expr_prop=1.1, inplace=False, **_KWARGS,
    )
    # masked rows are kept, as NaN
    assert len(result) == 2 * 5 * 5
    assert result["g"].isna().all(), "all pairs should be NaN when expr_prop exceeds any possible proportion"

    result_partial = lric(
        adata, resource=resource, groupby="cell_type",
        cell_types=cell_types, min_cells=5, expr_prop=0.01, inplace=False, **_KWARGS,
    )
    assert len(result_partial) == 2 * 5 * 5

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
    )["g"].to_numpy()
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
    assert _directed_pairs(result) == {("A", "B"), ("B", "A")}
    mat, g_pcf, g_expr = (result[c].to_numpy() for c in ("g", "g_pcf", "g_expr"))
    keep = ~np.isnan(mat)
    assert keep.any()
    np.testing.assert_allclose(g_expr[keep], 1.0, atol=1e-9)
    np.testing.assert_allclose(mat[keep], g_pcf[keep], rtol=1e-5)
    np.testing.assert_allclose(g_pcf[keep], 1.0, atol=0.1)

def test_lric_no_lr_pairs_raises(adata):
    # a resource with none of its genes in `adata.var_names` is caught by the
    # shared `assert_covered` check (same as `_inflow`/`_spatial_bivariate`),
    # before the LRIC-specific "no LR pairs" check is ever reached.
    bad = pd.DataFrame({"ligand": ["NOTEXIST1"], "receptor": ["NOTEXIST2"]})
    with raises(ValueError, match="Please check if appropriate organism/ID type"):
        lric(adata, resource=bad, inplace=False, verbose=False)
    with raises(ValueError, match="Please check if appropriate organism/ID type"):
        lric(adata, resource=bad, groupby="cell_type", inplace=False, verbose=False)


import os
import pathlib

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy import sparse

from liana.method.sp._misty._misty_constructs import lrMistyDataByCellType
from liana.method.sp._misty._single_view_models import LinearModel

test_path = pathlib.Path(__file__).parent

def _resource():
    return pd.DataFrame(
        {
            "ligand": ["ligA", "ligB", "ligC", "ligD"],
            "receptor": ["protE", "protF", "protE", "protF"],
        }
    )


def _adata():
    adata = sc.read_h5ad(os.path.join(test_path, "data", "synthetic.h5ad"))
    adata = sc.pp.subsample(adata, n_obs=300, copy=True)

    # Keep the on-disk synthetic data unchanged, but split part of the B
    # population into a third cell type for the cell-type-specific tests.
    cell_types = adata.obs["cell_type"].astype(str).to_numpy()
    b_cells = np.flatnonzero(cell_types == "B")
    cell_types[b_cells[: len(b_cells) // 2]] = "C"
    adata.obs["cell_type"] = pd.Categorical(cell_types)

    return adata


def _dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def _target_specific_adata():
    adata = _adata()
    labels = adata.obs["cell_type"].to_numpy()
    X = _dense(adata.X).astype(np.float32, copy=True)

    ligand_indices = {
        ligand: adata.var_names.get_loc(ligand)
        for ligand in ("ligA", "ligB", "ligC", "ligD")
    }
    for index in ligand_indices.values():
        X[:, index] = 0

    # Make ligA available only in B and ligB available only in C. A positive
    # nz_threshold then removes the non-relevant ligand from each sender view.
    X[labels == "B", ligand_indices["ligA"]] = 1
    X[labels == "C", ligand_indices["ligB"]] = 1

    rng = np.random.default_rng(42)
    receiver_mask = labels == "A"
    X[receiver_mask, adata.var_names.get_loc("protE")] = rng.normal(
        size=receiver_mask.sum()
    )
    X[receiver_mask, adata.var_names.get_loc("protF")] = rng.normal(
        size=receiver_mask.sum()
    )
    adata.X = X

    return adata


def test_lr_misty_by_cell_type_views():
    adata = _adata()

    misty = lrMistyDataByCellType(
        adata=adata,
        resource=_resource(),
        receiver_celltype="A",
        celltype_key="cell_type",
        nz_threshold=0,
        bandwidth=10,
        cutoff=0,
    )

    assert list(misty.mod.keys()) == ["intra", "extra_A", "extra_B", "extra_C"]
    assert "_misty_ligands_by_receptor" in misty.uns
    assert "_misty_resource" not in misty.uns
    assert misty.mod["intra"].n_obs == adata.n_obs
    assert misty.mod["extra_A"].n_obs == adata.n_obs
    assert misty.mod["extra_B"].n_obs == adata.n_obs
    assert misty.mod["extra_C"].n_obs == adata.n_obs
    assert misty.mod["intra"].obs["_misty_receiver"].sum() > 0

    labels = adata.obs["cell_type"].to_numpy()

    for sender in ("A","B", "C"):
        extra = _dense(misty.mod[f"extra_{sender}"].X)
        sender_mask = labels == sender
        assert np.allclose(extra[~sender_mask], 0)


def test_lr_misty_by_cell_type_filters_predictors_by_resource():
    misty = lrMistyDataByCellType(
        adata=_adata(),
        resource=_resource(),
        receiver_celltype="A",
        celltype_key="cell_type",
        nz_threshold=0,
        bandwidth=10,
        cutoff=0,
    )

    misty(
        model=LinearModel,
        maskby="_misty_receiver",
        k_cv=3,
    )

    interactions = misty.uns["interactions"]
    extra_interactions = interactions[
        interactions["view"] == "extra_B"
    ]

    allowed = {
        ("protE", "ligA"),
        ("protE", "ligC"),
        ("protF", "ligB"),
        ("protF", "ligD"),
    }

    observed = set(
        zip(
            extra_interactions["target"],
            extra_interactions["predictor"],
            strict=True,
        )
    )

    assert observed <= allowed


def test_lr_misty_by_cell_type_can_include_receiver_extra_view():
    misty = lrMistyDataByCellType(
        adata=_adata(),
        resource=_resource(),
        receiver_celltype="A",
        celltype_key="cell_type",
        include_receiver_extra=True,
        nz_threshold=0,
        bandwidth=10,
        cutoff=0,
    )

    assert list(misty.mod.keys()) == [
        "intra",
        "extra_A",
        "extra_B",
        "extra_C",
    ]


def test_lr_misty_by_cell_type_defaults_to_receiver_mask():
    misty = lrMistyDataByCellType(
        adata=_adata(),
        resource=_resource(),
        receiver_celltype="A",
        celltype_key="cell_type",
        nz_threshold=0,
        bandwidth=10,
        cutoff=0,
    )

    # The constructor-specific receiver mask should be used automatically.
    misty(model=LinearModel, k_cv=3)

    assert "target_metrics" in misty.uns
    assert "interactions" in misty.uns
    assert set(misty.uns["target_metrics"]["receiver_celltype"]) == {"A"}
    assert set(misty.uns["interactions"]["receiver_celltype"]) == {"A"}
    assert set(
        misty.uns["interactions"]["sender_celltype"].dropna()
    ) == {"A", "B", "C"}


def test_lr_misty_by_cell_type_rejects_a_different_mask():
    misty = lrMistyDataByCellType(
        adata=_adata(),
        resource=_resource(),
        receiver_celltype="A",
        celltype_key="cell_type",
        nz_threshold=0,
        bandwidth=10,
        cutoff=0,
    )

    with pytest.raises(
        ValueError,
        match="requires maskby='_misty_receiver'",
    ):
        misty(model=LinearModel, maskby="cell_type", k_cv=3)


def test_lr_misty_by_cell_type_uses_target_specific_sender_views():
    resource = pd.DataFrame(
        {
            "ligand": ["ligA", "ligB"],
            "receptor": ["protE", "protF"],
        }
    )
    misty = lrMistyDataByCellType(
        adata=_target_specific_adata(),
        resource=resource,
        receiver_celltype="A",
        celltype_key="cell_type",
        nz_threshold=0,
        bandwidth=10,
        cutoff=0,
    )

    # Make the active sender view differ by target. This isolates the
    # target_view_str bookkeeping in MistyData.__call__ from feature
    # filtering during view construction.
    misty.mod["extra_B"] = misty.mod["extra_B"][:, ["ligA"]].copy()
    misty.mod["extra_C"] = misty.mod["extra_C"][:, ["ligB"]].copy()

    assert misty.mod["extra_B"].var_names.tolist() == ["ligA"]
    assert misty.mod["extra_C"].var_names.tolist() == ["ligB"]

    misty(model=LinearModel, k_cv=3)
    target_metrics = misty.uns["target_metrics"].set_index("target")

    assert pd.notna(target_metrics.loc["protE", "extra_B"])
    assert pd.isna(target_metrics.loc["protE", "extra_C"])
    assert pd.isna(target_metrics.loc["protF", "extra_B"])
    assert pd.notna(target_metrics.loc["protF", "extra_C"])


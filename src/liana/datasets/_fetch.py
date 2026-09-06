from __future__ import annotations

from functools import cache
from importlib.resources import as_file, files
from typing import TYPE_CHECKING, cast

import scanpy as sc
from scverse_misc.datasets import fetch, parse_registry, register_loader

if TYPE_CHECKING:
    from pathlib import Path

    from anndata import AnnData
    from mudata import MuData
    from scverse_misc.datasets import DatasetEntry, DownloadCB


@register_loader("mudata")
def _load_mudata(entry: DatasetEntry, target: Path, download: DownloadCB, /, **kwargs: object) -> MuData:
    """Download the `.h5ad` file of each modality of `entry` and assemble them into a :class:`~mudata.MuData`.

    The modality names are taken from the entry's `modalities` metadata, which maps each modality to the file it is read from.
    """
    import anndata as ad
    from mudata import MuData

    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")

    modalities = cast("dict[str, str]", entry.metadata["modalities"])
    return MuData({mod: ad.read_h5ad(download(entry.file(name=filename))) for mod, filename in modalities.items()})


@cache
def _registry() -> tuple[str | None, dict[str, DatasetEntry]]:
    with as_file(files(__package__) / "registry.yaml") as path:
        return parse_registry(path)


def fetch_dataset(name: str) -> AnnData | MuData:
    """Download `name` into :attr:`scanpy.settings.datasetdir` if it is not cached there yet, and read it."""
    base_url, datasets = _registry()
    sc.settings.datasetdir.mkdir(parents=True, exist_ok=True)
    return fetch(datasets[name], sc.settings.datasetdir, base_url=base_url)

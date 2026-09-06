from urllib.parse import urlparse

import pytest

import liana as li
from liana.datasets._fetch import _registry


def test_registry_base_url() -> None:
    base_url, _ = _registry()
    assert base_url == "https://exampledata.scverse.org/liana"


@pytest.mark.parametrize("name", _registry()[1])
def test_registry_entry(name: str) -> None:
    base_url, datasets = _registry()
    entry = datasets[name]
    # every dataset is exposed under the function of the same name
    assert callable(getattr(li.ds, name))
    assert entry.type in {"anndata", "mudata"}
    assert entry.files
    for file in entry.files:
        # a sha256 is what lets a truncated download be detected instead of surfacing as a parse error
        assert file.sha256, f"{name}/{file.name} is missing a sha256"
        for url in [file.resolve_url(base_url), *(file.fallback_urls or [])]:
            assert urlparse(url).scheme in {"http", "https", "ftp"}
    if entry.type == "mudata":
        modalities = entry.metadata["modalities"]
        filenames = {file.name for file in entry.files}
        assert set(modalities.values()) == filenames

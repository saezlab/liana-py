"""Fixtures for the tests that need liana's external resources.

These are the only tests that reach the network, and they cache what they
download under ``tests/.cache`` so a full run pays for it once.
"""

import gzip
import os
import pathlib

import pytest


@pytest.fixture(scope="session")
def download_cache():
    """Directory the tests download their (large) external files to, once."""
    cache = pathlib.Path(__file__).parents[1] / ".cache"
    cache.mkdir(exist_ok=True)

    return cache


@pytest.fixture(scope="session")
def metalinks_db(download_cache):
    """Path to MetaLinksDB, downloaded on first use.

    ``_download_metalinksdb`` has no path argument and always writes to the
    working directory, hence the ``chdir``.
    """
    from liana.resource.get_metalinks import _download_metalinksdb

    cwd = os.getcwd()
    os.chdir(download_cache)
    try:
        return _download_metalinksdb(verbose=False)
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="session")
def hcop_file(download_cache):
    """Path the human-mouse HCOP table is cached at, downloaded on first use.

    ``get_hcop_orthologs`` does the downloading itself, and skips it when the
    file is already there -- so an interrupted download would otherwise be
    served from the cache forever. Drop it if it does not open.
    """
    path = download_cache / "human_mouse_hcop_fifteen_column.txt.gz"

    if path.exists():
        try:
            with gzip.open(path, "rb") as f:
                f.read(1024)
        except (OSError, EOFError):
            path.unlink()

    return str(path)

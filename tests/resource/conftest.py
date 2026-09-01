"""Fixtures for the tests that need liana's external resources.

These are the only tests that reach the network, and they cache what they
download under ``tests/.cache``.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import sqlite3

import pytest


def _readable_sqlite(path: pathlib.Path) -> bool:
    """Whether `path` is a non-empty SQLite database that passes a quick check.

    SQLite reads an empty file as a valid empty database, hence the size check.
    """
    if path.stat().st_size == 0:
        return False

    try:
        with contextlib.closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            return bool(conn.execute("pragma quick_check").fetchone()[0] == "ok")
    except sqlite3.Error:
        return False


@pytest.fixture(scope="session")
def download_cache() -> pathlib.Path:
    """Directory the tests download their (large) external files to, once."""
    cache = pathlib.Path(__file__).parents[1] / ".cache"
    cache.mkdir(exist_ok=True)

    return cache


@pytest.fixture(scope="session")
def metalinks_db(download_cache: pathlib.Path) -> str:
    """Path to MetaLinksDB, downloaded on first use.

    ``_download_metalinksdb`` has no path argument and always writes to the
    working directory, hence the ``chdir``. It only rejects an empty file, so
    a download cut off part-way would otherwise be cached forever.
    """
    from liana.resource.get_metalinks import _download_metalinksdb

    path = download_cache / "metalinksdb.db"
    if path.exists() and not _readable_sqlite(path):
        path.unlink()

    cwd = os.getcwd()
    os.chdir(download_cache)
    try:
        return _download_metalinksdb(verbose=False)
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="session")
def hcop_file(download_cache: pathlib.Path) -> str:
    """Path the human-mouse HCOP table is cached at, downloaded on first use."""
    from liana.resource import get_hcop_orthologs

    path = download_cache / "human_mouse_hcop_fifteen_column.txt.gz"

    if not path.exists():
        part = path.with_name(".part-" + path.name)
        part.unlink(missing_ok=True)
        try:
            get_hcop_orthologs(target_organism="mouse", filename=str(part), min_evidence=0)
            os.replace(part, path)
        finally:
            part.unlink(missing_ok=True)

    return str(path)

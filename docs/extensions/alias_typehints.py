"""Render the annotations `sphinx_autodoc_typehints` gets wrong.

A PEP 695 ``type`` alias is formatted as the class it is an instance of, so it renders as the bare text "TypeAliasType" (or "GenericAlias", when the alias is subscripted, as :data:`numpy.typing.NDArray` is) and the reference dangles.
"""

from dataclasses import dataclass
from types import GenericAlias
from typing import Any, TypeAliasType, get_args, get_origin

from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx_autodoc_typehints import format_annotation


@dataclass
class _AliasFormatter:
    """Formats the two cases above and defers everything else to `scanpydoc`.

    Recursing through :func:`~sphinx_autodoc_typehints.format_annotation` re-enters this hook, so nested aliases are handled too.
    """

    inner: Any

    def __call__(self, annotation: object, config: Config, **kwargs: object) -> str | None:
        if isinstance(annotation, TypeAliasType):
            return _alias(annotation, config)
        if isinstance(annotation, GenericAlias) and isinstance(alias := get_origin(annotation), TypeAliasType):
            # A subscripted alias, e.g. `NDArray[np.floating]`.
            args = ", ".join(format_annotation(a, config) for a in get_args(annotation))
            return rf"{_alias(alias, config)}\ \[{args}]"
        return None if self.inner is None else self.inner(annotation, config, **kwargs)

    def __getstate__(self) -> dict[str, Any]:
        """Sphinx pickles the config, and neither this nor what it wraps is needed once the doctrees are written."""
        return {}


def _alias(alias: TypeAliasType, config: Config) -> str:
    """Render one alias: liana's own are private, so they are expanded into what they stand for; anything else keeps its name.

    An alias' type parameters cannot be substituted through any public API, so a subscripted one has to keep its name regardless.
    `qualname_overrides` maps that name to wherever the alias is actually documented.
    """
    if alias.__module__.split(".")[0] == "liana":
        return format_annotation(alias.__value__, config)
    return rf":py:class:`~{alias.__module__}.{alias.__name__}`"


def setup(app: Sphinx) -> None:
    """`scanpydoc.elegant_typehints` overwrites ``typehints_formatter`` in its own setup, so wrap whatever it left behind."""
    app.config["typehints_formatter"] = _AliasFormatter(app.config["typehints_formatter"])

from typing import Any

from ._annulus import annulus
from ._circle_plot import circle
from ._connectivity_plot import connectivity
from ._dotplot import dotplot, dotplot_by_sample
from ._feature_by_group import feature_by_group
from ._lric_plot import lric_divergence, lric_lineplot
from ._misty_plots import misty_contributions, misty_interactions, misty_target_metrics
from ._tileplot import tileplot

__all__ = [
    "annulus",
    "circle",
    "connectivity",
    "dotplot",
    "dotplot_by_sample",
    "feature_by_group",
    "lric_divergence",
    "lric_lineplot",
    "misty_contributions",
    "misty_interactions",
    "misty_target_metrics",
    "tileplot",
]

_RENAMED = {
    "annulus_plot": "annulus",
    "circle_plot": "circle",
    "contributions": "misty_contributions",
    "interactions": "misty_interactions",
    "lric_divergence_plot": "lric_divergence",
    "target_metrics": "misty_target_metrics",
}
"""The 2.0 names, mapped onto the ones that follow the module's convention.

Plot functions are bare nouns, as in :mod:`scanpy.pl`, and carry the prefix of the method they belong to when they only apply to that method.
"""


def __getattr__(name: str) -> Any:
    if (renamed := _RENAMED.get(name)) is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from warnings import warn

    warn(f"`liana.pl.{name}` is deprecated; use `liana.pl.{renamed}` instead.", FutureWarning, stacklevel=2)

    return globals()[renamed]

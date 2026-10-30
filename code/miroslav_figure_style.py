"""Conservative Matplotlib baseline for publication-ready scientific figures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt


MM_TO_INCH = 1.0 / 25.4

OKABE_ITO = (
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
)


@dataclass(frozen=True)
class PublisherWidths:
    one_column_mm: float
    two_column_mm: float


WIDTHS = {
    "wiley-adom": PublisherWidths(85.0, 178.0),
    "springer": PublisherWidths(85.0, 170.0),
}


def set_publication_style() -> None:
    """Apply readable defaults intended for final-size journal figures."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "lines.markersize": 5.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def figure_size(
    publisher: Literal["wiley-adom", "springer"] = "wiley-adom",
    span: Literal["one", "two"] = "one",
    aspect: float = 0.75,
) -> tuple[float, float]:
    """Return (width, height) in inches for a final-size figure."""
    if aspect <= 0:
        raise ValueError("aspect must be positive")
    profile = WIDTHS[publisher]
    width_mm = profile.one_column_mm if span == "one" else profile.two_column_mm
    return width_mm * MM_TO_INCH, width_mm * aspect * MM_TO_INCH


def new_figure(
    publisher: Literal["wiley-adom", "springer"] = "wiley-adom",
    span: Literal["one", "two"] = "one",
    aspect: float = 0.75,
    **subplots_kwargs,
):
    """Create a styled figure and axes at the intended publication width."""
    set_publication_style()
    return plt.subplots(
        figsize=figure_size(publisher=publisher, span=span, aspect=aspect),
        constrained_layout=True,
        **subplots_kwargs,
    )

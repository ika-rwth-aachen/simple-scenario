from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

from contextlib import contextmanager
from pathlib import Path


def get_rcParams(dpi: int = 600, hide_ticks: bool = False) -> dict:  # noqa: N802
    # Lines should be 1px
    # Linewidth is given in points
    # 1 Point = DPI/72 px
    # 1 px = 72 / DPI Points

    px_in_pts = 72 / dpi
    rcParams = {}  # noqa: N806
    rcParams["lines.linewidth"] = px_in_pts
    rcParams["lines.markersize"] = px_in_pts
    rcParams["patch.linewidth"] = px_in_pts
    rcParams["axes.linewidth"] = 2 * px_in_pts
    rcParams["xtick.major.width"] = px_in_pts
    rcParams["ytick.major.width"] = px_in_pts
    rcParams["font.size"] = 30 * px_in_pts

    rcParams["legend.frameon"] = False
    rcParams["legend.loc"] = "lower right"

    if hide_ticks:
        rcParams["axes.spines.left"] = False
        rcParams["axes.spines.right"] = False
        rcParams["axes.spines.bottom"] = False
        rcParams["axes.spines.top"] = False
        rcParams["xtick.bottom"] = False
        rcParams["ytick.left"] = False
        rcParams["xtick.labelbottom"] = False
        rcParams["ytick.labelleft"] = False

    return rcParams


@contextmanager
def create_plot_ax(
    plot_dir: str | Path, plot_name: str, dpi: int = 600, figw: int = 9, figh: int = 9
) -> plt.Axes:
    if not isinstance(plot_dir, (Path, str)):
        msg = "'plot_dir' must be str or Path"
        raise TypeError(msg)
    plot_dir = Path(plot_dir)
    if not plot_dir.exists():
        msg = "Given 'plot_dir' does not exist. Please create it first."
        raise FileNotFoundError(msg)

    with plt.style.context("default"):
        with mpl.rc_context():
            f, ax = plt.subplots(figsize=(figw, figh), dpi=dpi)

            try:
                yield ax
            finally:
                f.savefig(plot_dir / f"{plot_name}.png", bbox_inches="tight")
                plt.close(f)


@contextmanager
def create_scenario_plot_ax(  # noqa: PLR0912
    plot_dir: str | Path,
    plot_name: str,
    dpi: int = 600,
    figw: int = 9,
    figh: int = 9,
    hide_ticks: bool = False,
    hide_legend: bool = False,
    hide_title: bool = False,
    hide_xlabel: bool = False,
    hide_ylabel: bool = False,
    hide_grid: bool = False,
    clean: bool = False,
) -> plt.Axes:
    if not isinstance(plot_dir, (Path, str)):
        msg = "'plot_dir' must be str or Path"
        raise TypeError(msg)
    plot_dir = Path(plot_dir)
    if not plot_dir.exists():
        msg = "Given 'plot_dir' does not exist. Please create it first."
        raise FileNotFoundError(msg)

    if clean:
        hide_ticks = True
        hide_legend = True
        hide_title = True
        hide_xlabel = True
        hide_ylabel = True
        hide_grid = True

    if plot_dir:
        if not isinstance(plot_dir, (Path, str)):
            msg = "'plot_dir' must be str or Path"
            raise TypeError(msg)
        plot_dir = Path(plot_dir)
        if not plot_dir.exists():
            msg = "Given 'plot_dir' does not exist. Please create it first."
            raise FileNotFoundError(msg)

    with plt.style.context("default"):
        with mpl.rc_context(get_rcParams(dpi=dpi, hide_ticks=hide_ticks)):
            f, ax = plt.subplots(figsize=(figw, figh), dpi=dpi)

            try:
                yield ax

            finally:
                # Handle legend
                legend = ax.get_legend()
                if legend:
                    if hide_legend:
                        legend.remove()
                    else:
                        legend.set_bbox_to_anchor((1, 1))
                        legend.set_ncols(3)

                if hide_title:
                    ax.set_title("")

                if hide_xlabel:
                    ax.set_xlabel("")

                if hide_ylabel:
                    ax.set_ylabel("")

                if hide_grid:
                    ax.grid(False)

                ax.set_aspect("equal")

                # Save to file
                f.savefig(plot_dir / f"{plot_name}.png", bbox_inches="tight")
                plt.close(f)

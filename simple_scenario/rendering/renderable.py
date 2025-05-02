from __future__ import annotations

import matplotlib.pyplot as plt

from abc import abstractmethod
from pathlib import Path

from .scenario_plot_context_manager import create_scenario_plot_ax


class Renderable:
    def render(
        self,
        plot_dir_or_ax: Path | plt.Axes,
        plot_name: str | None = None,
        dpi: int = 600,
        figw: int = 9,
        figh: int = 9,
        clean: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Either plot to file or into a given ax.
        """
        mode = self._derive_render_mode(plot_dir_or_ax, plot_name)

        if mode == "ax":
            self._plot_in_ax(plot_dir_or_ax, *args, **kwargs)

        elif mode == "plot_dir":
            with create_scenario_plot_ax(
                plot_dir_or_ax, plot_name, dpi=dpi, figw=figw, figh=figh, clean=clean
            ) as ax:
                self._plot_in_ax(ax, *args, **kwargs)
                self._format_ax(ax, *args, **kwargs)

    def _derive_render_mode(
        self, plot_dir_or_ax: Path | plt.Axes, plot_name: str
    ) -> None:
        if isinstance(plot_dir_or_ax, plt.Axes):
            return "ax"

        if isinstance(plot_dir_or_ax, (Path, str)):
            if not Path(plot_dir_or_ax).exists():
                msg = "Given 'plot_dir' does not exist. Please create it first."
                raise FileNotFoundError(msg)

            if plot_name is None:
                msg = "'plot_name' must be set when giving 'plot_dir' via 'plot_dir_or_ax'."
                raise ValueError(msg)

            return "plot_dir"

        msg = "Unsupported type of 'plot_dir_or_ax'"
        raise TypeError(msg)

    @abstractmethod
    def _plot_in_ax(self, ax: plt.Axes, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def _format_ax(self, ax: plt.Axes, *args, **kwargs) -> None:
        raise NotImplementedError

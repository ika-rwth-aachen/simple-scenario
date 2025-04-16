from __future__ import annotations

import lanelet2
import numpy as np

from abc import abstractmethod
from typing import TYPE_CHECKING

from lanelet2.core import (
    getId,
    LaneletMap,
    LineString3d,
    Point3d,
    BasicPoint2d,
)
from lanelet2.geometry import findNearest

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from pathlib import Path

from ..rendering import Renderable


class Road(Renderable):
    @abstractmethod
    def copy(self) -> Road:
        """
        Create a copy of the road object.
        """
        ...

    @property
    @abstractmethod
    def config(self) -> dict:
        """
        Return the config of the road object as a dict.
        """
        ...

    @property
    @abstractmethod
    def lanelet_map(self) -> LaneletMap: ...

    @property
    @abstractmethod
    def origin_lat(self) -> Path:
        """
        Latitude of the origin of the map.
        """
        ...

    @property
    @abstractmethod
    def origin_lon(self) -> Path:
        """
        Longitude of the origin of the map.
        """
        ...

    def get_boundary_rect(self) -> tuple[float, float, float, float]:
        """
        Find the boundary rect of the road.
        Returns (xmin, ymin, xmax, ymax)
        """

        xmin = None
        ymin = None
        xmax = None
        ymax = None

        for lanelet in self.lanelet_map.laneletLayer:
            for line in [lanelet.leftBound, lanelet.rightBound]:
                for pt in line:
                    if xmin is None or pt.x < xmin:
                        xmin = pt.x
                    if ymin is None or pt.y < ymin:
                        ymin = pt.y
                    if xmax is None or pt.x > xmax:
                        xmax = pt.x
                    if ymax is None or pt.y > ymax:
                        ymax = pt.y

        return (xmin, ymin, xmax, ymax)

    @abstractmethod
    def save_lanelet2_map(self, save_dir: str | Path, map_name: str) -> Path:
        """
        Save the lanelet2 map to the specified directory.
        """
        ...

    @abstractmethod
    def save_opendrive_map(self, save_dir: str | Path, map_name: str) -> Path:
        """
        Save the opendrive map to the specified directory.
        """
        ...

    @staticmethod
    def linestring2array(linestring: LineString3d) -> np.array:
        """
        Convert a linestring to an array of x, y points.
        """
        return np.array([[pt.x, pt.y] for pt in linestring])

    @staticmethod
    def array2linestring(array: np.array) -> LineString3d:
        """
        Convert an array of x, y points to a linestring.
        """
        return LineString3d(
            getId(), [Point3d(getId(), pt[0], pt[1], 0.0) for pt in array]
        )

    @staticmethod
    def from_frenet_to_cart(
        linestring: LineString3d, s: np.ndarray, t: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert frenet coordinates (s, t) along a linestring to cartesian coordinates (x, y).
        """

        def check_array(array_or_number: np.ndarray | float) -> np.ndarray:
            if isinstance(array_or_number, (float, int)):
                array_or_number = np.array([array_or_number]).astype(float)
            return array_or_number

        s = check_array(s)
        t = check_array(t)

        if s.shape != t.shape:
            raise Exception

        linestring2d = lanelet2.geometry.to2D(linestring)

        x = np.zeros_like(s)
        y = np.zeros_like(t)

        for i in range(s.shape[0]):
            arc = lanelet2.geometry.ArcCoordinates()
            arc.length = s[i]
            arc.distance = t[i]

            cart = lanelet2.geometry.fromArcCoordinates(linestring2d, arc)

            x[i] = cart.x
            y[i] = cart.y

        if x.shape[0] == 1:
            x = x[0]
            y = y[0]

        return x, y

    @staticmethod
    def from_cart_to_frenet(
        linestring: LineString3d, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert cartesian coordinates (x, y) to frenet coordinates (s, t) along a linestring.
        """

        def check_array(array_or_number: np.ndarray | float) -> np.ndarray:
            if isinstance(array_or_number, (float, int)):
                array_or_number = np.array([array_or_number]).astype(float)
            return array_or_number

        x = check_array(x)
        y = check_array(y)

        if x.shape != y.shape:
            raise Exception

        linestring2d = lanelet2.geometry.to2D(linestring)

        s = np.zeros_like(x)
        t = np.zeros_like(y)

        for i in range(x.shape[0]):
            arc = lanelet2.geometry.toArcCoordinates(
                linestring2d, BasicPoint2d(x[0], y[0])
            )

            s[i] = arc.length
            t[i] = arc.distance

        if s.shape[0] == 1:
            s = s[0]
            t = t[0]

        return s, t

    def find_lanelet_id_by_position(self, x: float, y: float) -> int:
        return self.find_lanelet_id_by_position_on_lanelet_map(self.lanelet_map, x, y)

    @staticmethod
    def find_lanelet_id_by_position_on_lanelet_map(
        lanelet_map: LaneletMap, x: float, y: float
    ) -> int:
        """
        Find ID of lanelet that x, y is inside.
        Return None if x,y is not inside any lanelet or in more than one lanelets.
        """
        candidates = findNearest(lanelet_map.laneletLayer, BasicPoint2d(x, y), 1)

        if len(candidates) == 0 or candidates[0][0] != 0:
            return None

        llt_id = candidates[0][1].id

        return llt_id

    @abstractmethod
    def from_llt_local_to_opendrive_local(
        self, x_llt2: float, y_llt2: float, heading_llt2: float | None = None
    ) -> tuple[float, float] | tuple[float, float, float]: ...

    def _plot_in_ax(self, ax: plt.Axes) -> None:
        """
        Plot the lanelet map in the given ax.
        """

        plotted_lanelines = set()

        for lanelet in self.lanelet_map.laneletLayer:
            leftbound = self.linestring2array(lanelet.leftBound)
            rightbound = self.linestring2array(lanelet.rightBound)

            clr = "k"

            # Find shape (dashed/solid)
            left_shape = "-"
            if (
                "subtype" in lanelet.leftBound.attributes
                and lanelet.leftBound.attributes["subtype"] == "dashed"
            ):
                left_shape = "--"

            right_shape = "-"
            if (
                "subtype" in lanelet.rightBound.attributes
                and lanelet.rightBound.attributes["subtype"] == "dashed"
            ):
                right_shape = "--"

            if lanelet.leftBound.id not in plotted_lanelines:
                ax.plot(leftbound[0, 0], leftbound[0, 1], "x", color=clr, zorder=10)
                ax.plot(
                    leftbound[:, 0],
                    leftbound[:, 1],
                    left_shape,
                    color=clr,
                    zorder=10,
                )
                ax.plot(leftbound[0, 0], leftbound[0, 1], "x", color=clr, zorder=10)
            if lanelet.rightBound.id not in plotted_lanelines:
                ax.plot(rightbound[0, 0], rightbound[0, 1], "x", color=clr, zorder=10)
                ax.plot(
                    rightbound[:, 0],
                    rightbound[:, 1],
                    right_shape,
                    color=clr,
                    zorder=10,
                )
                ax.plot(rightbound[0, 0], rightbound[0, 1], "x", color=clr, zorder=10)

            plotted_lanelines.add(lanelet.leftBound.id)
            plotted_lanelines.add(lanelet.rightBound.id)

    def _format_ax(self, ax: plt.Axes) -> None:
        """
        Format the ax after plotting.
        """

        ax.set(title="Road", xlabel="X position in m", ylabel="Y position in m")

        # Set axis limits
        margin = 10
        xmin, ymin, xmax, ymax = self.get_boundary_rect()
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)

        ax.set_aspect("equal")

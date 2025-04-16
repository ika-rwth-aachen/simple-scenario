from __future__ import annotations

import numpy as np

from matplotlib.patches import Rectangle
from typing import TYPE_CHECKING

from .rendering import Renderable
from .vehicle import Vehicle

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from .road.road import Road


class EgoConfiguration(Renderable):
    def __init__(
        self,
        lanelet_id: int,
        s0: float,
        t0: float,
        v0: float,
        target_s: float,
        target_t: float,
        target_lanelet_id: int | None = None,
        vehicle_type_name: str = "medium",
        length: float | None = None,
        width: float | None = None,
        v_lon_max: float | None = None,
        a_lon_min: float | None = None,
        a_lon_max: float | None = None,
    ) -> None:
        """
        lanelet_id: ID of initial lanelet the ego vehicle is starting on
        s0: Initial longitudinal position relative to the given lanelet in m
        t0: Initial lateral offset to the center line of the given lanelet in m
        v0: Initial speed in m/s
        target_s: Target longitudinal position in lanelet with target_lanelet_id
        target_lanelet_id: ID of the target lanelet, if None, lanelet_id is used
        vehicle_type: Type string of the vehicle, if "custom", length, width, v_lon_max, a_lon_min, a_lon_max must be given
        length: Length of the vehicle in m, if None, it is taken from the vehicle type
        width: Width of the vehicle in m, if None, it is taken from the vehicle type
        v_lon_max: Maximum longitudinal speed of the vehicle in m/s, if None, it is taken from the vehicle type
        a_lon_min: Minimum longitudinal acceleration of the vehicle in m/s^2, if None, it is taken from the vehicle type
        a_lon_max: Maximum longitudinal acceleration of the vehicle in m/s^2, if None, it is taken from the vehicle type
        """

        self._config = {
            "lanelet_id": lanelet_id,
            "s0": s0,
            "t0": t0,
            "v0": v0,
            "vehicle_type_name": vehicle_type_name,
            "target_s": target_s,
            "target_t": target_t,
            "target_lanelet_id": target_lanelet_id,
            "length": length,
            "width": width,
            "v_lon_max": v_lon_max,
            "a_lon_min": a_lon_min,
            "a_lon_max": a_lon_max,
        }

        self._lanelet_id = lanelet_id
        self._s0 = s0
        self._t0 = t0
        self._v0 = v0
        self._vehicle_type_name = vehicle_type_name
        self._target_s = target_s
        self._target_t = target_t
        self._target_lanelet_id = (
            target_lanelet_id if target_lanelet_id is not None else lanelet_id
        )

        if vehicle_type_name not in Vehicle.VEHICLE_TYPE_PARAMETERS:
            msg = f"Vehicle type {vehicle_type_name} not found. Available types: {Vehicle.VEHICLE_TYPE_PARAMETERS.keys()}"
            raise ValueError(msg)

        # Vehicle parameters
        if vehicle_type_name == "custom":
            if any(
                param is None
                for param in (length, width, v_lon_max, a_lon_min, a_lon_max)
            ):
                msg = "For custom vehicle, length, width, v_lon_max, a_lon_min, a_lon_max must be given."
                raise ValueError(msg)
            if length <= 0 or width <= 0:
                msg = "For custom vehicle, length and width must be greater than 0."
                raise ValueError(msg)
            if v_lon_max < 0:
                msg = "For custom vehicle, v_lon_max must be greater than 0."
                raise ValueError(msg)
            if a_lon_min > a_lon_max:
                msg = "For custom vehicle, a_lon_min must be less than a_lon_max."
                raise ValueError(msg)
            self._length = length
            self._width = width
            self._v_lon_max = v_lon_max
            self._a_lon_min = a_lon_min
            self._a_lon_max = a_lon_max
        else:
            vehicle_parameters = Vehicle.get_vehicle_parameters_of_vehicle_type(
                vehicle_type_name
            )
            self._length = vehicle_parameters.l
            self._width = vehicle_parameters.w
            self._v_lon_max = vehicle_parameters.longitudinal.v_max
            self._a_lon_min = -vehicle_parameters.longitudinal.a_max
            self._a_lon_max = vehicle_parameters.longitudinal.a_max

        # Compiled values
        self._compiled = False
        self._x0 = None
        self._y0 = None
        self._heading0 = None
        self._target_x = None
        self._target_y = None

    @property
    def config(self) -> dict:
        return self._config

    @property
    def lanelet_id(self) -> int:
        return self._lanelet_id

    @property
    def s0(self) -> float:
        return self._s0

    @property
    def t0(self) -> float:
        return self._t0

    @property
    def v0(self) -> float:
        return self._v0

    @property
    def target_lanelet_id(self) -> int:
        return self._target_lanelet_id

    @property
    def target_s(self) -> float:
        return self._target_s

    @property
    def target_t(self) -> float:
        return self._target_t

    @property
    def length(self) -> float:
        return self._length

    @property
    def width(self) -> float:
        return self._width

    @property
    def v_lon_min(self) -> float:
        return 0

    @property
    def v_lon_max(self) -> float:
        return self._v_lon_max

    @property
    def a_lon_min(self) -> float:
        return self._a_lon_min

    @property
    def a_lon_max(self) -> float:
        return self._a_lon_max

    @property
    def x0(self) -> float:
        self._check_is_compiled()
        return self._x0

    @property
    def y0(self) -> float:
        self._check_is_compiled()
        return self._y0

    @property
    def heading0(self) -> float:
        self._check_is_compiled()
        return self._heading0

    @property
    def target_x(self) -> float:
        """
        x coordinate of the target position.
        """
        self._check_is_compiled()
        return self._target_x

    @property
    def target_y(self) -> float:
        """
        y coordinate of the target position.
        """
        self._check_is_compiled()
        return self._target_y

    def get_boundary_rect(self) -> tuple[float, float, float, float]:
        """
        Find the boundary rect.
        Returns (xmin, ymin, xmax, ymax).
        """
        xmin = min(self.x0, self.target_x)
        xmax = max(self.x0, self.target_x)
        ymin = min(self.y0, self.target_y)
        ymax = max(self.y0, self.target_y)
        return (xmin, ymin, xmax, ymax)

    @property
    def compiled(self) -> bool:
        return self._compiled

    def _check_is_compiled(self) -> None:
        if not self._compiled:
            msg = "Please call .compile() before accessing this data."
            raise Exception(msg)

    def compile(self, road: Road) -> None:
        # Initial situation
        lanelet = road.lanelet_map.laneletLayer[self._lanelet_id]

        x0, y0 = road.from_frenet_to_cart(lanelet.centerline, self._s0, self._t0)

        # Assumption: heading is along the lanelet
        # Moved along lanelet for 1s
        x1, y1 = road.from_frenet_to_cart(lanelet.centerline, self._s0 + 0.5, self._t0)
        heading0 = np.arctan2(y1 - y0, x1 - x0)

        # Target situation
        target_lanelet = road.lanelet_map.laneletLayer[self._target_lanelet_id]
        target_x, target_y = road.from_frenet_to_cart(
            target_lanelet.centerline, self._target_s, self._target_t
        )

        # Compiled values
        self._x0 = x0
        self._y0 = y0
        self._heading0 = heading0
        self._target_x = target_x
        self._target_y = target_y

        self._compiled = True

    def _plot_in_ax(self, ax: plt.Axes, *args, **kwargs) -> None:  # noqa: ARG002
        # Ego starting state

        def get_rotated_bbox_pts(
            center_x: float,
            center_y: float,
            heading: float,
            length: float,
            width: float,
        ) -> np.ndarray:
            T = np.array(  # noqa: N806
                [
                    [np.cos(heading), -np.sin(heading), 0, center_x],
                    [np.sin(heading), np.cos(heading), 0, center_y],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )

            xmin = -length / 2
            xmax = length / 2
            ymin = -width / 2
            ymax = width / 2

            pt0 = np.dot(T, np.array([xmin, ymin, 0, 1]))[:2]
            pt1 = np.dot(T, np.array([xmax, ymin, 0, 1]))[:2]
            pt2 = np.dot(T, np.array([xmax, ymax, 0, 1]))[:2]
            pt3 = np.dot(T, np.array([xmin, ymax, 0, 1]))[:2]

            pts = np.array([pt0, pt1, pt2, pt3])
            return pts

        pts = get_rotated_bbox_pts(
            self._x0, self._y0, self._heading0, self.length, self.width
        )

        pts_closed = np.vstack((pts, pts[0]))
        ax.plot(pts_closed[:, 0], pts_closed[:, 1], "b-", zorder=20, label="Ego start")

        rect_patch = Rectangle(
            (self._x0 - self.length / 2, self._y0 - self.width / 2),
            self.length,
            self.width,
            angle=np.rad2deg(self._heading0),
            rotation_point="center",
            alpha=0.25,
            zorder=20,
            facecolor="b",
            edgecolor="b",
            label="Ego start",
        )
        ax.add_patch(rect_patch)

        # Ego target position
        ax.plot(
            self._target_x,
            self._target_y,
            "gx",
            markersize=5,
            zorder=20,
            label="Ego target",
        )

    def _format_ax(self, ax: plt.Axes, *args, **kwargs) -> None:  # noqa: ARG002
        ax.grid()
        ax.legend()

        ax.set(
            title="Ego configuration",
            xlabel="X position in m",
            ylabel="Y position in m",
        )

        ax.set_aspect("equal")

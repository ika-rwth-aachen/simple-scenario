import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Rectangle

from .road import Road
from .rendering import Renderable
from .vehicle import Vehicle


class EgoConfiguration(Renderable):
    def __init__(
        self,
        lanelet_id: int,
        s0: float,
        t0: float,
        v0: float,
        vehicle_type_name: str = "medium",
    ) -> None:
        """
        lanelet_id: ID of initial lanelet the ego vehicle is starting on
        s0: Initial longitudinal position relative to the given lanelet in m
        t0: Initial lateral offset to the center line of the given lanelet in m
        v0: Initial speed in m/s
        vehicle_type: Type string of the vehicle.
        """

        self._config = {
            "lanelet_id": lanelet_id,
            "s0": s0,
            "t0": t0,
            "v0": v0,
            "vehicle_type_name": vehicle_type_name,
        }

        self._lanelet_id = lanelet_id
        self._s0 = s0
        self._t0 = t0
        self._v0 = v0
        self._vehicle_type_name = vehicle_type_name

        # Create dummy vehicle to get vehicle length and width (for thw calculation)
        self._vehicle = Vehicle(
            -1,
            self._lanelet_id,
            self._s0,
            self._t0,
            self._v0,
            vehicle_type_name=self._vehicle_type_name,
        )

        # Compiled values
        self._compiled = False
        self._x0 = None
        self._y0 = None
        self._heading0 = None

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
    def length(self) -> float:
        return self._vehicle.length

    @property
    def width(self) -> float:
        return self._vehicle.width

    @property
    def v_lon_min(self) -> float:
        return 0

    @property
    def v_lon_max(self) -> float:
        return self._vehicle.vehicle_parameters.longitudinal.v_max

    @property
    def a_lon_min(self) -> float:
        return -self._vehicle.vehicle_parameters.longitudinal.a_max

    @property
    def a_lon_max(self) -> float:
        return self._vehicle.vehicle_parameters.longitudinal.a_max

    @property
    def x0(self) -> float:
        if not self._compiled:
            msg = "Please call .compile() before accessing this data."
            raise Exception(msg)
        return self._x0

    @property
    def y0(self) -> float:
        if not self._compiled:
            msg = "Please call .compile() before accessing this data."
            raise Exception(msg)
        return self._y0

    @property
    def heading0(self) -> float:
        if not self._compiled:
            msg = "Please call .compile() before accessing this data."
            raise Exception(msg)
        return self._heading0

    @property
    def compiled(self) -> bool:
        return self._compiled

    def compile(self, road: Road) -> None:
        lanelet = road.lanelet_map.laneletLayer[self._lanelet_id]

        x0, y0 = road.from_frenet_to_cart(lanelet.centerline, self._s0, self._t0)

        # Assumption: heading is along the lanelet
        # Moved after 1s
        s1 = self._s0 + self._v0
        t1 = self._t0

        x1, y1 = road.from_frenet_to_cart(lanelet.centerline, s1, t1)

        heading0 = np.arctan2(y1 - y0, x1 - x0)

        self._x0 = x0
        self._y0 = y0
        self._heading0 = heading0

        self._compiled = True

    def _plot_in_ax(self, ax: plt.Axes, *args, **kwargs) -> None:  # noqa: ARG002
        # PLOT
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
            angle=self._heading0,
            rotation_point="center",
            alpha=0.25,
            zorder=20,
            facecolor="b",
            edgecolor="b",
            label="Ego start",
        )
        ax.add_patch(rect_patch)

    def _format_ax(self, ax: plt.Axes, *args, **kwargs) -> None:  # noqa: ARG002
        ax.grid()
        ax.legend()

        ax.set(
            title="Ego configuration",
            xlabel="X position in m",
            ylabel="Y position in m",
        )

        ax.set_aspect("equal")

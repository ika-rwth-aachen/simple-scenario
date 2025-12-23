from __future__ import annotations

import numpy as np

from matplotlib.patches import Rectangle
from typing import TYPE_CHECKING

from .rendering import Renderable
from .vehicle import Vehicle

import lanelet2
import lanelet2.geometry
from lanelet2.core import GPSPoint

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from .road.road import Road


class EgoConfiguration(Renderable):
    def __init__(
        self,
        start_lanelet_id: int | None = None,
        start_s: float | None = None,
        start_t: float | None = None,
        start_x: float | None = None,
        start_y: float | None = None,
        start_lat: float | None = None,
        start_lon: float | None = None,
        target_lanelet_id: int | None = None,
        target_s: float | None = None,
        target_t: float | None = None,
        target_x: float | None = None,
        target_y: float | None = None,
        target_lat: float | None = None,
        target_lon: float | None = None,
        v0: float | None = None,
        vehicle_type_name: str = "medium",
        length: float | None = None,
        width: float | None = None,
        v_lon_max: float | None = None,
        a_lon_min: float | None = None,
        a_lon_max: float | None = None,
        controller: str | None = None,
    ) -> None:
        """
        start_lanelet_id: ID of initial lanelet the ego vehicle is starting on
        start_s: Initial longitudinal position relative to the given lanelet in m
        start_t: Initial lateral offset to the center line of the given lanelet in m
        start_x: Start x coordinate, used instead of start_lanelet_id/start_s/start_t if provided
        start_y: Start y coordinate, used instead of start_lanelet_id/start_s/start_t if provided
        start_lat: Start latitude coordinate, used instead of start_lanelet_id/start_s/start_t if provided
        start_lon: Start longitude coordinate, used instead of start_lanelet_id/start_s/start_t if provided
        target_lanelet_id: ID of the target lanelet, if None, start_lanelet_id is used
        target_s: Target longitudinal position in lanelet with target_lanelet_id
        target_t: Target lateral offset to the center line of the given lanelet in m
        target_x: Target x coordinate, used instead of target_s/target_t/target_lanelet_id if provided
        target_y: Target y coordinate, used instead of target_s/target_t/target_lanelet_id if provided
        target_lat: Target latitude coordinate, used instead of target_s/target_t/target_lanelet_id if provided
        target_lon: Target longitude coordinate, used instead of target_s/target_t/target_lanelet_id if provided
        v0: Initial speed in m/s
        vehicle_type_name: Type string of the vehicle, must be in Vehicle
        length: Length of the vehicle in m, if None, it is taken from the vehicle type
        width: Width of the vehicle in m, if None, it is taken from the vehicle type
        v_lon_max: Maximum longitudinal speed of the vehicle in m/s, if None, it is taken from the vehicle type
        a_lon_min: Minimum longitudinal acceleration of the vehicle in m/s^2, if None, it is taken from the vehicle type
        a_lon_max: Maximum longitudinal acceleration of the vehicle in m/s^2, if None, it is taken from the vehicle type
        controller: Controller for the ego vehicle
        """

        self._config = {
            "start_lanelet_id": start_lanelet_id,
            "start_s": start_s,
            "start_t": start_t,
            "start_x": start_x,
            "start_y": start_y,
            "start_lat": start_lat,
            "start_lon": start_lon,
            "target_lanelet_id": target_lanelet_id,
            "target_s": target_s,
            "target_t": target_t,
            "target_x": target_x,
            "target_y": target_y,
            "target_lat": target_lat,
            "target_lon": target_lon,
            "v0": v0,
            "vehicle_type_name": vehicle_type_name,
            "length": length,
            "width": width,
            "v_lon_max": v_lon_max,
            "a_lon_min": a_lon_min,
            "a_lon_max": a_lon_max,
            "controller": controller,
        }

        self._start_lanelet_id = start_lanelet_id
        self._start_s = start_s
        self._start_t = start_t
        self._start_x = start_x
        self._start_y = start_y
        self._start_lat = start_lat
        self._start_lon = start_lon
        self._target_lanelet_id = target_lanelet_id if target_lanelet_id is not None else start_lanelet_id
        self._target_s = target_s
        self._target_t = target_t
        self._target_x = target_x
        self._target_y = target_y
        self._target_lat = target_lat
        self._target_lon = target_lon
        self._v0 = v0
        self._vehicle_type_name = vehicle_type_name
        self._controller = controller

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

        self._traffic_rules = lanelet2.traffic_rules.create(
            lanelet2.traffic_rules.Locations.Germany,
            lanelet2.traffic_rules.Participants.Vehicle,
        )
        # Compiled values
        self._compiled = False
        self._start_heading = None
        self._route_length = None

    @property
    def config(self) -> dict:
        """
        Return the configuration as a dict.
        """
        return self._config

    @property
    def start_lanelet_id(self) -> int:
        """
        Lanelet ID of the start position.
        """
        self._check_is_compiled()
        return self._start_lanelet_id

    @property
    def start_s(self) -> float:
        """
        Start longitudinal position in lanelet coordinates.
        """
        self._check_is_compiled()
        return self._start_s

    @property
    def start_t(self) -> float:
        """
        Start lateral offset in lanelet coordinates.
        """
        self._check_is_compiled()
        return self._start_t

    @property
    def start_x(self) -> float:
        """
        x coordinate of the start position.
        """
        self._check_is_compiled()
        return self._start_x

    @property
    def start_y(self) -> float:
        """
        y coordinate of the start position.
        """
        self._check_is_compiled()
        return self._start_y

    @property
    def start_lat(self) -> float:
        """
        lat coordinate of the start position.
        """
        self._check_is_compiled()
        return self._start_lat

    @property
    def start_lon(self) -> float:
        """
        lon coordinate of the start position.
        """
        self._check_is_compiled()
        return self._start_lon

    @property
    def target_lanelet_id(self) -> int:
        """
        Lanelet ID of the target position.
        """
        if self._target_lanelet_id is None:
            self._check_is_compiled()
        return self._target_lanelet_id

    @property
    def target_s(self) -> float:
        """
        Target longitudinal position in lanelet coordinates.
        """
        if self._target_s is None:
            self._check_is_compiled()
        return self._target_s

    @property
    def target_t(self) -> float:
        """
        Target lateral offset in lanelet coordinates.
        """
        if self._target_t is None:
            self._check_is_compiled()
        return self._target_t

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

    @property
    def target_lat(self) -> float:
        """
        lat coordinate of the target position.
        """
        self._check_is_compiled()
        return self._target_lat

    @property
    def target_lon(self) -> float:
        """
        lon coordinate of the target position.
        """
        self._check_is_compiled()
        return self._target_lon

    @property
    def v0(self) -> float:
        """
        Initial speed in m/s.
        """
        return self._v0

    @property
    def length(self) -> float:
        """
        Vehicle length in meters.
        """
        return self._length

    @property
    def width(self) -> float:
        """
        Vehicle width in meters.
        """
        return self._width

    @property
    def v_lon_min(self) -> float:
        """
        Minimum longitudinal speed in m/s.
        """
        return 0

    @property
    def v_lon_max(self) -> float:
        """
        Maximum longitudinal speed in m/s.
        """
        return self._v_lon_max

    @property
    def a_lon_min(self) -> float:
        """
        Minimum longitudinal acceleration in m/s^2.
        """
        return self._a_lon_min

    @property
    def a_lon_max(self) -> float:
        """
        Maximum longitudinal acceleration in m/s^2.
        """
        return self._a_lon_max

    @property
    def start_heading(self) -> float:
        """
        Start heading in radians.
        """
        self._check_is_compiled()
        return self._start_heading

    @property
    def controller(self) -> str:
        """
        controller for ego vehicle.
        """
        self._check_is_compiled()
        return self._controller

    @property
    def route_length(self) -> float:
        """
        Length of the shortest route from start to target in meters.
        """
        self._check_is_compiled()
        return self._route_length

    def get_boundary_rect(self) -> tuple[float, float, float, float]:
        """
        Find the boundary rect.
        Returns (xmin, ymin, xmax, ymax).
        """
        xmin = min(self.start_x, self.target_x)
        xmax = max(self.start_x, self.target_x)
        ymin = min(self.start_y, self.target_y)
        ymax = max(self.start_y, self.target_y)
        return (xmin, ymin, xmax, ymax)

    @property
    def compiled(self) -> bool:
        """
        True if compiled values are available.
        """
        return self._compiled

    def _check_is_compiled(self) -> None:
        if not self._compiled:
            msg = "Please call .compile() before accessing this data."
            raise Exception(msg)

    def _project_latlon_to_llt_xy(
        self, road: Road, lat: float, lon: float
    ) -> tuple[float, float]:
        projector = getattr(road, "llt_utm_projector", None)
        if projector is None:
            msg = "Lat/lon requires a lanelet2 projector."
            raise ValueError(msg)
        gps = GPSPoint(lat, lon, 0.0)
        utm = projector.forward(gps)
        return utm.x, utm.y

    def _resolve_start_position(self, road: Road) -> lanelet2.core.Lanelet:
        if self._start_lat is not None and self._start_lon is not None:
            self._start_x, self._start_y = self._project_latlon_to_llt_xy(
                road, self._start_lat, self._start_lon
            )

        if self._start_x is not None and self._start_y is not None:
            self._start_lanelet_id = road.find_lanelet_id_by_position(
                self._start_x, self._start_y
            )
            if self._start_lanelet_id is None:
                msg = "Start position is not located in any lanelet."
                raise ValueError(msg)
            lanelet = road.lanelet_map.laneletLayer[self._start_lanelet_id]
            self._start_s, self._start_t = road.from_cart_to_frenet(
                lanelet.centerline, self._start_x, self._start_y
            )

        if (
            self._start_lanelet_id is not None
            and self._start_s is not None
            and self._start_t is not None
            and (self._start_x is None or self._start_y is None)
        ):
            lanelet = road.lanelet_map.laneletLayer[self._start_lanelet_id]
            self._start_x, self._start_y = road.from_frenet_to_cart(
                lanelet.centerline, self._start_s, self._start_t
            )

        if self._start_lanelet_id is None:
            msg = "Start lanelet ID could not be determined."
            raise ValueError(msg)
        if self._start_s is None:
            msg = "Start longitudinal position could not be determined."
            raise ValueError(msg)
        if self._start_t is None:
            msg = "Start lateral offset could not be determined."
            raise ValueError(msg)

        return road.lanelet_map.laneletLayer[self._start_lanelet_id]

    def _resolve_target_position(self, road: Road) -> lanelet2.core.Lanelet:
        if self._target_lat is not None and self._target_lon is not None:
            self._target_x, self._target_y = self._project_latlon_to_llt_xy(
                road, self._target_lat, self._target_lon
            )

        if self._target_x is not None and self._target_y is not None:
            self._target_lanelet_id = road.find_lanelet_id_by_position(
                self._target_x, self._target_y
            )
            if self._target_lanelet_id is None:
                msg = "Target position is not located in any lanelet."
                raise ValueError(msg)
            target_lanelet = road.lanelet_map.laneletLayer[self._target_lanelet_id]
            self._target_s, self._target_t = road.from_cart_to_frenet(
                target_lanelet.centerline, self._target_x, self._target_y
            )

        if (
            self._target_lanelet_id is not None
            and self._target_s is not None
            and self._target_t is not None
            and (self._target_x is None or self._target_y is None)
        ):
            target_lanelet = road.lanelet_map.laneletLayer[self._target_lanelet_id]
            self._target_x, self._target_y = road.from_frenet_to_cart(
                target_lanelet.centerline, self._target_s, self._target_t
            )

        if self._target_lanelet_id is None:
            msg = "Target lanelet ID could not be determined."
            raise ValueError(msg)
        if self._target_s is None:
            msg = "Target longitudinal position could not be determined."
            raise ValueError(msg)
        if self._target_t is None:
            msg = "Target lateral offset could not be determined."
            raise ValueError(msg)

        return road.lanelet_map.laneletLayer[self._target_lanelet_id]

    def compile(self, road: Road) -> None:
        lanelet = self._resolve_start_position(road)

        # Assumption: heading is along the lanelet
        # Moved along lanelet for 1s
        x1, y1 = road.from_frenet_to_cart(
            lanelet.centerline, self._start_s + 0.5, self._start_t
        )
        start_heading = np.arctan2(y1 - self._start_y, x1 - self._start_x)

        target_lanelet = self._resolve_target_position(road)

        # Calculate route length
        routing_graph = lanelet2.routing.RoutingGraph(
            road.lanelet_map, self._traffic_rules
        )
        route = routing_graph.getRoute(lanelet, target_lanelet)
        if route is None:
            msg = (
                "No valid route found. Please check the route "
                f"(start_lanelet_id: {self._start_lanelet_id} -> target_lanelet_id: {self._target_lanelet_id})."
            )
            raise ValueError(msg)

        target_lanelet_length = lanelet2.geometry.length2d(target_lanelet)
        route_length = route.length2d()
        route_length -= self._start_s
        route_length -= target_lanelet_length - min(self._target_s, target_lanelet_length)

        if route_length < -1e-6:
            msg = (
                "Computed route length is negative. Please check that the target is "
                "ahead of the start position along the route."
            )
            raise ValueError(msg)

        # Compiled values
        self._start_heading = start_heading
        self._route_length = max(0.0, route_length)
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
            self.start_x, self.start_y, self._start_heading, self.length, self.width
        )

        pts_closed = np.vstack((pts, pts[0]))
        ax.plot(pts_closed[:, 0], pts_closed[:, 1], "b-", zorder=20, label="Ego start")

        rect_patch = Rectangle(
            (self.start_x - self.length / 2, self.start_y - self.width / 2),
            self.length,
            self.width,
            angle=np.rad2deg(self._start_heading),
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

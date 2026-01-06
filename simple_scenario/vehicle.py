from __future__ import annotations

import lanelet2
import lanelet2.geometry
import numpy as np

from matplotlib.patches import Rectangle
from typing import TYPE_CHECKING
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.parameters_vehicle3 import parameters_vehicle3

from .road.road import Road
from .rendering import Renderable

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from omegaconf import DictConfig
    from lanelet2.core import LaneletMap


class Vehicle(Renderable):
    VEHICLE_TYPE_PARAMETERS = {
        "small": parameters_vehicle1,  # VehicleType.FORD_ESCORT,
        "medium": parameters_vehicle2,  # VehicleType.BMW_320i,
        "van": parameters_vehicle3,  # VehicleType.VW_VANAGON
        # "semi_trailer": 4  # Currently not available
        "custom": None,  # Take width and length from parameters
    }

    ACCELERATION_PROFILES = ["constant"]

    LC_TYPES = ("polynomial", "vy")

    def __init__(
        self,
        vehicle_id: int,
        *,
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
        a0: float = 0,
        a_delay: float = 0,
        a_profile: str = "constant",
        lc_direction: int = 0,
        lc_delay: float = 0,
        lc_duration: float = 3,
        lc_type: str = "polynomial",
        lc_vy: float = 0,
        inverse_driving_direction: bool = False,
        vehicle_type_name: str = "medium",
        length: float | None = None,
        width: float | None = None,
        from_data: bool = False,
        depends_on_ego: bool = False,
        duration: float | None = None,
    ) -> None:
        """
        vehicle_id: Unique ID of the vehicle
        start_lanelet_id: ID of initial lanelet the vehicle is starting on
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
        a0: Acceleration in m/s^2
        a_delay: Time to wait before applying acceleration
        a_profile: How the acceleration changes over time
        lc_direction: Direction of lane change. 0: no lane change, 1: left, -1: right
        lc_delay: Time to wait before lane change
        lc_duration: Time the lane change takes to be completed
        lc_type: How the lc is done
        lc_vy: Only used if lc_type is "vy"
        inverse_driving_direction: If True, the vehicle is driving in the opposite direction
        vehicle_type_name: Type of the vehicle
        length: Length of the vehicle, only used if vehicle_type_name is "custom"
        width: Width of the vehicle, only used if vehicle_type_name is "custom"
        from_data: If True, the vehicle is created from data and no compilation is needed
        depends_on_ego: If True, the vehicle's behavior depends on the ego vehicle movement
        duration: Optional duration for this vehicle's trajectory in seconds
        """

        if a_profile not in self.ACCELERATION_PROFILES:
            msg = f"Unsupported acceleration profile specifed: {a_profile}. Choose from: {self.ACCELERATION_PROFILES}."
            raise AttributeError(msg)

        if lc_direction not in (-1, 0, 1):
            msg = "lc_direction can be -1, 0, or 1"
            raise ValueError(msg)

        if lc_type not in self.LC_TYPES:
            msg = f"Unsupported lc type specifed: {lc_type}. Choose from: {self.LC_TYPES}."
            raise AttributeError(msg)

        if vehicle_type_name not in self.VEHICLE_TYPE_PARAMETERS:
            msg = "Vehicle type is not available"
            raise ValueError(msg)

        if vehicle_type_name == "custom" and length is None and width is None:
            msg = "If vehicle type is 'custom', parameters 'length' and 'width' must be not None"
            raise ValueError(msg)

        if vehicle_type_name != "custom" and length is not None and width is not None:
            msg = "If vehicle type is not 'custom', parameters 'length' and 'width' must be None"
            raise ValueError(msg)

        self._config = {
            "vehicle_id": vehicle_id,
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
            "a0": a0,
            "a_delay": a_delay,
            "a_profile": a_profile,
            "lc_direction": lc_direction,
            "lc_delay": lc_delay,
            "lc_duration": lc_duration,
            "lc_type": lc_type,
            "lc_vy": lc_vy,
            "inverse_driving_direction": inverse_driving_direction,
            "vehicle_type_name": vehicle_type_name,
            "length": length,
            "width": width,
            "from_data": from_data,
            "depends_on_ego": depends_on_ego,
            "duration": duration,
        }

        self._start_position = {
            "lanelet_id": start_lanelet_id,
            "s": start_s,
            "t": start_t,
            "x": start_x,
            "y": start_y,
            "lat": start_lat,
            "lon": start_lon,
        }

        self._target_position = {
            "lanelet_id": target_lanelet_id if target_lanelet_id is not None else start_lanelet_id,
            "s": target_s,
            "t": target_t,
            "x": target_x,
            "y": target_y,
            "lat": target_lat,
            "lon": target_lon,
        }

        self._vehicle_id = vehicle_id
        self._v0 = v0
        self._a0 = a0
        self._a_delay = a_delay
        self._a_profile = a_profile
        self._lc_direction = lc_direction
        self._lc_delay = lc_delay
        self._lc_duration = lc_duration
        self._lc_type = lc_type
        self._lc_vy = lc_vy
        self._inverse_driving_direction = inverse_driving_direction
        self._vehicle_type_name = vehicle_type_name
        self._vehicle_parameters = None
        if self._vehicle_type_name == "custom":
            self._length = length
            self._width = width
        else:
            vehicle_parameters = self.get_vehicle_parameters_of_vehicle_type(
                vehicle_type_name
            )
            self._length = vehicle_parameters.l
            self._width = vehicle_parameters.w
        self._depends_on_ego = depends_on_ego
        self._duration = duration

        self._traffic_rules = lanelet2.traffic_rules.create(
            lanelet2.traffic_rules.Locations.Germany,
            lanelet2.traffic_rules.Participants.Vehicle,
        )

        # Compiled values
        self._compiled = False
        self._x = None
        self._y = None
        self._heading = None
        self._v = None
        self._a = None

        # If the vehicle is created directly from data, this is True
        self._initialized_from_data = from_data

    @classmethod
    def get_vehicle_parameters_of_vehicle_type(cls, vehicle_type_name: str) -> dict:
        if vehicle_type_name not in cls.VEHICLE_TYPE_PARAMETERS:
            msg = "Vehicle type is not available"
            raise ValueError(msg)

        if vehicle_type_name == "custom":
            msg = "If vehicle type is 'custom', parameters must be selected manually."
            raise ValueError(msg)

        vehicle_parameters = cls.VEHICLE_TYPE_PARAMETERS[vehicle_type_name]()
        return vehicle_parameters

    @classmethod
    def from_data(
        cls,
        vehicle_id: int,
        x: np.ndarray,
        y: np.ndarray,
        heading: np.ndarray,
        v: np.ndarray,
        a: np.ndarray,
        vehicle_type_name: str = "medium",
        length: float | None = None,
        width: float | None = None,
        depends_on_ego: bool = False,
        duration: float | None = None,
    ) -> Vehicle:
        if x.shape[0] == 0:
            msg = "All data arrays must have some values."
            raise ValueError(msg)

        if len(x.shape) != 1:
            msg = "All data arrays must be of shape (X,)."
            raise ValueError(msg)

        if not (x.shape == y.shape == heading.shape == v.shape == a.shape):
            msg = "All data arrays must have same shape"
            raise ValueError(msg)

        vehicle = Vehicle(
            vehicle_id=vehicle_id,
            v0=v[0],
            a0=a[0],
            vehicle_type_name=vehicle_type_name,
            length=length,
            width=width,
            from_data=True,
            depends_on_ego=depends_on_ego,
            duration=duration
        )

        vehicle.set_data(x, y, heading, v, a)

        return vehicle

    @property
    def config(self) -> dict:
        return self._config

    @property
    def id(self) -> int:
        return self._vehicle_id

    @property
    def vehicle_parameters(self) -> DictConfig:
        if self._vehicle_type_name == "custom":
            msg = (
                "If vehicle_type_name is 'custom', vehicle_parameters are not available"
            )
            raise ValueError(msg)
        return self._vehicle_parameters

    @property
    def start_lanelet_id(self) -> int:
        if self._initialized_from_data:
            msg = "Not possible to access start_lanelet_id when initialized from data."
            raise Exception(msg)
        return self._start_position["lanelet_id"]

    @property
    def start_s(self) -> float:
        if self._initialized_from_data:
            msg = "Not possible to access start_s when initialized from data."
            raise Exception(msg)
        return self._start_position["s"]

    @property
    def start_t(self) -> float:
        if self._initialized_from_data:
            msg = "Not possible to access start_t when initialized from data."
            raise Exception(msg)
        return self._start_position["t"]

    @property
    def start_x(self) -> float:
        if self._initialized_from_data:
            msg = "Not possible to access start_x when initialized from data."
            raise Exception(msg)
        return self._start_position["x"]

    @property
    def start_y(self) -> float:
        if self._initialized_from_data:
            msg = "Not possible to access start_y when initialized from data."
            raise Exception(msg)
        return self._start_position["y"]

    @property
    def start_lat(self) -> float:
        if self._initialized_from_data:
            msg = "Not possible to access start_lat when initialized from data."
            raise Exception(msg)
        return self._start_position["lat"]

    @property
    def start_lon(self) -> float:
        if self._initialized_from_data:
            msg = "Not possible to access start_lon when initialized from data."
            raise Exception(msg)
        return self._start_position["lon"]

    @property
    def target_lanelet_id(self) -> int:
        if self._initialized_from_data:
            msg = "Not possible to access target_lanelet_id when initialized from data."
            raise Exception(msg)
        return self._target_position["lanelet_id"]

    @property
    def target_s(self) -> float:
        if self._initialized_from_data:
            msg = "Not possible to access target_s when initialized from data."
            raise Exception(msg)
        return self._target_position["s"]

    @property
    def target_t(self) -> float:
        if self._initialized_from_data:
            msg = "Not possible to access target_t when initialized from data."
            raise Exception(msg)
        return self._target_position["t"]

    @property
    def target_x(self) -> float:
        if self._initialized_from_data:
            msg = "Not possible to access target_x when initialized from data."
            raise Exception(msg)
        return self._target_position["x"]

    @property
    def target_y(self) -> float:
        if self._initialized_from_data:
            msg = "Not possible to access target_y when initialized from data."
            raise Exception(msg)
        return self._target_position["y"]

    @property
    def target_lat(self) -> float:
        if self._initialized_from_data:
            msg = "Not possible to access target_lat when initialized from data."
            raise Exception(msg)
        return self._target_position["lat"]

    @property
    def target_lon(self) -> float:
        if self._initialized_from_data:
            msg = "Not possible to access target_lon when initialized from data."
            raise Exception(msg)
        return self._target_position["lon"]

    @property
    def v0(self) -> float:
        return self._v0

    @property
    def a0(self) -> float:
        return self._a0

    @property
    def inverse_driving_direction(self) -> bool:
        return self._inverse_driving_direction

    @property
    def vehicle_type_name(self) -> str:
        return self._vehicle_type_name

    @property
    def length(self) -> float:
        return self._length

    @property
    def width(self) -> float:
        return self._width

    @property
    def depends_on_ego(self) -> bool:
        return self._depends_on_ego

    @property
    def duration(self) -> float | None:
        return self._duration

    @property
    def x(self) -> np.ndarray:
        self._check_is_compiled()
        return self._x

    @property
    def y(self) -> np.ndarray:
        self._check_is_compiled()
        return self._y

    @property
    def heading(self) -> np.ndarray:
        self._check_is_compiled()
        return self._heading

    @property
    def v(self) -> np.ndarray:
        self._check_is_compiled()
        return self._v

    @property
    def a(self) -> np.ndarray:
        self._check_is_compiled()
        return self._a

    @property
    def initialized_from_data(self) -> bool:
        return self._initialized_from_data

    def _check_is_compiled(self) -> None:
        if not self._compiled:
            msg = "Please call .compile() before accessing this data."
            raise Exception(msg)

    def set_data(
        self, x: np.array, y: np.array, heading: np.array, v: np.array, a: np.array
    ) -> None:
        if not self._initialized_from_data:
            msg = "Cannot set data unless the vehicle is initialized with 'from_data=True'."
            raise Exception(msg)

        self._x = x
        self._y = y
        self._heading = heading
        self._v = v
        self._a = a

        self._compiled = True

    @property
    def compiled(self) -> bool:
        return self._compiled

    def compile(  # noqa: PLR0912
        self,
        road: Road,
        duration: float,
        dt: float,
    ) -> None:
        """
        Create absolute cartesian coordinates etc
        """

        road.resolve_position(self._start_position)
        road.resolve_position(self._target_position)

        full_n_steps = self._n_steps_from_duration_dt(duration, dt)
        active_duration = self._duration if self._duration is not None else duration

        # -- Build time-series arrays --
        n_steps = self._n_steps_from_duration_dt(active_duration, dt)
        if n_steps <= 0:
            pad_len = max(0, full_n_steps)
            pad = np.full((pad_len,), np.nan)
            self._a = pad.copy()
            self._v = pad.copy()
            self._x = pad.copy()
            self._y = pad.copy()
            self._heading = pad.copy()
            self._compiled = True
            return

        # a
        acceleration_vector = self._get_full_acceleration_vector(n_steps, dt)
        speed_change_vector = acceleration_vector * dt

        # v
        speed_vector = self._v0 * np.ones_like(speed_change_vector)
        speed_vector[1:] = speed_vector[1:] + np.cumsum(speed_change_vector)[:-1]
        speed_vector[speed_vector < 1e-2] = 0.0

        # s
        driving_direction = -1 if self._inverse_driving_direction else 1
        lon_position_change_vector = driving_direction * speed_vector * dt
        lon_position_vector = self._start_position["s"] * np.ones_like(speed_change_vector)
        lon_position_vector[1:] = (
            lon_position_vector[1:] + np.cumsum(lon_position_change_vector)[:-1]
        )

        # t
        lat_offset_vector = self._start_position["t"] * np.ones((n_steps,))

        # -- Determine route --
        source_lanelet = road.lanelet_map.laneletLayer[self._start_position["lanelet_id"]]
        if self._target_position["lanelet_id"] is not None:
            target_lanelet = road.lanelet_map.laneletLayer[self._target_position["lanelet_id"]]
        else:
            target_lanelet = source_lanelet

        # Establish routing_graph, route without lane changes, get shortest path and transform it into a lanelet_sequence
        routing_graph = lanelet2.routing.RoutingGraph(road.lanelet_map, self._traffic_rules)
        route = routing_graph.getRoute(source_lanelet, target_lanelet)

        if route is None:
            msg = f"Vehicle {self._vehicle_id}: No valid route found. Please check the route (start_lanelet_id: {self._start_position['lanelet_id']} -> target_lanelet_id: {self._target_position['lanelet_id']})."
            raise ValueError(msg)

        # FOR DEBUGGIN MAPS
        # projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(50.9098472225444, 6.22742895630397))  # noqa: ERA001
        # lanelet2.io.write("route.osm", route.laneletSubmap().laneletMap(), projector)  # noqa: ERA001
        # lanelet2.io.write("routing_graph.osm", routing_graph.getDebugLaneletMap(0), projector)  # noqa: ERA001
        # FOR DEBUGGIN MAPS

        # -- Handle lane change --
        if self._lc_direction in (-1, 1):
            # Make sure that the requested movement is possible on the route
            # Change the lat offset values according to requested lane change

            # -- Section up to lane change --
            # Find shortest path from source_lanelet to target_lanelet in the route (including lane changes)
            lanelet_path = route.shortestPath()
            # Get initial part of the route until the first lane change
            lanelet_sequence = lanelet_path.getRemainingLane(source_lanelet)

            # Find lanelet at lc position
            lc_start_step = int(self._lc_delay / dt)
            lc_start_s = lon_position_vector[lc_start_step]

            if lc_start_s > lanelet2.geometry.length(lanelet_sequence.centerline):
                msg = f"Vehicle {self._vehicle_id}: Path up to lane change position is not long enough. Please check the route (lanelet_id: {self._start_position['lanelet_id']} -> target_lanelet_id: {self._target_position['lanelet_id']})."
                raise ValueError(msg)

            # -- Lane change section --

            # Find lanelet at lc position
            lc_start_x, lc_start_y = Road.from_frenet_to_cart(
                lanelet_sequence.centerline, lc_start_s, 0
            )
            lc_source_llt_id = Road.find_lanelet_id_by_position_on_lanelet_map(
                road.lanelet_map, lc_start_x, lc_start_y
            )
            if lc_source_llt_id is None:
                msg = f"Vehicle {self._vehicle_id}: Lane change after {self._lc_delay}s at s={lc_start_s}: No valid start lanelet at this position. Maybe there are more than one?"
                raise ValueError(msg)
            lc_source_lanelet = road.lanelet_map.laneletLayer[lc_source_llt_id]

            # Check that there is a valid neighbour for the lane change
            if self._lc_direction == -1:
                lc_target_lanelet = routing_graph.right(lc_source_lanelet)

            elif self._lc_direction == 1:
                lc_target_lanelet = routing_graph.left(lc_source_lanelet)

            if lc_target_lanelet is None:
                direction = "left" if self._lc_direction == 1 else "right"
                msg = f"Vehicle {self._vehicle_id}: Lane change after {self._lc_delay}s at s={lc_start_s} from lanelet {lc_source_llt_id} to the {direction} not possible. No valid neighbour lanelet. (x={lc_start_x}, y={lc_start_y})"
                raise ValueError(msg)

            # Perform the lane change
            if self._lc_type == "polynomial":
                lc_traj = self._generate_lc_trajectory(
                    road.lanelet_map,
                    lon_position_vector[lc_start_step],
                    speed_vector[lc_start_step],
                    lc_source_lanelet.id,
                    lc_target_lanelet.id,
                    self._lc_duration,
                    dt,
                )

                n_lc_steps = lc_traj.shape[0]

                lat_offset_vector[lc_start_step : lc_start_step + n_lc_steps] = lc_traj[
                    :, 1
                ]

            elif self._lc_type == "vy":
                lat_offset_change_vector = (
                    self._lc_vy
                    * dt
                    * np.ones_like(lat_offset_vector)
                    * self._lc_direction
                )
                lat_offset_vector[1:] = (
                    lat_offset_vector[1:] + np.cumsum(lat_offset_change_vector)[:-1]
                )

                # End of LC
                # Find max possible t positon
                x_lc1, y_lc1 = Road.from_frenet_to_cart(
                    lc_source_lanelet.centerline, self._start_position["s"], 0
                )
                _, t_lc1 = Road.from_cart_to_frenet(
                    lc_target_lanelet.centerline, x_lc1, y_lc1
                )
                max_t = t_lc1  # - self.width / 2

                if self._lc_direction == 1:
                    lat_offset_vector[lat_offset_vector > max_t] = max_t
                else:
                    lat_offset_vector[lat_offset_vector < max_t] = max_t

            # -- Section after lane change --

            # Check whether the rest of the route is long enough

            # Find x, y position after lane change
            lc_end_idx = lc_start_step + n_lc_steps - 1
            lc_end_s = lon_position_vector[lc_end_idx]
            lc_end_t = lat_offset_vector[lc_end_idx]
            lc_end_x, lc_end_y = Road.from_frenet_to_cart(
                lc_source_lanelet.centerline, lc_end_s, lc_end_t
            )

            # Find s, t position w.r.t. the lc_target_lanelet
            lc_end_s_in_lc_target_lanelet, lc_end_t_in_lc_target_lanelet = (
                Road.from_cart_to_frenet(
                    lc_target_lanelet.centerline, lc_end_x, lc_end_y
                )
            )

            # Update the remainder of the lon_position_vector and lat_offset_vector to be relative to the new lanelet
            # There may be a s-offset between the lanelets
            lon_position_vector[lc_start_step + n_lc_steps :] = lon_position_vector[
                lc_start_step + n_lc_steps :
            ] + (lc_end_s_in_lc_target_lanelet - lc_end_s)
            # Assuming that the vehicle is staying a the offset t from the end of the lane change
            lat_offset_vector[lc_start_step + n_lc_steps :] = (
                lc_end_t_in_lc_target_lanelet
            )

            # Get remaining route until the target_lanelet
            route_after_lc = routing_graph.getRoute(lc_target_lanelet, target_lanelet)
            shortest_path_after_lc = route_after_lc.shortestPath()
            # Only keep non-lc path
            lanelet_sequence_after_lc = shortest_path_after_lc.getRemainingLane(
                lc_target_lanelet
            )

            remaining_path_length = (
                lanelet2.geometry.length(lanelet_sequence_after_lc.centerline)
                - lc_end_s_in_lc_target_lanelet
            )

            remaining_driven_dist = lon_position_vector[-1] - lc_end_s
            if remaining_driven_dist > remaining_path_length:
                msg = f"Vehicle {self._vehicle_id} is reaching the end of the given path. Please check the route (lanelet_id: {self._start_position['lanelet_id']} -> target_lanelet_id: {self._target_position['lanelet_id']})."
                raise ValueError(msg)

            # -- Compute global x, y for the whole lane change trajectory --

            x = np.zeros_like(lon_position_vector)
            y = np.zeros_like(lat_offset_vector)

            # Until end of lane change
            x[: lc_end_idx + 1], y[: lc_end_idx + 1] = Road.from_frenet_to_cart(
                lanelet_sequence.centerline,
                lon_position_vector[: lc_end_idx + 1],
                lat_offset_vector[: lc_end_idx + 1],
            )
            # After lane change
            x[lc_end_idx + 1 :], y[lc_end_idx + 1 :] = Road.from_frenet_to_cart(
                lanelet_sequence_after_lc.centerline,
                lon_position_vector[lc_end_idx + 1 :],
                lat_offset_vector[lc_end_idx + 1 :],
            )

        else:
            # -- No lane change --

            # Find shortest path from source_lanelet to target_lanelet in the route (including lane changes)
            lanelet_path = route.shortestPath()
            # Get initial part of the route until the first lane change
            lanelet_sequence = lanelet_path.getRemainingLane(source_lanelet)

            # Check that the lanelet_sequence is long enough
            total_driven_dist = lon_position_vector[-1]
            if total_driven_dist > lanelet2.geometry.length(
                lanelet_sequence.centerline
            ):
                msg = f"Vehicle {self._vehicle_id} is reaching the end of the given path. Please check the route (lanelet_id: {self._start_position['lanelet_id']} -> target_lanelet_id: {self._target_position['lanelet_id']})."
                raise ValueError(msg)

            # Positions to cartesian
            x, y = Road.from_frenet_to_cart(
                lanelet_sequence.centerline, lon_position_vector, lat_offset_vector
            )

        # Heading, Assumption: Heading does not change in last time step
        if self._lc_type == "vy":
            if driving_direction == 1:
                heading = np.zeros_like(acceleration_vector)
            elif driving_direction == -1:
                heading = np.pi * np.ones_like(acceleration_vector)
        else:
            # Get initial heading guess from the road
            x_diff0, y_diff0 = Road.from_frenet_to_cart(
                lanelet_sequence.centerline,
                np.array([lon_position_vector[0], lon_position_vector[0] + 0.5]),
                np.zeros((2,)),
            )
            initial_heading_guess = np.arctan2(np.diff(y_diff0), np.diff(x_diff0))
            # Handle standstills (heading remains constant)
            standstill = np.isclose(
                np.diff(x, append=x[-1]) + np.diff(y, append=y[-1]), 0, atol=1e-3
            )
            if np.all(standstill):
                heading = initial_heading_guess * np.ones_like(x)
            else:
                heading = np.arctan2(np.diff(y, append=y[-1]), np.diff(x, append=x[-1]))

                standstill_start_idcs = np.where(
                    np.diff(standstill.astype(int), prepend=0) == 1
                )[0]
                standstill_end_idcs = np.where(
                    np.diff(standstill.astype(int), append=0) == -1
                )[0]
                for start_idx, end_idx in zip(
                    standstill_start_idcs, standstill_end_idcs
                ):
                    if start_idx == 0:
                        value = initial_heading_guess
                    else:
                        value = heading[start_idx - 1]
                    heading[start_idx : end_idx + 1] = value

        self._a = acceleration_vector
        self._v = speed_vector
        self._x = x
        self._y = y
        self._heading = heading

        if n_steps < full_n_steps:
            pad_len = full_n_steps - n_steps
            pad = np.full((pad_len,), np.nan)
            self._a = np.concatenate((self._a, pad))
            self._v = np.concatenate((self._v, pad))
            self._x = np.concatenate((self._x, pad))
            self._y = np.concatenate((self._y, pad))
            self._heading = np.concatenate((self._heading, pad))

        self._compiled = True

    def get_boundary_rect(self) -> tuple[float, float, float, float]:
        valid_mask = ~np.isnan(self._x) & ~np.isnan(self._y)
        if not np.any(valid_mask):
            return (np.nan, np.nan, np.nan, np.nan)
        return (
            np.nanmin(self._x[valid_mask]),
            np.nanmin(self._y[valid_mask]),
            np.nanmax(self._x[valid_mask]),
            np.nanmax(self._y[valid_mask]),
        )

    @staticmethod
    def _n_steps_from_duration_dt(duration: float, dt: float) -> int:
        n_steps = int(np.floor(duration / dt))
        return n_steps

    @staticmethod
    def _elapsed_time_from_n_steps_dt(n_steps: float, dt: float) -> float:
        elapsed_time = np.arange(n_steps, dt)
        return elapsed_time

    def _get_full_acceleration_vector(self, n_steps: int, dt: float) -> np.ndarray:
        if self._a_profile == "constant":
            acceleration_vector = self._a0 * np.ones((n_steps,))

            # Delay
            delay_steps = int(self._a_delay / dt)
            acceleration_vector[:delay_steps] = 0.0

        return acceleration_vector

    def _generate_lc_trajectory(
        self,
        lanelet_map: LaneletMap,
        s_lc0: float,
        v0: float,
        llt_id_lc0: int,
        llt_id_lc1: int,
        lc_duration: float,
        dt: float,
    ) -> np.ndarray:
        """
        Return LC trajectory in frenet frame of llt_lc0 (before lc)
        TODO: Adapt for lower speeds (see sad-rl for solution)
        """
        # !! COPIED FROM: https://gitlab.ika.rwth-aachen.de/lva/pilots/-/blob/main/pilots/highway_pilot/highway_pilot.py?ref_type=heads#L591
        # CHANGED!

        llt_lc0 = lanelet_map.laneletLayer[llt_id_lc0]
        llt_lc1 = lanelet_map.laneletLayer[llt_id_lc1]

        # Keep velocity, constant acceleration over lc_duration
        a = 0
        v1 = v0 + a * lc_duration

        # s position after lc
        s_lc1 = s_lc0 + ((v0 + v1) / 2) * lc_duration

        # Lateral positions in ref_frame
        x_lc0, y_lc0 = Road.from_frenet_to_cart(llt_lc0.centerline, s_lc0, 0)
        x_lc1, y_lc1 = Road.from_frenet_to_cart(llt_lc1.centerline, s_lc1, 0)

        _, t_lc0 = Road.from_cart_to_frenet(llt_lc0.centerline, x_lc0, y_lc0)
        _, t_lc1 = Road.from_cart_to_frenet(llt_lc0.centerline, x_lc1, y_lc1)

        # 5th order polynomial
        d = lc_duration
        p = np.array(
            [
                [0**5, 0**4, 0**3, 0**2, 0**1, 1],  # lateral start position
                [d**5, d**4, d**3, d**2, d**1, 1],  # lateral end positioin
                [0, 0, 0, 0, 1, 0],  # lateral start velocity
                [5 * d**4, 4 * d**3, 3 * d**2, 2 * d, 1, 0],  # lateral end velocity
                [0, 0, 0, 2, 0, 0],  # lateral start acceleration
                [20 * d**3, 12 * d**2, 6 * d, 2, 0, 0],
            ]
        )  # lateral end acceleration
        pv = np.linalg.solve(p, [t_lc0, t_lc1, 0.0, 0.0, 0, 0])

        trajectory_pts = []
        time_steps = np.arange(0, lc_duration + dt, dt)

        for t in time_steps:
            delta_s = s_lc0 + v0 * t + 0.5 * a * t * t
            delta_t = (
                pv[0] * t**5
                + pv[1] * t**4
                + pv[2] * t**3
                + pv[3] * t**2
                + pv[4] * t**1
                + pv[5]
            )
            trajectory_pts.append([delta_s, delta_t])

        trajectory_pts = np.array(trajectory_pts)

        return trajectory_pts

    def _plot_in_ax(
        self,
        ax: plt.Axes,
        timestep: int | None = None,
        include_trace: bool = False,
        color: str | None = None,
    ) -> None:
        if timestep and not (0 <= timestep < self._x.shape[0]):
            msg = "timestep is out of possible range."
            raise ValueError(msg)

        if color is None:
            color = "r"

        if timestep is None:
            if np.isnan(self._x[0]) or np.isnan(self._y[0]):
                return
            x = self._x[0]
            y = self._y[0]
            heading = self._heading[0]
        else:
            if np.isnan(self._x[timestep]) or np.isnan(self._y[timestep]):
                return
            x = self._x[timestep]
            y = self._y[timestep]
            heading = self._heading[timestep]

        rect_patch = Rectangle(
            (x - self.length / 2, y - self.width / 2),
            self.length,
            self.width,
            angle=np.rad2deg(heading),
            rotation_point="center",
            alpha=0.8,
            zorder=30,
            facecolor=color,
            label=f"Vehicle {self._vehicle_id}",
        )

        ax.add_patch(rect_patch)

        # Plot trajectory
        if include_trace:
            if timestep is None:
                ax.plot(
                    self._x, self._y, "-", color=rect_patch.get_facecolor(), zorder=31
                )
            else:
                ax.plot(
                    self._x[timestep:],
                    self._y[timestep:],
                    "-",
                    color=rect_patch.get_facecolor(),
                    zorder=31,
                )
        elif timestep is None:
            ax.plot(self._x, self._y, "x", color=rect_patch.get_facecolor(), zorder=31)

    def _format_ax(self, ax: plt.Axes, *args, **kwargs) -> None:  # noqa: ARG002
        margin = 5
        valid_mask = ~np.isnan(self._x) & ~np.isnan(self._y)
        if not np.any(valid_mask):
            return
        ax.set_xlim(
            np.nanmin(self._x[valid_mask]) - margin,
            np.nanmax(self._x[valid_mask]) + margin,
        )
        ax.set_ylim(
            np.nanmin(self._y[valid_mask]) - margin,
            np.nanmax(self._y[valid_mask]) + margin,
        )

        ax.set_aspect("equal")

        ax.set(title="Vehicle", xlabel="X position in m", ylabel="Y position in m")

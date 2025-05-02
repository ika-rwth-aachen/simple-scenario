from __future__ import annotations

import lanelet2

import numpy as np
from matplotlib.patches import Rectangle
from typing import TYPE_CHECKING
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.parameters_vehicle3 import parameters_vehicle3

from .rendering import Renderable

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from omegaconf import DictConfig
    from .road import Road


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
        lanelet_id: int,
        s0: float,
        t0: float,
        v0: float,
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
    ) -> None:
        """
        lanelet_id: ID of initial lanelet the vehicle is starting on
        s0: Initial longitudinal position relative to the given lanelet in m
        t0: Initial lateral offset to the center line of the given lanelet in m
        v0: Initial speed in m/s
        a: Acceleration in m/s^2
        a_delay: Time to wait before applying acceleration
        a_profile: How the acceleration changes over time
        lc_direction: Direction of lane change. 0: no lane change, 1: left, -1: right
        lc_delay: Time to wait before lane change
        lc_duration: Time the lane change takes to be completed
        lc_type: How the lc is done
        lc_vy: Only used if lc_type is "vy"
        vehicle_type_name: Type of the vehicle
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
            "lanelet_id": lanelet_id,
            "s0": s0,
            "t0": t0,
            "v0": v0,
            "a0": a0,
            "a_delay": a_delay,
            "a_profile": a_profile,
            "lc_direction": lc_direction,
            "lc_delay": lc_delay,
            "lc_duration": lc_duration,
            "lc_type": lc_type,
            "lc_vy": lc_vy,
            "vehicle_type_name": vehicle_type_name,
        }

        self._vehicle_id = vehicle_id
        self._lanelet_id = lanelet_id
        self._s0 = s0
        self._t0 = t0
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
            self._vehicle_parameters = self.VEHICLE_TYPE_PARAMETERS[
                self._vehicle_type_name
            ]()
            self._length = self._vehicle_parameters.l
            self._width = self._vehicle_parameters.w

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
    def from_data(
        cls,
        vehicle_id: int,
        lanelet_id: int,
        x: np.ndarray,
        y: np.ndarray,
        heading: np.ndarray,
        v: np.ndarray,
        a: np.ndarray,
        vehicle_type_name: str = "medium",
        length: float | None = None,
        width: float | None = None,
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
            vehicle_id,
            lanelet_id,
            None,
            None,
            heading[0],
            v[0],
            vehicle_type_name=vehicle_type_name,
            length=length,
            width=width,
            from_data=True,
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
    def lanelet_id(self) -> int:
        return self._lanelet_id

    @property
    def s0(self) -> float:
        if self._initialized_from_data:
            msg = "Not possible to access s0 when initialized from data."
            raise Exception(msg)
        return self._s0

    @property
    def t0(self) -> float:
        if self._initialized_from_data:
            msg = "Not possible to access t0 when initialized from data."
            raise Exception(msg)
        return self._t0

    @property
    def v0(self) -> float:
        return self._v0

    @property
    def length(self) -> float:
        return self._length

    @property
    def width(self) -> float:
        return self._width

    @property
    def x(self) -> np.ndarray:
        if not self._compiled:
            msg = "Please call .compile() before accessing this data."
            raise Exception(msg)
        return self._x

    @property
    def y(self) -> np.ndarray:
        if not self._compiled:
            msg = "Please call .compile() before accessing this data."
            raise Exception(msg)
        return self._y

    @property
    def heading(self) -> np.ndarray:
        if not self._compiled:
            msg = "Please call .compile() before accessing this data."
            raise Exception(msg)
        return self._heading

    @property
    def v(self) -> np.ndarray:
        if not self._compiled:
            msg = "Please call .compile() before accessing this data."
            raise Exception(msg)
        return self._v

    @property
    def a(self) -> np.ndarray:
        if not self._compiled:
            msg = "Please call .compile() before accessing this data."
            raise Exception(msg)
        return self._a

    @property
    def inverse_driving_direction(self) -> bool:
        return self._inverse_driving_direction

    @property
    def vehicle_type_name(self) -> str:
        return self._vehicle_type_name

    @property
    def initialized_from_data(self) -> bool:
        return self._initialized_from_data

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

    def compile(self, road: Road, duration: float, dt: float) -> None:  # noqa: PLR0912
        """
        Create absolute cartesian coordinates etc
        """

        # NEED LANELET2 MAP FIRST
        lanelet_map = road.lanelet_map

        # Find lanelet by id
        lanelet = lanelet_map.laneletLayer[self._lanelet_id]

        n_steps = self._n_steps_from_duration_dt(duration, dt)

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
        lon_position_vector = self._s0 * np.ones_like(speed_change_vector)
        lon_position_vector[1:] = (
            lon_position_vector[1:] + np.cumsum(lon_position_change_vector)[:-1]
        )

        # t
        lat_offset_vector = self._t0 * np.ones((n_steps,))
        lc_start_step = int(self._lc_delay / dt)

        # Lane change
        if self._lc_direction in (-1, 1):
            if self._lc_direction == -1:
                # Assumption: right lanelet ID is one less than initial lanelet id
                target_lanelet_id = self._lanelet_id - 1

                if target_lanelet_id not in lanelet_map.laneletLayer:
                    msg = "Lane change to the right not possible. No lanelet to the right."
                    raise ValueError(msg)

                target_lanelet = lanelet_map.laneletLayer[target_lanelet_id]

                if not lanelet2.geometry.rightOf(target_lanelet, lanelet):
                    msg = "Right lanelet has another ID than expected."
                    raise Exception(msg)

            elif self._lc_direction == 1:
                # Assumption: right lanelet ID is one more than initial lanelet id
                target_lanelet_id = self._lanelet_id + 1

                if target_lanelet_id not in lanelet_map.laneletLayer:
                    msg = (
                        "Lane change to the left not possible. No lanelet to the left."
                    )
                    raise ValueError(msg)

                target_lanelet = lanelet_map.laneletLayer[target_lanelet_id]

                if not lanelet2.geometry.leftOf(target_lanelet, lanelet):
                    msg = "Left lanelet has another ID than expected."
                    raise Exception(msg)

            if self._lc_type == "polynomial":
                lc_traj = self._generate_lc_trajectory(
                    road,
                    lon_position_vector[lc_start_step],
                    speed_vector[lc_start_step],
                    self._lanelet_id,
                    target_lanelet_id,
                    self._lc_duration,
                    dt,
                )

                n_lc_steps = lc_traj.shape[0]

                lat_offset_vector[lc_start_step : lc_start_step + n_lc_steps] = lc_traj[
                    :, 1
                ]
                lat_offset_vector[lc_start_step + n_lc_steps :] = lc_traj[-1, 1]

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
                initial_llt = road.lanelet_map.laneletLayer[self._lanelet_id]
                target_llt = road.lanelet_map.laneletLayer[target_lanelet_id]
                x_lc1, y_lc1 = road.from_frenet_to_cart(
                    target_llt.centerline, self._s0, 0
                )
                _, t_lc1 = road.from_cart_to_frenet(
                    initial_llt.centerline, x_lc1, y_lc1
                )
                max_t = t_lc1  # - self.width / 2

                if self._lc_direction == 1:
                    lat_offset_vector[lat_offset_vector > max_t] = max_t
                else:
                    lat_offset_vector[lat_offset_vector < max_t] = max_t

        # Positions to cartesian
        x, y = road.from_frenet_to_cart(
            lanelet.centerline, lon_position_vector, lat_offset_vector
        )

        # Heading, Assumption: Heading does not change in last time step
        if self._lc_type == "vy":
            if driving_direction == 1:
                heading = np.zeros_like(acceleration_vector)
            elif driving_direction == -1:
                heading = np.pi * np.ones_like(acceleration_vector)
            else:
                raise RuntimeError
        else:
            # Get initial heading guess from the road
            x_diff0, y_diff0 = road.from_frenet_to_cart(
                lanelet.centerline,
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

        self._compiled = True

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
        road: Road,
        s_lc0: float,
        v0: float,
        llt_id_lc0: int,
        llt_id_lc1: int,
        lc_duration: float,
        dt: float,
    ) -> np.ndarray:
        """
        Return LC trajectory in frenet frame of llt_lc0 (before lc)
        """
        # !! COPIED FROM: https://gitlab.ika.rwth-aachen.de/lva/pilots/-/blob/main/pilots/highway_pilot/highway_pilot.py?ref_type=heads#L591
        # CHANGED!

        llt_lc0 = road.lanelet_map.laneletLayer[llt_id_lc0]
        llt_lc1 = road.lanelet_map.laneletLayer[llt_id_lc1]

        # Keep velocity, constant acceleration over lc_duration
        a = 0
        v1 = v0 + a * lc_duration

        # s position after lc
        s_lc1 = s_lc0 + ((v0 + v1) / 2) * lc_duration

        # Lateral positions in ref_frame
        x_lc0, y_lc0 = road.from_frenet_to_cart(llt_lc0.centerline, s_lc0, 0)
        x_lc1, y_lc1 = road.from_frenet_to_cart(llt_lc1.centerline, s_lc1, 0)

        _, t_lc0 = road.from_cart_to_frenet(llt_lc0.centerline, x_lc0, y_lc0)
        _, t_lc1 = road.from_cart_to_frenet(llt_lc0.centerline, x_lc1, y_lc1)

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
            x = self._x[0]
            y = self._y[0]
            heading = self._heading[0]
        else:
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
        ax.set_xlim(np.min(self._x) - margin, np.max(self._x) + margin)
        ax.set_ylim(np.min(self._y) - margin, np.max(self._y) + margin)

        ax.set_aspect("equal")

        ax.set(title="Vehicle", xlabel="X position in m", ylabel="Y position in m")

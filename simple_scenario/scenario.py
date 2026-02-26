from __future__ import annotations

import io
import json
import matplotlib.pyplot as plt
import numpy as np
import warnings

from copy import deepcopy
from loguru import logger
from matplotlib.transforms import Bbox
from pathlib import Path
from PIL import Image
from scenariogeneration import xosc
from tqdm import tqdm

from . import CR_AVAILABLE

if CR_AVAILABLE:
    from commonroad_dc import pycrcc
    from commonroad.common.file_reader import CommonRoadFileReader
    from commonroad.common.solution import VehicleType
    from commonroad.scenario.trajectory import Trajectory, State
    from commonroad_dc.boundary import boundary
    from commonroad_dc.collision.trajectory_queries.trajectory_queries import (
        obb_enclosure_polygons_static,
    )
    from commonroad_dc.feasibility.feasibility_checker import trajectory_feasibility
    from commonroad_dc.feasibility.vehicle_dynamics import (
        VehicleDynamics,
        VehicleParameterMapping,
    )
    from shapely.errors import ShapelyDeprecationWarning

    from .lanelet_network_wrapper import LaneletNetworkWrapper
    from .commonroad_interface import CommonroadInterface

from .from_data_config_error import FromDataConfigError
from .ego_configuration import EgoConfiguration
from .rendering import Renderable, get_rcParams
from .road import (
    SyntheticRoad,
    StraightSegment,
    ClothoidSegment,
    ArcSegment,
    MappedRoad,
)
from .vehicle import Vehicle


class Scenario(Renderable):
    COMPILE_MODES = (
        "config",  # Such that it can be used to init again with Scenario.from_config()
        "cr",  # Save to commroad xml
        "openx",
        "lanelet2",
    )

    def __init__(
        self,
        scenario_id: str,
        road: SyntheticRoad,
        ego_configuration: EgoConfiguration,
        vehicles: list[Vehicle] | None,
        duration: float,
        dt: float = 0.1,
        from_data: bool = False,
        check_feasibility: bool = False,
        reference_to_map: str | Path | None = None,
        carla_metrics: list[str] | None = None,
        stop_on_ego_route_complete: bool = False,
    ) -> None:
        logger.info(f"Create simple scenario '{scenario_id}'")

        if abs(round(duration / dt) - duration / dt) > 1e-6:
            msg = "'duration' must be a multiple of 'dt'."
            raise ValueError(msg)

        if vehicles is None:
            vehicles = []

        vehicle_ids = [vehicle.id for vehicle in vehicles]
        if len(set(vehicle_ids)) != len(vehicle_ids):
            msg = f"There are multiple vehicles with the same id. (Unique IDs: {set(vehicle_ids)}, all IDs: {vehicle_ids})"
            raise ValueError(msg)

        # Save parameter values in config
        self._config = {
            "scenario_id": scenario_id,
            "road": road.config,
            "ego_configuration": ego_configuration.config,
            "vehicles": [v.config for v in vehicles],
            "duration": duration,
            "dt": dt,
            "reference_to_map": reference_to_map,
            "carla_metrics": carla_metrics,
            "stop_on_ego_route_complete": stop_on_ego_route_complete,
        }

        self._scenario_id = scenario_id
        self._road = road
        self._ego_configuration = ego_configuration
        self._vehicles = vehicles
        self._duration = duration
        self._dt = dt
        self._reference_to_map = reference_to_map
        self._carla_metrics = carla_metrics
        self._stop_on_ego_route_complete = stop_on_ego_route_complete

        # CR interface
        self._cr_interface = None

        self._compiled = False
        self._compile()

        self._initialized_from_data = from_data

        self._is_feasible = None
        if check_feasibility:
            self._is_feasible = self.check_feasibility()

    def __str__(self) -> str:
        return f"<Scenario {self._scenario_id} ({self._duration:.2f}s, dt: {self._dt:.2f}s, {len(self._vehicles)} vehicles)>"

    def __repr__(self) -> str:
        return self.__str__()

    def copy(
        self, copied_scenario_id_suffix: str = "copy", including_vehicles: bool = True
    ) -> Scenario:
        try:
            scenario_config_empty = deepcopy(self.config)
            scenario_config_empty["scenario_id"] += f"_{copied_scenario_id_suffix}"
            if not including_vehicles:
                scenario_config_empty["vehicles"] = []
            scenario_copy = Scenario.from_config(scenario_config_empty)

        except FromDataConfigError:
            # Scenario has been created from data, cannot use config
            road_copy = self._road.copy()

            vehicles = None
            if including_vehicles:
                vehicles = deepcopy(self._vehicles)

            scenario_copy = Scenario(
                scenario_id=f"{self._scenario_id}_{copied_scenario_id_suffix}",
                road=road_copy,
                ego_configuration=deepcopy(self._ego_configuration),
                vehicles=vehicles,
                duration=self._duration,
                dt=self._dt,
                from_data=True,
            )
        return scenario_copy

    @property
    def config(self) -> dict:
        if self._initialized_from_data:
            msg = "Not possible to access config when initialized from data."
            raise FromDataConfigError(msg)
        return self._config

    @classmethod
    def from_x(
        cls, scenario_or_config_or_file: Scenario | Path | str | dict
    ) -> Scenario:
        """
        Magic function that determines input.
        """

        if isinstance(scenario_or_config_or_file, cls):
            scenario = scenario_or_config_or_file

        elif isinstance(scenario_or_config_or_file, dict):
            scenario = cls.from_config(scenario_or_config_or_file)

        elif isinstance(scenario_or_config_or_file, (Path, str)):
            scenario_or_config_or_file = Path(scenario_or_config_or_file)

            if scenario_or_config_or_file.suffix == ".json":
                scenario = cls.from_config_file(scenario_or_config_or_file)

            elif scenario_or_config_or_file.suffix == ".xml":
                scenario = cls.from_cr_xml(scenario_or_config_or_file)

        else:
            msg = "Invalid value for 'scenario_or_config_or_file'."
            raise TypeError(msg)

        return scenario

    @classmethod
    def from_config_file(cls, config_file: tuple[str, Path]) -> Scenario:
        """
        Load from a json config file.
        """

        if not isinstance(config_file, (Path, str)):
            msg = "Wrong input type for config_file. Please provide a str or Path to the json."
            raise TypeError(msg)

        config_file = Path(config_file)

        if config_file.suffix != ".json":
            msg = "Please provid a path to a json file as config_file."
            raise TypeError(msg)

        with config_file.open("r") as f:
            config = json.load(f)

        # Relative to absolute paths
        if "from" in config.get("road", {}):
            for map_file_key in ("lanelet2_map_file", "opendrive_map_file"):
                map_file = config["road"]["from"].get(map_file_key, None)
                if map_file is not None:
                    config["road"]["from"][map_file_key] = (
                        config_file.parent / map_file
                    ).resolve()

        scenario = cls.from_config(config)

        return scenario

    @classmethod
    def from_config(cls, config: dict) -> Scenario:
        """
        All input paramters of all init functions of road, vehicles etc
        """

        compiled_config = deepcopy(config)

        # -- Road --
        # Check whether it is a real road or synthetic road
        # real road uses "from"
        is_real_road = (
            len(compiled_config["road"]) == 1 and "from" in compiled_config["road"]
        )

        if is_real_road:
            road = MappedRoad(**compiled_config["road"]["from"])

        else:
            if len(config["road"]["segments"]) == 1:
                segments = [StraightSegment(**config["road"]["segments"][0])]
            elif len(config["road"]["segments"]) == 3:
                segments = [
                    StraightSegment(**config["road"]["segments"][0]),
                    ClothoidSegment(**config["road"]["segments"][1]),
                    ArcSegment(**config["road"]["segments"][2]),
                ]
            else:
                msg = "Only 1 or 3 segments road segments are supported via config."
                raise ValueError(msg)
            compiled_config["road"]["segments"] = segments

            road = SyntheticRoad(**compiled_config["road"])

        compiled_config["road"] = road

        # -- Ego configuration --
        ego_configuration = EgoConfiguration(**config["ego_configuration"])
        compiled_config["ego_configuration"] = ego_configuration

        # -- Vehicles --
        if "vehicles" in config:
            vehicles = [Vehicle(**vehicle_config) for vehicle_config in config["vehicles"]]
        else:
            vehicles = []
        compiled_config["vehicles"] = vehicles

        # -- openDRIVE reference --
        if config.get("reference_to_map"):
            compiled_config["reference_to_map"] = config["reference_to_map"]

        # -- CARLA metrics --
        if config.get("carla_metrics"):
            compiled_config["carla_metrics"] = config["carla_metrics"]

        # -- Stop on ego route complete --
        if config.get("stop_on_ego_route_complete") is not None:
            compiled_config["stop_on_ego_route_complete"] = config[
                "stop_on_ego_route_complete"
            ]

        # -- Scenario --
        scenario = cls(**compiled_config)

        return scenario

    @classmethod
    def from_cr_xml(cls, cr_xml_path: str | Path) -> Scenario:  # noqa: PLR0912
        if not isinstance(cr_xml_path, (Path, str)):
            msg = "Wrong input type for cr_xml_path. Please provide a str or Path to the xml."
            raise TypeError(msg)

        cr_xml_path = Path(cr_xml_path)

        if cr_xml_path.suffix != ".xml":
            msg = f"Wrong file type. Need xml. Have: {cr_xml_path.suffix}."
            raise ValueError(msg)

        if not cr_xml_path.exists():
            msg = f"File at cr_xml_path={cr_xml_path} does not exist."
            raise FileNotFoundError(msg)

        if not CR_AVAILABLE:
            msg = "Please install the commonroad extra to use this feature. `pip install simple_scenario[commonroad]`"
            raise ModuleNotFoundError(msg)

        # Load file
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ShapelyDeprecationWarning)
            scenario, planning_problem_set = CommonRoadFileReader(
                str(cr_xml_path)
            ).open()

        # Extract planning problem
        planning_problems = list(planning_problem_set.planning_problem_dict.values())
        if len(planning_problems) != 1:
            msg = "Need exactly one planning problem"
            raise Exception(msg)
        planning_problem = planning_problems[0]

        # Road
        lanelet_network_wrapper = LaneletNetworkWrapper(scenario.lanelet_network)

        n_lanes = len(lanelet_network_wrapper.lanelet_list)
        # Check that is it a straight road
        road_lengths = [
            lanelet.distance[-1] - lanelet.distance[0]
            for lanelet in lanelet_network_wrapper.lanelet_list
        ]
        is_straight_road = all(road_lengths == road_lengths[0])
        if not is_straight_road:
            msg = "Can only handle straight roads atm."
            raise NotImplementedError(msg)
        road_length = road_lengths[0]

        lane_widths = []
        for lanelet in lanelet_network_wrapper.lanelet_list:
            diff = lanelet.left_vertices - lanelet.right_vertices
            dist = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
            if not np.all(dist == dist[0]):
                msg = "All lanes must have the same lane width throughout the lane."
                raise NotImplementedError(msg)

            lane_widths.append(dist[0])

        if not all(lane_widths == lane_widths[0]):
            msg = "All lanes must have the same lane width"
            raise NotImplementedError(msg)
        lane_width = lane_widths[0]

        speed_limits = []
        for lanelet in lanelet_network_wrapper.lanelet_list:
            centerline = lanelet_network_wrapper.get_llt_centerline(lanelet.lanelet_id)
            speed_limit = lanelet_network_wrapper.get_speed_limit_by_pos(
                centerline[0, 0], centerline[0, 1]
            )
            speed_limits.append(speed_limit)

        if not np.all(np.isclose(speed_limits, speed_limits[0])):
            msg = "The speed limit must be the same on all lanes"
            raise NotImplementedError(msg)
        speed_limit = speed_limits[0]

        # Starting position
        p0 = lanelet_network_wrapper.lanelet_list[-1].left_vertices[0]

        # Ego configuration
        ego_initial_pos_cart = planning_problem.initial_state.position
        ego_initial_lanelet_id = lanelet_network_wrapper.find_lanelet_id_by_position(
            *ego_initial_pos_cart
        )
        ego_initial_pos_frenet = lanelet_network_wrapper.from_cart_to_llt_frenet(
            ego_initial_lanelet_id, *ego_initial_pos_cart
        )
        # Extract goal region
        if len(planning_problem.goal.state_list) != 1:
            msg = "Planning problem must contain exactly one goal region."
            raise NotImplementedError(msg)
        goal_region = planning_problem.goal.state_list[0]
        # Assumption: A slim rectangle spanning over all lanes of the road at some s
        goal_region_center_x, goal_region_center_y = np.mean(
            goal_region.position.vertices, axis=0
        )
        goal_region_center_s, goal_region_center_t = (
            lanelet_network_wrapper.from_cart_to_ref_frenet(
                goal_region_center_x, goal_region_center_y
            )
        )

        ego_configuration = EgoConfiguration(
            start_lanelet_id=ego_initial_lanelet_id,
            start_s=ego_initial_pos_frenet[0],
            start_t=ego_initial_pos_frenet[1],
            v0=planning_problem.initial_state.velocity,
            target_s=goal_region_center_s,
            target_t=goal_region_center_t,
        )

        road = SyntheticRoad(
            n_lanes,
            lane_width,
            [StraightSegment(road_length)],
            speed_limit,
            x0=p0[0],
            y0=p0[1],
        )

        # Vehicles
        if len(scenario.obstacles) == 0:
            msg = "There must be at least one vehicle."
            raise NotImplementedError(msg)

        durations = []
        vehicles = []
        for obstacle in scenario.obstacles:
            # Caution: Initial state is separate from rest of trajectory
            all_states = [
                obstacle.initial_state,
                *obstacle.prediction.trajectory.state_list,
            ]
            n_states = len(all_states)
            durations.append(n_states)

            all_x = [obstacle.initial_state.position[0]]
            all_y = [obstacle.initial_state.position[1]]
            all_heading = [obstacle.initial_state.orientation]
            all_v = [obstacle.initial_state.velocity]

            for state in obstacle.prediction.trajectory.state_list:
                all_x.append(state.position[0])
                all_y.append(state.position[1])
                all_heading.append(state.orientation)
                all_v.append(state.velocity)

            all_x = np.array(all_x)
            all_y = np.array(all_y)
            all_heading = np.array(all_heading)
            all_v = np.array(all_v)

            # Calculate acceleration
            all_a = np.diff(all_v)
            all_a = np.append(all_a, all_a[-1])

            vehicle = Vehicle.from_data(
                vehicle_id=obstacle.obstacle_id,
                x=all_x,
                y=all_y,
                heading=all_heading,
                v=all_v,
                a=all_a,
            )

            vehicles.append(vehicle)

        if len(durations) > 1 and not np.all(np.array(durations) == durations[0]):
            msg = "All vehicles must have the same amount of states."
            raise NotImplementedError(msg)
        duration_steps = durations[0]

        dt = scenario.dt
        duration = duration_steps * dt

        scenario = cls(
            str(scenario.scenario_id),
            road,
            ego_configuration,
            vehicles,
            duration,
            dt,
            from_data=True,
        )

        return scenario

    def _compile(self) -> None:
        """
        Create absolute cartesian trajectories
        """

        # Compile ego confiuration
        if not self._ego_configuration.compiled:
            self._ego_configuration.compile(self._road)

        # Compile vehicles
        for vehicle in self._vehicles:
            if not vehicle.compiled:
                vehicle.compile(
                    self._road,
                    self._duration,
                    self._dt,
                )

        self._compiled = True

    @property
    def id(self) -> str:
        return self._scenario_id

    @property
    def road(self) -> SyntheticRoad:
        return self._road

    @property
    def ego_configuration(self) -> EgoConfiguration:
        return self._ego_configuration

    @property
    def vehicles(self) -> list[Vehicle]:
        return self._vehicles

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def step_start(self) -> int:
        """
        The first step of the scenario.
        """
        return 0

    @property
    def step_end(self) -> int:
        """
        The last step of the scenario. (Index, hint: If you are using )
        """
        return self.n_steps - 1

    @property
    def n_steps(self) -> int:
        """
        Total number of steps in the scenario.
        """
        return round(self._duration / self._dt)

    @property
    def steps(self) -> np.ndarray:
        """
        Array with all steps (step indices) in the scenario.
        """
        return np.arange(self.step_start, self.step_end + 1)

    def get_boundary_rect(
        self, show_full_road: bool = False
    ) -> tuple[float, float, float, float]:
        # Ego configuration
        xmin, ymin, xmax, ymax = self._ego_configuration.get_boundary_rect()

        # Vehicles
        for vehicle in self._vehicles:
            vehicle_xmin, vehicle_ymin, vehicle_xmax, vehicle_ymax = (
                vehicle.get_boundary_rect()
            )
            if np.isnan(
                [vehicle_xmin, vehicle_ymin, vehicle_xmax, vehicle_ymax]
            ).any():
                continue
            xmin = min(xmin, vehicle_xmin)
            ymin = min(ymin, vehicle_ymin)
            xmax = max(xmax, vehicle_xmax)
            ymax = max(ymax, vehicle_ymax)

        # Road
        if show_full_road:
            road_xmin, road_ymin, road_xmax, road_ymax = self._road.get_boundary_rect()
            xmin = min(xmin, road_xmin)
            ymin = min(ymin, road_ymin)
            xmax = max(xmax, road_xmax)
            ymax = max(ymax, road_ymax)

        return (xmin, ymin, xmax, ymax)

    @property
    def cr_interface(self) -> None:
        msg = "The cr_interface property is deprecated, because it is slow and slows down debugging. Use get_cr_interface() instead."
        raise RuntimeError(msg)

    def get_cr_interface(self) -> CommonroadInterface:
        if not CR_AVAILABLE:
            msg = "Please install the commonroad extra for this feature: pip install simple_scenario[commonroad]"
            raise ModuleNotFoundError(msg)

        if self._cr_interface is None:
            self._cr_interface = self._create_cr_interface()
        return self._cr_interface

    def _create_cr_interface(self) -> CommonroadInterface:
        commonroad_interface = CommonroadInterface(
            self._config, self._ego_configuration, self._vehicles, self._road
        )

        return commonroad_interface

    def is_feasible(
        self, check_trajectories: bool = True, check_collision: bool = True
    ) -> bool:
        if self._is_feasible is None:
            self._is_feasible = self.check_feasibility(
                check_trajectories, check_collision
            )
        return self._is_feasible

    def check_feasibility(  # noqa: PLR0912
        self, check_trajectories: bool = True, check_collision: bool = True
    ) -> bool:
        logger.debug(f"Scenario '{self._scenario_id}': Check feasibility")

        if not CR_AVAILABLE:
            msg = "Please install the commonroad extra to use this feature. `pip install simple_scenario[commonroad]`"
            raise ModuleNotFoundError(msg)

        if check_trajectories:
            allowed_error_posx = 5e-2  # 5cm
            allowed_error_posy = 5e-2  # 5cm
            allowed_error_orientation = 3e-2

            vehicle_dynamic = VehicleDynamics.KS(VehicleType.BMW_320i)
            vehicle_parameters = VehicleParameterMapping.from_vehicle_type(
                VehicleType.BMW_320i
            )
            l_wb = vehicle_parameters.a + vehicle_parameters.b

            for vehicle in self._vehicles:
                valid_mask = ~np.isnan(vehicle.x) & ~np.isnan(vehicle.y)
                if not np.any(valid_mask):
                    continue

                x = vehicle.x[valid_mask]
                y = vehicle.y[valid_mask]
                heading = vehicle.heading[valid_mask]
                v = vehicle.v[valid_mask]

                psi_dot = np.diff(heading, append=heading[-1]) / self._dt
                safe_v = np.nan * np.ones_like(v)
                safe_v[~np.isclose(v, 0, atol=1e-3)] = v[
                    ~np.isclose(v, 0, atol=1e-3)
                ]
                steering_angle = np.arctan(np.divide(psi_dot * l_wb, safe_v))
                steering_angle[np.isnan(steering_angle)] = 0.0

                new_state_list = []
                for i, _ in enumerate(x):
                    new_state_list.append(
                        State(
                            position=np.array((x[i], y[i])),
                            steering_angle=steering_angle[i],
                            velocity=v[i],
                            orientation=heading[i],
                            time_step=i,
                        )
                    )

                object_trajectory = Trajectory(0, new_state_list)
                feasible, _ = trajectory_feasibility(
                    object_trajectory,
                    vehicle_dynamic,
                    self._dt,
                    e=np.array(
                        [
                            allowed_error_posx,
                            allowed_error_posy,
                            allowed_error_orientation,
                        ]
                    ),
                )

                if not feasible:
                    msg = f"scenario '{self._scenario_id}': Not feasible, because of trajectory."
                    logger.error(msg)
                    return False

        if check_collision:
            # Check collision (see https://gitlab.ika.rwth-aachen.de/lva/scenario-sim-env/-/blob/main/scenario_sim_env/simulation_core.py?ref_type=heads#L217)
            cr_scenario = self.get_cr_interface().scenario

            # Road
            road_inclusion_polygon_group = boundary.create_road_polygons(
                cr_scenario,
                method="lane_polygons",
                buffer=1,
                resample=1,
                triangulate=False,
            )
            _, road_boundary_collision_object = boundary.create_road_boundary_obstacle(
                cr_scenario
            )

            # Collision objects
            timesteps = int(self._duration / self._dt)
            for timestep in range(timesteps):
                # Create collision objects
                collision_objs = []
                for do in cr_scenario.dynamic_obstacles:
                    if timestep == 0:
                        state = do.initial_state
                    elif (
                        do.prediction.initial_time_step
                        <= timestep
                        <= do.prediction.final_time_step
                    ):
                        state = do.prediction.trajectory.state_at_time_step(timestep)
                    else:
                        continue

                    collision_obj = pycrcc.RectOBB(
                        do.obstacle_shape.length / 2,
                        do.obstacle_shape.width / 2,
                        state.orientation,
                        state.position[0],
                        state.position[1],
                    )
                    collision_objs.append(collision_obj)

                for collision_obj in collision_objs:
                    # Check road collisiion
                    if timestep == 0:
                        is_offroad = not obb_enclosure_polygons_static(
                            road_inclusion_polygon_group, collision_obj
                        )

                    is_offroad = is_offroad or road_boundary_collision_object.collide(
                        collision_obj
                    )

                    if is_offroad:
                        msg = f"scenario '{self._scenario_id}': Not feasible, because of offroad (timestep: {timestep})."
                        logger.error(msg)
                        return False

                    # Check vehicle collision

                    for other_collision_obj in collision_objs:
                        if collision_obj is other_collision_obj:
                            continue

                        is_collision = collision_obj.collide(other_collision_obj)

                        if is_collision:
                            msg = f"scenario '{self._scenario_id}': Not feasible, because of collision (timestep: {timestep})."
                            logger.error(msg)
                            return False

        return True

    def render(
        self,
        plot_dir_or_ax: Path | plt.Axes,
        plot_name_suffix: str | None = None,
        dpi: int = 600,
        figw: int = 9,
        figh: int = 9,
        clean: bool = False,
        *args,
        **kwargs,
    ) -> None:

        plot_name = f"{self._scenario_id}"
        if plot_name_suffix:
            plot_name += f"_{plot_name_suffix}"

        super().render(
            plot_dir_or_ax,
            *args,
            plot_name=plot_name,
            dpi=dpi,
            figw=figw,
            figh=figh,
            clean=clean,
            **kwargs,
        )

    def _plot_in_ax(
        self,
        ax: plt.Axes,
        timestep: int | None = None,
        *args,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> None:
        # Plot road
        self._road.render(ax)

        # Plot ego configuration
        self._ego_configuration.render(ax)

        # Plot vehicles and their trajectories
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i_vehicle, vehicle in enumerate(self._vehicles):
            vehicle.render(ax, timestep=timestep, color=colors[i_vehicle % len(colors)])

    def _format_ax(
        self,
        ax: plt.Axes,
        timestep: int | None = None,
        *args,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> None:
        # Set axis limits
        margin = 10
        xmin, ymin, xmax, ymax = self.get_boundary_rect()
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)

        title = f"Scenario '{self._scenario_id}'"
        if timestep:
            title += f" ({timestep * self._dt:.1f}s)"
        ax.set_title(title)

        ax.legend(ncols=3)

        ax.set_aspect("equal")

    def render_gif(
        self,
        save_dir: Path,
        include_traces: bool = True,
        plot_name_suffix: str = "",
        dpi: int = 300,
        crop_gif: bool = True,
    ) -> Path:
        logger.debug(f"Create gif of scenario '{self._scenario_id}'")

        if isinstance(self._road, MappedRoad):
            msg = "Cannot create gif of mapped road. (Would work, but takes a long time, because the road may be very large.)"
            raise NotImplementedError(msg)

        save_dir = Path(save_dir)

        frames = []

        for timestep in tqdm(self.steps, desc="Creating gif frames"):
            with plt.rc_context(get_rcParams(dpi=dpi, hide_ticks=True)):
                f, ax = plt.subplots()

                # Plot road
                self._road.render(ax)

                # Plot ego configuration
                self._ego_configuration.render(ax)

                # Plot vehicles and their trajectories
                colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                for i_vehicle, vehicle in enumerate(self._vehicles):
                    vehicle.render(
                        ax,
                        timestep=timestep,
                        include_trace=include_traces,
                        color=colors[i_vehicle % len(colors)],
                    )

                # Set axis limits
                margin = 10
                xmin, ymin, xmax, ymax = self.get_boundary_rect()
                ax.set_xlim(xmin - margin, xmax + margin)
                ax.set_ylim(ymin - margin, ymax + margin)

                ax.text(
                    0.01,
                    0.99,
                    f"{np.round((timestep * self._dt), 1)}s",
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                )

                ax.set_aspect("equal")

                f.set_dpi(dpi)

                f_size_inches = f.get_size_inches()
                fw = f_size_inches[0]
                fh = f_size_inches[1]
                rect = (0, 0, fw, fh)

                if crop_gif:
                    # Create rect out of tightbox (inches)
                    tightbox = f.get_tightbbox()
                    # rect : tuple (left, bottom, right, top), default: (0, 0, 1, 1)
                    rect = np.array(
                        [
                            (tightbox.x0 / fw) * fw,
                            (tightbox.y0 / fh) * fh,
                            (tightbox.x1 / fw) * fw,
                            (tightbox.y1 / fh) * fh,
                        ]
                    )
                    rect = np.round(rect, 1)
                    rect = tuple(rect)

                with io.BytesIO() as io_buf:
                    f.savefig(
                        io_buf,
                        format="raw",
                        dpi=dpi,
                        bbox_inches=Bbox.from_extents(*rect),
                    )
                    io_buf.seek(0)
                    img_data = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)

                height_px = int((rect[3] - rect[1]) * dpi)
                width_px = int((rect[2] - rect[0]) * dpi)
                img_arr = np.reshape(img_data, newshape=(height_px, width_px, -1))

                frame = Image.fromarray(img_arr)
                frames.append(frame)

                plt.close(f)

        plot_name = f"{self._scenario_id}"
        if plot_name_suffix:
            plot_name += f"_{plot_name_suffix}"

        gif_filepath = save_dir / f"{plot_name}.gif"

        logger.debug(f"Start gif file creation: {gif_filepath}")
        # Create the GIF file
        with io.BytesIO() as gif_buffer:
            frames[0].save(
                gif_buffer,
                save_all=True,
                append_images=frames[1:],
                format="gif",
                loop=0,
            )
            gif_file_content = gif_buffer.getvalue()

        with gif_filepath.open("wb") as f:
            f.write(gif_file_content)

        logger.debug(f"Successfully created gif file: {gif_filepath}")

        return gif_filepath

    def save(self, result_dir: Path | str | list, mode: str = "config") -> None:
        """
        modes: see self.COMPILE_MODES
        """

        if mode not in self.COMPILE_MODES:
            msg = "Compile mode not available"
            raise ValueError(msg)

        if isinstance(result_dir, list) and not (
            mode == "openx" and 0 < len(result_dir) <= 2
        ):
            msg = "Too less or many result directories provided."
            raise ValueError(msg)

        result_dir = result_dir if isinstance(result_dir, list) else [result_dir]
        result_dir = [Path(d) if d is not None else d for d in result_dir]

        save_handlers = {
            "config": self._save_config,
            "cr": self._save_cr,
            "openx": self._save_openx,
            "lanelet2": self._save_lanelet2,
        }
        save_handlers[mode](result_dir)

    def _save_config(self, result_dir: list[Path | None]) -> None:
        if self._initialized_from_data:
            msg = "Cannot save to config if the scenario has been initialized from data."
            raise ValueError(msg)

        config_file = result_dir[0] / f"{self._scenario_id}.json"

        with config_file.open("w") as f:
            json.dump(self._config, f, indent=2)

    def _save_cr(self, result_dir: list[Path | None]) -> None:
        self.get_cr_interface().to_xml(result_dir[0])

    def _save_openx(self, result_dir: list[Path | None]) -> None:
        if self._initialized_from_data:
            msg = "Cannot save to openx if the scenario has been initialized from data."
            raise ValueError(msg)

        scr_dir = result_dir[0]
        odr_dir = result_dir[0] if len(result_dir) == 1 else result_dir[1]

        # OpenDRIVE
        if odr_dir is not None:
            odr_path = self._road.save_opendrive_map(odr_dir, self._scenario_id)

        # OpenSCENARIO
        if self._reference_to_map:
            odr_path = Path(self._reference_to_map)

        if scr_dir is not None:
            osc = self._create_openscenario(odr_path)
            osc.write_xml(str(scr_dir / f"{self._scenario_id}.xosc"))

    def _save_lanelet2(self, result_dir: list[Path | None]) -> None:
        if self._initialized_from_data:
            msg = "Cannot save to lanelet2 if the scenario has been initialized from data."
            raise ValueError(msg)

        self._road.save_lanelet2_map(result_dir[0], self._scenario_id)

    def _create_openscenario(self, odr_path: str | Path) -> xosc.Scenario:
        """
        OpenScenario for esmini
        Inspired by https://gitlab.ika.rwth-aachen.de/scenario-based-validation/smartervalidation/-/blob/MA-Warmuth/smartervalidation/simulation/esmini_simulation.py?ref_type=heads#L81

        Idea: OSC file consists of just one story that includes FollowTrajectoryActions for all vehicles in the scenario.

        xosc+xodr files need to be placed together into a subfolder in esminis resource folder

        Run by `esmini --osc "D:\Programme\esmini-2.37.10\resources\test_openx_export_default\test.xosc" --window 60 60 1024 576`
        """  # noqa: W605

        vehicle_catalog_path = "../xosc/Catalogs/Vehicles"
        model_blue_car_path = "../models/car_blue.osgb"
        model_red_car_path = "../models/car_red.osgb"

        scenario_parameters = xosc.ParameterDeclarations()

        catalog = xosc.Catalog()
        catalog.add_catalog("VehicleCatalog", vehicle_catalog_path)

        road_network = xosc.RoadNetwork(roadfile=str(odr_path))

        entities = self._create_openx_entities(model_blue_car_path, model_red_car_path)
        init, step_time = self._create_openx_init()
        storyboard = self._create_openx_storyboard(init)
        story = self._create_openx_story(step_time)
        storyboard.add_story(story)

        osc_scenario = xosc.Scenario(
            name=self._scenario_id,
            author="simple_scenario",
            parameters=scenario_parameters,
            entities=entities,
            storyboard=storyboard,
            roadnetwork=road_network,
            catalog=catalog,
            osc_minor_version=1,
        )

        return osc_scenario

    def _create_openx_entities(
        self,
        model_blue_car_path: str,
        model_red_car_path: str,
    ) -> xosc.Entities:
        entities = xosc.Entities()

        ego_boundingbox = xosc.BoundingBox(
            width=self._ego_configuration.width,
            length=self._ego_configuration.length,
            height=1.8,
            x_center=2.0,
            y_center=0,
            z_center=0.9,
        )
        ego_front_axle = xosc.Axle(0.523598775598, 0.8, 1.68, 2.98, 0.4)
        ego_rear_axle = xosc.Axle(0.523598775598, 0.8, 1.68, 0, 0.4)
        ego_vehicle_object = xosc.Vehicle(
            name="car_white",
            vehicle_type=xosc.VehicleCategory.car,
            boundingbox=ego_boundingbox,
            frontaxle=ego_front_axle,
            rearaxle=ego_rear_axle,
            max_speed=self._ego_configuration.v_lon_max,
            max_acceleration=self._ego_configuration.a_lon_max,
            max_deceleration=-self._ego_configuration.a_lon_min,
        )
        ego_vehicle_object.add_property_file(model_blue_car_path)
        ego_vehicle_id = "ego_vehicle"
        ego_vehicle_object.add_property("model_id", "ego")
        ego_vehicle_object.add_property("type", ego_vehicle_id)

        entities.add_scenario_object(ego_vehicle_id, ego_vehicle_object)

        max_acceleration = 9.81
        max_deceleration = 9.81

        for vehicle in self._vehicles:
            vehicle_boundingbox = xosc.BoundingBox(
                width=vehicle.width,
                length=vehicle.length,
                height=1.5,
                x_center=1.3,
                y_center=0,
                z_center=0.8,
            )
            vehicle_front_axle = xosc.Axle(0.523598775598, 0.8, 1.68, 2.98, 0.4)
            vehicle_rear_axle = xosc.Axle(0.523598775598, 0.8, 1.68, 0, 0.4)

            vehicle_object = xosc.Vehicle(
                name="car_red",
                vehicle_type=xosc.VehicleCategory.car,
                boundingbox=vehicle_boundingbox,
                frontaxle=vehicle_front_axle,
                rearaxle=vehicle_rear_axle,
                max_speed=69,
                max_acceleration=max_acceleration,
                max_deceleration=max_deceleration,
            )
            vehicle_object.add_property_file(model_red_car_path)
            vehicle_object.add_property("model_id", f"other_{vehicle.id}")
            vehicle_object.add_property("type", "other_vehicle")
            entities.add_scenario_object(f"other_{vehicle.id}", vehicle_object)

        return entities

    @staticmethod
    def _create_world_position(
        x: float,
        y: float,
        *,
        h: float | None = None,
        z: float | None = None,
    ) -> xosc.WorldPosition:
        kwargs = {}
        if h is not None:
            kwargs["h"] = h
        if z is not None:
            kwargs["z"] = z
        return xosc.WorldPosition(x, y, **kwargs)

    def _create_openx_init(
        self,
    ) -> tuple[xosc.Init, xosc.TransitionDynamics]:
        init = xosc.Init()
        step_time = xosc.TransitionDynamics(
            xosc.DynamicsShapes.step, xosc.DynamicsDimension.time, 1
        )

        ego_initial_speed_action = xosc.AbsoluteSpeedAction(
            self._ego_configuration.v0, step_time
        )
        init.add_init_action("ego_vehicle", ego_initial_speed_action)

        ego_start_x, ego_start_y, ego_start_heading = (
            self._road.from_llt_local_to_opendrive_local(
                self._ego_configuration.start_x,
                self._ego_configuration.start_y,
                self._ego_configuration.start_heading,
            )
        )
        ego_start_position_action = xosc.TeleportAction(
            self._create_world_position(
                ego_start_x,
                ego_start_y,
                h=ego_start_heading,
                z=self._ego_configuration.z,
            )
        )
        init.add_init_action("ego_vehicle", ego_start_position_action)

        if self._ego_configuration.controller:
            controller_value = self._ego_configuration.controller
            controller_props = xosc.Properties()
            controller_props.add_property(name="module", value=controller_value)

            controller_props.add_property(
                name="initial_speed",
                value=str(self._ego_configuration.v0),
            )
            controller = xosc.Controller("CustomController", controller_props)

            ego_override_controller_value_action = xosc.OverrideControllerValueAction()
            ego_override_controller_value_action.throttle_active = False
            ego_override_controller_value_action.brake_active = False
            ego_override_controller_value_action.gear_active = False

            ego_controller_action = xosc.ControllerAction(
                assignControllerAction=xosc.AssignControllerAction(
                    controller=controller
                ),
                overrideControllerValueAction=ego_override_controller_value_action,
            )
            init.add_init_action("ego_vehicle", ego_controller_action)

        for vehicle in self._vehicles:
            if not vehicle.depends_on_ego:
                vehicle_initial_speed_action = xosc.AbsoluteSpeedAction(
                    vehicle.v0, step_time
                )
                init.add_init_action(
                    f"other_{vehicle.id}", vehicle_initial_speed_action
                )

            xodr_local_x, xodr_local_y, xodr_local_heading = (
                self._road.from_llt_local_to_opendrive_local(
                    vehicle.x[0], vehicle.y[0], vehicle.heading[0]
                )
            )
            vehicle_initial_position_action = xosc.TeleportAction(
                self._create_world_position(
                    xodr_local_x,
                    xodr_local_y,
                    h=xodr_local_heading,
                    z=vehicle.z,
                )
            )
            init.add_init_action(
                f"other_{vehicle.id}", vehicle_initial_position_action
            )

        return init, step_time

    def _create_openx_storyboard(self, init: xosc.Init) -> xosc.StoryBoard:
        if self._carla_metrics:
            stoptrigger_storyboard = xosc.ConditionGroup("stop")
            for test in self._carla_metrics:
                name = test["name"]
                delay = float(test["delay"]) if test.get("delay") else 0.0
                condition_edge = (
                    getattr(xosc.ConditionEdge, test["conditionEdge"])
                    if test.get("conditionEdge")
                    else xosc.ConditionEdge.rising
                )
                reference_parameter = (
                    test["parameterRef"] if test.get("parameterRef") else ""
                )
                value = int(test["value"]) if test.get("value") else 0
                rule = (
                    getattr(xosc.Rule, test["rule"])
                    if test.get("rule")
                    else xosc.Rule.lessThan
                )

                if name == "criteria_DrivenDistanceTest" and value == 0.0:
                    value = 0.95 * self._ego_configuration.route_length

                stoptrigger = xosc.ValueTrigger(
                    name,
                    delay,
                    condition_edge,
                    xosc.ParameterCondition(reference_parameter, value, rule),
                    "stop",
                )
                stoptrigger_storyboard.add_condition(stoptrigger)

            return xosc.StoryBoard(init, stoptrigger_storyboard)

        return xosc.StoryBoard(init)

    def _create_openx_story(
        self, step_time: xosc.TransitionDynamics
    ) -> xosc.Story:
        storyparam = xosc.ParameterDeclarations()
        story = xosc.Story(f"Act_scenario_{self._scenario_id}", storyparam)

        stoptrigger_act = self._create_openx_stoptrigger_act()
        act = xosc.Act(f"Act_scenario_{self._scenario_id}", stoptrigger=stoptrigger_act)

        self._add_ego_route_to_act(act)

        for vehicle in self._vehicles:
            self._add_vehicle_follow_trajectory(act, vehicle, step_time)

        story.add_act(act)

        return story

    def _create_openx_stoptrigger_act(self) -> xosc.Trigger:
        stoptrigger_act = xosc.Trigger("stop")
        stoptrigger_time = xosc.ValueTrigger(
            "StoptriggerTime",
            0,
            xosc.ConditionEdge.rising,
            xosc.SimulationTimeCondition(self._duration, xosc.Rule.greaterThan),
            triggeringpoint="stop",
        )
        stoptrigger_time_group = xosc.ConditionGroup("stop")
        stoptrigger_time_group.add_condition(stoptrigger_time)
        stoptrigger_act.add_conditiongroup(stoptrigger_time_group)

        if self._stop_on_ego_route_complete:
            stoptrigger_ego_route_done = xosc.ValueTrigger(
                "StoptriggerEgoRoute",
                0,
                xosc.ConditionEdge.rising,
                xosc.StoryboardElementStateCondition(
                    xosc.StoryboardElementType.event,
                    "Event_ego_vehicle",
                    xosc.StoryboardElementState.completeState,
                ),
                triggeringpoint="stop",
            )
            stoptrigger_route_group = xosc.ConditionGroup("stop")
            stoptrigger_route_group.add_condition(stoptrigger_ego_route_done)
            stoptrigger_act.add_conditiongroup(stoptrigger_route_group)

        return stoptrigger_act

    def _add_ego_route_to_act(self, act: xosc.Act) -> None:
        maneuver_group = xosc.ManeuverGroup("ManeuverGroup_ego_vehicle")
        maneuver_group.add_actor("ego_vehicle")

        maneuver = xosc.Maneuver("Maneuver_ego_vehicle")
        event = xosc.Event("Event_ego_vehicle", xosc.Priority.parallel)

        ego_route = xosc.Route("Route_ego_vehicle", closed=False)

        ego_start_x, ego_start_y, ego_start_heading = (
            self._road.from_llt_local_to_opendrive_local(
                self._ego_configuration.start_x,
                self._ego_configuration.start_y,
                self._ego_configuration.start_heading,
            )
        )
        ego_target_x, ego_target_y = (
            self._road.from_llt_local_to_opendrive_local(
                self._ego_configuration.target_x,
                self._ego_configuration.target_y,
            )
        )

        vehicle_position = self._create_world_position(
            ego_start_x,
            ego_start_y,
            h=ego_start_heading,
            z=self._ego_configuration.z,
        )
        ego_route.add_waypoint(vehicle_position, "shortest")

        vehicle_position = self._create_world_position(
            ego_target_x,
            ego_target_y,
            z=self._ego_configuration.z,
        )
        ego_route.add_waypoint(vehicle_position, "shortest")

        action = xosc.AssignRouteAction(ego_route)
        event.add_action("AssignRouteAction_ego_vehicle", action)

        maneuver.add_event(event)
        maneuver_group.add_maneuver(maneuver)
        act.add_maneuver_group(maneuver_group)

    def _add_vehicle_follow_trajectory(
        self,
        act: xosc.Act,
        vehicle: Vehicle,
        step_time: xosc.TransitionDynamics,
    ) -> None:
        maneuver_group = xosc.ManeuverGroup(f"ManeuverGroup_vehicle_{vehicle.id}")
        maneuver_group.add_actor(f"other_{vehicle.id}")

        maneuver = xosc.Maneuver(f"Maneuver_vehicle_{vehicle.id}")
        event = xosc.Event(f"Event_vehicle_{vehicle.id}", xosc.Priority.overwrite)

        vehicle_trajectory = xosc.Trajectory(
            f"Trajectory_vehicle_{vehicle.id}", closed=False
        )

        vehicle_positions = []
        valid_indices = np.where(~np.isnan(vehicle.x))[0]
        if valid_indices.size == 0:
            return
        for i in valid_indices:
            xodr_local_x, xodr_local_y, xodr_local_heading = (
                self._road.from_llt_local_to_opendrive_local(
                    vehicle.x[i], vehicle.y[i], vehicle.heading[i]
                )
            )
            vehicle_position = self._create_world_position(
                xodr_local_x,
                xodr_local_y,
                h=xodr_local_heading,
                z=vehicle.z,
            )
            vehicle_positions.append(vehicle_position)

        vehicle_step_times = (valid_indices * self._dt).tolist()
        vehicle_polyline = xosc.Polyline(vehicle_step_times, vehicle_positions)
        vehicle_trajectory.add_shape(vehicle_polyline)

        vehicle_follow_trajectory_action = xosc.FollowTrajectoryAction(
            vehicle_trajectory,
            xosc.FollowingMode.position,
            reference_domain=xosc.ReferenceContext.absolute,
            scale=1,
            offset=0,
        )
        event.add_action(
            f"FollowTrajectoryAction_vehicle_{vehicle.id}",
            vehicle_follow_trajectory_action,
        )

        if vehicle.depends_on_ego:
            vehicle_speed_action = xosc.AbsoluteSpeedAction(vehicle.v0, step_time)
            event.add_action(
                f"SpeedAction_vehicle_{vehicle.id}",
                vehicle_speed_action,
            )

            condition = xosc.EntityTrigger(
                f"EntityDistanceCondition_vehicle_{vehicle.id}",
                0,
                xosc.ConditionEdge.rising,
                xosc.TraveledDistanceCondition(1.0),
                "ego_vehicle",
                xosc.TriggeringEntitiesRule.any,
            )

            condition_group = xosc.ConditionGroup()
            condition_group.add_condition(condition)

            start_trigger = xosc.Trigger("start")
            start_trigger.add_conditiongroup(condition_group)
            event.add_trigger(start_trigger)

        maneuver.add_event(event)
        maneuver_group.add_maneuver(maneuver)
        act.add_maneuver_group(maneuver_group)

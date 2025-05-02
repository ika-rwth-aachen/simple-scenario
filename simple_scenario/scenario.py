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
from .road import Road, StraightSegment, ClothoidSegment, ArcSegment
from .vehicle import Vehicle


class Scenario(Renderable):
    COMPILE_MODES = (
        "config",  # Such that it can be used to init again with Scenario.from_config()
        "cr",  # Save to commroad xml
        "openx",
    )

    def __init__(
        self,
        scenario_id: str,
        road: Road,
        ego_configuration: EgoConfiguration,
        vehicles: list[Vehicle] | None,
        duration: float,
        dt: float = 0.1,
        from_data: bool = False,
        check_feasibility: bool = False,
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
        }

        self._scenario_id = scenario_id
        self._road = road
        self._ego_configuration = ego_configuration
        self._vehicles = vehicles
        self._duration = duration
        self._dt = dt

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

        scenario = cls.from_config(config)

        return scenario

    @classmethod
    def from_config(cls, config: dict) -> Scenario:
        """
        All input paramters of all init functions of road, vehicles etc
        """

        compiled_config = deepcopy(config)

        # Road
        if len(config["road"]["segments"]) == 1:
            segments = [StraightSegment(**config["road"]["segments"][0])]
        elif len(config["road"]["segments"]) == 3:
            segments = [
                StraightSegment(**config["road"]["segments"][0]),
                ClothoidSegment(**config["road"]["segments"][1]),
                ArcSegment(**config["road"]["segments"][2]),
            ]
        else:
            raise ValueError
        compiled_config["road"]["segments"] = segments

        road = Road(**compiled_config["road"])
        compiled_config["road"] = road

        # Ego configuration
        ego_configuration = EgoConfiguration(**config["ego_configuration"])
        compiled_config["ego_configuration"] = ego_configuration

        # Vehicles
        vehicles = [Vehicle(**vehicle_config) for vehicle_config in config["vehicles"]]

        compiled_config["vehicles"] = vehicles

        # Scenario
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
        ego_configuration = EgoConfiguration(
            ego_initial_lanelet_id,
            *ego_initial_pos_frenet,
            planning_problem.initial_state.velocity,
        )

        if len(planning_problem.goal.state_list) != 1:
            msg = "Planning problem must contain exactly one goal region."
            raise NotImplementedError(msg)
        goal_region = planning_problem.goal.state_list[0]
        # Assumption: A slim rectangle spanning over all lanes of the road at some s
        goal_region_x = goal_region.position.vertices[:, 0].mean()
        goal_region_s, _ = lanelet_network_wrapper.from_cart_to_ref_frenet(
            goal_region_x, 0
        )

        road = Road(
            n_lanes,
            lane_width,
            [StraightSegment(road_length)],
            speed_limit,
            goal_region_s,
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

            obstacle_lanelet_id = lanelet_network_wrapper.find_lanelet_id_by_position(
                *obstacle.initial_state.position
            )

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
                obstacle.obstacle_id,
                obstacle_lanelet_id,
                all_x,
                all_y,
                all_heading,
                all_v,
                all_a,
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
                vehicle.compile(self._road, self._duration, self._dt)

        self._compiled = True

    @property
    def id(self) -> str:
        return self._scenario_id

    @property
    def road(self) -> Road:
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
                psi_dot = (
                    np.diff(vehicle.heading, append=vehicle.heading[-1]) / self._dt
                )
                safe_v = np.nan * np.ones_like(vehicle.v)
                safe_v[~np.isclose(vehicle.v, 0, atol=1e-3)] = vehicle.v[
                    ~np.isclose(vehicle.v, 0, atol=1e-3)
                ]
                steering_angle = np.arctan(np.divide(psi_dot * l_wb, safe_v))
                steering_angle[np.isnan(steering_angle)] = 0.0

                new_state_list = []
                for i, _ in enumerate(vehicle.x):
                    new_state_list.append(
                        State(
                            position=np.array((vehicle.x[i], vehicle.y[i])),
                            steering_angle=steering_angle[i],
                            velocity=vehicle.v[i],
                            orientation=vehicle.heading[i],
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
                        msg = f"scenario '{self._scenario_id}': Dynamic obstacle is not defined for the complete scenario."
                        raise RuntimeError(msg)

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
        # Find road boundaries
        road_boundary = self._road.boundary_line
        ax.set_xlim(
            np.min(road_boundary[:, 0]) - margin, np.max(road_boundary[:, 0]) + margin
        )
        ax.set_ylim(
            np.min(road_boundary[:, 1]) - margin, np.max(road_boundary[:, 1]) + margin
        )

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

        save_dir = Path(save_dir)

        frames = []

        for timestep in self.steps:
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
                # Find road boundaries
                road_boundary = self._road.boundary_line
                ax.set_xlim(
                    np.min(road_boundary[:, 0]) - margin,
                    np.max(road_boundary[:, 0]) + margin,
                )
                ax.set_ylim(
                    np.min(road_boundary[:, 1]) - margin,
                    np.max(road_boundary[:, 1]) + margin,
                )

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

    def save(self, result_dir: Path, mode: str = "config") -> None:
        """
        mode: "config", "cr" or "openx"
        """

        result_dir = Path(result_dir)

        if mode not in self.COMPILE_MODES:
            msg = "Compile mode not available"
            raise ValueError(msg)

        if mode == "config":
            if self._initialized_from_data:
                msg = "Cannot save to config if the scenario has been initialized from data."
                raise ValueError(msg)

            config_file = result_dir / f"{self._scenario_id}.json"

            with config_file.open("w") as f:
                json.dump(self._config, f, indent=2)

        elif mode == "cr":
            self.get_cr_interface().to_xml(result_dir)

        elif mode == "openx":
            if self._initialized_from_data:
                msg = "Cannot save to openx if the scenario has been initialized from data."
                raise ValueError(msg)

            # CAUTION: Please read the instructions in _create_openscenario() to run in esmini

            # OpenDRIVE
            odr_path = result_dir / f"{self._scenario_id}.xodr"
            odr = self._road.create_opendrive_map()
            odr.write_xml(str(odr_path))

            # OpenSCENARIO
            osc = self._create_openscenario(odr_path)
            osc.write_xml(str(result_dir / f"{self._scenario_id}.xosc"))

    def _create_openscenario(self, odr_path: str | Path) -> xosc.Scenario:
        """
        OpenScenario for esmini
        Inspired by https://gitlab.ika.rwth-aachen.de/scenario-based-validation/smartervalidation/-/blob/MA-Warmuth/smartervalidation/simulation/esmini_simulation.py?ref_type=heads#L81

        Idea: OSC file consists of just one story that includes FollowTrajectoryActions for all vehicles in the scenario.

        xosc+xodr files need to be placed together into a subfolder in esminis resource folder

        Run by `esmini --osc "D:\Programme\esmini-2.37.10\resources\test_openx_export_default\test.xosc" --window 60 60 1024 576`
        """  # noqa: W605

        # Paths (relative to xodr dir of esmini folder)
        vehicle_catalog_path = "../xosc/Catalogs/Vehicles"
        model_blue_car_path = "../models/car_blue.osgb"
        model_red_car_path = "../models/car_red.osgb"

        # Create parameters of the scenario
        scenario_parameters = xosc.ParameterDeclarations()

        # Create catalogs
        catalog = xosc.Catalog()
        catalog.add_catalog("VehicleCatalog", vehicle_catalog_path)

        # Create road network from opendrive file
        road_network = xosc.RoadNetwork(roadfile=odr_path.name)

        # Create entities
        entities = xosc.Entities()

        # Create ego vehicle object
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
        ego_vehicle_object.add_property("model_id", "ego")

        # Add entities
        entities.add_scenario_object("ego", ego_vehicle_object)  # , controller)

        # Create other vehicle objects
        max_acceleration = 9.81
        max_deceleration = 9.81

        vehicle_objects = {}
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
            vehicle_object.add_property("model_id", str(vehicle.id))
            vehicle_objects[vehicle.id] = vehicle_object
            entities.add_scenario_object(str(vehicle.id), vehicle_object)

        # Create the init part of the storyboard
        init = xosc.Init()
        step_time = xosc.TransitionDynamics(
            xosc.DynamicsShapes.step, xosc.DynamicsDimension.time, 1
        )

        # Ego
        ego_initial_speed_action = xosc.AbsoluteSpeedAction(
            self._ego_configuration.v0, step_time
        )
        # In Simple Scenario lane ids start with 1000 from the right (driving direction)
        # In ODR, they start with -1 from the left and get smaller to the right
        ego_initial_lane_id = (
            self._ego_configuration.lanelet_id - 1000
        ) - self._road.n_lanes
        ego_start_position_action = xosc.TeleportAction(
            xosc.LanePosition(
                self._ego_configuration.s0,
                self._ego_configuration.t0,
                ego_initial_lane_id,
                1,
            )
        )

        init.add_init_action("ego", ego_initial_speed_action)
        init.add_init_action("ego", ego_start_position_action)

        # Other vehicles
        for vehicle in self._vehicles:
            vehicle_initial_speed_action = xosc.AbsoluteSpeedAction(
                vehicle.v0, step_time
            )
            vehicle_initial_lane_id = (vehicle.lanelet_id - 1000) - self._road.n_lanes
            vehicle_initial_position_action = xosc.TeleportAction(
                xosc.LanePosition(vehicle.s0, vehicle.t0, vehicle_initial_lane_id, 1)
            )

            init.add_init_action(str(vehicle.id), vehicle_initial_speed_action)
            init.add_init_action(str(vehicle.id), vehicle_initial_position_action)

        ## Init the storyboard
        stoptrigger_storyboard = xosc.ValueTrigger(
            "StoptriggerTime",
            0,
            xosc.ConditionEdge.rising,
            xosc.SimulationTimeCondition(self._duration, xosc.Rule.greaterThan),
            triggeringpoint="stop",
        )
        storyboard = xosc.StoryBoard(init, stoptrigger_storyboard)

        ## Init the story
        storyparam = xosc.ParameterDeclarations()
        story = xosc.Story(f"Act_scenario_{self._scenario_id}", storyparam)

        # Init the Act
        act = xosc.Act(f"Act_scenario_{self._scenario_id}")

        # Add FollowTrajectoryActions for all vehicles to the event
        scenario_step_times = (self.steps * self._dt).tolist()

        for vehicle in self._vehicles:
            # Init the maneuvergroup
            maneuver_group = xosc.ManeuverGroup(f"ManeuverGroup_vehicle_{vehicle.id}")

            # Init the maneuver
            maneuver = xosc.Maneuver(f"Maneuver_vehicle_{vehicle.id}")

            # Create the event
            event = xosc.Event(f"Event_vehicle_{vehicle.id}", xosc.Priority.override)

            # Add the actor to the maneuver group
            maneuver_group.add_actor(str(vehicle.id))

            # Init trajectory
            vehicle_trajectory = xosc.Trajectory(
                f"Trajectory_vehicle_{vehicle.id}", closed=False
            )

            # Create polyline
            vehicle_positions = []
            for i, _ in enumerate(vehicle.x):
                vehicle_position = xosc.WorldPosition(
                    vehicle.x[i], vehicle.y[i], h=vehicle.heading[i]
                )
                vehicle_positions.append(vehicle_position)

            vehicle_polyline = xosc.Polyline(scenario_step_times, vehicle_positions)

            # Fill trajectory with polyline
            vehicle_trajectory.add_shape(vehicle_polyline)

            # Create follow trajectory action with trajectory, reference_domain, scale and offset need to be set such that the speed is also adjusted in the simulation (here: esmini).
            vehicle_follow_trajectory_action = xosc.FollowTrajectoryAction(
                vehicle_trajectory,
                xosc.FollowingMode.position,
                reference_domain=xosc.ReferenceContext.absolute,
                scale=1,
                offset=0,
            )

            # Add the action to the event
            event.add_action(
                f"FollowTrajectoryAction_vehicle_{vehicle.id}",
                vehicle_follow_trajectory_action,
            )

            # Add the event to the maneuver
            maneuver.add_event(event)
            # Add the maneuver to the maneuver group
            maneuver_group.add_maneuver(maneuver)

            # Add the maneuver group to the act
            act.add_maneuver_group(maneuver_group)

        # Add the act to the story
        story.add_act(act)
        # Add the story to the storyboard
        storyboard.add_story(story)

        ## Put everything together to create the scenario
        osc_scenario = xosc.Scenario(
            name=self._scenario_id,
            author="simple_scenario",
            parameters=scenario_parameters,
            entities=entities,
            storyboard=storyboard,
            roadnetwork=road_network,
            catalog=catalog,
        )

        return osc_scenario

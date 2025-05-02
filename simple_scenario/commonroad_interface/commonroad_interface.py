from __future__ import annotations

import numpy as np
import warnings

from loguru import logger
from typing import TYPE_CHECKING

from commonroad.common.file_reader import DynamicObstacleFactory
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.common.util import AngleInterval, Interval
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.scenario.scenario import Scenario, ScenarioID, Tag
from commonroad.scenario.trajectory import State, Trajectory
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.geometry.shape import Rectangle
from commonroad.planning.goal import GoalRegion
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from commonroad_dc.geometry.util import resample_polyline

from .lanelet_network_creator import LaneletNetworkCreator

from ..lanelet_network_wrapper import LaneletNetworkWrapper

if TYPE_CHECKING:
    from pathlib import Path
    from ..ego_configuration import EgoConfiguration
    from ..road.road import Road
    from ..vehicle import Vehicle


class CommonroadInterface:
    """
    Interface to commonroad.
    Create scenario, planning problem set etc from concrete scenario
    """

    def __init__(
        self,
        simple_scenario_config: dict,
        ego_configuration: EgoConfiguration,
        vehicles: list[Vehicle],
        road: Road,
    ) -> None:
        logger.debug(
            f"Scenario '{simple_scenario_config['scenario_id']}': Create CR interface"
        )

        self._scenario_config = simple_scenario_config
        self._ego_configuration = ego_configuration
        self._vehicles = vehicles
        self._road = road

        # Result
        self._cr_scenario = None
        self._cr_planning_problem_set = None
        self._cr_planning_problem = None
        self._lanelet_network_wrapper = None
        self._ego_llt_id = None
        self._ego_llt_frenet_frame = None
        self._speed_limit_kph = None

        self._compile()

    @property
    def scenario(self) -> Scenario:
        return self._cr_scenario

    @property
    def planning_problem_set(self) -> PlanningProblemSet:
        return self._cr_planning_problem_set

    @property
    def planning_problem(self) -> PlanningProblem:
        if self._cr_planning_problem is None:
            # Extract single planning problem from set
            planning_problems = list(
                self._cr_planning_problem_set.planning_problem_dict.values()
            )
            if len(planning_problems) != 1:
                msg = "Need exactly one planning problem"
                raise NotImplementedError(msg)
            self._cr_planning_problem = planning_problems[0]
        return self._cr_planning_problem

    @property
    def lanelet_network_wrapper(self) -> LaneletNetworkWrapper:
        if self._lanelet_network_wrapper is None:
            self._lanelet_network_wrapper = LaneletNetworkWrapper(
                self._cr_scenario.lanelet_network
            )
        return self._lanelet_network_wrapper

    @property
    def ego_llt_id(self) -> int:
        if self._ego_llt_id is None:
            self._ego_llt_id = self.lanelet_network_wrapper.find_lanelet_id_by_position(
                *self.planning_problem.initial_state.position
            )
        return self._ego_llt_id

    @property
    def ego_llt_frenet_frame(self) -> CurvilinearCoordinateSystem:
        if self._ego_llt_frenet_frame is None:
            # Prepare curvilinear CS
            # Curvilinear CS ref line is the centerline of the lanelet the ego vehicle starts in (WHY NOT THE REFLINE?)
            # Find ego llt id
            ego_llt_id = self.lanelet_network_wrapper.find_lanelet_id_by_position(
                *self.planning_problem.initial_state.position
            )
            # Ego llt coordinate system
            self._ego_llt_frenet_frame = (
                self.lanelet_network_wrapper._get_llt_frenet_frame(ego_llt_id)  # noqa: SLF001
            )
        return self._ego_llt_frenet_frame

    @property
    def speed_limit_kph(self) -> float:
        if self._speed_limit_kph is None:
            self._speed_limit_kph = self.lanelet_network_wrapper.get_speed_limit_by_pos(
                *self.planning_problem.initial_state.position
            )
        return self._speed_limit_kph

    def _compile(self, configuration_id: int = 0) -> None:
        """
        More information: https://gitlab.lrz.de/tum-cps/commonroad-scenarios/-/blob/master/documentation/XML_commonRoad_2020a.pdf
        """

        # Init scenario object
        self._cr_scenario = self._create_cr_scenario_object(configuration_id)

        # Add everything to scenario
        # Layer 1: Road
        self._add_road_layer()

        # Layer 4: Moving object
        self._add_moving_objects_layer()

        # Add planning problem set
        self._cr_planning_problem_set = self._create_cr_planning_problem_set()

    def _generate_object_id(self) -> int:
        """Starts with 1."""
        return self._cr_scenario.generate_object_id()

    def _create_cr_scenario_object(self, configuration_id: int) -> Scenario:
        """Create ScenarioID object"""

        # ScenarioID
        # Test if the scenario_id is already a valid CR ID
        warnings.filterwarnings("error")
        try:
            scenario_id = ScenarioID.from_benchmark_id(
                self._scenario_config["scenario_id"], "2020a"
            )
        except UserWarning:
            map_name = f"{self._scenario_config['scenario_id']}"
            obstacle_behaviour = (
                "T"  # Single Trajectories (P=Predicted, S=Set-based Occupancy)
            )
            scenario_id = ScenarioID(
                configuration_id=configuration_id,
                map_name=map_name,
                map_id=0,
                obstacle_behavior=obstacle_behaviour,
                prediction_id=0,
            )

        warnings.filterwarnings("default")

        # Scenario
        scenario_dt = self._scenario_config["dt"]
        author = "simple_scenario"
        affiliation = "ika"
        source = "synthetical"
        tags = {Tag.HIGHWAY}  # crss.Tag.CRITICAL
        scenario = Scenario(
            scenario_dt,
            scenario_id=scenario_id,
            author=author,
            tags=tags,
            affiliation=affiliation,
            source=source,
        )

        return scenario

    def _add_road_layer(self) -> None:
        lanelet_network_creator = LaneletNetworkCreator(self._road)

        self._cr_scenario.add_objects(lanelet_network_creator.lanelet_network)

    def _add_moving_objects_layer(self) -> None:
        # Create all moving objects in one loop

        logger.debug("Create moving objects")

        if self._vehicles is None:
            logger.debug("No moving objects.")
            return

        for vehicle in self._vehicles:
            logger.debug("Create moving object with id: {}", vehicle.id)

            object_id = self._generate_object_id()

            moving_object_shape = Rectangle(width=vehicle.width, length=vehicle.length)

            initial_state = None
            moving_object_states = []

            n_timesteps = vehicle.x.shape[0]

            # Fill states
            for timestep in range(n_timesteps):
                if np.isnan(vehicle.x[timestep]):
                    continue

                state = State(
                    position=np.array([vehicle.x[timestep], vehicle.y[timestep]]),
                    velocity=vehicle.v[timestep],
                    orientation=vehicle.heading[timestep],
                    yaw_rate=0.0,
                    slip_angle=0.0,
                    time_step=int(timestep),
                )

                if initial_state is None:
                    initial_state = state
                else:
                    moving_object_states.append(state)

            initial_trajectory_timestep = moving_object_states[0].time_step
            moving_object_trajectory = Trajectory(
                initial_trajectory_timestep, moving_object_states
            )

            # assignments, took from vehicle.py lines 467ff.
            lanelet_network = self._cr_scenario.lanelet_network
            # state list assignments
            # find_obstacle_shape_lanelets modifies lanelet_network and adds obstacle ids to identified lanelets. not written to xml
            shape_lanelet_assignment = (
                DynamicObstacleFactory.find_obstacle_shape_lanelets(
                    initial_state,
                    moving_object_states,
                    lanelet_network,
                    object_id,
                    moving_object_shape,
                )
            )
            # find_obstacle_center_lanelets does not modify lanelet_network
            center_lanelet_assignment = (
                DynamicObstacleFactory.find_obstacle_center_lanelets(
                    initial_state, moving_object_states, lanelet_network
                )
            )

            rotated_shape = moving_object_shape.rotate_translate_local(
                initial_state.position, initial_state.orientation
            )
            initial_shape_lanelet_ids = set(
                lanelet_network.find_lanelet_by_shape(rotated_shape)
            )
            initial_center_lanelet_ids = set(
                lanelet_network.find_lanelet_by_position([initial_state.position])[0]
            )

            prediction = TrajectoryPrediction(
                trajectory=moving_object_trajectory,
                shape=moving_object_shape,
                center_lanelet_assignment=shape_lanelet_assignment,
                shape_lanelet_assignment=center_lanelet_assignment,
            )

            moving_object_obstacle = DynamicObstacle(
                obstacle_id=object_id,
                obstacle_type=ObstacleType.CAR,
                obstacle_shape=moving_object_shape,
                initial_state=initial_state,
                prediction=prediction,
                initial_center_lanelet_ids=initial_center_lanelet_ids,
                initial_shape_lanelet_ids=initial_shape_lanelet_ids,
            )

            self._cr_scenario.add_objects(moving_object_obstacle)

    def _create_cr_planning_problem_set(self) -> PlanningProblemSet:
        # Read from scenario info
        duration = self._scenario_config["duration"]
        dt = self._scenario_config["dt"]

        # Create
        planning_problem_id = self._generate_object_id()

        # Ego initial state
        ego_initial_state = State(
            position=np.array([self._ego_configuration.x0, self._ego_configuration.y0]),
            velocity=self._ego_configuration.v0,
            orientation=self._ego_configuration.heading0,
            yaw_rate=0.0,
            slip_angle=0.0,
            time_step=0,
            steering_angle=0.0,
            steering_angle_speed=0.0,
        )

        # Goal regions
        goal_states = []

        max_time_step = int(np.floor(duration / dt))

        # Make end of road goal region
        goal_area_length = 0.1
        goal_area_width = max(self._road.offset_lines.keys())

        # In ref_line frenet coordinates
        goal_area_pos_s = self._road.goal_position
        goal_area_pos_t = -goal_area_width / 2

        # To cart
        ref_line_frenet_cs = CurvilinearCoordinateSystem(
            resample_polyline(self._road.ref_line, 0.5)
        )
        goal_area_center_cart = np.array(
            ref_line_frenet_cs.convert_list_of_points_to_cartesian_coords(
                [[goal_area_pos_s, goal_area_pos_t]], 1
            )[0]
        )

        goal_road_tangent = ref_line_frenet_cs.tangent(goal_area_pos_s)
        goal_area_theta = np.arctan2(goal_road_tangent[1], goal_road_tangent[0])

        goal_shape = Rectangle(
            length=goal_area_length,
            width=goal_area_width,
            orientation=goal_area_theta,
            center=goal_area_center_cart,
        )

        road_end_goal_state = State(
            position=goal_shape,
            time_step=Interval(0, max_time_step),
            orientation=AngleInterval(
                goal_area_theta - np.pi / 8, goal_area_theta + np.pi / 8
            ),
        )
        goal_states.append(road_end_goal_state)

        goal_region = GoalRegion(goal_states)

        # To planning problem and pp set
        planning_problem = PlanningProblem(
            planning_problem_id, ego_initial_state, goal_region
        )
        planning_problem_set = PlanningProblemSet([planning_problem])

        self._cr_planning_problem_set = planning_problem_set

        return self._cr_planning_problem_set

    def to_xml(self, xml_save_dir: Path) -> Path:
        xml_save_path = xml_save_dir / f"{self._cr_scenario.scenario_id}.xml"

        logger.info("Write scenario to cr xml: {}", xml_save_path)
        file_writer = CommonRoadFileWriter(
            self._cr_scenario, self._cr_planning_problem_set
        )
        file_writer.write_to_file(
            str(xml_save_path), OverwriteExistingFile.SKIP, check_validity=True
        )

        return xml_save_path

    @property
    def scenario_id_str(self) -> str:
        if self._cr_scenario is None:
            return None

        return str(self._cr_scenario.scenario_id)

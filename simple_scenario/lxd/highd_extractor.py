from __future__ import annotations

import numpy as np

from loguru import logger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Generator
    from pathlib import Path

from .. import CR_AVAILABLE, LXD_AVAILABLE

if CR_AVAILABLE:
    from ..lanelet_network_wrapper import LaneletNetworkWrapper
    from ..commonroad_interface import LaneletNetworkCreator

if LXD_AVAILABLE:
    from lxd_io import Dataset

from .. import EgoConfiguration, Vehicle, Scenario
from ..road import Road


class HighdExtractor:
    def __init__(self, dataset_dir: str | Path) -> None:
        if not (CR_AVAILABLE and LXD_AVAILABLE):
            msg = "Please install the commonroad and lxd extras to use this feature. `pip install simple_scenario[commonroad, lxd]`"
            raise ModuleNotFoundError(msg)

        self._dataset = Dataset(dataset_dir)

        self._lower_roads_per_recording = {}
        self._lower_road_wrappers_per_recording = {}
        self._upper_roads_per_recording = {}
        self._upper_road_wrappers_per_recording = {}

        self._default_speed_limit = 130
        self._start_line_s = 500 + 50  # 500 m (because the road starts at -500 m)

    def extract_simple_scenarios(self) -> dict[int, Generator[Scenario]]:
        all_scenarios = {}

        for recording_id in self._dataset.recording_ids:
            all_scenarios[recording_id] = self.extract_recording_scenarios(recording_id)

        return all_scenarios

    def get_specific_scenario(
        self, recording_id: int, track_id: int
    ) -> tuple[str, Scenario | None, str | None]:
        return self._extract_track_scenario(recording_id, track_id)

    def extract_recording_scenarios(
        self,
        recording_id: int,
        first_track_idx_to_process: int = -1,
        last_track_idx_to_process: int = -1,
    ) -> Generator[tuple[str, Scenario | None, str | None]]:
        recording = self._dataset.get_recording(recording_id)

        for track_idx, track_id in enumerate(recording.track_ids):
            # Skip track_idcs that should not be processed here.
            if (
                first_track_idx_to_process != -1
                and last_track_idx_to_process != -1
                and not (
                    first_track_idx_to_process <= track_idx <= last_track_idx_to_process
                )
            ):
                continue

            try:
                scenario_id, scenario, extraction_error_msg = (
                    self._extract_track_scenario(recording_id, track_id)
                )
            except Exception as e:  # noqa: BLE001
                logger.error(
                    f"Recording {recording.id}: Skip track with idx '{track_idx}' id '{track_id}', because of exception during extraction: {e}"
                )
                scenario_id = f"highd_r{recording_id:02d}_t{track_id:05d}_error"
                scenario = None
                extraction_error_msg = "exception_during_extraction"

            yield scenario_id, scenario, extraction_error_msg

    def _extract_track_scenario(  # noqa: PLR0912
        self, recording_id: int, track_id: int
    ) -> tuple[str, Scenario | None, str | None]:
        recording = self._dataset.get_recording(recording_id)
        recording_frames_array = np.array(
            [*recording.frames, recording.frames[-1] + 1]
        )  # append one frame, because in some recordings, the duration parameter seems to be off by 1 frame

        # Speed limit for ADS is 130 km/h.
        speed_limit = recording.get_meta_data("speedLimit")
        if speed_limit == -1:
            speed_limit = self._default_speed_limit
        else:
            speed_limit = round(speed_limit * 3.6)

        # Create Road objects
        if recording_id not in self._lower_roads_per_recording:
            lower_road = Road.from_highd_parameters(
                recording.get_meta_data("lowerLaneMarkings"),
                "lower",
                speed_limit=speed_limit,
            )
            upper_road = Road.from_highd_parameters(
                recording.get_meta_data("upperLaneMarkings"),
                "upper",
                speed_limit=speed_limit,
            )

            # Create CR lanelet networks and wrap them
            lower_road_wrapper = LaneletNetworkWrapper(
                LaneletNetworkCreator(lower_road).lanelet_network
            )
            upper_road_wrapper = LaneletNetworkWrapper(
                LaneletNetworkCreator(upper_road).lanelet_network
            )

            self._lower_roads_per_recording[recording_id] = lower_road
            self._lower_road_wrappers_per_recording[recording_id] = lower_road_wrapper
            self._upper_roads_per_recording[recording_id] = upper_road
            self._upper_road_wrappers_per_recording[recording_id] = upper_road_wrapper
        else:
            lower_road = self._lower_roads_per_recording[recording_id]
            lower_road_wrapper = self._lower_road_wrappers_per_recording[recording_id]
            upper_road = self._upper_roads_per_recording[recording_id]
            upper_road_wrapper = self._upper_road_wrappers_per_recording[recording_id]

        track = recording.get_track(track_id)

        # -- Road --

        # Find the relevant road (lower or upper)
        local_trajectory = track.get_local_trajectory()
        local_trajectory[:, -1] = -local_trajectory[
            :, -1
        ]  # Invert, because we want the y axis to point upwards (map is already flipped)

        first_position_x = local_trajectory[0, 0]
        first_position_y = local_trajectory[0, 1]

        ego_driving_direction = track.get_meta_data("drivingDirection")
        if ego_driving_direction == 1:
            relevant_road_name = "upper"
            relevant_road = upper_road
            relevant_wrapper = upper_road_wrapper
        elif ego_driving_direction == 2:
            relevant_road_name = "lower"
            relevant_road = lower_road
            relevant_wrapper = lower_road_wrapper
        else:
            msg = f"drivingDirection should be 1 (upper road) or 2 (lower road), but not {ego_driving_direction}"
            raise ValueError(msg)

        scenario_id = f"highd_r{recording_id:02d}_t{track_id:05d}_{relevant_road_name}"

        # Check whether it is starting before the start line and reaches the goal line
        last_position_x = local_trajectory[-1, 0]
        last_position_y = local_trajectory[-1, 1]
        first_position_s = relevant_wrapper.from_cart_to_ref_frenet(
            first_position_x, first_position_y
        )[0]
        last_position_s = relevant_wrapper.from_cart_to_ref_frenet(
            last_position_x, last_position_y
        )[0]

        # Remove trucks
        track_class = track.get_meta_data("class")
        if track_class != "Car":
            logger.info(
                f"Recording {recording_id:02d}: Skip track {track_id:04d}, because it is not a Car, but a {track_class}."
            )
            return scenario_id, None, "track_class_not_car"

        if (
            first_position_s > self._start_line_s
            or last_position_s < relevant_road.goal_position
        ):
            logger.info(
                f"Recording {recording_id:02d}: Skip track (id={track_id}), because it is too short."
            )
            return scenario_id, None, "track_too_short"

        logger.info(f"Extract simple scenario for track {track_id:05d}")

        # -- EgoConfiguration --
        # Create the ego configuration; starting point of the ego is the moment it crosses the start_line
        ego_ref_frenet = relevant_wrapper.point_array_from_cart_to_ref_frenet(
            local_trajectory
        )

        ego_start_idx = np.where(ego_ref_frenet[:, 0] > self._start_line_s)[0][0]
        ego_end_idx = np.where(ego_ref_frenet[:, 0] > relevant_road.goal_position)[0][0]

        ego_start_position_x = local_trajectory[ego_start_idx, 0]
        ego_start_position_y = local_trajectory[ego_start_idx, 1]
        ego_start_llt_id = relevant_wrapper.find_lanelet_id_by_position(
            ego_start_position_x, ego_start_position_y
        )

        ego_start_llt_s, ego_start_llt_t = relevant_wrapper.from_cart_to_llt_frenet(
            ego_start_llt_id, ego_start_position_x, ego_start_position_y
        )

        ego_v = track.get_data("xVelocity")
        if relevant_road_name == "upper":
            ego_v = -ego_v

        ego_start_v = ego_v[ego_start_idx]

        ego_configuration = EgoConfiguration(
            ego_start_llt_id, ego_start_llt_s, ego_start_llt_t, ego_start_v
        )

        # -- Vehicles --
        vehicles = []

        # Find frames that belong to ego_start_idx and ego_end_idx
        # From ego local to global frame
        initial_frame = track.get_meta_data("initialFrame")
        scenario_start_frame = initial_frame + ego_start_idx
        scenario_end_frame = initial_frame + ego_end_idx
        n_steps_reference = scenario_end_frame - scenario_start_frame + 1
        # Increase duration by 25% (distance is reduced from 400 to 300 meters, 25%)
        n_steps = int(np.ceil(n_steps_reference * 1.25))
        scenario_lifetime = np.zeros_like(recording_frames_array, dtype=bool)
        # The frames are not the indices of the frames array, because for highD, the first frame is 1 (and not 0)
        scenario_start_frame_idx = np.where(
            recording_frames_array == scenario_start_frame
        )[0][0]
        scenario_end_frame_idx = np.where(recording_frames_array == scenario_end_frame)[
            0
        ][0]
        scenario_lifetime[
            scenario_start_frame_idx : scenario_end_frame_idx + n_steps + 1
        ] = 1

        # Create Vehicle objects
        other_relevant_track_ids_ = []
        for frame in range(scenario_start_frame, scenario_end_frame + 1):
            other_relevant_track_ids_.extend(recording.get_track_ids_at_frame(frame))
        other_relevant_track_ids = sorted(set(other_relevant_track_ids_))

        cur_vehicle_id = 1000
        for other_track_id in other_relevant_track_ids:
            # Skip ego vehicle
            if other_track_id == track_id:
                continue

            other_track = recording.get_track(other_track_id)

            if other_track.get_meta_data("drivingDirection") != ego_driving_direction:
                continue

            # The other_track is maybe not present for the complete duration!
            # Append / prepend np.nan for unknown states
            other_track_initial_frame = other_track.get_meta_data("initialFrame")
            other_track_final_frame = other_track.get_meta_data("finalFrame")
            other_track_lifetime = np.zeros_like(recording_frames_array, dtype=bool)
            # The frames are not the indices of the frames array, because for highD, the first frame is 1 (and not 0)
            other_track_initial_frame_idx = np.where(
                recording_frames_array == other_track_initial_frame
            )[0][0]
            other_track_final_frame_idx = np.where(
                recording_frames_array == other_track_final_frame
            )[0][0]
            other_track_lifetime[
                other_track_initial_frame_idx : other_track_final_frame_idx + 1
            ] = 1

            other_track_length = other_track.get_meta_data("width")
            other_track_width = other_track.get_meta_data("height")
            other_track_x = other_track.get_data("x") + other_track_length / 2
            other_track_y = -(other_track.get_data("y") + other_track_width / 2)
            other_track_v = other_track.get_data("xVelocity")
            other_track_a = other_track.get_data("xAcceleration")
            other_track_heading = np.zeros_like(other_track_a)
            if relevant_road_name == "upper":
                other_track_v = -other_track_v
                other_track_a = -other_track_a
                other_track_heading += np.pi

            # Arrays for the whole recording lifetime
            other_track_x_whole_recording = np.nan * np.ones_like(
                recording_frames_array
            )
            other_track_x_whole_recording[other_track_lifetime] = other_track_x
            other_track_y_whole_recording = np.nan * np.ones_like(
                recording_frames_array
            )
            other_track_y_whole_recording[other_track_lifetime] = other_track_y
            other_track_v_whole_recording = np.nan * np.ones_like(
                recording_frames_array
            )
            other_track_v_whole_recording[other_track_lifetime] = other_track_v
            other_track_a_whole_recording = np.nan * np.ones_like(
                recording_frames_array
            )
            other_track_a_whole_recording[other_track_lifetime] = other_track_a
            other_track_heading_whole_recording = np.nan * np.ones_like(
                recording_frames_array
            )
            other_track_heading_whole_recording[other_track_lifetime] = (
                other_track_heading
            )

            # Take only the part that is in the scenario
            other_track_x_during_scenario = other_track_x_whole_recording[
                scenario_lifetime
            ]
            other_track_y_during_scenario = other_track_y_whole_recording[
                scenario_lifetime
            ]
            other_track_v_during_scenario = other_track_v_whole_recording[
                scenario_lifetime
            ]
            other_track_a_during_scenario = other_track_a_whole_recording[
                scenario_lifetime
            ]
            other_track_heading_during_scenario = other_track_heading_whole_recording[
                scenario_lifetime
            ]

            # Need at least two time steps with non-nan values
            if np.count_nonzero(~np.isnan(other_track_x_during_scenario)) < 2:
                continue

            vehicle_id = cur_vehicle_id
            # Find lanelet id for first time step the vehicle exists in the scenario
            # Find first non nan value
            other_track_first_valid_pos_x = other_track_x_during_scenario[
                ~np.isnan(other_track_x_during_scenario)
            ][0]
            other_track_first_valid_pos_y = other_track_y_during_scenario[
                ~np.isnan(other_track_y_during_scenario)
            ][0]
            other_track_initial_llt_id = relevant_road.find_lanelet_id_by_position(
                other_track_first_valid_pos_x, other_track_first_valid_pos_y
            )

            other_vehicle = Vehicle.from_data(
                vehicle_id,
                other_track_initial_llt_id,
                other_track_x_during_scenario,
                other_track_y_during_scenario,
                other_track_heading_during_scenario,
                other_track_v_during_scenario,
                other_track_a_during_scenario,
                vehicle_type_name="custom",
                length=other_track_length,
                width=other_track_width,
            )
            vehicles.append(other_vehicle)
            cur_vehicle_id += 1

        dt = 1 / recording.get_meta_data("frameRate")
        duration = n_steps * dt

        scenario = Scenario(
            scenario_id,
            relevant_road.copy(),
            ego_configuration,
            vehicles,
            from_data=True,
            duration=duration,
            dt=dt,
        )

        return scenario_id, scenario, None

import numpy as np
import warnings

from commonroad.scenario.lanelet import (
    LaneletNetwork,
    Lanelet,
    LaneletType,
    LineMarking,
)
from commonroad.scenario.traffic_sign import TrafficSignElement, TrafficSignIDZamunda
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from shapely.errors import ShapelyDeprecationWarning

from .traffic_sign_fixed import TraffiSignFixed
from ..road.road import Road


class LaneletNetworkCreator:
    def __init__(self, road: Road) -> None:
        self._lanelet_network = self._create_lanelet_network(road)

    @property
    def lanelet_network(self) -> LaneletNetwork:
        return self._lanelet_network

    def _create_lanelet_network(self, road: Road) -> LaneletNetwork:
        """
        Generate CR-LaneletNetwork from Road object
        """

        n_lanes = road.n_lanes
        all_lines = road.offset_lines
        all_lines[0.0] = road.ref_line

        offsets = sorted(all_lines.keys(), reverse=True)

        lanelet_list = []

        # CAUTION: Do not use lanelet ID 0. This will break the adj_left, adj_right relations when writing to xml file
        # see class LaneletXMLNode code in CRFileWriter
        lanelet_min_id = 1000
        lanelet_ids = range(lanelet_min_id, n_lanes + lanelet_min_id)

        for i_lanelet, lanelet_id in enumerate(lanelet_ids):
            # Neighbours
            if (lanelet_id - 1) in lanelet_ids:
                adj_right = lanelet_id - 1
                adj_right_direction = True
                right_marking = LineMarking.DASHED
            else:
                adj_right = None
                adj_right_direction = False
                right_marking = LineMarking.SOLID

            if (lanelet_id + 1) in lanelet_ids:
                adj_left = lanelet_id + 1
                adj_left_direction = True
                left_marking = LineMarking.DASHED
            else:
                adj_left = None
                adj_left_direction = False
                left_marking = LineMarking.SOLID

            # (Virtual) speed limit sign
            speed_limit_traffic_sign = TraffiSignFixed(
                max(lanelet_ids) + 100,
                [
                    TrafficSignElement(
                        TrafficSignIDZamunda["MAX_SPEED"], [str(road.speed_limit)]
                    )
                ],
                {lanelet_id},
                position=np.array([0, 0]),
                virtual=True,
            )

            # Select lines
            left_line_offset = offsets[2 + 2 * i_lanelet]
            center_line_offset = offsets[1 + 2 * i_lanelet]
            right_line_offset = offsets[2 * i_lanelet]

            left_vertices = all_lines[left_line_offset]
            center_vertices = all_lines[center_line_offset]
            right_vertices = all_lines[right_line_offset]

            # Create lanelet object
            lanelet = Lanelet(
                left_vertices,
                center_vertices,
                right_vertices,
                lanelet_id,
                adjacent_left=adj_left,
                adjacent_left_same_direction=adj_left_direction,
                adjacent_right=adj_right,
                adjacent_right_same_direction=adj_right_direction,
                line_marking_left_vertices=left_marking,
                line_marking_right_vertices=right_marking,
                lanelet_type={LaneletType.HIGHWAY},
                traffic_signs={speed_limit_traffic_sign},
            )

            lanelet_list.append(lanelet)

        # Create lanelet
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ShapelyDeprecationWarning)
            lanelet_network = LaneletNetwork.create_from_lanelet_list(lanelet_list)

        # Add traffic signs
        for lanelet in lanelet_list:
            for ts in lanelet.traffic_signs:
                # Catch warning that the traffic light is added multiple times to each lanelet.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    lanelet_network.add_traffic_sign(
                        ts, set()
                    )  # references already existant in lanelet, but traffic sign object missing

        return lanelet_network

    @staticmethod
    def arc_heading_at_s(
        curvilinear_cs: CurvilinearCoordinateSystem, s: float
    ) -> float:
        """Returns tangent angle at s in rad"""

        tanget_vector = curvilinear_cs.tangent(s)
        orientation = np.arctan2(tanget_vector[1], tanget_vector[0])

        return orientation

from dataclasses import dataclass


@dataclass
class NeighbourLanes:
    left_lane_available: bool
    left_lane_lanelet_id: int
    right_lane_available: bool
    right_lane_lanelet_id: int

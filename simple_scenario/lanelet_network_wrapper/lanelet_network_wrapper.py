from __future__ import annotations

import numpy as np

from copy import deepcopy
from functools import lru_cache
from typing import TYPE_CHECKING

from commonroad_dc.geometry.util import resample_polyline
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem

if TYPE_CHECKING:
    from commonroad.scenario.lanelet import Lanelet, LaneletNetwork

from .neighbour_lanes import NeighbourLanes
from .surrounding_vehicles import SurroundingVehicles
from .traffic_sign_interpreter_fixed import TrafficSigInterpreterFixed
from .out_of_projection_domain_exception import OutOfProjectionDomainError


class LaneletNetworkWrapper:
    """
    Wrapper around CR laneletnetwork with easier handling of frenet coordinate systems.
    """

    def __init__(self, lanelet_network: LaneletNetwork) -> None:
        self._lanelet_network = lanelet_network

        self._lanelet_list = self._lanelet_network.lanelets
        # Assumption: leftmost llt is the last in the list.
        self._ref_line = self._lanelet_list[-1].left_vertices

        self._lanelets_by_llt_id = {
            llt.lanelet_id: llt for llt in self._lanelet_network.lanelets
        }

        self._frenet_frames = {"ref": self._create_frenet_frame(self._ref_line)}

        self._traffic_sign_interpreter = TrafficSigInterpreterFixed(
            "ZAM", self._lanelet_network
        )

    # -- General access to the map --

    @property
    def lanelet_list(self) -> list:
        return self._lanelet_list

    def get_lanelet_dict(self) -> dict:
        lanelet_dict = {llt.lanelet_id: llt for llt in self._lanelet_list}
        return lanelet_dict

    # -- Map matching ---

    @lru_cache(maxsize=100000)
    def find_lanelet_id_by_position(self, x: float, y: float) -> int:
        # Assumption: there are not overlapping lanelets
        llt_id = self._lanelet_network.find_lanelet_by_position([[x, y]])[0][0]

        return llt_id

    def get_lanelet_by_id(self, llt_id: int) -> Lanelet:
        self._check_llt_id(llt_id)

        return self._lanelets_by_llt_id[llt_id]

    def get_speed_limit_by_pos(self, x: float, y: float) -> float:
        lanelet_id = self.find_lanelet_id_by_position(x, y)
        speed_limit = self._traffic_sign_interpreter.speed_limit(
            frozenset([lanelet_id])
        )

        return speed_limit

    # -- Neighbour lanes --

    def find_available_neighbour_lanes(self, ego_lanelet_id: int) -> NeighbourLanes:
        lanelet = self.get_lanelet_by_id(ego_lanelet_id)

        left_lane_available = False
        left_lane_lanelet_id = -1
        if lanelet.adj_left and lanelet.adj_left_same_direction:
            left_lane_available = True
            left_lane_lanelet_id = lanelet.adj_left

        right_lane_lanelet_id = -1
        right_lane_available = False
        if lanelet.adj_right and lanelet.adj_right_same_direction:
            right_lane_available = True
            right_lane_lanelet_id = lanelet.adj_right

        neighbour_lanes = NeighbourLanes(
            left_lane_available,
            left_lane_lanelet_id,
            right_lane_available,
            right_lane_lanelet_id,
        )

        return neighbour_lanes

    # -- Surrounding vehicles --

    def find_surrounding_vehicles(
        self, ego_x: float, ego_y: float, dynamic_objects: dict
    ) -> SurroundingVehicles:
        ego_lanelet_id = self.find_lanelet_id_by_position(ego_x, ego_y)
        ego_ref_s, _ = self.from_cart_to_ref_frenet(ego_x, ego_y)

        # Map all dynamic objects to lanelet ids
        lanelet_id_to_do_id = {}
        do_id_to_lanelet_id = {}
        for do_id, do in dynamic_objects.items():
            do_lanelet_id = self.find_lanelet_id_by_position(do["x"], do["y"])
            # Add to do dict
            do["lanelet_id"] = do_lanelet_id
            # Add to mapping dicts
            if do_lanelet_id not in lanelet_id_to_do_id:
                lanelet_id_to_do_id[do_lanelet_id] = []
            lanelet_id_to_do_id[do_lanelet_id].append(do_id)

            do_id_to_lanelet_id[do_id] = do_lanelet_id

        # -- Identify surrounding objects ---

        # Identify lead object
        lead_vehicle = self._find_lead_vehicle(
            ego_ref_s, ego_lanelet_id, dynamic_objects, lanelet_id_to_do_id
        )

        # Left front etc..
        left_lead_vehicle, left_rear_vehicle = self._find_adjacent_vehicles(
            "left", ego_ref_s, ego_lanelet_id, dynamic_objects, lanelet_id_to_do_id
        )
        right_lead_vehicle, right_rear_vehicle = self._find_adjacent_vehicles(
            "right", ego_ref_s, ego_lanelet_id, dynamic_objects, lanelet_id_to_do_id
        )

        surrounding_vehicles = SurroundingVehicles(
            lead_vehicle,
            left_lead_vehicle,
            left_rear_vehicle,
            right_lead_vehicle,
            right_rear_vehicle,
        )

        return surrounding_vehicles

    def _find_lead_vehicle(
        self,
        ego_ref_s: float,
        ego_lanelet_id: int,
        dynamic_objects: dict,
        lanelet_id_to_do_id: dict,
    ) -> dict:
        # Init
        lead_vehicle_id = None
        lead_dx = None
        lead_obj = None

        if not lanelet_id_to_do_id:
            return None

        # Find objects in ego lanelet (or TODO: its successors)
        relevant_do_ids = lanelet_id_to_do_id.get(ego_lanelet_id, [])
        for do_id in relevant_do_ids:
            do = dynamic_objects[do_id]

            do_s, _ = self.from_cart_to_ref_frenet(do["x"], do["y"])

            if do_s > ego_ref_s:
                dx = do_s - ego_ref_s

                if lead_dx is None or dx < lead_dx:
                    lead_vehicle_id = do_id
                    lead_dx = dx
                    lead_obj = do

        # Find in following lanelets of ego lanelet
        if lead_vehicle_id is None:
            ego_lanelet = self.get_lanelet_by_id(ego_lanelet_id)

            ego_lanelet_successor = ego_lanelet.successor

            if ego_lanelet_successor:
                msg = "Please implement"
                raise NotImplementedError(msg)

        return lead_obj

    def _find_adjacent_vehicles(
        self,
        side: str,
        ego_ref_s: float,
        ego_lanelet_id: int,
        dynamic_objects: dict,
        lanelet_id_to_do_id: dict,
    ) -> tuple[dict, dict]:
        """
        Can only handle one set of parallel lanelets.
        Does not look at consecutive lanelets.
        """

        if side not in ("left", "right"):
            msg = "side must be left or right"
            raise ValueError(msg)

        ego_lanelet = self.get_lanelet_by_id(ego_lanelet_id)

        adj_lanelet_same_direction_map = {
            "left": ego_lanelet.adj_left_same_direction,
            "right": ego_lanelet.adj_right_same_direction,
        }

        adj_lanelet_id_map = {
            "left": ego_lanelet.adj_left,
            "right": ego_lanelet.adj_right,
        }

        adj_lead_id = None
        adj_lead_dx = None
        adj_lead_obj = None

        adj_rear_id = None
        adj_rear_dx = None
        adj_rear_obj = None

        if adj_lanelet_same_direction_map[side]:
            adj_lanelet_id = adj_lanelet_id_map[side]
            relevant_do_ids = lanelet_id_to_do_id.get(adj_lanelet_id, [])

            for do_id in relevant_do_ids:
                do = dynamic_objects[do_id]

                do_s, _ = self.from_cart_to_ref_frenet(do["x"], do["y"])

                if do_s >= ego_ref_s:
                    dx = do_s - ego_ref_s

                    if adj_lead_id is None or dx < adj_lead_dx:
                        adj_lead_id = do_id
                        adj_lead_dx = dx
                        adj_lead_obj = do
                else:
                    dx = ego_ref_s - do_s

                    if adj_rear_id is None or dx < adj_rear_dx:
                        adj_rear_id = do_id
                        adj_rear_dx = dx
                        adj_rear_obj = do

        return adj_lead_obj, adj_rear_obj

    # -- Centerlines --

    @property
    def ref_line(self) -> np.ndarray:
        return deepcopy(self._ref_line)

    def get_llt_centerline(self, llt_id: int) -> np.ndarray:
        self._check_llt_id(llt_id)
        return deepcopy(self._lanelets_by_llt_id[llt_id].center_vertices)

    # -- Reference frenet coordinate system --

    def point_array_from_cart_to_ref_frenet(
        self, pt_array_cart: np.ndarray
    ) -> np.ndarray:
        return self._pt_array_to_frenet(self._frenet_frames["ref"], pt_array_cart)

    def point_array_from_ref_frenet_to_cart(
        self, pt_array_frenet: np.ndarray
    ) -> np.ndarray:
        return self._pt_array_to_cart(self._frenet_frames["ref"], pt_array_frenet)

    def from_cart_to_ref_frenet(self, x: float, y: float) -> tuple[float, float]:
        return self._to_frenet(self._frenet_frames["ref"], x, y)

    def from_ref_frenet_to_cart(self, s: float, t: float) -> tuple[float, float]:
        return self._to_cart(self._frenet_frames["ref"], s, t)

    # -- Lanelet centerline frenet coordinate systems --

    def point_array_from_cart_to_llt_frenet(
        self, llt_id: int, pt_array_cart: np.ndarray
    ) -> np.ndarray:
        self._check_llt_id(llt_id)

        frenet_frame = self._get_llt_frenet_frame(llt_id)

        pt_array_frenet = self._pt_array_to_frenet(frenet_frame, pt_array_cart)

        return pt_array_frenet

    def point_array_from_llt_frenet_to_cart(
        self, llt_id: int, pt_array_frenet: np.ndarray
    ) -> np.ndarray:
        self._check_llt_id(llt_id)

        frenet_frame = self._get_llt_frenet_frame(llt_id)

        pt_array_cart = self._pt_array_to_cart(frenet_frame, pt_array_frenet)

        return pt_array_cart

    def from_cart_to_llt_frenet(
        self, llt_id: int, x: float, y: float
    ) -> tuple[float, float]:
        self._check_llt_id(llt_id)

        frenet_frame = self._get_llt_frenet_frame(llt_id)

        s, t = self._to_frenet(frenet_frame, x, y)

        return s, t

    def from_llt_frenet_to_cart(
        self, llt_id: int, s: float, t: float
    ) -> tuple[float, float]:
        self._check_llt_id(llt_id)

        frenet_frame = self._get_llt_frenet_frame(llt_id)

        x, y = self._to_cart(frenet_frame, s, t)

        return x, y

    # -- Heading --

    def heading_along_llt(self, llt_id: int, s_array: np.ndarray) -> np.ndarray:
        self._check_llt_id(llt_id)
        self._check_1d_array(s_array)

        frenet_cs = self._frenet_frames[llt_id]

        tangents = np.array([frenet_cs.tangent(s) for s in s_array])
        tangent_headings = np.arctan2(tangents[:, 1], tangents[:, 0])

        return tangent_headings

    # -- Internal --

    def _check_llt_id(self, llt_id: int) -> bool:
        if llt_id not in self._lanelets_by_llt_id:
            msg = f"Invalid llt_id: {llt_id}"
            raise ValueError(msg)
        return True

    def _get_llt_frenet_frame(self, llt_id: int) -> CurvilinearCoordinateSystem:
        if llt_id not in self._frenet_frames:
            frenet_frame = self._create_frenet_frame(
                self._lanelets_by_llt_id[llt_id].center_vertices
            )
            self._frenet_frames[llt_id] = frenet_frame

        return self._frenet_frames[llt_id]

    @staticmethod
    def _create_frenet_frame(
        polyline: np.ndarray, sampling_dist: float = 0.5
    ) -> CurvilinearCoordinateSystem:
        LaneletNetworkWrapper._check_2d_array(polyline)

        reference_line = resample_polyline(polyline, sampling_dist)

        frenet_frame = CurvilinearCoordinateSystem(reference_line)

        frenet_frame.compute_and_set_curvature()

        return frenet_frame

    @staticmethod
    def _pt_array_to_cart(
        frenet_cs: CurvilinearCoordinateSystem, pt_array_frenet: np.ndarray
    ) -> np.ndarray:
        LaneletNetworkWrapper._check_2d_array(pt_array_frenet)

        pt_array_cart = []
        for p in pt_array_frenet:
            x, y = LaneletNetworkWrapper._to_cart(frenet_cs, p[0], p[1])
            pt_array_cart.append([x, y])
        pt_array_cart = np.array(pt_array_cart)

        return pt_array_cart

    @staticmethod
    def _pt_array_to_frenet(
        frenet_cs: CurvilinearCoordinateSystem, pt_array_cart: np.ndarray
    ) -> np.ndarray:
        LaneletNetworkWrapper._check_2d_array(pt_array_cart)

        pt_array_frenet = []
        for p in pt_array_cart:
            x, y = LaneletNetworkWrapper._to_frenet(frenet_cs, p[0], p[1])
            pt_array_frenet.append([x, y])
        pt_array_frenet = np.array(pt_array_frenet)

        return pt_array_frenet

    @staticmethod
    def _check_1d_array(candidate: np.ndarray) -> bool:
        if not (isinstance(candidate, np.ndarray) and len(candidate.shape) == 1):
            msg = "Input must be a numpy array with shape: (X,)."
            raise Exception(msg)
        return True

    @staticmethod
    def _check_2d_array(candidate: np.ndarray) -> bool:
        if not (
            isinstance(candidate, np.ndarray)
            and len(candidate.shape) == 2
            and candidate.shape[1] == 2
        ):
            msg = "Input must be a numpy array with shape: (X, 2)."
            raise Exception(msg)
        return True

    @staticmethod
    def _to_cart(
        frenet_cs: CurvilinearCoordinateSystem, s: float, t: float
    ) -> tuple[float, float]:
        if s == 0:
            s = 1e-3

        if s > frenet_cs.length():
            msg = f"s is too high ({s} > {frenet_cs.length()})"
            raise OutOfProjectionDomainError(msg)

        plist = frenet_cs.convert_list_of_points_to_cartesian_coords([[s, t]], 1)
        x = plist[0][0]
        y = plist[0][1]

        x = LaneletNetworkWrapper.round_to_cm(x)
        y = LaneletNetworkWrapper.round_to_cm(y)

        return x, y

    @staticmethod
    def _to_frenet(
        frenet_cs: CurvilinearCoordinateSystem, x: float, y: float
    ) -> tuple[float, float]:
        plist = frenet_cs.convert_list_of_points_to_curvilinear_coords([[x, y]], 1)
        s = plist[0][0]
        t = plist[0][1]

        s = LaneletNetworkWrapper.round_to_cm(s)
        t = LaneletNetworkWrapper.round_to_cm(t)

        return s, t

    @staticmethod
    def round_to_cm(a: float) -> float:
        """Round to 1cm precision."""
        return np.round(a, decimals=2)

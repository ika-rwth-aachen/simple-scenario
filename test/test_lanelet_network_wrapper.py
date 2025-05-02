import pytest

from pathlib import Path

from simple_scenario import CR_AVAILABLE

if CR_AVAILABLE:
    from commonroad.common.file_reader import CommonRoadFileReader

    from simple_scenario import LaneletNetworkWrapper


@pytest.fixture
def dynamic_objects_dict() -> dict:
    return {
        "right_rear": {
            "x": 20.0,
            "y": -4.0,
        },
        "right_lead": {
            "x": 100.0,
            "y": -4.0,
        },
        "left_rear": {
            "x": 20.0,
            "y": -2.0,
        },
        "left_lead": {
            "x": 100.0,
            "y": -2.0,
        },
    }


class TestLaneletNetworkWrapper:
    DATA_DIR = Path(__file__).parent / "assets"
    RESULT_DIR = Path(__file__).parent / "results"

    @pytest.mark.skipif(not CR_AVAILABLE, reason="commonroad extra is not installed")
    def test_lanelet_network_wrapper(self, dynamic_objects_dict):
        xml = self.DATA_DIR / "ZAM_ValidationScenario+No+PaperScenario1-0_0_T-0.xml"

        scenario, _ = CommonRoadFileReader(str(xml)).open()

        lanelet_network_wrapper = LaneletNetworkWrapper(scenario.lanelet_network)

        assert lanelet_network_wrapper.lanelet_list

        assert lanelet_network_wrapper.from_ref_frenet_to_cart(0, 0)

        assert len(lanelet_network_wrapper.get_lanelet_dict()) == 2

        lanelet_network_wrapper.get_lanelet_by_id(1000)

        with pytest.raises(ValueError, match=r"Invalid llt_id: \d+"):
            lanelet_network_wrapper.get_lanelet_by_id(0)

        available_neighbour_lanes = (
            lanelet_network_wrapper.find_available_neighbour_lanes(1000)
        )
        assert available_neighbour_lanes.left_lane_available
        assert available_neighbour_lanes.left_lane_lanelet_id == 1001
        assert not available_neighbour_lanes.right_lane_available
        assert available_neighbour_lanes.right_lane_lanelet_id == -1

        ego_x = 50.0
        ego_y = -4.0
        surrounding_vehicles = lanelet_network_wrapper.find_surrounding_vehicles(
            ego_x, ego_y, dynamic_objects_dict
        )

        assert surrounding_vehicles.lead is not None
        assert surrounding_vehicles.left_lead is not None
        assert surrounding_vehicles.left_rear is not None
        assert surrounding_vehicles.right_lead is None
        assert surrounding_vehicles.right_rear is None

        with pytest.raises(ValueError, match="side must be left or right"):
            lanelet_network_wrapper._find_adjacent_vehicles(  # noqa: SLF001
                "hi", None, None, None, None
            )

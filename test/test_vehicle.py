from pathlib import Path

import numpy as np
import pytest

from simple_scenario import Vehicle
from simple_scenario.rendering import create_scenario_plot_ax
from simple_scenario.road import SyntheticRoad, StraightSegment


class TestVehicle:
    RESULT_DIR = Path(__file__).parent / "results" / "test_vehicle"
    RESULT_DIR.mkdir(exist_ok=True)

    def test_vehicle(self):
        result_dir = self.RESULT_DIR / "test_vehicle"
        result_dir.mkdir(exist_ok=True)

        print("Create vehicle")

        vehicle = Vehicle(
            0,
            start_lanelet_id=1000,
            start_s=10,
            start_t=0,
            v0=27.78,
        )

        # Access parameters
        print(f"Length: {vehicle.length}")
        print(f"Width: {vehicle.width}")

        # For compilation, we need a road
        print("Compile vehicle")
        road = SyntheticRoad(3, 3.75, [StraightSegment(300)])

        duration = 10
        dt = 0.1
        vehicle.compile(road, duration, dt)

        vehicle.render(result_dir, plot_name="vehicle_default")
        vehicle.render(result_dir, plot_name="vehicle_clean", clean=True)

        with create_scenario_plot_ax(result_dir, "vehicle_on_road_default") as ax:
            vehicle.render(ax)
            road.render(ax)
        with create_scenario_plot_ax(
            result_dir, "vehicle_on_road_clean", clean=True
        ) as ax:
            vehicle.render(ax)
            road.render(ax)

    def test_vehicle_lc_left(self):
        result_dir = self.RESULT_DIR / "test_vehicle_lc_left"
        result_dir.mkdir(exist_ok=True)

        print("Create vehicle")

        vehicle = Vehicle(
            0,
            start_lanelet_id=1000,
            start_s=10,
            start_t=0,
            v0=27.78,
            lc_direction=1,
            lc_delay=2,
        )

        # Access parameters
        print(f"Length: {vehicle.length}")
        print(f"Width: {vehicle.width}")

        # For compilation, we need a road
        print("Compile vehicle")
        road = SyntheticRoad(3, 3.75, [StraightSegment(300)])

        duration = 10
        dt = 0.1
        vehicle.compile(road, duration, dt)

        vehicle.render(result_dir, plot_name="vehicle_lc_left_default")
        vehicle.render(result_dir, plot_name="vehicle_lc_left_clean", clean=True)

        with create_scenario_plot_ax(
            result_dir, "vehicle_lc_left_on_road_default"
        ) as ax:
            vehicle.render(ax)
            road.render(ax)
        with create_scenario_plot_ax(
            result_dir, "vehicle_lc_left_on_road_clean", clean=True
        ) as ax:
            vehicle.render(ax)
            road.render(ax)

    def test_vehicle_lc_right(self):
        result_dir = self.RESULT_DIR / "test_vehicle_lc_right"
        result_dir.mkdir(exist_ok=True)

        print("Create vehicle")

        vehicle = Vehicle(
            0,
            start_lanelet_id=1001,
            start_s=10,
            start_t=0,
            v0=27.78,
            lc_direction=-1,
            lc_delay=2,
        )

        # Access parameters
        print(f"Length: {vehicle.length}")
        print(f"Width: {vehicle.width}")

        # For compilation, we need a road
        print("Compile vehicle")
        road = SyntheticRoad(3, 3.75, [StraightSegment(300)])

        duration = 10
        dt = 0.1
        vehicle.compile(road, duration, dt)

        vehicle.render(result_dir, plot_name="vehicle_lc_right_default")
        vehicle.render(result_dir, plot_name="vehicle_lc_right_clean", clean=True)

        with create_scenario_plot_ax(
            result_dir, "vehicle_lc_right_on_road_default"
        ) as ax:
            vehicle.render(ax)
            road.render(ax)
        with create_scenario_plot_ax(
            result_dir, "vehicle_lc_right_on_road_clean", clean=True
        ) as ax:
            vehicle.render(ax)
            road.render(ax)


if __name__ == "__main__":
    tester = TestVehicle()
    tester.test_vehicle()
    tester.test_vehicle_lc_left()
    tester.test_vehicle_lc_right()


class TestVehicleLaneChange:
    """Tests that verify lane-change behaviour for both polynomial and vy types."""

    LANE_WIDTH = 3.75
    N_LANES = 3
    ROAD_LENGTH = 300
    DURATION = 10.0
    DT = 0.1
    V0 = 27.78  # ~100 km/h

    @staticmethod
    def _build_road():
        return SyntheticRoad(
            TestVehicleLaneChange.N_LANES,
            TestVehicleLaneChange.LANE_WIDTH,
            [StraightSegment(TestVehicleLaneChange.ROAD_LENGTH)],
        )

    # -- polynomial lane-change tests --

    @pytest.mark.parametrize(
        "start_lanelet_id, lc_direction, expected_target_lanelet_id",
        [
            (1000, 1, 1001),   # left from rightmost lane
            (1001, -1, 1000),  # right from center lane
        ],
    )
    def test_polynomial_lc_reaches_target_lane(
        self, start_lanelet_id, lc_direction, expected_target_lanelet_id
    ):
        road = self._build_road()
        lc_delay = 2.0
        lc_duration = 3.0

        vehicle = Vehicle(
            0,
            start_lanelet_id=start_lanelet_id,
            start_s=10,
            start_t=0,
            v0=self.V0,
            lc_direction=lc_direction,
            lc_delay=lc_delay,
            lc_duration=lc_duration,
            lc_type="polynomial",
        )
        vehicle.compile(road, self.DURATION, self.DT)

        # Before LC starts the vehicle should still be on the source lanelet
        pre_lc_step = int(lc_delay / self.DT) - 1
        pre_lc_lanelet = road.find_lanelet_id_by_position(
            vehicle.x[pre_lc_step], vehicle.y[pre_lc_step]
        )
        assert pre_lc_lanelet == start_lanelet_id

        # Well after the LC finishes the vehicle should be on the target lanelet
        post_lc_step = int((lc_delay + lc_duration) / self.DT) + 5
        post_lc_lanelet = road.find_lanelet_id_by_position(
            vehicle.x[post_lc_step], vehicle.y[post_lc_step]
        )
        assert post_lc_lanelet == expected_target_lanelet_id

        # Vehicle should remain on the target lanelet until the end
        final_lanelet = road.find_lanelet_id_by_position(
            vehicle.x[-1], vehicle.y[-1]
        )
        assert final_lanelet == expected_target_lanelet_id

    def test_polynomial_lc_lateral_offset_smooth(self):
        """The y-trajectory during a polynomial LC should be continuous (no jumps)."""
        road = self._build_road()
        lc_delay = 2.0
        lc_duration = 3.0

        vehicle = Vehicle(
            0,
            start_lanelet_id=1000,
            start_s=10,
            start_t=0,
            v0=self.V0,
            lc_direction=1,
            lc_delay=lc_delay,
            lc_duration=lc_duration,
            lc_type="polynomial",
        )
        vehicle.compile(road, self.DURATION, self.DT)

        # Max step-to-step y-change should be bounded (no discontinuities)
        y_diffs = np.abs(np.diff(vehicle.y))
        assert np.all(y_diffs < self.LANE_WIDTH / 2), "y-trajectory has a discontinuity"

    # -- vy lane-change tests --

    @pytest.mark.parametrize(
        "start_lanelet_id, lc_direction, expected_target_lanelet_id",
        [
            (1000, 1, 1001),   # left from rightmost lane
            (1001, -1, 1000),  # right from center lane
        ],
    )
    def test_vy_lc_reaches_target_lane(
        self, start_lanelet_id, lc_direction, expected_target_lanelet_id
    ):
        road = self._build_road()
        lc_delay = 2.0
        lc_vy = 1.25  # lane change takes ~3s for 3.75m width

        vehicle = Vehicle(
            0,
            start_lanelet_id=start_lanelet_id,
            start_s=10,
            start_t=0,
            v0=self.V0,
            lc_direction=lc_direction,
            lc_delay=lc_delay,
            lc_type="vy",
            lc_vy=lc_vy,
        )
        vehicle.compile(road, self.DURATION, self.DT)

        # Before LC starts the vehicle should be on the source lanelet
        pre_lc_step = int(lc_delay / self.DT) - 1
        pre_lc_lanelet = road.find_lanelet_id_by_position(
            vehicle.x[pre_lc_step], vehicle.y[pre_lc_step]
        )
        assert pre_lc_lanelet == start_lanelet_id

        # After LC the vehicle should be on the target lanelet
        expected_lc_steps = int(np.ceil(self.LANE_WIDTH / (lc_vy * self.DT)))
        post_lc_step = int(lc_delay / self.DT) + expected_lc_steps + 5
        post_lc_lanelet = road.find_lanelet_id_by_position(
            vehicle.x[post_lc_step], vehicle.y[post_lc_step]
        )
        assert post_lc_lanelet == expected_target_lanelet_id

        # Vehicle should remain on the target lanelet until the end
        final_lanelet = road.find_lanelet_id_by_position(
            vehicle.x[-1], vehicle.y[-1]
        )
        assert final_lanelet == expected_target_lanelet_id

    def test_vy_lc_lateral_offset_clamps(self):
        """After the vy LC finishes the y-position must stay constant (clamped)."""
        road = self._build_road()
        lc_delay = 2.0
        lc_vy = 1.25

        vehicle = Vehicle(
            0,
            start_lanelet_id=1000,
            start_s=10,
            start_t=0,
            v0=self.V0,
            lc_direction=1,
            lc_delay=lc_delay,
            lc_type="vy",
            lc_vy=lc_vy,
        )
        vehicle.compile(road, self.DURATION, self.DT)

        # Determine the step where the LC should be finished
        expected_lc_steps = int(np.ceil(self.LANE_WIDTH / (lc_vy * self.DT)))
        settled_step = int(lc_delay / self.DT) + expected_lc_steps + 2
        settled_y = vehicle.y[settled_step:]

        # All post-LC y-values should be approximately the same (straight road)
        assert np.allclose(settled_y, settled_y[0], atol=0.05), (
            "y-position should stay constant after vy lane change completes"
        )

    def test_vy_lc_zero_delay(self):
        """vy lane change with zero delay should still work correctly."""
        road = self._build_road()
        lc_vy = 1.25

        vehicle = Vehicle(
            0,
            start_lanelet_id=1000,
            start_s=10,
            start_t=0,
            v0=self.V0,
            lc_direction=1,
            lc_delay=0,
            lc_type="vy",
            lc_vy=lc_vy,
        )
        vehicle.compile(road, self.DURATION, self.DT)

        # Vehicle should end up on the target lanelet
        final_lanelet = road.find_lanelet_id_by_position(
            vehicle.x[-1], vehicle.y[-1]
        )
        assert final_lanelet == 1001

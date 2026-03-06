from pathlib import Path

from simple_scenario import Scenario, EgoConfiguration, Vehicle
from simple_scenario.road import SyntheticRoad, StraightSegment


class TestOncomingTraffic:
    RESULT_DIR = Path(__file__).parent / "results" / "test_oncoming_traffic"
    RESULT_DIR.mkdir(exist_ok=True)

    def test_scenario_gen(self):
        result_dir = self.RESULT_DIR / "test_scenario_gen"
        result_dir.mkdir(exist_ok=True)

        road = SyntheticRoad(2, 3.75, [StraightSegment(500)])

        ego = EgoConfiguration(
            start_lanelet_id=1000,
            start_s=50,
            start_t=0,
            v0=100 / 3.6,
            target_s=400,
            target_t=0,
        )

        overtaking_vehicle = Vehicle(
            vehicle_id=0,
            start_lanelet_id=1001,
            start_s=400,
            start_t=0,
            v0=100 / 3.6,
            lc_delay=3,
            lc_direction=-1,
            inverse_driving_direction=True,
        )
        slow_vehicle = Vehicle(
            vehicle_id=1,
            start_lanelet_id=1001,
            start_s=330,
            start_t=0.5,
            v0=60 / 3.6,
            inverse_driving_direction=True,
        )

        scenario = Scenario(
            "oncoming_traffic", road, ego, [overtaking_vehicle, slow_vehicle], 13
        )

        scenario.render_gif(result_dir)

        for vehicle in scenario.vehicles:
            if vehicle.inverse_driving_direction:
                assert vehicle.x[-1] < vehicle.x[0], (
                    f"Vehicle {vehicle.id} should drive in inverse driving direction, but it is not."
                )

        print("TEST PASSED")


if __name__ == "__main__":
    tester = TestOncomingTraffic()
    tester.test_scenario_gen()

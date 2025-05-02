from pathlib import Path

from simple_scenario import Scenario, EgoConfiguration, Vehicle
from simple_scenario.road import Road, StraightSegment


class TestOncomingTraffic:
    RESULT_DIR = Path(__file__).parent / "results" / "test_oncoming_traffic"
    RESULT_DIR.mkdir(exist_ok=True)

    def test_scenario_gen(self):
        result_dir = self.RESULT_DIR / "test_scenario_gen"
        result_dir.mkdir(exist_ok=True)

        road = Road(2, 3.75, [StraightSegment(500)])

        ego = EgoConfiguration(1000, 50, 0, 100 / 3.6)

        overtaking_vehicle = Vehicle(
            0,
            1001,
            400,
            0,
            100 / 3.6,
            lc_delay=3,
            lc_direction=-1,
            inverse_driving_direction=True,
        )
        slow_vehicle = Vehicle(
            1, 1001, 330, 0.5, 60 / 3.6, inverse_driving_direction=True
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

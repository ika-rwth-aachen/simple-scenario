from pathlib import Path

from simple_scenario import Vehicle
from simple_scenario.rendering import create_scenario_plot_ax
from simple_scenario.road import Road, StraightSegment


class TestVehicle:
    RESULT_DIR = Path(__file__).parent / "results" / "test_vehicle"
    RESULT_DIR.mkdir(exist_ok=True)

    def test_vehicle(self):
        result_dir = self.RESULT_DIR / "test_vehicle"
        result_dir.mkdir(exist_ok=True)

        print("Create vehicle")

        vehicle = Vehicle(0, 1000, 10, 0, 27.78)

        # Access parameters
        print(f"Length: {vehicle.length}")
        print(f"Width: {vehicle.width}")

        # For compilation, we need a road
        print("Compile vehicle")
        road = Road(3, 3.75, [StraightSegment(300)])

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

        vehicle = Vehicle(0, 1000, 10, 0, 27.78, lc_direction=1, lc_delay=2)

        # Access parameters
        print(f"Length: {vehicle.length}")
        print(f"Width: {vehicle.width}")

        # For compilation, we need a road
        print("Compile vehicle")
        road = Road(3, 3.75, [StraightSegment(300)])

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

        vehicle = Vehicle(0, 1001, 10, 0, 27.78, lc_direction=-1, lc_delay=2)

        # Access parameters
        print(f"Length: {vehicle.length}")
        print(f"Width: {vehicle.width}")

        # For compilation, we need a road
        print("Compile vehicle")
        road = Road(3, 3.75, [StraightSegment(300)])

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

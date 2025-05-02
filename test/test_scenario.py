import json
import numpy as np
import pytest

from pathlib import Path

from simple_scenario import (
    Scenario,
    EgoConfiguration,
    Vehicle,
    CR_AVAILABLE,
)
from simple_scenario.rendering import create_scenario_plot_ax
from simple_scenario.road import Road, StraightSegment


class TestScenario:
    DATA_DIR = Path(__file__).parent / "assets"
    RESULT_DIR = Path(__file__).parent / "results" / "test_scenario"
    RESULT_DIR.mkdir(exist_ok=True)

    @staticmethod
    def _check_feasible(scenario: Scenario):
        if CR_AVAILABLE:
            assert scenario.is_feasible()
        else:
            with pytest.raises(ModuleNotFoundError):
                assert scenario.is_feasible()

    @staticmethod
    def _get_cr_interface(scenario: Scenario):
        if CR_AVAILABLE:
            scenario.get_cr_interface()
        else:
            with pytest.raises(ModuleNotFoundError):
                scenario.get_cr_interface()

    @staticmethod
    def _save_to_file(scenario: Scenario, result_dir: Path):
        if scenario._initialized_from_data:  # noqa: SLF001
            with pytest.raises(
                ValueError,
                match="Cannot save to config if the scenario has been initialized from data.",
            ):
                scenario.save(result_dir)
            with pytest.raises(
                ValueError,
                match="Cannot save to openx if the scenario has been initialized from data.",
            ):
                scenario.save(result_dir, mode="openx")
        else:
            scenario.save(result_dir)
            scenario.save(result_dir, mode="openx")

        if CR_AVAILABLE:
            scenario.save(result_dir, mode="cr")
        else:
            with pytest.raises(ModuleNotFoundError):
                scenario.save(result_dir, mode="cr")

    def test_scenario_creation(self):
        """
        Create a simple scenario in a pythonic way and save it to file (osm and json file!)
        """

        result_dir = self.RESULT_DIR / "test_scenario_creation"
        result_dir.mkdir(exist_ok=True)

        dt = 0.1
        duration = 10

        # Create road
        road = Road(3, 3.75, [StraightSegment(500, heading=0)], goal_position=450)

        # Create ego configuration
        ego_configuration = EgoConfiguration(1000, 50, 0, 27.78)

        # Create vehicles

        # A vehicle in front of the ego vehicle
        vehicle0_thw0 = 3
        vehicle0_s0 = ego_configuration.s0 + ego_configuration.v0 * vehicle0_thw0
        vehicle0 = Vehicle(
            0, ego_configuration.lanelet_id, vehicle0_s0, 0, ego_configuration.v0
        )

        # A vehicle in front of vehicle 0
        vehicle1_thw0 = 3
        vehicle1_s0 = vehicle0.s0 + vehicle0.v0 * vehicle1_thw0
        vehicle1 = Vehicle(1, ego_configuration.lanelet_id, vehicle1_s0, 0, vehicle0.v0)

        # A vehicle on another lane
        vehicle2 = Vehicle(2, 1001, ego_configuration.s0, 0, ego_configuration.v0)

        vehicles = [vehicle0, vehicle1, vehicle2]

        # Create scenario
        scenario = Scenario("test", road, ego_configuration, vehicles, duration, dt=dt)

        # Render overview
        scenario.render(result_dir, "default")
        scenario.render(result_dir, "clean", clean=True)
        # Render timestep
        scenario.render(result_dir, timestep=50, plot_name_suffix="timestep_50")
        # Render GIF
        scenario.render_gif(result_dir, dpi=300)

        # Test ax mode
        with create_scenario_plot_ax(
            result_dir, f"scenario_{scenario.id}_axmode"
        ) as ax:
            scenario.render(ax)

        # Get cr interface
        self._get_cr_interface(scenario)

        # Save to file
        self._save_to_file(scenario, result_dir)

        # Check feasibility
        self._check_feasible(scenario)

    def test_scenario_creation_big_road(self):
        """
        Create a simple scenario in a pythonic way and save it to file (osm and json file!)
        """

        result_dir = self.RESULT_DIR / "test_scenario_creation_big_road"
        result_dir.mkdir(exist_ok=True)

        dt = 0.1
        duration = 10

        # Create road
        road = Road(5, 3.75, [StraightSegment(500, heading=0)], goal_position=450)

        # Create ego configuration
        ego_configuration = EgoConfiguration(1000, 50, 0, 27.78)

        # Create vehicles

        # A vehicle in front of the ego vehicle
        vehicle0_thw0 = 3
        vehicle0_s0 = ego_configuration.s0 + ego_configuration.v0 * vehicle0_thw0
        vehicle0 = Vehicle(
            0, ego_configuration.lanelet_id, vehicle0_s0, 0, ego_configuration.v0
        )

        # A vehicle in front of vehicle 0
        vehicle1_thw0 = 3
        vehicle1_s0 = vehicle0.s0 + vehicle0.v0 * vehicle1_thw0
        vehicle1 = Vehicle(1, ego_configuration.lanelet_id, vehicle1_s0, 0, vehicle0.v0)

        # A vehicle on another lane
        vehicle2 = Vehicle(2, 1001, ego_configuration.s0, 0, ego_configuration.v0)

        vehicles = [vehicle0, vehicle1, vehicle2]

        # Create scenario
        scenario = Scenario(
            "test_big_road", road, ego_configuration, vehicles, duration, dt=dt
        )

        # Render overview
        scenario.render(result_dir, "default")
        scenario.render(result_dir, "clean", clean=True)
        # Render timestep
        scenario.render(result_dir, "timestep_50", timestep=50)
        # Render GIF
        scenario.render_gif(result_dir, dpi=300)

        # Test ax mode
        with create_scenario_plot_ax(
            result_dir, f"scenario_{scenario.id}_axmode"
        ) as ax:
            scenario.render(ax)

        # Get cr interface
        self._get_cr_interface(scenario)

        # Save to file
        self._save_to_file(scenario, result_dir)

        # Check feasibility
        self._check_feasible(scenario)

    def test_load_from_config(self):
        result_dir = self.RESULT_DIR / "test_load_from_config"
        result_dir.mkdir(exist_ok=True)

        config_file = self.DATA_DIR / "scenario_config.json"

        with config_file.open("r") as f:
            config = json.load(f)

        scenario = Scenario.from_config(config)

        scenario.render(result_dir, "from_config")

        # Save to file
        self._save_to_file(scenario, result_dir)

        # Check feasibility
        self._check_feasible(scenario)

    def test_load_from_config_file(self):
        result_dir = self.RESULT_DIR / "test_load_from_config_file"
        result_dir.mkdir(exist_ok=True)

        config_file = self.DATA_DIR / "scenario_config.json"

        scenario = Scenario.from_config_file(config_file)

        scenario.render(result_dir, "from_config_file")

        # Get cr interface
        self._get_cr_interface(scenario)

        # Save to file
        self._save_to_file(scenario, result_dir)

        # Check feasibility
        self._check_feasible(scenario)

    def test_load_from_cr_xml(self):
        result_dir = self.RESULT_DIR / "test_load_from_config_file"
        result_dir.mkdir(exist_ok=True)

        xml = self.DATA_DIR / "ZAM_ValidationScenario+No+PaperScenario1-0_0_T-0.xml"

        if CR_AVAILABLE:
            scenario = Scenario.from_cr_xml(xml)

            scenario.render(result_dir, "from_cr_xml")

            # Save to file
            self._save_to_file(scenario, result_dir)

            # Check feasibility
            self._check_feasible(scenario)
        else:
            with pytest.raises(ModuleNotFoundError):
                scenario = Scenario.from_cr_xml(xml)

    def test_load_from_x_config(self):
        result_dir = self.RESULT_DIR / "test_load_from_x_config"
        result_dir.mkdir(exist_ok=True)

        config_file = self.DATA_DIR / "scenario_config.json"

        with config_file.open("r") as f:
            config = json.load(f)

        scenario = Scenario.from_x(config)

        scenario.render(result_dir, "from_x_config")

        # Get cr interface
        self._get_cr_interface(scenario)

        # Save to file
        self._save_to_file(scenario, result_dir)

        # Check feasibility
        self._check_feasible(scenario)

    def test_load_from_x_config_file(self):
        result_dir = self.RESULT_DIR / "test_load_from_x_config_file"
        result_dir.mkdir(exist_ok=True)

        config_file = self.DATA_DIR / "scenario_config.json"

        scenario = Scenario.from_x(config_file)

        scenario.render(result_dir, "from_x_config_file")

        # Get cr interface
        self._get_cr_interface(scenario)

        # Save to file
        self._save_to_file(scenario, result_dir)

        # Check feasibility
        self._check_feasible(scenario)

    def test_load_from_x_cr_xml(self):
        result_dir = self.RESULT_DIR / "test_load_from_x_cr_xml"
        result_dir.mkdir(exist_ok=True)

        xml = self.DATA_DIR / "ZAM_ValidationScenario+No+PaperScenario1-0_0_T-0.xml"

        if CR_AVAILABLE:
            scenario = Scenario.from_x(xml)

            scenario.render(result_dir, "from_x_cr_xml")

            # Get cr interface
            self._get_cr_interface(scenario)

            # Save to file
            self._save_to_file(scenario, result_dir)

            # Check feasibility
            self._check_feasible(scenario)
        else:
            with pytest.raises(ModuleNotFoundError):
                scenario = Scenario.from_x(xml)

    def test_load_from_x_object(self):
        result_dir = self.RESULT_DIR / "test_load_from_x_object"
        result_dir.mkdir(exist_ok=True)

        config_file = self.DATA_DIR / "scenario_config.json"

        scenario = Scenario.from_x(config_file)
        same_scenario = Scenario.from_x(scenario)

        same_scenario.render(result_dir, "test_load_from_x_object")

        # Get cr interface
        self._get_cr_interface(scenario)

        # Save to file
        self._save_to_file(scenario, result_dir)

        # Check feasibility
        self._check_feasible(scenario)

    def test_scenario_cr_interface(self):
        """
        Create a simple scenario in a pythonic way and save it to file (osm and json file!)
        """

        dt = 0.1
        duration = 10

        # Create road
        road = Road(3, 3.75, [StraightSegment(500, heading=0)], goal_position=450)

        # Create ego configuration
        ego_configuration = EgoConfiguration(1000, 50, 0, 27.78)

        # Create vehicles

        # A vehicle in front of the ego vehicle
        vehicle0_thw0 = 3
        vehicle0_s0 = ego_configuration.s0 + ego_configuration.v0 * vehicle0_thw0
        vehicle0 = Vehicle(
            0, ego_configuration.lanelet_id, vehicle0_s0, 0, ego_configuration.v0
        )

        # A vehicle in front of vehicle 0
        vehicle1_thw0 = 3
        vehicle1_s0 = vehicle0.s0 + vehicle0.v0 * vehicle1_thw0
        vehicle1 = Vehicle(1, ego_configuration.lanelet_id, vehicle1_s0, 0, vehicle0.v0)

        # A vehicle on another lane
        vehicle2 = Vehicle(2, 1001, ego_configuration.s0, 0, ego_configuration.v0)

        vehicles = [vehicle0, vehicle1, vehicle2]

        # Create scenario
        scenario = Scenario("test", road, ego_configuration, vehicles, duration, dt=dt)

        # Get cr interface
        self._get_cr_interface(scenario)

        # Check feasibility
        self._check_feasible(scenario)

    def test_heading_calcuation_standstill(self):
        config_json = self.DATA_DIR / "scenario_standstill.json"
        scenario = Scenario.from_x(config_json)

        assert np.all(scenario.vehicles[0].heading) != 0
        self._get_cr_interface(scenario)
        self._check_feasible(scenario)


if __name__ == "__main__":
    tester = TestScenario()
    tester.test_heading_calcuation_standstill()

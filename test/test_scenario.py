from __future__ import annotations

import json
import numpy as np
import pytest
import xml.etree.ElementTree as ET

from pathlib import Path

from simple_scenario import (
    Scenario,
    EgoConfiguration,
    Vehicle,
    CR_AVAILABLE,
)
from simple_scenario.rendering import create_scenario_plot_ax
from simple_scenario.road import SyntheticRoad, StraightSegment


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
            with pytest.raises(
                ValueError,
                match="Cannot save to lanelet2 if the scenario has been initialized from data.",
            ):
                scenario.save(result_dir, mode="lanelet2")
        else:
            scenario.save(result_dir)
            scenario.save(result_dir, mode="openx")
            scenario.save(result_dir, mode="lanelet2")

        if CR_AVAILABLE:
            scenario.save(result_dir, mode="cr")
        else:
            with pytest.raises(ModuleNotFoundError):
                scenario.save(result_dir, mode="cr")

    @staticmethod
    def _get_world_positions(xosc_path: Path) -> list:
        root = ET.parse(xosc_path).getroot()  # noqa: S314
        return list(root.findall(".//WorldPosition"))

    @staticmethod
    def _minimal_openx_config(
        scenario_id: str,
        ego_z: float | None = None,
        vehicle_z: float | None = None,
    ) -> dict:
        config = {
            "scenario_id": scenario_id,
            "road": {
                "n_lanes": 2,
                "lane_width": 3.75,
                "segments": [{"length": 200, "heading": 0.0}],
                "speed_limit": 120,
                "x0": 0,
                "y0": 0,
            },
            "ego_configuration": {
                "start_lanelet_id": 1000,
                "start_s": 10,
                "start_t": 0,
                "target_s": 120,
                "target_t": 0,
                "v0": 10,
                "vehicle_type_name": "medium",
            },
            "vehicles": [
                {
                    "vehicle_id": 1,
                    "start_lanelet_id": 1000,
                    "start_s": 20,
                    "start_t": 0,
                    "v0": 8,
                    "vehicle_type_name": "medium",
                }
            ],
            "duration": 1.0,
            "dt": 0.2,
        }

        if ego_z is not None:
            config["ego_configuration"]["z"] = ego_z
        if vehicle_z is not None:
            config["vehicles"][0]["z"] = vehicle_z

        return config

    def test_scenario_creation(self):
        """
        Create a simple scenario in a pythonic way and save it to file (osm and json file!)
        """

        result_dir = self.RESULT_DIR / "test_scenario_creation"
        result_dir.mkdir(exist_ok=True)

        dt = 0.1
        duration = 10

        # Create road
        road = SyntheticRoad(3, 3.75, [StraightSegment(500, heading=0)])

        # Create ego configuration
        ego_configuration = EgoConfiguration(
            start_lanelet_id=1000,
            start_s=50,
            start_t=0,
            v0=27.78,
            target_s=450,
            target_t=0,
        )
        ego_configuration.compile(road)

        # Create vehicles

        # A vehicle in front of the ego vehicle
        vehicle0_thw0 = 3
        vehicle0_start_s = (
            ego_configuration.start_s + ego_configuration.v0 * vehicle0_thw0
        )
        vehicle0 = Vehicle(
            0,
            start_lanelet_id=ego_configuration.start_lanelet_id,
            start_s=vehicle0_start_s,
            start_t=0,
            v0=ego_configuration.v0,
        )

        # A vehicle in front of vehicle 0
        vehicle1_thw0 = 3
        vehicle1_start_s = vehicle0.start_s + vehicle0.v0 * vehicle1_thw0
        vehicle1 = Vehicle(
            1,
            start_lanelet_id=ego_configuration.start_lanelet_id,
            start_s=vehicle1_start_s,
            start_t=0,
            v0=vehicle0.v0,
        )

        # A vehicle on another lane
        vehicle2 = Vehicle(
            2,
            start_lanelet_id=1001,
            start_s=ego_configuration.start_s,
            start_t=0,
            v0=ego_configuration.v0,
        )

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
        road = SyntheticRoad(5, 3.75, [StraightSegment(500, heading=0)])

        # Create ego configuration
        ego_configuration = EgoConfiguration(
            start_lanelet_id=1000,
            start_s=50,
            start_t=0,
            v0=27.78,
            target_s=450,
            target_t=0,
        )
        ego_configuration.compile(road)

        # Create vehicles

        # A vehicle in front of the ego vehicle
        vehicle0_thw0 = 3
        vehicle0_start_s = (
            ego_configuration.start_s + ego_configuration.v0 * vehicle0_thw0
        )
        vehicle0 = Vehicle(
            0,
            start_lanelet_id=ego_configuration.start_lanelet_id,
            start_s=vehicle0_start_s,
            start_t=0,
            v0=ego_configuration.v0,
        )

        # A vehicle in front of vehicle 0
        vehicle1_thw0 = 3
        vehicle1_start_s = vehicle0.start_s + vehicle0.v0 * vehicle1_thw0
        vehicle1 = Vehicle(
            1,
            start_lanelet_id=ego_configuration.start_lanelet_id,
            start_s=vehicle1_start_s,
            start_t=0,
            v0=vehicle0.v0,
        )

        # A vehicle on another lane
        vehicle2 = Vehicle(
            2,
            start_lanelet_id=1001,
            start_s=ego_configuration.start_s,
            start_t=0,
            v0=ego_configuration.v0,
        )

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

    def test_openx_export_with_z(self):
        result_dir = self.RESULT_DIR / "test_openx_export_with_z"
        result_dir.mkdir(exist_ok=True)

        scenario = Scenario.from_config(
            self._minimal_openx_config("test_openx_z", ego_z=1.25, vehicle_z=2.5)
        )
        scenario.save(result_dir, mode="openx")

        assert np.isclose(scenario.ego_configuration.z, 1.25)
        assert np.isclose(scenario.vehicles[0].z, 2.5)

        xosc_path = result_dir / "test_openx_z.xosc"
        world_positions = self._get_world_positions(xosc_path)
        z_values = [
            float(position.attrib["z"])
            for position in world_positions
            if "z" in position.attrib
        ]

        assert len(z_values) == len(world_positions)
        assert any(np.isclose(z, 1.25) for z in z_values)
        assert any(np.isclose(z, 2.5) for z in z_values)

    def test_openx_export_without_z_keeps_previous_behavior(self):
        result_dir = self.RESULT_DIR / "test_openx_export_without_z"
        result_dir.mkdir(exist_ok=True)

        scenario = Scenario.from_config(self._minimal_openx_config("test_openx_no_z"))
        scenario.save(result_dir, mode="openx")

        xosc_path = result_dir / "test_openx_no_z.xosc"
        world_positions = self._get_world_positions(xosc_path)

        assert all("z" not in position.attrib for position in world_positions)

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
        road = SyntheticRoad(3, 3.75, [StraightSegment(500, heading=0)])

        # Create ego configuration
        ego_configuration = EgoConfiguration(
            start_lanelet_id=1000,
            start_s=50,
            start_t=0,
            v0=27.78,
            target_s=450,
            target_t=0,
        )
        ego_configuration.compile(road)

        # Create vehicles

        # A vehicle in front of the ego vehicle
        vehicle0_thw0 = 3
        vehicle0_start_s = (
            ego_configuration.start_s + ego_configuration.v0 * vehicle0_thw0
        )
        vehicle0 = Vehicle(
            0,
            start_lanelet_id=ego_configuration.start_lanelet_id,
            start_s=vehicle0_start_s,
            start_t=0,
            v0=ego_configuration.v0,
        )

        # A vehicle in front of vehicle 0
        vehicle1_thw0 = 3
        vehicle1_start_s = vehicle0.start_s + vehicle0.v0 * vehicle1_thw0
        vehicle1 = Vehicle(
            1,
            start_lanelet_id=ego_configuration.start_lanelet_id,
            start_s=vehicle1_start_s,
            start_t=0,
            v0=vehicle0.v0,
        )

        # A vehicle on another lane
        vehicle2 = Vehicle(
            2,
            start_lanelet_id=1001,
            start_s=ego_configuration.start_s,
            start_t=0,
            v0=ego_configuration.v0,
        )

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

        assert np.all(scenario.vehicles[0].heading != 0)
        self._get_cr_interface(scenario)

        self._check_feasible(scenario)


if __name__ == "__main__":
    tester = TestScenario()
    tester.test_scenario_creation()

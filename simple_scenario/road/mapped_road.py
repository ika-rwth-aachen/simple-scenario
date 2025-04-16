from __future__ import annotations

import lanelet2
import numpy as np
import pyproj
import shutil

from lanelet2.core import LaneletMap, BasicPoint3d
from lanelet2.projection import UtmProjector
from lxml import etree
from pathlib import Path

from .road import Road


class MappedRoad(Road):
    def __init__(
        self, opendrive_map_file: str | Path, lanelet2_map_file: str | Path
    ) -> None:
        # Check inputs
        opendrive_map_file = Path(opendrive_map_file)
        lanelet2_map_file = Path(lanelet2_map_file)

        for file in (opendrive_map_file, lanelet2_map_file):
            if not file.exists():
                msg = f"File {file} does not exist"
                raise FileNotFoundError(msg)

        self._opendrive_map_file = opendrive_map_file
        self._lanelet2_map_file = lanelet2_map_file

        # Find xodr's proj
        self._xodr_proj_str = self._load_xodr_proj_str(self._opendrive_map_file)
        self._xodr_proj = pyproj.Proj(projparams=self._xodr_proj_str)
        # Extrat lat, lon origin
        self._origin_lon, self._origin_lat = self._xodr_proj(0, 0, inverse=True)
        # Find corresponding utm epsg
        utm_crs_list = pyproj.database.query_utm_crs_info(
            datum_name="WGS 84",
            area_of_interest=pyproj.aoi.AreaOfInterest(
                west_lon_degree=self._origin_lon,
                south_lat_degree=self._origin_lat,
                east_lon_degree=self._origin_lon,
                north_lat_degree=self._origin_lat,
            ),
        )
        if len(utm_crs_list) != 1:
            msg = "Cannot find exactly one UTM zone for the given OpenDrive origin"
            raise ValueError(msg)
        utm_epsg_code = int(utm_crs_list[0].code)
        self._utm_proj = pyproj.Proj(utm_epsg_code)

        # Load lanelet2 map
        self._llt_utm_projector = UtmProjector(
            lanelet2.io.Origin(self._origin_lat, self._origin_lon)
        )
        self._lanelet_map = lanelet2.io.load(
            str(self._lanelet2_map_file),
            self._llt_utm_projector,
        )

    def _load_xodr_proj_str(self, opendrive_map_file: str | Path) -> str:
        """
        Read origin lat lon from the opendrive map file's geoReference proj string.
        """

        # Find proj string
        tree = etree.parse(str(opendrive_map_file))
        proj_str = tree.getroot().find(".//geoReference").text

        if not proj_str:
            msg = "Could not find geoReference string in opendrive map"
            raise ValueError(msg)

        return proj_str

    def copy(self) -> MappedRoad:
        return MappedRoad(
            opendrive_map_file=self._opendrive_map_file,
            lanelet2_map_file=self._lanelet2_map_file,
        )

    @property
    def config(self) -> dict:
        return {
            "opendrive_map_file": str(self._opendrive_map_file),
            "lanelet2_map_file": str(self._lanelet2_map_file),
        }

    @property
    def opendrive_map_file(self) -> Path:
        return self._opendrive_map_file

    @property
    def lanelet2_map_file(self) -> Path:
        return self._lanelet2_map_file

    @property
    def origin_lat(self) -> float:
        return self._origin_lat

    @property
    def origin_lon(self) -> float:
        return self._origin_lon

    @property
    def lanelet_map(self) -> LaneletMap:
        return self._lanelet_map

    def from_llt_local_to_opendrive_local(
        self, x_llt2: float, y_llt2: float, heading_llt2: float | None = None
    ) -> tuple[float, float] | tuple[float, float, float]:
        """
        Transform a position from the lanelet maps local coordinate frame to the opendrive maps local coordinate frame.
        The lanelet2 map uses a transform from geographic coordinates (lat, lon) to a coordinate frame in UTM coordinates (relative to the origin).
        The opendrive map uses a transform given in the xodr file.

        Idea: Transform from llt2 local coordinates to lat, lon, then transform from lat, lon to xodr local coordinates.
        """

        # From llt2's local utm coordinates to lat, lon
        gps_point = self._llt_utm_projector.reverse(BasicPoint3d(x_llt2, y_llt2, 0))
        lat, lon = gps_point.lat, gps_point.lon

        # From lat, lon to xodr's local coordinates
        x_xodr, y_xodr = self._xodr_proj(lon, lat)

        if heading_llt2 is None:
            return x_xodr, y_xodr

        # Grid convergence
        grid_convergence_llt2 = np.deg2rad(
            self._utm_proj.get_factors(lon, lat).meridian_convergence
        )
        grid_convergence_xodr = np.deg2rad(
            self._xodr_proj.get_factors(lon, lat).meridian_convergence
        )
        heading_xodr = heading_llt2 - grid_convergence_llt2 + grid_convergence_xodr

        # Alternative: 2 point method (same results)
        # pt2x = x_llt2 + np.cos(heading_llt2)  # noqa: ERA001
        # pt2y = y_llt2 + np.sin(heading_llt2)  # noqa: ERA001
        # gps_point2 = self._llt_utm_projector.reverse(BasicPoint3d(pt2x, pt2y, 0))  # noqa: ERA001
        # heading_ = np.arctan2(pt2y - y_llt2, pt2x - x_llt2)  # noqa: ERA001
        # pt2x_xodr, pt2y_xodr = self._xodr_proj(gps_point2.lon, gps_point2.lat)  # noqa: ERA001
        # heading_xodr_ = np.arctan2(pt2y_xodr - y_xodr, pt2x_xodr - x_xodr)  # noqa: ERA001

        return x_xodr, y_xodr, heading_xodr

    def save_lanelet2_map(
        self, save_dir: str | Path, map_name: str | None = None
    ) -> Path:
        """
        Save the lanelet2 map to the specified directory.
        """
        save_dir = Path(save_dir)
        if not save_dir.is_dir():
            msg = "Save directory does not exist"
            raise FileNotFoundError(msg)

        if map_name is None:
            map_name = self._lanelet2_map_file.stem

        save_file = save_dir / (map_name + self._lanelet2_map_file.suffix)

        # Copy the original file to the save_dir
        shutil.copyfile(self._lanelet2_map_file, save_file)

        return save_file

    def save_opendrive_map(
        self, save_dir: str | Path, map_name: str | None = None
    ) -> Path:
        """
        Save the opendrive map to the specified directory.
        """
        save_dir = Path(save_dir)
        if not save_dir.is_dir():
            msg = "Save directory does not exist"
            raise FileNotFoundError(msg)

        if map_name is None:
            map_name = self._opendrive_map_file.stem

        save_file = save_dir / (map_name + self._opendrive_map_file.suffix)

        # Copy the original file to the save_dir
        shutil.copyfile(self._opendrive_map_file, save_file)

        return save_file

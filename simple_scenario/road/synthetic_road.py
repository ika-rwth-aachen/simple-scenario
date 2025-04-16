from __future__ import annotations

import lanelet2
import numpy as np

from copy import deepcopy
from lanelet2.core import (
    getId,
    Lanelet,
    LaneletMap,
    LineString3d,
    Point3d,
    AttributeMap,
)
from lanelet2.projection import UtmProjector

from loguru import logger
from pathlib import Path
from scenariogeneration import xodr
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

from .road import Road
from .road_segment import RoadSegment
from .straight_segment import StraightSegment
from .clothoid_segment import ClothoidSegment
from .arc_segment import ArcSegment
from .polyline_segment import PolylineSegment
from ..rendering import create_plot_ax


class SyntheticRoad(Road):
    ALLOWED_SEGMENT_SEQUENCES = (
        (StraightSegment,),
        (StraightSegment, ClothoidSegment, ArcSegment),
        (PolylineSegment,),
    )

    # GPS origin centered in UTM 32N
    _ORIGIN_LAT = 50.0
    _ORIGIN_LON = 9.0

    def __init__(
        self,
        n_lanes: int,
        lane_width: float,
        segments: list[RoadSegment],
        speed_limit: float = 120,
        x0: float = 0,
        y0: float = 0,
    ) -> None:
        """
        n_lanes: Number of lanes in t-direction
        lane_width: Lane width of each of the lanes in m
        segments: List of segments in s-direction
        speed_limit: Speed limit on the road section in km/h
        x0: Start position (x) of the road in cartesian coordinates in m
        y0: Start position (y) of the road in cartesian coordinates in m
        """

        if not isinstance(segments, list):
            msg = "segments need to be a list"
            raise TypeError(msg)
        if len(segments) == 0:
            msg = "There needs to be at least one segment"
            raise TypeError(msg)
        if not all(isinstance(s, RoadSegment) for s in segments):
            msg = "Every segment needs to be a RoadSegment"
            raise TypeError(msg)

        self._config = {
            "n_lanes": n_lanes,
            "lane_width": lane_width,
            "segments": [s.config for s in segments],
            "speed_limit": speed_limit,
            "x0": x0,
            "y0": y0,
        }

        self._n_lanes = n_lanes
        self._lane_width = lane_width
        self._segments = segments
        self._speed_limit = speed_limit
        self._x0 = x0
        self._y0 = y0

        self._create_road_from_segments()

        self._overall_ref_line = None
        self._ref_line_length = None
        self._overall_ref_line_curvature = None
        self._overall_ref_line_heading = None
        self._overall_offset_lines = {}
        self._boundary_line = None

        self._lanelet_map = self._create_lanelet_map()

    def copy(self) -> SyntheticRoad:
        road_copy = SyntheticRoad(
            self.n_lanes,
            self.lane_width,
            segments=deepcopy(self.segments),
            speed_limit=self.speed_limit,
            x0=self.x0,
            y0=self.y0,
        )
        return road_copy

    @property
    def config(self) -> dict:
        return self._config

    @property
    def n_lanes(self) -> int:
        return self._n_lanes

    @property
    def lane_width(self) -> int:
        return self._lane_width

    @property
    def speed_limit(self) -> int:
        return self._speed_limit

    @property
    def segments(self) -> list[RoadSegment]:
        return deepcopy(self._segments)

    @property
    def x0(self) -> float:
        return self._x0

    @property
    def y0(self) -> float:
        return self._y0

    def _create_lanelet_map(self) -> LaneletMap:
        """
        Generate Lanelet2 map from road lines
        """

        all_lines = self.offset_lines
        all_lines[0.0] = self.ref_line

        offsets = sorted(all_lines.keys(), reverse=True)

        # -- Linestrings --
        all_linestrings = {}

        n_linestrings = len(offsets)
        for i_linestring, offset in enumerate(offsets):
            line = all_lines[offset]

            linestring_points = [Point3d(getId(), pt[0], pt[1], 0.0) for pt in line]

            # Check if border
            if i_linestring == 0 or i_linestring == n_linestrings - 1:
                attributes_dict = {"type": "line_thin", "subtype": "solid"}
            elif i_linestring % 2 == 0:
                attributes_dict = {"type": "line_thin", "subtype": "dashed"}
            else:
                attributes_dict = {}

            new_linestring = LineString3d(
                getId(), linestring_points, AttributeMap(attributes_dict)
            )
            all_linestrings[offset] = new_linestring

        # -- Lanelets --

        all_lanelets = []

        n_lanelets = self._n_lanes

        lanelet_base_id = 1000

        for i_lanelet in range(n_lanelets):
            # Attributes
            attributes_dict = {
                "type": "lanelet",
                "subtype": "highway",
                "location": "nonurban",
                "region": "de",
                "one_way": "yes",
                "speed_limit": f"{self._speed_limit}",
            }

            # Linestrings
            left_line_offset = offsets[2 + 2 * i_lanelet]
            center_line_offset = offsets[1 + 2 * i_lanelet]
            right_line_offset = offsets[0 + 2 * i_lanelet]

            left_linestring = all_linestrings[left_line_offset]
            center_linestring = all_linestrings[center_line_offset]
            right_linestring = all_linestrings[right_line_offset]

            new_lanelet = Lanelet(
                lanelet_base_id + i_lanelet,
                left_linestring,
                right_linestring,
                AttributeMap(attributes_dict),
            )
            new_lanelet.centerline = center_linestring

            all_lanelets.append(new_lanelet)

        # -- Lanelet map --

        lanelet_map = lanelet2.core.createMapFromLanelets(all_lanelets)

        return lanelet_map

    @property
    def lanelet_map(self) -> LaneletMap:
        return self._lanelet_map

    @property
    def origin_lat(self) -> float:
        return self._ORIGIN_LAT

    @property
    def origin_lon(self) -> float:
        return self._ORIGIN_LON

    @property
    def ref_line(self) -> np.ndarray:
        if self._overall_ref_line is None:
            segment_ref_lines = [seg.ref_line for seg in self._segments]
            overall_ref_line = np.concatenate(segment_ref_lines)
            self._overall_ref_line = overall_ref_line

        return deepcopy(self._overall_ref_line)

    @property
    def ref_line_curvature(self) -> np.ndarray:
        if self._overall_ref_line_curvature is None:
            segment_ref_line_curvatures = [
                seg.ref_line_curvature for seg in self._segments
            ]
            overall_curvature = np.concatenate(segment_ref_line_curvatures)
            self._overall_ref_line_curvature = overall_curvature

        return deepcopy(self._overall_ref_line_curvature)

    @property
    def ref_line_heading(self) -> np.ndarray:
        if self._overall_ref_line_heading is None:
            segment_ref_line_heading = [seg.ref_line_heading for seg in self._segments]
            overall_heading = np.concatenate(segment_ref_line_heading)
            self._overall_ref_line_heading = overall_heading

        return deepcopy(self._overall_ref_line_heading)

    @property
    def ref_line_length(self) -> float:
        if self._ref_line_length is None:
            self._ref_line_length = float(
                np.sum(
                    np.sqrt(
                        np.diff(self.ref_line[:, 0]) ** 2
                        + np.diff(self.ref_line[:, 1]) ** 2
                    )
                )
            )

        return self._ref_line_length

    @property
    def offset_lines(self) -> dict:
        if not self._overall_offset_lines:
            offset_line_list_per_offset = {}

            for seg in self._segments:
                for offset, line in seg.offset_lines.items():
                    if offset not in offset_line_list_per_offset:
                        offset_line_list_per_offset[offset] = []
                    offset_line_list_per_offset[offset].append(line)

            overall_offset_line_per_offset = {
                offset: np.concatenate(lines)
                for offset, lines in offset_line_list_per_offset.items()
            }
            self._overall_offset_lines = overall_offset_line_per_offset

        return deepcopy(self._overall_offset_lines)

    @property
    def boundary_line(self) -> np.ndarray:
        if self._boundary_line is None:
            outer_line0 = self._overall_ref_line

            all_t_offsets = sorted(self._overall_offset_lines.keys())
            highest_t = all_t_offsets[-1]
            outer_line1 = np.flipud(self._overall_offset_lines[highest_t])

            self._boundary_line = np.vstack(
                (outer_line0, outer_line1, outer_line0[0])
            )  # closed

        return self._boundary_line

    def _create_road_from_segments(self) -> None:
        segment_sequence = tuple([s.__class__ for s in self._segments])

        logger.debug("Road segment sequence: {}", segment_sequence)

        if segment_sequence not in self.ALLOWED_SEGMENT_SEQUENCES:
            msg = f"Give segment sequence ({segment_sequence}) is not in allowed segment sequence: ({self.ALLOWED_SEGMENT_SEQUENCES})"
            raise Exception(msg)

        if segment_sequence == self.ALLOWED_SEGMENT_SEQUENCES[0]:
            straight_segment = self._segments[0]

            logger.debug("Create overall ref_line")
            straight_ref_line = straight_segment.compute_ref_line(self._x0, self._y0)

            # Create offset lines
            lateral_offsets = []
            for i_lane in range(self._n_lanes):
                # Center line offset
                lateral_offsets.append((i_lane + 0.5) * self._lane_width)
                # Right boundary offset
                lateral_offsets.append((i_lane + 1) * self._lane_width)

            logger.debug("Create offset_lines at {}", lateral_offsets)

            for lateral_offset in lateral_offsets:
                straight_segment.compute_offset_line(lateral_offset)

        elif segment_sequence == self.ALLOWED_SEGMENT_SEQUENCES[1]:
            # Create segment objects
            straight_segment = self._segments[0]
            clothoid_segment = self._segments[1]
            arc_segment = self._segments[2]

            # Create ref_lines
            logger.debug("Create overall ref_line")
            straight_ref_line = straight_segment.compute_ref_line(self._x0, self._y0)
            clothoid_ref_line = clothoid_segment.compute_ref_line(
                straight_ref_line[-1, 0],
                straight_ref_line[-1, 1],
                straight_segment.heading,
                arc_segment.radius,
            )
            arc_segment.compute_ref_line(
                clothoid_ref_line[-1, 0],
                clothoid_ref_line[-1, 1],
                clothoid_segment.ref_clothoid.ThetaEnd,
            )

            # Create offset lines
            lateral_offsets = []
            for i_lane in range(self._n_lanes):
                # Center line offset
                lateral_offsets.append((i_lane + 0.5) * self._lane_width)
                # Right boundary offset
                lateral_offsets.append((i_lane + 1) * self._lane_width)

            logger.debug("Create offset_lines at {}", lateral_offsets)

            for lateral_offset in lateral_offsets:
                straight_segment.compute_offset_line(lateral_offset)
                clothoid_segment.compute_offset_line(lateral_offset)
                arc_segment.compute_offset_line(lateral_offset)

        elif segment_sequence == self.ALLOWED_SEGMENT_SEQUENCES[2]:
            # ref_line and offset_lines are already available, no need to do anything here
            pass

    def from_llt_local_to_opendrive_local(
        self, x_llt2: float, y_llt2: float, heading_llt2: float | None = None
    ) -> tuple[float, float] | tuple[float, float, float]:
        if heading_llt2 is None:
            return x_llt2, y_llt2

        return x_llt2, y_llt2, heading_llt2

    def save_opendrive_map(self, save_dir: str | Path, map_name: str) -> Path:
        """
        Create an opendrive map from the road object.
        Use odrviewer.io to visualize it.
        """

        save_dir = Path(save_dir)
        if not save_dir.is_dir():
            msg = "Save directory does not exist"
            raise FileNotFoundError(msg)

        odr = self._create_opendrive_map(map_name)

        odr_path = save_dir / f"{map_name}.xodr"
        odr.write_xml(str(odr_path))

        return odr_path

    def _create_opendrive_map(self, map_name: str) -> xodr.OpenDrive:
        if self._segments == self.ALLOWED_SEGMENT_SEQUENCES[2]:
            raise NotImplementedError

        geo_reference = f"<![CDATA[+proj=tmerc +lat_0={self._ORIGIN_LAT} +lon_0={self._ORIGIN_LON} +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +geoidgrids=egm96_15.gtx +vunits=m +no_defs ]]>"

        # Create odr object
        odr = xodr.OpenDrive(map_name, geo_reference=geo_reference)

        # Add segments
        odr_road_segments = []

        for segment in self._segments:
            if isinstance(segment, StraightSegment):
                odr_road_segment = xodr.Line(segment.length)
            elif isinstance(segment, ArcSegment):
                odr_road_segment = xodr.Arc(segment.curvature, segment.length)
            elif isinstance(segment, ClothoidSegment):
                odr_road_segment = xodr.Spiral(
                    segment.curvature_start, segment.curvature_end, segment.length
                )
            else:
                raise NotImplementedError

            odr_road_segments.append(odr_road_segment)

        # Create road
        road = xodr.create_road(
            odr_road_segments,
            1,
            left_lanes=0,
            right_lanes=self.n_lanes,
            lane_width=self.lane_width,
        )

        # Add road
        odr.add_road(road)

        # Magic
        odr.adjust_roads_and_lanes()

        return odr

    def save_lanelet2_map(self, result_dir: str | Path, map_name: str) -> Path:
        result_dir = Path(result_dir)

        lanelet2_map_file = result_dir / f"{map_name}.osm"

        projector = UtmProjector(lanelet2.io.Origin(self._ORIGIN_LAT, self._ORIGIN_LON))

        lanelet2.io.write(str(lanelet2_map_file), self._lanelet_map, projector)

        return lanelet2_map_file

    def _plot_in_ax(
        self, ax: plt.Axes, use_lanelet: bool = True, verbose: bool = False
    ) -> None:
        # Start actual plotting
        if use_lanelet:
            super()._plot_in_ax(ax)

        elif verbose:
            colors = ["g", "b", "r"]

            for i, seg in enumerate(self._segments):
                ref_line = seg.ref_line

                ax.plot(
                    ref_line[:, 0],
                    ref_line[:, 1],
                    f"{colors[i]}-",
                    linewidth=2,
                    label=f"ref_line: {seg}",
                )
                ax.plot(
                    ref_line[0, 0],
                    ref_line[0, 1],
                    f"{colors[i]}x",
                    label=f"start of ref_line: {seg}",
                )
                ax.plot(
                    ref_line[-1, 0],
                    ref_line[-1, 1],
                    f"{colors[i]}o",
                    label=f"end of ref_line: {seg}",
                )

                for line in seg.offset_lines.values():
                    ax.plot(line[:, 0], line[:, 1], f"{colors[i]}-")
                    ax.plot(line[0, 0], line[0, 1], f"{colors[i]}x")
                    ax.plot(line[-1, 0], line[-1, 1], f"{colors[i]}o")

        else:
            # Plot refline
            ax.plot(
                self.ref_line[:, 0],
                self.ref_line[:, 1],
                "k-",
                zorder=11,
                label="Lane line",
            )

            # Plot all other
            for line in self.offset_lines.values():
                ax.plot(line[:, 0], line[:, 1], "k-", zorder=10)

    def _format_ax(
        self, ax: plt.Axes, use_lanelet: bool = True, verbose: bool = False
    ) -> None:
        # Formatting for plotting to file

        additional_info = []

        if use_lanelet:
            additional_info.append("lanelet2")
        if verbose:
            additional_info.append("verbose")

        title = f"Road ({', '.join(additional_info)})"

        ax.set(title=title, xlabel="X position in m", ylabel="Y position in m")
        ax.legend()
        # Set axis limits
        margin = 5
        # Find road boundaries
        road_boundary = self.boundary_line
        ax.set_xlim(
            np.min(road_boundary[:, 0]) - margin, np.max(road_boundary[:, 0]) + margin
        )
        ax.set_ylim(
            np.min(road_boundary[:, 1]) - margin, np.max(road_boundary[:, 1]) + margin
        )

        ax.set_aspect("equal")

    def render_curvature(self, plot_dir: Path, plot_name: str = "road") -> None:
        with create_plot_ax(plot_dir, f"{plot_name}_curvature") as ax:
            ax.plot(self.ref_line_curvature, label="Overall curvature")

            ax.set(
                xlabel="s in m",
                ylabel="Curvature in 1/m",
                title="Curvature of the ref_line along s",
            )
            ax.grid()
            ax.legend()

    def render_heading(self, plot_dir: Path, plot_name: str = "road") -> None:
        with create_plot_ax(plot_dir, f"{plot_name}_heading") as ax:
            ax.plot(np.rad2deg(self.ref_line_heading), label="Overall heading")

            ax.set(
                xlabel="s in m",
                ylabel="Heading in deg",
                title="Heading of the ref_line along s",
            )
            ax.grid()
            ax.legend()

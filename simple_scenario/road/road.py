from __future__ import annotations

import lanelet2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy
from lanelet2.core import (
    getId,
    Lanelet,
    LaneletMap,
    LineString3d,
    Point3d,
    AttributeMap,
    BasicPoint2d,
)
from lanelet2.geometry import findNearest
from loguru import logger
from scenariogeneration import xodr
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from .road_segment import RoadSegment
from .straight_segment import StraightSegment
from .clothoid_segment import ClothoidSegment
from .arc_segment import ArcSegment
from .polyline_segment import PolylineSegment
from ..rendering import Renderable, create_plot_ax


class Road(Renderable):
    ALLOWED_SEGMENT_SEQUENCES = (
        (StraightSegment,),
        (StraightSegment, ClothoidSegment, ArcSegment),
        (PolylineSegment,),
    )

    def __init__(
        self,
        n_lanes: int,
        lane_width: float,
        segments: list[RoadSegment],
        speed_limit: float = 120,
        goal_position: float | None = None,
        x0: float = 0,
        y0: float = 0,
    ) -> None:
        """
        n_lanes: Number of lanes in t-direction
        lane_width: Lane width of each of the lanes in m
        segments: List of segments in s-direction
        speed_limit: Speed limit on the road section in km/h
        goal_position: The s-position of the goal region in m
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
            "goal_position": goal_position,
            "x0": x0,
            "y0": y0,
        }

        self._n_lanes = n_lanes
        self._lane_width = lane_width
        self._segments = segments
        self._speed_limit = speed_limit
        self._goal_position = goal_position
        self._x0 = x0
        self._y0 = y0

        self._create_road_from_segments()

        self._overall_ref_line = None
        self._ref_line_length = None
        self._overall_ref_line_curvature = None
        self._overall_ref_line_heading = None
        self._overall_offset_lines = {}
        self._boundary_line = None

        if self._goal_position is None:
            self._goal_position = 0.8 * self.ref_line_length

        self._lanelet_map = self._create_lanelet_map()

    def copy(self) -> Road:
        road_copy = Road(
            self.n_lanes,
            self.lane_width,
            segments=deepcopy(self.segments),
            speed_limit=self.speed_limit,
            goal_position=self.goal_position,
            x0=self.x0,
            y0=self.y0,
        )
        return road_copy

    @property
    def config(self) -> dict:
        return self._config

    @classmethod
    def from_highd_parameters(
        cls,
        lane_markings_str: str,
        road_part: str,
        speed_limit: int,
        goal_position_from_end_of_road: float = -100,
    ) -> Road:
        """
        Create a Road object from the highD map parameters.
        """

        possible_road_parts = ("lower", "upper")

        if road_part not in possible_road_parts:
            raise ValueError

        dx = 5
        x_buffer = 500
        x_min = -x_buffer
        x_max = 450 + x_buffer
        x_values = np.arange(x_min, x_max + dx, dx).astype(float)

        lane_marking_offsets = [-float(elem) for elem in lane_markings_str.split(";")]

        all_linestrings = {}

        for offset in lane_marking_offsets:
            linestring_points_x = x_values.copy().tolist()
            linestring_points_y = [offset] * len(linestring_points_x)

            linestring = [linestring_points_x, linestring_points_y]
            all_linestrings[offset] = linestring

        # Add centerlines
        n_lanes = len(lane_marking_offsets) - 1
        for i_lanelet in range(n_lanes):
            left_line_offset = lane_marking_offsets[i_lanelet]
            right_line_offset = lane_marking_offsets[i_lanelet + 1]
            centerline_offset = round(
                left_line_offset + (right_line_offset - left_line_offset) / 2, 2
            )
            centerline_x = x_values.copy().tolist()
            centerline_y = [centerline_offset] * len(centerline_x)
            linestring = [centerline_x, centerline_y]

            all_linestrings[centerline_offset] = linestring
        all_linestrings = dict(sorted(all_linestrings.items()))

        if road_part == "lower":
            ref_line_y = max(lane_marking_offsets)  # leftmost in driving direction
            ref_line = all_linestrings[ref_line_y]
            offset_lines = {
                abs(round(offset - ref_line_y, 2)): linestring
                for offset, linestring in all_linestrings.items()
                if offset != ref_line_y
            }
            offset_lines = dict(sorted(offset_lines.items()))
            x0 = x_min
            y0 = ref_line_y
        else:
            # Invert all linestrings
            inverted_linestrings = {}
            for offset, linestring in all_linestrings.items():
                xvals = np.flip(linestring[0])
                new_linestring = [xvals.tolist(), linestring[1]]
                inverted_linestrings[offset] = new_linestring

            ref_line_y = min(lane_marking_offsets)  # leftmost in driving direction
            ref_line = inverted_linestrings[ref_line_y]
            offset_lines = {
                abs(round(ref_line_y - offset, 2)): linestring
                for offset, linestring in inverted_linestrings.items()
                if offset != ref_line_y
            }
            offset_lines = dict(sorted(offset_lines.items()))
            x0 = x_max
            y0 = ref_line_y

        polyline_segment = PolylineSegment(ref_line=ref_line, offset_lines=offset_lines)

        goal_position_s = x_max + goal_position_from_end_of_road
        lane_width = None

        road = cls(
            n_lanes=n_lanes,
            lane_width=lane_width,
            segments=[polyline_segment],
            speed_limit=speed_limit,
            goal_position=goal_position_s,
            x0=x0,
            y0=y0,
        )

        return road

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

    @staticmethod
    def linestring2array(linestring: LineString3d) -> np.array:
        return np.array([[pt.x, pt.y] for pt in linestring])

    @staticmethod
    def array2linestring(array: np.array) -> LineString3d:
        return LineString3d(
            getId(), [Point3d(getId(), pt[0], pt[1], 0.0) for pt in array]
        )

    @staticmethod
    def from_frenet_to_cart(
        linestring: LineString3d, s: np.ndarray, t: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        def check_array(array_or_number: np.ndarray | float) -> np.ndarray:
            if isinstance(array_or_number, (float, int)):
                array_or_number = np.array([array_or_number]).astype(float)
            return array_or_number

        s = check_array(s)
        t = check_array(t)

        if s.shape != t.shape:
            raise Exception

        linestring2d = lanelet2.geometry.to2D(linestring)

        x = np.zeros_like(s)
        y = np.zeros_like(t)

        for i in range(s.shape[0]):
            arc = lanelet2.geometry.ArcCoordinates()
            arc.length = s[i]
            arc.distance = t[i]

            cart = lanelet2.geometry.fromArcCoordinates(linestring2d, arc)

            x[i] = cart.x
            y[i] = cart.y

        if x.shape[0] == 1:
            x = x[0]
            y = y[0]

        return x, y

    @staticmethod
    def from_cart_to_frenet(
        linestring: LineString3d, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        def check_array(array_or_number: np.ndarray | float) -> np.ndarray:
            if isinstance(array_or_number, (float, int)):
                array_or_number = np.array([array_or_number]).astype(float)
            return array_or_number

        x = check_array(x)
        y = check_array(y)

        if x.shape != y.shape:
            raise Exception

        linestring2d = lanelet2.geometry.to2D(linestring)

        s = np.zeros_like(x)
        t = np.zeros_like(y)

        for i in range(x.shape[0]):
            arc = lanelet2.geometry.toArcCoordinates(
                linestring2d, BasicPoint2d(x[0], y[0])
            )

            s[i] = arc.length
            t[i] = arc.distance

        if s.shape[0] == 1:
            s = s[0]
            t = t[0]

        return s, t

    def find_lanelet_id_by_position(self, x: float, y: float) -> int:
        candidates = findNearest(self._lanelet_map.laneletLayer, BasicPoint2d(x, y), 1)

        if len(candidates) == 0 or candidates[0][0] != 0:
            return None

        llt_id = candidates[0][1].id

        return llt_id

    @property
    def goal_position(self) -> float:
        return self._goal_position

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

    def create_opendrive_map(self) -> xodr.OpenDrive:
        """
        Create an opendrive map from the road object.
        Use odrviewer.io to visualize it.
        """

        if self._segments == self.ALLOWED_SEGMENT_SEQUENCES[2]:
            raise NotImplementedError

        # Create odr object
        odr = xodr.OpenDrive("my_road")

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

    def _plot_in_ax(  # noqa: PLR0912
        self, ax: plt.Axes, use_lanelet: bool = True, verbose: bool = False
    ) -> None:
        # Start actual plotting
        if use_lanelet:
            plotted_lanelines = []

            for i, lanelet in enumerate(self._lanelet_map.laneletLayer):
                leftbound = self.linestring2array(lanelet.leftBound)
                centerline = self.linestring2array(lanelet.centerline)
                rightbound = self.linestring2array(lanelet.rightBound)

                if verbose:
                    ax.plot(
                        leftbound[:, 0],
                        leftbound[:, 1],
                        "--",
                        label=f"{lanelet.id} (left)",
                    )
                    ax.plot(
                        centerline[:, 0],
                        centerline[:, 1],
                        label=f"{lanelet.id} (center)",
                    )
                    ax.plot(
                        rightbound[:, 0],
                        rightbound[:, 1],
                        label=f"{lanelet.id} (right)",
                    )

                    ax.text(leftbound[-1, 0], leftbound[-1, 1], f"{lanelet.id} (left)")
                    ax.text(
                        centerline[-1, 0], centerline[-1, 1], f"{lanelet.id} (center)"
                    )
                    ax.text(
                        rightbound[-1, 0], rightbound[-1, 1], f"{lanelet.id} (right)"
                    )

                else:
                    left_label = None
                    right_label = None
                    if i == 0:
                        left_label = "Lane line"
                        right_label = None

                    clr = "k"

                    # Find shape (dashed/solid)
                    left_shape = "-"
                    if lanelet.leftBound.attributes["subtype"] == "dashed":
                        left_shape = "--"

                    right_shape = "-"
                    if lanelet.rightBound.attributes["subtype"] == "dashed":
                        right_shape = "--"

                    if lanelet.leftBound not in plotted_lanelines:
                        ax.plot(
                            leftbound[:, 0],
                            leftbound[:, 1],
                            left_shape,
                            color=clr,
                            label=left_label,
                            zorder=10,
                        )
                    if lanelet.rightBound not in plotted_lanelines:
                        ax.plot(
                            rightbound[:, 0],
                            rightbound[:, 1],
                            right_shape,
                            color=clr,
                            label=right_label,
                            zorder=10,
                        )

                    plotted_lanelines.append(lanelet.leftBound)
                    plotted_lanelines.append(lanelet.rightBound)

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

        # Plot goal position
        s_goal = self._goal_position
        all_t_offsets = sorted(self._overall_offset_lines.keys())
        highest_t = all_t_offsets[-1]

        x_low, y_low = self.from_frenet_to_cart(
            self.array2linestring(self._overall_ref_line), s_goal, 0
        )
        x_high, y_high = self.from_frenet_to_cart(
            self.array2linestring(self._overall_ref_line), s_goal, -highest_t
        )

        ax.plot(
            [x_low, x_high],
            [y_low, y_high],
            "g-",
            zorder=10,
            linewidth=2 * mpl.rcParams["lines.linewidth"],
            label="Goal",
        )

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

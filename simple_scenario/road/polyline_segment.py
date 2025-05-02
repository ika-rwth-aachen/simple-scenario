from __future__ import annotations

import numpy as np

from .road_segment import RoadSegment


class PolylineSegment(RoadSegment):
    def __init__(
        self,
        ref_line: list[float, float],
        offset_lines: dict[float, list[float, float]],
    ) -> None:
        """
        ref_line: Polyline defining the ref_line of the road segment
        offset_lines: List of polylines defining the offset lines of the road segment
        """

        self._config = {"ref_line": ref_line, "offset_lines": offset_lines}

        super().__init__()

        self._ref_line = np.array(ref_line).T
        self._offset_lines = {
            offset: np.array(offset_line).T
            for offset, offset_line in offset_lines.items()
        }

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}>"

    @property
    def config(self) -> dict:
        return self._config

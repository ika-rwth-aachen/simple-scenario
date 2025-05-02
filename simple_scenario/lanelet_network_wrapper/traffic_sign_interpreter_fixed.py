"""
This file is a modified version of the following file: https://gitlab.lrz.de/tum-cps/commonroad_io/-/blob/cc4c594c9d5409a5621535e804b58bdacdf533f6/commonroad/scenario/traffic_sign_interpreter.py

The original file (see above) is licensed under the following license:

BSD 3-Clause License

Copyright 2021 Technical University of Munich, Professorship of Cyber-Physical Systems.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from functools import lru_cache
from typing import Union, FrozenSet
from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter


class TrafficSigInterpreterFixed(TrafficSigInterpreter):
    """
    Fix speed limit look up (expects ID instead of traffic sign, but this is only available in the scenario if it is saved once to XML before).
    During creation, the TrafficSign is needed..
    """

    @lru_cache(maxsize=1024)
    def speed_limit(self, lanelet_ids: FrozenSet[int]) -> Union[float, None]:
        """
        Extracts the maximum speed limit of provided lanelets

        :param lanelet_ids: set of lanelets which should be considered
        :returns: speed limit of provided lanelets or None if no speed limit exists

        """
        speed_limits = []
        for lanelet_id in lanelet_ids:
            lanelet = self._lanelet_network.find_lanelet_by_id(lanelet_id)
            for traffic_sign_id in lanelet.traffic_signs:

                # --- THE FIX ---
                traffic_sign_id_ = traffic_sign_id
                if not isinstance(traffic_sign_id, int):
                    # It is a TrafficSign object
                    traffic_sign_id_ = traffic_sign_id.traffic_sign_id
                # --- THE FIX ---

                traffic_sign = self._lanelet_network.find_traffic_sign_by_id(traffic_sign_id_)
                for elem in traffic_sign.traffic_sign_elements:
                    if elem.traffic_sign_element_id == self.traffic_sign_ids.MAX_SPEED:
                        speed_limits.append(float(elem.additional_values[0]))

        if len(speed_limits) == 0:
            speed_limit = None
        else:
            speed_limit = min(speed_limits)
        return speed_limit

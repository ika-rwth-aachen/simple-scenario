from commonroad.scenario.traffic_sign import TrafficSign


class TraffiSignFixed(TrafficSign):
    def __str__(self) -> str:
        """Somehow the normal TrafficLight uses the str() in the xml file for the ref.."""
        return str(self.traffic_sign_id)

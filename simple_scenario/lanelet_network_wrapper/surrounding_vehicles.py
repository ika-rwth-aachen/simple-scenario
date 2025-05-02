from dataclasses import dataclass


@dataclass(frozen=True)
class SurroundingVehicles:
    lead: dict
    left_lead: dict
    left_rear: dict
    right_lead: dict
    right_rear: dict

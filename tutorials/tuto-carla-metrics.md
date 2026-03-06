# CARLA Metrics

In CARLA, various metric criteria can be defined to evaluate the performance of the `ego_vehicle` during scenario execution using the carla-scenario-runner. To set up these metrics, the config file must include a `carla_metrics` section containing a list of all desired testing metrics. 

## Configuration Example

E.g. the config file should be extended as shown below. Note that `criteria_DrivenDistanceTest` requires additional parameters compared th the standard boolean tests.

```json
{
    ...
    "carla_metrics": [
        {
            "name": "criteria_RunningStopTest"
        },
        {
            "name": "criteria_RunningRedLightTest"
        },
        {
            "name": "criteria_CollisionTest"
        },
        {
            "name": "criteria_WrongLaneTest"
        },
        {
            "name": "criteria_OnSidewalkTest"
        },
        {
            "name": "criteria_DrivenDistanceTest",
            "parameterRef": "distance_success",
            "rule": "greaterThan",
            "value": 100.0
        }
    ]
}
```

## Supported Criteria

All critera are CARLA-specific. They can be categorized into standard checks and parametric checks.

### Standard Checks

These criteria act as boolean flags (pass/fail) based on specific infractions.

- **`criteria_RunningStopTest`:** Checks if the `ego_vehicle` runs a stop sign.

- **`criteria_RunningRedLightTest`:** Checks if the `ego_vehicle` runs a red traffic light.

- **`criteria_CollisionTest`:** Monitors for collisions involving the `ego_vehicle`.

- **`criteria_WrongLaneTest`:** Checks if the `ego_vehicle` drives in the wrong lane.

- **`criteria_OnSidewalkTest`:** Checks if the `ego_vehicle` drives on the sidewalk.

### Parametric Checks

These criteria require specific values and rules to evaluate success.

- **`criteria_DrivenDistanceTest`:** Evaluates if the `ego_vehicle` has driven a specified distance. This criterion requires the following additional parameters:

    - `value`: The target distance (e.g. `100.0`)

    - `parameterRef`: The internal metric reference (e.g. `"distance_success"`)

    - `rule`: The logic used for comparison (e.g. `"greaterThan"`)

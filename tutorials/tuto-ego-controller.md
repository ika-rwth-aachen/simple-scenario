# Customize ego controller

To customize the `ego_vehicle` controller, add the parameter `"controller"` to the `ego_configuration` object in the config file.

## Default Behavior

If the `"controller"` parameter is not specified, the system does not set a custom controller.

## Configuration Example

The following snippet demonstrates how to explicitly define the controller within the configuration:

```json
    ...,
    "ego_configuration": {
        "lanelet_id": 1000,
        "target_lanelet_id": 1001,
        "s0": 5,
        "t0": 0,
        "v0": 10,
        "vehicle_type_name": "medium",
        "target_s": 20,
        "controller": "ros_vehicle_control_route_action.py"
    },
    ...
```

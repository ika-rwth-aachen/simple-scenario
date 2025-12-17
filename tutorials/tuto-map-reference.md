# Setting Custom Map References in OpenSCENARIO files

The map reference within an OpenSCENARIO file can be customized using the `reference_to_map` parameter.

- **Default Behavior:** By default, the system sets the reference to the newly exported OpenDRIVE file.
- **Custom References:** The parameter can be used to point to pre-existing maps within simulators, such as CARLA (e.g. `Town10HD`) or relating `.xodr` files.

> [!WARNING]
> If the OpenDRIVE export is suppressed (see [Customize OpenX Export](./tuto-save-openx.md)) but the `reference_to_map` parameter is defined, the system automatically enforce an OpenDRIVE export in the same directory as the OpenSCENARIO file.
> This safeguard ensures the exported OpenSCENARIO files remain functional by guaranteeing a valid map reference exists.

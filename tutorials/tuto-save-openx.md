# Customize OpenX Export

To save a scenario to OpenDRIVE and OpenSCENARIO formats the `save(...)` function with the parameter `mode="openx"` is used.

### 1. Saving to a Single Directory

If `result_dir` is a **`string`**, both the OpenDRIVE and OpenSCENARIO files are saved to the specified directory.

```python
# Example: Save both files to the "output" folder
Scenario.save(mode="openx", result_dir="output/")
```
### 2. Saving Separately (or Skipping)

To handle the files independently, pass a list with two elements to `result_dir`. The order is strictly defined:

- **Index 0:** Path for OpenSCENARIO
- **Index 1:** Path for OpenDRIVE

To suppress the generation of a specific file type set its corresponding list entry to `None`.

```python
# Index 0 = OpenSCENARIO path, Index 1 = OpenDRIVE path
Scenario.save(mode="openx", result_dir=["/scenarios/town.xosc", "/maps/town.xodr"])
```

```python
# Index 0 is set to None, so the OpenSCENARIO file is not generated
Scenario.save(mode="openx", result_dir=[None, "/maps/town.xodr"])
```

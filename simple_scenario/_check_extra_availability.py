# Check extra availability
# commonroad extra
try:
    import commonroad
    import commonroad_dc
    import shapely

    CR_AVAILABLE = True
except ModuleNotFoundError:
    CR_AVAILABLE = False

# lxd extra
try:
    import lxd_io

    LXD_AVAILABLE = True
except ModuleNotFoundError:
    LXD_AVAILABLE = False

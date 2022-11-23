import h5py as h5

with h5.File('/home/jfgf/catkin_ws/src/algalbloom-tracking/output/mission.m5', 'r') as f:
    # lon = f["lon"][()]
    # lat = f["lat"][()]
    # chl = f["chl"][()]
    # time = f["time"][()]
    # traj = f["traj"][()]
    # delta_vals = f["measurement_vals"][()]
    # grad_vals = f["grad_vals"][()]
    # delta_ref = f.attrs["delta_ref"]
    # meas_per = f.attrs["meas_period"]
    # t_idx = f.attrs["t_idx"]
    print(f.attrs.items())
    # print(f.attrs["meas_period"])
    attributes = f.attrs.items()
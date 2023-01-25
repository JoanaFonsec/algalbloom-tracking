from argparse import ArgumentParser
import h5py as h5
import numpy as np
import geopy.distance
import matplotlib.pyplot as plt

import gp4aes.plotter.mission_plotter as plot_mission

# Setup plotting style
plt.style.reload_library()
plt.style.use(['science', 'no-latex'])
plt.rcParams.update({'xtick.labelsize': 20,
                    'ytick.labelsize': 20,
                    'axes.titlesize': 20,
                    'axes.labelsize': 20,
                    'legend.fontsize': 20,
                    'legend.frameon' : True
                    })

# Read runtime arguments
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the HDF5 file containing the processed data.")
    parser.add_argument("--pdf", action='store_true', help="Save plots in pdf format")
    parser.add_argument("--prefix",  type=str, help="Name used as prefix when saving plots.")
    parser.add_argument('-z','--zoom', nargs='+', help='Zoom on a particlar region of the map [x0,y0,width,height]', \
        required=False,type=lambda s: [float(item) for item in s.split(',')])
    parser.add_argument('-t','--time', nargs='+', help='Specify the time range in hours for plotting', \
        required=False,type=lambda s: [float(item) for item in s.split(',')])
    return parser.parse_args()

# Parse in arguments
args = parse_args()

# Read h5 file
with h5.File(args.path, 'r') as f:
    lon = f["lon"][()]
    lat = f["lat"][()]
    chl = f["chl"][()]
    time = f["time"][()]
    position = f["traj"][()]
    measurements = f["measurement_vals"][()]
    gradient = f["grad_vals"][()]
    chl_ref = f.attrs["delta_ref"]
    #time_step = f.attrs["time_step"]
    meas_per = f.attrs["meas_period"]
    t_idx = f.attrs["t_idx"]

# Fixing parameters
meas_per = int(meas_per)

# Set plot format
extension = 'png'
if args.pdf:
    extension = 'pdf'

# Set prefix for plot names
plot_name_prefix = ""
if args.prefix:
    plot_name_prefix = args.prefix + "-"

######################### MODES: Mode 1 is most simulations and Mode 2 is simulations with 100x scale to match experiments
MODE = 1

if MODE == 1:
    # Plot offsets - MODE 1
    l_offset = 2.215 # 9.44 
    lat_offset = 2.185 # 3.24 
elif MODE == 2:
    # Plot offsets - MODE 2
    l_offset = 7.18635 # 7.2788 
    lat_offset = 1.05063 # 1.0505 
################################


# Create zoom 1 and 2 time axis
zoom1_start = 5130
zoom1_end = 8750
zoom2_start = 6310
zoom2_end = 6515

# Update lon and lat
lon = lon +l_offset
lat = lat +lat_offset

# Trim zeros and last entry so that all meas/grads are matched
position = position[~np.all(position == 0, axis=1)]
position_offset = np.concatenate((l_offset*np.ones((position.shape[0], 1)), lat_offset*np.ones((position.shape[0], 1))), axis=1)
position = position + position_offset
time_step = 1

############################################ PRINTS
# Attributes and length os variables
print("delta_ref :", chl_ref)
print("time_step :", time_step)
print("meas_per :", meas_per)
print('len(position) ', len(position[:, 0])-1, ' len(grad) ', len(gradient[:, 0]), ' len(measurements) ', len(measurements))

# Average speed
distances_between_samples = np.array([])
for i in range(0,len(position[:, 0])-2):
    distance = geopy.distance.geodesic((position[i,1],position[i,0]), (position[i+1,1],position[i+1,0])).m
    distances_between_samples = np.append(distances_between_samples,distance)
print("Average speed: {} m/s".format(np.mean(distances_between_samples)))

############################################ PLOTS

# Call plotter class
plotter = plot_mission.Plotter(position, lon, lat, chl[:,:,t_idx], gradient, measurements, chl_ref, meas_per, time_step, zoom1_start, zoom1_end, zoom2_start, zoom2_end)

#a) Mission overview
fig_trajectory = plotter.mission_overview()
fig_trajectory.savefig("plots/{}{}.{}".format(plot_name_prefix, "big_map",extension),bbox_inches='tight')

#c) Gradient zoom1
fig_gradient = plotter.gradient_comparison()
fig_gradient.savefig("plots/{}{}.{}".format(plot_name_prefix, "gradient",extension),bbox_inches='tight')

#b) Zoom1 map with gradient
fig_zoom_gradient = plotter.zoom1()
fig_zoom_gradient.savefig("plots/{}{}.{}".format(plot_name_prefix, "zoom1_map",extension),bbox_inches='tight')

#d) Chl zoom1
fig_chl = plotter.chl_comparison()
fig_chl.savefig("plots/{}{}.{}".format(plot_name_prefix, "measurements",extension),bbox_inches='tight')

#f) Control law zoom2
fig_control = plotter.control_input()
fig_control.savefig("plots/{}{}.{}".format(plot_name_prefix, "control",extension),bbox_inches='tight')

#e) Zoom2 map with control law 
fig_zoom_control = plotter.zoom2()
fig_zoom_control.savefig("plots/{}{}.{}".format(plot_name_prefix, "zoom2_map",extension),bbox_inches='tight')

plt.show()

# Distance to front
# fig_distance = plotter.distance_to_front()
# fig_distance.savefig("plots/{}{}.{}".format(plot_name_prefix, "distance",extension),bbox_inches='tight')
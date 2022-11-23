#!/usr/bin/env python3

# Original author : Alexandre Rocha
# https://github.com/avrocha/front-tracking-algorithm

from calendar import c
import sys
from copy import deepcopy
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from matplotlib import animation

from scipy.interpolate import RegularGridInterpolator
from scipy import spatial

from argparse import ArgumentParser
import geopy.distance


######################### MODES
MODE = 2

if MODE == 1:
    # Plot offsets - MODE 1
    l_offset = 2.215 # 9.44 
    lat_offset = 2.185 # 3.24 
elif MODE == 2:
    # Plot offsets - MODE 2
    l_offset = 7.18635 # 7.2788 
    lat_offset = 1.05063 # 1.0505 
################################

# Setup plotting style

# https://github.com/garrettj403/SciencePlots/issues/15

plt.style.reload_library()
plt.style.use(['science', 'no-latex'])
plt.rcParams.update({'xtick.labelsize': 20,
                    'ytick.labelsize': 20,
                    'axes.titlesize': 20,
                    'axes.labelsize': 20,
                    'legend.fontsize': 20,
                    'legend.frameon' : True
                    })

################################################################################ GET VARIABLES #######################################################

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

# Set plot format
extension = 'png'
if args.pdf:
    extension = 'pdf'

# Set prefix for plot names
plot_name_prefix = ""
if args.prefix:
    plot_name_prefix = args.prefix + "-"

# Save figure function
def save_figure(fig, name):
    fig.savefig("../plots/{}{}.{}".format(plot_name_prefix,name,extension),bbox_inches='tight')

##### Read h5 file
with h5.File(args.path, 'r') as f:
    lon = f["lon"][()]
    lat = f["lat"][()]
    chl = f["chl"][()]
    time = f["time"][()]
    traj = f["traj"][()]
    delta_vals = f["measurement_vals"][()]
    grad_vals = f["grad_vals"][()]
    delta_ref = f.attrs["delta_ref"]
    meas_per = f.attrs["meas_period"]
    t_idx = f.attrs["t_idx"]

    attributes = f.attrs.items()

################################################################################ INIT #######################################################

# Update lon and lat
lon = lon +l_offset
lat = lat +lat_offset

# Trim zeros
traj = traj[~np.all(traj == 0, axis=1)]

#if traj.shape[1] == 3:
#    t_idx = np.argmin(np.abs(traj[n+offset,-1] - time))

# Print attributes
if sys.version_info.major == 2: 
    for att in attributes:
        print("{} : {}".format(att[0],att[1]))

# Get start and end time from arguments
time_step = 1
start_time = 0
end_time = float(time_step*float((len(traj[:, 0])-1)/3600.0))
if args.time:
    start_time = float(args.time[0][0])
    end_time = float(args.time[0][1])

# Assume start and end time are given in hours and time-step is meas_per
idx_start = int(start_time * 3600 / meas_per)
idx_end = int(end_time * 3600 / meas_per)
vector_length = idx_end - idx_start

# Determine index of traj where AUV reaches the front
idx_trig = 0
for i in range(len(delta_vals)):
    if delta_ref - 5e-3 <= delta_vals[i]:
        idx_trig = i
        break

# Create mission time axis
print('Actual start and end for traj: ', start_time, end_time)
print('Index start and end for traj: ', meas_per*idx_start, meas_per*idx_end)
it = np.linspace(start_time, end_time, vector_length)
idx_trig_time = it[idx_trig]

# Print times to check
print('it ', it[0], it[-1], len(it), ' idx_start ' ,idx_start, ' idx_end ', idx_end)
print('len(traj) ', len(traj[:, 0]), ' len(grad) ', len(grad_vals[:, 0]), ' len(delta_vals) ', len(delta_vals))

# Adjust all data
traj = traj[int(meas_per*idx_start):int(meas_per*idx_end),:]
delta_vals = delta_vals[idx_start:idx_end]
grad_vals = grad_vals[idx_start:idx_end,:]

print('len(traj) ', len(traj[:, 0]), ' len(grad) ', len(grad_vals[:, 0]), ' len(delta_vals) ', len(delta_vals))


# Print out distance travelled
distances_between_samples = np.array([])
for i in range(0, int(meas_per*vector_length-1)):

    # Get distance between each point in the trajectory
    distance = geopy.distance.geodesic((traj[i,1],traj[i,0]), (traj[i+1,1],traj[i+1,0])).m
    distances_between_samples = np.append(distances_between_samples,distance)

# Print out average speed, trajectory sampling rate is set at 1 Hz
print("Average speed: {} m/s".format(np.mean(distances_between_samples)))


############################################################################################### PLOT FUNCTIONS ####################################################

def plot_trajectory(axis, show_contour_legend = False):
    """ Plot SAM trajectory """
    xx, yy = np.meshgrid(lon, lat, indexing='ij')
    p = axis.pcolormesh(xx, yy, chl[:,:,t_idx], cmap='viridis', shading='auto', vmin=0, vmax=10)
    cs = axis.contour(xx, yy, chl[:,:,t_idx], levels=[delta_ref])
    axis.plot(traj[:,0], traj[:,1], 'r', linewidth=3) # if Experiment
    #axis.plot(traj[:,0]+l_offset, traj[:,1]+lat_offset, 'r', linewidth=3)

    path = None
    if show_contour_legend:

        # Determine which path is the longest (treat this as the gradient path)
        longest_path = 0
        for i in cs.collections[0].get_paths():
            path_length = i.vertices.shape[0]
            if path_length>longest_path:
                longest_path = path_length
                path = i    

        path = path.vertices

    return p,path

def plot_inset(axis,inset,zoom):
    """Add inset (zoom) to plot
    

    Parameters
    ----------
    inset : list, required
        Position of where to placed 'zoomed' axis. 
        [x0, y0, width, height] where [0,5,0,5,0.3,0.3] represents a region in the top right hand corner
        30% the width and height of the original plot.
    zoom : list, required
        coordinates of the original plot that should be zoomed into
        [x_lim_0,x_lim_1,y_lim_0,y_lim_1]
    """
    
    # Create an inset axis at coordinates [inset]
    axin = axis.inset_axes(inset) 

    # Plot the data on the inset axis\stable\gallery\lines_bars_and_markers\joinstyle.html
    plot_trajectory(axis=axin)

    # Zoom in on the noisy data in the inset axis
    axin.set_xlim(zoom[0], zoom[1])
    axin.set_ylim(zoom[2], zoom[3])

    # Hide inset axis ticks
    axin.set_xticks([])
    axin.set_yticks([])

    # Add the lines to indicate where the inset axis is coming from
    # axis.indicate_inset(axin,edgecolor="black",linestyle="-.")
    axis.indicate_inset_zoom(axin,edgecolor="black",linestyle="-.")    


############################################################################################# PLOT TRAJECTORY ###################################################

##### Plot trajectory
fig, ax = plt.subplots(figsize=(15, 7))
p,ref_path = plot_trajectory(axis=ax,show_contour_legend=True)
ax.set_aspect('equal')
cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
cp = fig.colorbar(p, cax=cax)
cp.set_label("Chl a concentration [mm/mm3]")
ax.set_xlabel("Longitude (degrees E)")
ax.set_ylabel("Latitude (degrees N)")
plt.grid(True)


if args.zoom:

    # Portion of figure to zoom
    a = 0.1
    b = 0.2

    # Get lenght of lat and lon
    lon_length = (lon[-1] - lon[0])
    lat_length = (lat[-1] - lat[0])

    # Get centre coordinates of where to zoom
    x_centre = lon_length*args.zoom[0][0] + lon[0]
    y_centre = lat_length*args.zoom[0][1] + lat[0]

    # Determine x_0 and y_0 of zoomed region
    x0 = x_centre - lon_length*a/2
    y0 = y_centre - lat_length*b/2

    # Determine x_1 and y_1 of zoomed region
    x1 = x_centre + lon_length*a/2
    y1 = y_centre + lat_length*b/2

    # Consider replacing with https://matplotlib.org/stable/gallery/axes_grid1/inset_locator_demo.html

    plot_inset(axis=ax,inset=[0.6, 0.3, a*3, b*3],zoom=[x0,x1,y0,y1])
    # plt.tight_layout()

save_figure(fig, "traj")

################################################################################ PLOT GRADIENT #######################################################
    
# gt => ground truth
gt_grad_vals = np.zeros([grad_vals.shape[0], 2])
dot_prod_cos = np.zeros(grad_vals.shape[0])
grad_vals_cos = np.zeros(grad_vals.shape[0])
gt_grad_vals_cos = np.zeros(grad_vals.shape[0])

if traj.shape[1] == 2:

    # Ground truth gradient
    gt_grad = np.gradient(chl[:,:,t_idx])
    gt_grad_norm = np.sqrt(gt_grad[0]**2 + gt_grad[1]**2)
    gt_gradient = (RegularGridInterpolator((lon, lat), gt_grad[0]/gt_grad_norm),
                RegularGridInterpolator((lon, lat), gt_grad[1]/gt_grad_norm))
    
    # Compute ground truth gradients
    for i in range(len(grad_vals)):
        x = int(meas_per*i)

        gt_grad_vals[i, 0] = gt_gradient[0]((traj[x,0], traj[x,1]))
        gt_grad_vals[i, 1] = gt_gradient[1]((traj[x,0], traj[x,1]))
        dot_prod_cos[i] = np.dot(grad_vals[i], gt_grad_vals[i]) / (np.linalg.norm(grad_vals[i]) * np.linalg.norm(gt_grad_vals[i]))

        grad_vals_cos[i] = grad_vals[i, 0] / np.linalg.norm(grad_vals[i])
        gt_grad_vals_cos[i] = gt_grad_vals[i, 0] / np.linalg.norm(gt_grad_vals[i])

    # Determine gradient angle
    new_it = deepcopy(it)
    print("IT Shape: ", it.shape)
    gt_grad_angles = np.arctan2(gt_grad_vals[:, 1],gt_grad_vals[:, 0])
    print("GT Grad Angles: ", gt_grad_angles.shape)
    pos = np.where(np.abs(np.diff(gt_grad_angles)) >= 4)[0] +1
    gt_grad_angles[pos] = np.nan
    print("GT Grad Angles Pos-insert: ", gt_grad_angles.shape)
    # it_gt_grad = np.insert(new_it, pos, np.nan)
    it_gt_grad = np.linspace(0, len(gt_grad_angles)/meas_per, len(gt_grad_angles))

    new_v2_it = deepcopy(it)
    grad_angles = np.arctan2(grad_vals[:, 1],grad_vals[:, 0])
    print("Grad Angles: ", grad_angles.shape)
    pos = np.where(np.abs(np.diff(grad_angles)) >= 4)[0]+1
    grad_angles[pos] = np.nan
    print("Grad Angles Pos-insert: ", grad_angles.shape)
    # it_grad = np.insert(new_v2_it, pos, np.nan)
    it_grad = np.linspace(0, len(grad_angles)/meas_per, len(grad_angles))
    print(it_grad.shape)
    print(it_gt_grad.shape)
    #exit()

grad_ref = np.ones(dot_prod_cos.shape)
error = np.mean(np.abs(dot_prod_cos[idx_trig:] - grad_ref[idx_trig:])/grad_ref[idx_trig:]) * 100

# Plot gradient angle
fig, ax = plt.subplots(figsize=(15, 7))
plt.plot(it_gt_grad, gt_grad_angles, 'r-', linewidth=1, label='Ground truth')
plt.plot(it_grad, grad_angles, 'k-', linewidth=1, label='Estimated from GP Model')
plt.xlabel('Mission time [h]')
plt.ylabel('Gradient [rad]')
# plt.axis([0, np.max(it), -1.2, 1.2])
plt.legend(loc=4, shadow=True)
plt.grid(True)
plt.axis([it[0], it_grad[-1], -4, 4])
save_figure(fig,"gradient_angle")
plt.show()
#exit()
################################################################################ PLOT CHL #######################################################

print(delta_vals[2],delta_vals[20],delta_vals[200])

if idx_trig > idx_start: 
    error = np.mean(np.abs(delta_vals[idx_trig:] - delta_ref)/delta_ref)*100
else:
    error = np.mean(np.abs(delta_vals - delta_ref)/delta_ref)*100

it_chl = np.linspace(0, len(grad_angles)/meas_per, len(delta_vals))
#print("Reference average relative error = %.4f %%" % (error))
fig, ax = plt.subplots(figsize=(15, 7))
plt.plot(it, np.tile(delta_ref, vector_length), 'r-', label="Chl concentration reference")
plt.plot(it_chl, delta_vals, 'k-', linewidth=1, label="Measured Chl concentration")
if idx_trig > idx_start:
    plt.plot(np.tile(it[idx_trig], 10), np.linspace(np.min(delta_vals), delta_ref*1.4, 10), 'r--')
plt.xlabel('Mission time [h]')
plt.ylabel('Chl a concentration [mm/mm3]')
# plt.axis([0, it[-1], np.min(delta_vals), 0.5+np.max(delta_vals)])
plt.axis([it[0], it[-1], 2, 9]) # 6,9 and 2,9
plt.legend(loc=4, shadow=True)
plt.grid(True)
save_figure(fig,"reference_tracking")
plt.show()

################################################################################ PLOT DISTANCE #######################################################

# Array to store distance
dist = np.zeros(int(meas_per*vector_length))

# Upsample
def upsample_refpath(conxy):
    conxyd = np.empty((0,2))
    discretmax = 0.00005

    for ii in range(conxy.shape[0]-1):
        vertdist = np.sqrt((conxy[ii, 0]-conxy[ii+1, 0])**2 + (conxy[ii, 1]-conxy[ii+1, 1])**2)
        discret = int(vertdist/discretmax)

        if discret < 0.5:
            conxyii = np.empty((0, 2))
        else:
            conxyii = np.hstack((np.linspace(conxy[ii, 0], conxy[ii+1, 0], discret).reshape(-1,1), np.linspace(conxy[ii, 1], conxy[ii+1, 1], discret).reshape(-1,1)))

        conxyd = np.vstack((conxyd, conxyii))
    return conxyd


# Data matrices
# true_path = np.array([traj[:,0]+ l_offset, traj[:,1]+ lat_offset])
true_path = np.array([traj[:,0], traj[:,1]])  # if Experiment
true_path = true_path.transpose()

# Tree struct
upsampled_ref_p = upsample_refpath(ref_path)
tree = spatial.KDTree(upsampled_ref_p)
projection = np.zeros(traj.shape)

for ind, point in enumerate(true_path):
    # Closest point between true path and ref path
    __ ,index = tree.query(point,2)
    v = upsampled_ref_p[index[0]]
    w = upsampled_ref_p[index[1]]

    # Find projection of point p onto line, where t = [(p-v) . (w-v)] / |w-v|^2
    l2 = np.linalg.norm(v - w)**2  # |w-v|^2
    t = max(0, min(1, np.dot(point - v, w - v) / l2))
    projection = v + t * (w - v)

    # Compute distance with point to segment
    distance = geopy.distance.geodesic(point, projection).m
    dist[ind] = distance

# Create special time vector that fits traj
time_traj = np.linspace(start_time, end_time, meas_per*vector_length)

fig, ax = plt.subplots(figsize=(15, 7))
plt.plot(time_traj,dist,'k')
plt.xlabel('Mission time [h]')
plt.ylabel('Distance to front [m]')
if idx_trig_time>idx_start:
    plt.plot(np.tile(idx_trig_time, 10), np.linspace(np.max(dist), 0, 10), 'r--')
plt.grid(True)
plt.axis([it[0], it[-1], 0, 3100]) # 0,370 and 0,3100
save_figure(fig,"distance_error")


################################################################################################# PLOT TRIMMED TRAJECTORY
# Plot and save trajectory for specific time range
lat_start = 21.1
lat_end = 21.17
lon_start = 61.525
lon_end = 61.565

# Plot mesh with improved resolution using interpolation
interpolating_function = RegularGridInterpolator((lon,lat), chl[:,:,t_idx])
fig, ax = plt.subplots(figsize=(15, 7))
xx, yy = np.meshgrid(np.linspace(lon[0], lon[-1], 8*lon.shape[0]), np.linspace(lat[0], lat[-1], 8*lat.shape[0]))
field = interpolating_function((xx, yy))

# Plot contour and trajectory
p = ax.pcolormesh(xx, yy, field, cmap='viridis', shading='auto', vmin=0, vmax=10)
cs = ax.contour(xx, yy, field, levels=[delta_ref])
ax.plot(traj[:,0]+l_offset, traj[:,1]+lat_offset, 'r', linewidth=3)

ax.set_aspect('equal')
ax.set_xlabel("Distance along longitude (km)")
ax.set_ylabel("Distance along latitude (km)")
ax.set_xlim([lat_start, lat_end])
ax.set_ylim([lon_start, lon_end])
plt.grid(True)

# Plot gradient arrows
for index in range(delta_vals.shape[0]):
    if index % 240 == 0 :
        x=traj[int(meas_per*index),0]+l_offset
        y=traj[int(meas_per*index),1]+lat_offset
        dx=grad_vals[index][0] / np.linalg.norm(grad_vals[index])
        dy=grad_vals[index][1] / np.linalg.norm(grad_vals[index])
        dx_gt = gt_grad_vals[index][0] #/ np.linalg.norm(gt_grad_vals[index])
        dy_gt = gt_grad_vals[index][1] #/ np.linalg.norm(gt_grad_vals[index])
        ax.arrow(x, y, 0.003*dx, 0.003*dy, width=.00008)
        ax.arrow(x, y, 0.003*dx_gt, 0.003*dy_gt, width=.00008, ec='red')

# Change distance from degres to meters
ax.set_xticks([21.1, 21.11, 21.12, 21.13, 21.14, 21.15, 21.16, 21.17])
ax.set_xticklabels([0, 0.5, 1.1, 1.6, 2.1, 2.7, 3.2, 3.7]) #([0, 530, 1060 , 1590, 2120, 2650, 3180, 3710])
ax.set_yticks([61.525, 61.53, 61.535, 61.54, 61.545, 61.55, 61.555, 61.56, 61.565])
ax.set_yticklabels([0, 0.5, 1.1, 1.7, 2.2, 2.8, 3.3, 3.9, 4.4]) #([0, 556, 1112, 1668, 2224, 2780, 3336, 3892, 4448])
save_figure(fig,"trajectory_trimmed")

plt.show()
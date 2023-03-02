import numpy as np
import rospy
import os
import time
import h5py as h5
import scipy.io
from scipy.interpolate import RegularGridInterpolator


def save_raw_mission_data(out_path,measurements,grads,delta_ref,traj, alpha_seek):
    """
    Write raw mission data to out_path\raw.m5 when algal bloom tracking node is closed
    """

    with h5.File(out_path+"/raw.h5", 'w') as f:
        f.create_dataset("traj", data=traj)
        f.create_dataset("measurement_vals", data=measurements)
        f.create_dataset("grad_vals", data=grads)
        f.attrs.create("delta_ref", data=delta_ref)
        f.attrs.create("alpha_seek", data=alpha_seek)

def save_mission(out_path,grid,meas_per,sample_params,track_params, t_idx):
    """
    Save mission measurements/traj/etc

    Works by reading the last saved output/raw.m5 file and combining this with the simulated grid of the current mission
    """

    # Check if the file raw.h5 exists - means the controller was closed first
    if not os.path.isfile(out_path+"/raw.h5"):
        rospy.logerr("Controller was not closed! Close controller first.")

        while not os.path.isfile(out_path+"/raw.h5"):
            rospy.loginfo("Waiting for controller to save file...")
            time.sleep(1)

    with h5.File(out_path+"/raw.h5", 'r') as f:
        traj = f["traj"][()]
        measurement_vals = f["measurement_vals"][()]
        grad_vals = f["grad_vals"][()]
        delta_ref = f.attrs["delta_ref"]
        alpha_seek = f.attrs["alpha_seek"]

    with h5.File(out_path+"/mission.m5", 'w') as f:
        f.create_dataset("traj", data=traj)
        f.create_dataset("chl", data=grid.values)
        f.create_dataset("lon", data=grid.grid[0])
        f.create_dataset("lat", data=grid.grid[1])
        if len(grid.grid) == 3:
            f.create_dataset("time", data=grid.grid[2])
        f.create_dataset("measurement_vals", data=measurement_vals)
        f.create_dataset("grad_vals", data=grad_vals)
        f.attrs.create("t_idx", data=t_idx)
        f.attrs.create("delta_ref", data=delta_ref)
        f.attrs.create("meas_period", data=meas_per) 
        f.attrs.create("alpha_seek", data=alpha_seek)

        for key in sample_params:
            f.attrs.create(key, data=sample_params[key]) 

        for key in track_params:
            f.attrs.create(key, data=track_params[key])

    # Delete controller file for it to not be mistaken next time...
    os.remove(out_path+"/raw.h5")

# Read matlab data
def read_mat_data_offset(timestamp,include_time=False,scale_factor=1,lat_start=0,lon_start = 0, base_path=None):

    # Get datapath
    if base_path is None:
        base_path = rospy.get_param('~data_file_base_path')

    # Read mat files
    chl = scipy.io.loadmat(base_path+'/chl.mat')['chl']
    lat = scipy.io.loadmat(base_path+'/lat.mat')['lat']
    lon = scipy.io.loadmat(base_path+'/lon.mat')['lon']
    time = scipy.io.loadmat(base_path+'/time.mat')['time']

    # Reshape
    lat = np.reshape(lat,[-1,])
    lon = np.reshape(lon,[-1,])
    chl = np.swapaxes(chl,0,2)
    time = np.reshape(time,[-1,])    

    # Scale data        
    lat = ((lat - lat[0])*scale_factor)+lat[0]
    lon = ((lon - lon[0])*scale_factor)+lon[0]

    # Shift the data
    lat_offset = lat_start - lat[0]
    lon_offset = lon_start - lon[0]
    lat = lat + lat_offset
    lon = lon + lon_offset

    # Logging
    rospy.loginfo('Scale factor : {}'.format(scale_factor))
    rospy.loginfo("Dimensions of lat {} - {}".format(lat[0],lat[-1]))
    rospy.loginfo("Dimensions of lon {} - {}".format(lon[0],lon[-1]))

    # Print
    if True:
        print('Scale factor : {}'.format(scale_factor))
        print("Dimensions of lat {} - {}".format(lat[0],lat[-1]))
        print("Dimensions of lon {} - {}".format(lon[0],lon[-1]))

    t_idx = np.argmin(np.abs(timestamp - time))

    if include_time is False:
        field = RegularGridInterpolator((lon, lat), chl[:,:,t_idx])
    else:
        field = RegularGridInterpolator((lon, lat, time), chl)

    return field, t_idx
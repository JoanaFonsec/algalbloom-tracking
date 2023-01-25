#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import numpy as np
import signal
import rospy
from std_msgs.msg import Header
from geographic_msgs.msg import GeoPointStamped
from smarc_msgs.msg import ChlorophyllSample
from sensor_msgs.msg import NavSatFix
import matplotlib.pyplot as plt

# From other files
from smarc_algal_bloom_tracking.util import read_mat_data, save_mission
from smarc_algal_bloom_tracking.publishers import publish_offset

fig,ax = plt.subplots()

# Constants
GRADIENT_TOPIC = '/sam/algae_tracking/gradient'
VITUAL_POSITION_TOPIC = '/sam/algae_tracking/vp'
LIVE_WP_BASE_TOPIC = 'sam/smarc_bt/live_wp/'
WAPOINT_TOPIC=LIVE_WP_BASE_TOPIC+'wp'

class substance_sampler_node(object):

    def __init__(self):
        """ Init the sampler"""

        # Parameters
        self.update_period = rospy.get_param('~sampling_time')                  # Update period
        # update_frequency = rospy.get_param('~sampling_frequency', None)    # Update frequency (takes precedence)

        #         # Sampling rate
        # if update_frequency is None:
        #     self.update_period = update_period
        # else:
        #     self.update_period = 1.0/update_frequency

        # Determine if data needs to be scaled
        self.scale_factor =  float(1)/float(rospy.get_param('~data_downs_scale_factor'))
        self.delta_ref = rospy.get_param('~delta_ref')

        # Init values     
        self.init = False

        # Real position
        self.lat = None
        self.lon = None

        self.origin_lat = None
        self.origin_lon = None
        self.data_rotate_angle = rospy.get_param('~data_rotate_angle')

        # WGS84 grid (lookup-table for sampling)
        self.include_time = False
        self.timestamp = 1618610399
        self.grid = read_mat_data(self.timestamp, include_time=self.include_time,scale_factor=self.scale_factor)
        # self.grid = None

        # Check if data should be offset
        self.gps_lat_offset = 0
        self.gps_lon_offset = 0
        self.lat_centre =  0
        self.lon_centre =  0
        self.offset_gps = rospy.get_param('~offset_gps')
        if self.offset_gps:
            self.lat_centre =  rospy.get_param('~starting_lat')
            self.lon_centre =  rospy.get_param('~starting_lon')

        self.gps_topic = rospy.get_param('~gps_topic', '/sam/core/gps')
        
        # Publishers and subscribers
        # self.dr_sub = rospy.Subscriber('/sam/dr/lat_lon', GeoPoint, self.lat_lon__cb, queue_size=2)
        # self.dr_sub = rospy.Subscriber('/sam/dr/lat_lon', GeoPoint, self.lat_lon__cb, queue_size=2)
        self.dr_sub = rospy.Subscriber(self.gps_topic, NavSatFix, self.lat_lon__cb,queue_size=2)
        self.chlorophyll_publisher = rospy.Publisher('/sam/algae_tracking/measurement', ChlorophyllSample, queue_size=1)
        self.lat_lon_offset_publisher = rospy.Publisher('/sam/algae_tracking/lat_lon_offset', GeoPointStamped, queue_size=2)

        # Plotting
        self.grid_plotted = False

    def lat_lon__cb(self,fb):

        # Determine the offset of the GPS
        if not self.init and self.offset_gps:
            self.gps_lat_offset = fb.latitude - self.lat_centre
            self.gps_lon_offset = fb.longitude - self.lon_centre

        # Publish offset (for the purpose of plottting in plot_live_grid)
        publish_offset(lat=self.gps_lat_offset,lon=self.gps_lon_offset,pub=self.lat_lon_offset_publisher)

        # Get position
        if fb.latitude > 1e-6 and fb.longitude > 1e-6:
            self.lat = fb.latitude - self.gps_lat_offset 
            self.lon = fb.longitude - self.gps_lon_offset
        else:
            rospy.logwarn("#PROBLEM# Received Zero GPS coordinates!")


        # Check offset correct set
        # if not self.init:

        #     # Determine offsets
        #     lat_error = (self.lat - self.gps_lat_offset) - self.lat_centre
        #     long_Error = (self.lon - self.gps_lon_offset) - self.lon_centre
        #     rospy.loginfo("Offset error : {}, {}".format(lat_error,long_Error))
        #     rospy.loginfo("Offset lat : {}".format(self.gps_lat_offset))
        #     rospy.loginfo("Offset lon : {}".format(self.gps_lon_offset))

        #     # Offset the data
        #     self.grid = read_mat_data(self.timestamp, include_time=self.include_time,scale_factor=self.scale_factor,lat_shift=self.gps_lat_offset,lon_shift=self.gps_lon_offset)

        #     # Set origin of rotation
        #     self.origin_lat = self.lat
        #     self.origin_lon = self.lon

        # Rotate data       
        # origin = (self.origin_lon,self.origin_lat)
        # point = (self.lon, self.lat)
        # angle = math.radians(self.data_rotate_angle)
        # self.lon, self.lat = rotate(origin, point, -angle)

        self.init = True

    def publish_sample(self):
        """ Publish Chlorophyll Sample"""

        # Do nothing if current lat/long not set
        if None in [self.lat, self.lon, self.gps_lat_offset, self.gps_lon_offset]:
            rospy.logwarn("Cannot take sample, current lat/lon is None : [{},{}]".format(self.lat,self.lon))
            return
        
        # Get current position
        current_position = [self.lon, self.lat]

        # Get sample
        try:
            std = 1e-3 # standard deviation of measurement
            val = self.grid(current_position) + np.random.normal(0, std)
        except Exception as e:
            rospy.logwarn("Caught in Exception!")
            rospy.logwarn(e)
            rospy.logwarn("Unable to attain sample at : {} {}".format(current_position[0], current_position[1]))
            rospy.loginfo("Grid offset {} - {}".format(self.gps_lon_offset, self.gps_lat_offset))
            rospy.loginfo("Current position {} - {}".format(self.lon, self.lat))
            return

        # Publish sample message
        sample = ChlorophyllSample()
        sample.header = Header()
        sample.header.stamp = rospy.Time.now()
        sample.lat = self.lat
        sample.lon = self.lon
        sample.sample = val

        # Publish message
        rospy.loginfo('Publishing sample : {} at {},{}'.format(sample.sample,sample.lat,sample.lon))
        self.chlorophyll_publisher.publish(sample)

    def run_node(self):
        """ Start sampling """

        rate = rospy.Rate(float(1)/self.update_period)
        while not rospy.is_shutdown():
            self.publish_sample()
            rate.sleep()

    def close_node(self,signum, frame):
        """
        Kill node and save data
        """
        
        rospy.logwarn("Closing node")
        out_path = rospy.get_param('~output_data_path')

        # Get all relevant ros params
        self.all_params = rospy.get_param_names()
        self.tracking_params = [a for a in self.all_params if "sam_gp4aes_controller" in a]
        self.sampler_params = [a for a in self.all_params if "substance_sampler" in a]

        # track_params
        track_params= {}
        for key in self.tracking_params:
            track_params[key] = rospy.get_param(key)

        # sample_params
        sample_params = {}
        for key in self.sampler_params:
            sample_params[key] = rospy.get_param(key)
        
        #print('meas_per at simulated_chl_sampler is ', self.update_period)

        try :
            save_mission(out_path=out_path,grid=self.grid,meas_per=self.update_period,sample_params=sample_params,track_params=track_params)
            rospy.logwarn("Data saved!")
        except Exception as e:
            rospy.logwarn(e)
            rospy.logwarn("Failed to save data")

        exit(1)

if __name__ == '__main__':

    rospy.init_node("substance_sampler")
    sampler = substance_sampler_node()

    # Attach exit handler
    signal.signal(signal.SIGINT, sampler.close_node)

    sampler.run_node()
        
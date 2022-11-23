#! /usr/bin/env python
import numpy as np

import rospy

# Smarc imports
from std_msgs.msg import Bool
from geographic_msgs.msg import GeoPoint, GeoPointStamped
from sensor_msgs.msg import NavSatFix
from smarc_msgs.msg import GotoWaypoint, GotoWaypointActionResult, ChlorophyllSample, AlgaeFrontGradient

from controller.positions import RelativePosition
from controller.controller_parameters import ControllerParameters
from controller.controller_state import ControllerState


class ChlorophyllController(object):

    def __init__(self):
        self.args = {}
        self.args['initial_heading'] = rospy.get_param('~initial_heading')                         # initial heading [degrees]
        self.args['delta_ref'] = rospy.get_param('~delta_ref')                                     # target chlorophyll value
        self.args['following_gain'] = rospy.get_param('~following_gain')                           # following gain
        self.args['seeking_gain'] = rospy.get_param('~seeking_gain')                               # seeking gain
        self.args['wp_distance'] = rospy.get_param('~wp_distance')                                 # wp_distance [m]
        self.args['estimation_trigger_val'] = rospy.get_param('~estimation_trigger_val')           # number of samples before estimation
        self.args['speed'] = rospy.get_param('~speed')                                             # waypoint following speed [m/s]
        self.args['travel_rpm'] = rospy.get_param('~travel_rpm')                                   # waypoint target rotation speed
        self.args['waypoint_tolerance'] = rospy.get_param('~waypoint_tolerance')                   # waypoint tolerance [m]
        self.args['range'] = rospy.get_param('~range')                                             # estimation circle radius [m]
        self.args['gradient_decay'] = rospy.get_param('~gradient_decay')
        self.args['n_meas'] = rospy.get_param('~n_meas')

        # Vehicle Controller
        self.ctl_rate = rospy.Rate(50)
        self.controller_state = ControllerState()
        self.controller_params = ControllerParameters()

        self.set_subscribers_publishers()
        self.set_services()

        # Main cycle
        self.run()

    ###############################################
    #           Callbacks Region                  #
    ###############################################
    def lat_lon__cb(self, fb):
        """        
        Latlon topic subscriber callback:
        Update virtual position of the robot using dead reckoning
        """

        # Get position and update it using fb.latitude and fb.longitude
        # Save the variables you need for the position with those.          

    def waypoint_reached__cb(self, fb):
        """
        Waypoint reached
        Logic checking for proximity threshold is handled by the line following action
        """

        # Determine if waypoint has been reached
        if fb.status.text == "WP Reached":
            # Update an internal variable that says that you achieved the current waypoint?
            pass
        pass

    def chlorophyl__cb(self, fb):
        """
        Callback when a sensor reading is received

        The sensor reading should be appended to the list of sensor readings, along with the associated
        lat lon position where the reading was taken.
        """

        # TODO : Make the measurement a service so that controller can set measurement period

        # Increment sample index

        # read values (the sensor is responsible for providing the Geo stamp i.e. lat lon co-ordinates)
        position = np.array([[fb.lon, fb.lat]])
        sample = fb.sample

        # Save the chlorophyll value to an internal variableif np.isnan(self.measurement):
        if np.isnan(sample):
            print("Warning: NaN value measured.")
            self.measurement = self.measurements[-1]  # Avoid plots problems
        else:
            self.measurement = sample
        pass

    ###############################################
    #           End Callbacks Region              #
    ###############################################

    def set_subscribers_publishers(self):
        """
        Helper function to create all publishers and subscribers.
        """
        # Subscribers
        self.state_sub = rospy.Subscriber('~measurements', ChlorophyllSample, self.measurement__cb)
        self.gps_position_sub = rospy.Subscriber('~gps', NavSatFix, self.lat_lon__cb)
        self.goal_reached_sub = rospy.Subscriber('~go_to_waypoint_result', GotoWaypointActionResult,
                                                 self.waypoint_reached__cb, queue_size=2)

        # Publishers
        self.enable_waypoint_pub = rospy.Publisher("~enable_live_waypoint", Bool, queue_size=1)
        self.waypoint_pub = rospy.Publisher("~live_waypoint", GotoWaypoint, queue_size=5)
        self.vp_pub = rospy.Publisher("~virtual_position", GeoPointStamped, queue_size=1)
        self.gradient_pub = rospy.Publisher("~gradient", AlgaeFrontGradient, queue_size=1)

    def set_services(self):
        """
        Helper function to create all services.
        """
        rospy.wait_for_service("~lat_lon_utm_srv", timeout=1)

    def run(self):

        while not rospy.is_shutdown():
            ##### Take measurement
            # self.measurement  exists! 

            ##### Init state - From beginning until 5% tolerance from front
            if (i < n_meas or measurements[-1] < 0.95*dynamics.delta_ref) and init_flag is True:
                gradient = np.append(gradient, init_heading[[0], :2] / np.linalg.norm(init_heading[0, :2]), axis=0)
                filtered_gradient = np.append(filtered_gradient, gradient[[-1],:], axis=0)
                filtered_measurements = np.append(filtered_measurements,measurements[-1])

            ##### Estimation state - From reaching the front till the end of the mission
            else:
                if init_flag is True:
                    print("Following the front...")
                    init_flag = False

                filtered_measurements = np.append(filtered_measurements, 
                                            np.average(measurements[- meas_filter_len:], weights=weights_meas))

                # Estimate and filter gradient
                gradient_calculation = np.array(est.est_grad(position[-n_meas:],filtered_measurements[-n_meas:])).squeeze().reshape(-1, 2)
                gradient = np.append(gradient, gradient_calculation / np.linalg.norm(gradient_calculation), axis=0)
                filtered_gradient = np.append(filtered_gradient, filtered_gradient[[-2], :]*alpha + gradient[[-1], :]*(1-alpha), axis=0)

            ##### Calculate next position
            control = dynamics(filtered_measurements[-1], filtered_gradient[-1,:], include_time=False)
            next_position = controller.next_position(position[-1, :],control)
            position = np.append(position, next_position, axis=0)

            if (lon[0] <= position[-1, 0] <= lon[-1]) and (lat[0] <= position[-1, 1] <= lat[-1]):
                print("Warning: trajectory got out of boundary limits.")
                break
            if next_position[0, 1] > 61.64:
                break

            self.ctl_rate.sleep()


if __name__ == '__main__':
    rospy.init_node('chlorophyll_controller')
    try:
        controller = ChlorophyllController()
    except rospy.ROSInterruptException:
        pass

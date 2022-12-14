#! /usr/bin/env python3
import numpy as np
import rospy

# Smarc imports
from std_msgs.msg import Bool
from geographic_msgs.msg import GeoPointStamped
from sensor_msgs.msg import NavSatFix
from smarc_msgs.msg import GotoWaypoint, GotoWaypointActionResult, ChlorophyllSample, AlgaeFrontGradient
from smarc_msgs.srv import LatLonToUTM

# GP4AES imports
import gp4aes.estimator.GPR as gpr
import gp4aes.controller.front_tracking as controller

# Publishers
from utils.publishers import publish_gradient, publish_waypoint, publish_vp

class FrontTracking(object):

    def __init__(self):
        self.initial_heading = rospy.get_param('~initial_heading')                         # initial heading [degrees]
        self.delta_ref = rospy.get_param('~delta_ref')                                     # target chlorophyll value
        self.wp_distance = rospy.get_param('~wp_distance')                                 # wp_distance [m]
        self.estimation_trigger_val = rospy.get_param('~estimation_trigger_val')           # number of samples before estimation
        self.speed = rospy.get_param('~speed')                                             # waypoint following speed [m/s]
        self.travel_rpm = rospy.get_param('~travel_rpm')                                   # waypoint target rotation speed
        self.waypoint_tolerance = rospy.get_param('~waypoint_tolerance')                   # waypoint tolerance [m]
        self.range = rospy.get_param('~range')                                             # estimation circle radius [m]
        self.kernel_params = rospy.get_param('~kernel_params')                             # kernel parameters obtained through training
        self.kernel = rospy.get_param('~kernel')                                           # name of the kernel to use: RQ or MAT
        self.std = rospy.get_param('~std')                                                 # measurement noise

        # Vehicle Controller
        self.ctl_rate = rospy.Rate(50)

        self.set_subscribers_publishers()
        self.set_services()

        # Main cycle
        self.run()

    ###############################################
    #           Callbacks Region                  #
    ###############################################
    def position__cb(self, fb):  
        """        
        Latlon topic subscriber callback:
        Update virtual position of the robot using dead reckoning
        """
        self.position = np.array([[fb.longitude, fb.latitude]])

    def waypoint_reached__cb(self, fb): 
        """
        Waypoint reached
        Logic checking for proximity threshold is handled by the line following action
        """
        if fb.status.text == "WP Reached":
            self.waypoint_reached = True
            pass
        pass

    def measurement__cb(self, fb): 
        """
        Callback when a sensor reading is received

        The sensor reading should be appended to the list of sensor readings, along with the associated
        lat lon position where the reading was taken.
        """

        # TODO : Make the measurement a service so that controller can set measurement period

        # read values (the sensor is responsible for providing the Geo stamp i.e. lat lon co-ordinates)
        self.position_measurement = np.array([[fb.lon, fb.lat]])
        sample = fb.sample

        # Save the measurement if not Nan
        if np.isnan(sample):
            print("Warning: NaN value measured.")
            self.measurement = self.measurement[-1]  # Avoid plots problems
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
        rospy.Subscriber('~measurement', ChlorophyllSample, self.measurement__cb)
        rospy.Subscriber('~gps', NavSatFix, self.position__cb)
        rospy.Subscriber('~go_to_waypoint_result', GotoWaypointActionResult, self.waypoint_reached__cb, queue_size=2)

        # Publishers
        self.enable_waypoint_pub = rospy.Publisher("~enable_live_waypoint", Bool, queue_size=1)
        self.waypoint_pub = rospy.Publisher("~live_waypoint", GotoWaypoint, queue_size=5)
        self.vp_pub = rospy.Publisher("~virtual_position", GeoPointStamped, queue_size=1)
        self.gradient_pub = rospy.Publisher("~gradient", AlgaeFrontGradient, queue_size=1)

    def set_services(self):
        """
        Helper function to create all services.
        """

        service = '/sam/dr/lat_lon_to_utm'
        try:
            rospy.wait_for_service(service, timeout=1)
        except:
            rospy.logwarn(str(service)+" service not found!")

        self.latlontoutm_service = rospy.ServiceProxy(service,LatLonToUTM)

    ############################################################################################################
    def run(self):

        ############ Tunable parameters
        # Dynamics
        alpha_seek = 50
        alpha_follow = 1
        dynamics = controller.Dynamics(alpha_seek, alpha_follow, self.delta_ref, self.wp_distance)

        # Trajectory parameters
        init_towards = np.array([[21, 61.492]])
        init_coords = np.array([[20.925, 61.492]])

        # Algorithm settings (commented values are of trajectory for IROS paper)
        n_meas = 200 # 125
        meas_filter_len = 3 # 3
        alpha = 0.97 # Gradient update factor, 0.95
        weights_meas = [0.2, 0.3, 0.5]
        init_flag = True

        ############ Initialize functions
        # Gaussian Process Regression
        est = gpr.GPEstimator(self.kernel, self.std, self.range, self.kernel_params)
        init_heading = np.array([[init_towards[0, 0] - init_coords[0, 0], init_towards[0, 1] - init_coords[0, 1]]])

        # Main variables: position, measurements, gradient, filtered_measurements, filtered_gradient
        gradient = np.empty((0,2))
        filtered_measurements = np.empty((0, 2))
        filtered_gradient = np.empty((0, 2))


        while not rospy.is_shutdown():

            ##### Init state - From beginning until 5% tolerance from front
            if (len(filtered_measurements) < n_meas or self.measurement[-1] < 0.95*dynamics.delta_ref) and init_flag is True:
                gradient = np.append(gradient, init_heading[[0], :2] / np.linalg.norm(init_heading[0, :2]), axis=0)
                filtered_gradient = np.append(filtered_gradient, gradient[[-1],:], axis=0)
                filtered_measurements = np.append(filtered_measurements,self.measurement[-1])

            ##### Estimation state - From reaching the front till the end of the mission
            else:
                if init_flag is True:
                    print("Following the front...")
                    init_flag = False

                filtered_measurements = np.append(filtered_measurements, np.average(self.measurement[- meas_filter_len:], weights=weights_meas))

                # Estimate and filter gradient
                gradient_calculation = np.array(est.est_grad(self.position_measurement[-n_meas:],filtered_measurements[-n_meas:])).squeeze().reshape(-1, 2)
                gradient = np.append(gradient, gradient_calculation / np.linalg.norm(gradient_calculation), axis=0)
                filtered_gradient = np.append(filtered_gradient, filtered_gradient[[-2], :]*alpha + gradient[[-1], :]*(1-alpha), axis=0)

            ##### Calculate next waypoint
            rospy.loginfo("Determining new waypoint")
            if self.waypoint_reached:
                control = dynamics(filtered_measurements[-1], filtered_gradient[-1,:], include_time=False)
                next_waypoint = controller.next_position(self.position[-1, :],control)
                self.waypoint_reached = False
                publish_waypoint(self.latlontoutm_service,next_waypoint,self.waypoint_pub,self.enable_waypoint_pub,self.travel_rpm,self.speed,self.waypoint_tolerance)


            # if (self.lon[0] <= self.position[-1, 0] <= self.lon[-1]) and (self.lat[0] <= self.position[-1, 1] <= self.lat[-1]):
            #     print("Warning: trajectory got out of boundary limits.")
            #     break
            if next_waypoint[0, 1] > 61.64:
                break

            self.ctl_rate.sleep()


if __name__ == '__main__':
    rospy.init_node('Front_Tracking')
    try:
        tracker = FrontTracking()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

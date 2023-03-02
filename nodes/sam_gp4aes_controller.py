#! /usr/bin/env python3
import numpy as np
import rospy
import signal
import math

# Smarc imports
from std_msgs.msg import Bool
from geographic_msgs.msg import GeoPointStamped
from sensor_msgs.msg import NavSatFix
from smarc_msgs.msg import ChlorophyllSample, AlgaeFrontGradient, ThrusterRPM, GotoWaypoint
from smarc_msgs.srv import LatLonToUTM
from std_msgs.msg import Float64

# GP4AES imports
import gp4aes.estimator.GPR as gpr
from gp4aes.estimator.LSQ import LSQ_estimation
import gp4aes.controller.front_tracking as controller

# Publishers & utils
from smarc_algal_bloom_tracking.publishers import publish_waypoint
from smarc_algal_bloom_tracking.util import save_raw_mission_data

class FrontTracking(object):

    def __init__(self):
        self.initial_heading = rospy.get_param('~initial_heading')                         # initial heading [degrees]
        self.delta_ref = rospy.get_param('~delta_ref')                                     # target chlorophyll value
        self.wp_distance = 50                                 # wp_distance [m]
        self.n_meas = rospy.get_param('~n_measurements')                                   # number of samples before estimation
        self.speed = rospy.get_param('~speed')                                             # waypoint following speed [m/s]
        self.travel_rpm_thruster_1 = rospy.get_param('~travel_rpm_thruster_1')
        self.travel_rpm_thruster_2 = rospy.get_param('~travel_rpm_thruster_2')
        self.waypoint_tolerance = 1                   # waypoint tolerance [m]
        self.range = rospy.get_param('~range')                                             # estimation circle radius [m]
        self.kernel_params = rospy.get_param('~kernel_params')                             # kernel parameters obtained through training
        self.kernel = rospy.get_param('~kernel')                                           # name of the kernel to use: RQ or MAT
        self.std = rospy.get_param('~std')                                                 # measurement noise
        self.alpha_seek = rospy.get_param('~alpha_seek')                                   # Controller param to seek front
        self.alpha_follow = rospy.get_param('~alpha_follow')                               # Controller param to follow front
        self.gps_topic = rospy.get_param('~gps_topic', '/sam/core/gps')
        
        # Vehicle Controller
        self.ctl_rate = rospy.Rate(50)
        self.control = np.array([100, 1])

        # Init variables
        self.measurement = None
        self.position_measurement = None
        self.position = None
        self.waypoint_reached = True

        self.gps_lat_offset = None
        self.gps_lon_offset = None

        # Flag to check if there is a new measurement
        self.received_new_measurement = False

        # Call subscribers, publishers, services
        self.set_subscribers_publishers()
        self.set_services()

    ###############################################
    #           Callbacks Region                  #
    ###############################################
    def position__cb(self, fb):  
        """        
        Latlon topic subscriber callback:
        Update virtual position of the robot using dead reckoning
        """
        if self.gps_lat_offset is None or self.gps_lon_offset is None:
            return

        if fb.latitude > 1e-6 and fb.longitude > 1e-6:
            if self.position is None:
                self.position = np.array([[fb.longitude - self.gps_lon_offset, fb.latitude - self.gps_lat_offset]])
            else:
                self.position = np.append(self.position, np.array([[fb.longitude - self.gps_lon_offset, fb.latitude - self.gps_lat_offset]]), axis=0)
        else:
            if self.position is not None:
                self.position = np.append(self.position, self.position[[-1],:].reshape((1, 2)), axis=0)
            rospy.logwarn("#PROBLEM# Received Zero GPS coordinates in Tracker!")


    def measurement__cb(self, fb): 
        """
        Callback when a sensor reading is received

        The sensor reading should be appended to the list of sensor readings, along with the associated
        lat lon position where the reading was taken.
        """

        # TODO : Make the measurement a service so that controller can set measurement period

        # read values (the sensor is responsible for providing the Geo stamp i.e. lat lon co-ordinates)
        position_measurement = np.array([[fb.lon, fb.lat]])
        sample = fb.sample

        # Save the measurement if not Nan
        if np.isnan(sample):
            print("Warning: NaN value measured.")
            if self.measurement is not None:
                self.measurement = np.append(self.measurement,self.measurement[-1])  # Avoid plots problems
                self.position_measurement = np.append(self.position_measurement,position_measurement, axis=0)
        else:
            if self.measurement is not None:
                self.measurement = np.append(self.measurement,sample)
                self.position_measurement = np.append(self.position_measurement,position_measurement, axis=0)
            else:
                self.measurement = np.array([sample])
                self.position_measurement = position_measurement

            # Indicate that we recived a new measurement
            self.received_new_measurement = True

        pass

    def gps_offset__cb(self, msg):
        """
        Call for GPS offset, given by substance sampler
        """
        self.gps_lat_offset = msg.position.latitude
        self.gps_lon_offset = msg.position.longitude

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
        rospy.Subscriber('~gps_offset', GeoPointStamped, self.gps_offset__cb, queue_size=2)

        # Publishers
        self.thruster_1_pub = rospy.Publisher("~thruster_1", ThrusterRPM, queue_size=1)
        self.thruster_2_pub = rospy.Publisher("~thruster_2", ThrusterRPM, queue_size=1)
        self.yaw_setpoint_pub = rospy.Publisher("~yaw_setpoint", Float64, queue_size=1)
        self.gradient_pub = rospy.Publisher("~gradient", AlgaeFrontGradient, queue_size=1)

        # Publisher for plotter
        self.waypoint_pub = rospy.Publisher("~live_waypoint", GotoWaypoint, queue_size=1)

    def set_services(self):
        """
        Helper function to create all services.
        """

        try:
            rospy.wait_for_service('~lat_lon_to_utm', timeout=1)
        except:
            rospy.logwarn(" service not found!")

        self.latlontoutm_service = rospy.ServiceProxy('~lat_lon_to_utm', LatLonToUTM)

    def publish_yaw_setpoint(self, control_vector):
        
        yaw_setpoint = math.atan2(control_vector[1], control_vector[0])

        thruster_cmd = ThrusterRPM()
        thruster_cmd.rpm = self.travel_rpm_thruster_1

        self.thruster_1_pub.publish(thruster_cmd)

        thruster_cmd.rpm = self.travel_rpm_thruster_2
        self.thruster_2_pub.publish(thruster_cmd)

        yaw_sp_cmd = Float64()
        yaw_sp_cmd.data = yaw_setpoint
        self.yaw_setpoint_pub.publish(yaw_sp_cmd)


    ############################################################################################################
    def run(self):

        ############ INIT functions
        # Dynamics
        dynamics = controller.Dynamics(self.alpha_seek, self.alpha_follow, self.delta_ref, self.wp_distance)
        # Gaussian Process
        est = gpr.GPEstimator(self.kernel, self.std, self.kernel_params)

        ############ Tunable parameters
        meas_filter_len = 3 
        alpha = 0.98
        weights_meas = [0.2, 0.3, 0.5]
        init_flag = True
        init_gradient = np.array([[1, 0]])

        ############ INIT vectors 
        gradient = np.empty((0,2))
        self.filtered_measurements = np.empty((0, 1))
        self.filtered_gradient = np.empty((0, 2))
        self.next_waypoint = np.array([[0, 0]])

        ############ Get the first measurement
        while self.measurement is None or self.position is None or self.gps_lat_offset is None or self.gps_lon_offset is None:
            rospy.logwarn("Waiting for valida data: ")
            rospy.logwarn("Measurement: {} - Position: {} - GPS Offset: {},{}".format(self.measurement, self.position, self.gps_lon_offset, self.gps_lat_offset))
            rospy.sleep(1)


        ########### MISSION CYCLE
        while not rospy.is_shutdown():

            if self.received_new_measurement:
                ##### Init state - From beginning until at front
                if (len(self.filtered_measurements) < self.n_meas or self.measurement[-1] < 0.99*self.delta_ref) and init_flag is True:
                    gradient = np.append(gradient, init_gradient[[0], :2] / np.linalg.norm(init_gradient[0, :2]), axis=0)
                    self.filtered_gradient = np.append(self.filtered_gradient, gradient[[-1],:], axis=0)
                    self.filtered_measurements = np.append(self.filtered_measurements,self.measurement[-1])
                    # Calculate next waypoint
                    self.control = np.array([100, 1])
                    self.next_waypoint = controller.next_position(self.position[-1, :],self.control)
                    
                ##### Main state - From reaching the front till the end 
                else:
                    if init_flag is True:
                        print("Following the front...")
                        init_flag = False

                    self.filtered_measurements = np.append(self.filtered_measurements, np.average(self.measurement[- meas_filter_len:], weights=weights_meas))

                    # Estimate and filter gradient
                    gradient_calculation = np.array(est.est_grad(self.position_measurement[-self.n_meas:, :].reshape((self.n_meas, 2)), self.filtered_measurements[-self.n_meas:].reshape((-1, 1)))).squeeze().reshape(-1, 2)
                    # gradient_calculation = np.array(LSQ_estimation(self.position[-self.n_meas:],self.filtered_measurements[-self.n_meas:])).squeeze().reshape(-1, 2)
                    gradient = np.append(gradient, gradient_calculation / np.linalg.norm(gradient_calculation), axis=0)
                    self.filtered_gradient = np.append(self.filtered_gradient, self.filtered_gradient[[-1], :]*alpha + gradient[[-1], :]*(1-alpha), axis=0)
                      
                    # Calculate next waypoint
                    self.control = dynamics(self.filtered_measurements[-1], self.filtered_gradient[-1,:], include_time=False)

                    self.next_waypoint = controller.next_position(self.position[-1, :],self.control)
                    
                print("measurement", self.filtered_measurements[-1], ", gradient", self.filtered_gradient[-1])
                
                # Convert back waypoints to simulator coordinates
                self.next_waypoint[0, 0] = self.next_waypoint[0, 0] + self.gps_lon_offset
                self.next_waypoint[0, 1] = self.next_waypoint[0, 1] + self.gps_lat_offset

                # Clear flag for next run
                self.received_new_measurement = False

            if self.next_waypoint[0, 1] > 61.64:
                break
            
            # Publish waypoint and control command
            publish_waypoint(self.latlontoutm_service,self.next_waypoint,self.waypoint_pub,self.travel_rpm_thruster_1,self.speed,self.waypoint_tolerance)
            self.publish_yaw_setpoint(self.control)
            self.ctl_rate.sleep()

    def close_node(self,signum, frame):
        """
        Stop following behaviour, save data output and close the node
        """
        rospy.logwarn("Closing node")
        rospy.logwarn("Attempting to end waypoint following")

        try :
            enable_waypoint_following = Bool()
            enable_waypoint_following.data = False
            self.enable_waypoint_pub.publish(enable_waypoint_following)
            rospy.logwarn("Waypoint following successfully disabled")
        except Exception as e:
            rospy.logwarn("Failed to disabled Waypoint following")

        out_path = rospy.get_param('~output_data_path')
        
        try :
            save_raw_mission_data(out_path=out_path, traj=self.position, measurements=self.filtered_measurements, grads=self.filtered_gradient, delta_ref=self.delta_ref, alpha_seek = self.alpha_seek)
            rospy.logwarn("Data saved!")

        except Exception as e:
            rospy.logwarn(e)
            rospy.logwarn("Failed to save data")

        exit(1)


if __name__ == '__main__':
    rospy.init_node('Front_Tracking')

    tracker = FrontTracking()

    signal.signal(signal.SIGINT, tracker.close_node)

    tracker.run()
    rospy.signal_shutdown()

import rospy

# SMaRC imports
from smarc_msgs.msg import AlgaeFrontGradient, GotoWaypoint
from smarc_msgs.srv import LatLonToUTM
from geographic_msgs.msg import GeoPoint, GeoPointStamped
from std_msgs.msg import Bool

"""
PUBLISHERS
"""

def publish_gradient(lat,lon,x,y,gradient_pub):
    """ Publish gradient information """

    msg = AlgaeFrontGradient()
    msg.header.stamp = rospy.Time.now()
    msg.lat = lat
    msg.lon = lon
    msg.x = x
    msg.y = y

    gradient_pub.publish(msg)  


def publish_offset(lat,lon,pub):
    """ Publish lat_lon offset"""

    msg = GeoPointStamped()
    msg.header.stamp = rospy.Time.now()
    msg.position.latitude = lat
    msg.position.longitude = lon
    msg.position.altitude = -1

    pub.publish(msg)   


def publish_vp(lat,lon,vp_pub):
    """ Publish current vp """

    msg = GeoPointStamped()
    msg.header.stamp = rospy.Time.now()
    msg.position.latitude = lat
    msg.position.longitude = lon
    msg.position.altitude = -1

    vp_pub.publish(msg)   


def publish_waypoint(latlontoutm_service,controller_params,waypoint_pub,enable_waypoint_pub,lat,lon,depth,travel_rpm):    
    """ Publish waypoint to SAM"""

    # Convert lat,lon to UTM
    x, y = latlon_to_utm(service=latlontoutm_service, lat=lat,lon=lon,z=depth)

    # Speed and z controls
    z_control_modes = [GotoWaypoint.Z_CONTROL_DEPTH]
    speed_control_mode = [GotoWaypoint.SPEED_CONTROL_RPM,GotoWaypoint.SPEED_CONTROL_SPEED]

    # Waypoint message
    msg = GotoWaypoint()
    msg.travel_depth = -1
    msg.goal_tolerance = controller_params.waypoint_tolerance
    msg.lat = lat
    msg.lon = lon
    msg.z_control_mode = z_control_modes[0]
    msg.travel_rpm = travel_rpm
    msg.speed_control_mode = speed_control_mode[0]
    msg.travel_speed = controller_params.speed
    msg.pose.header.frame_id = 'utm'
    msg.pose.header.stamp = rospy.Time(0)
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y

    # Enable waypoint following
    enable_waypoint_following = Bool()
    enable_waypoint_following.data = True
    enable_waypoint_pub.publish(enable_waypoint_following)

    # Publish waypoint
    waypoint_pub.publish(msg)
    rospy.loginfo('Published waypoint : {},{}'.format(lat,lon))



"""
SERVICES
"""

def latlon_to_utm(service,lat,lon,z,in_degrees=False):

    """
    Use proxy service to convert lat/lon to utmZ
    """

    try:
        rospy.wait_for_service(service, timeout=1)
    except:
        rospy.logwarn(str(service)+" service not found!")
        return (None, None)
    try:
        latlontoutm_service = rospy.ServiceProxy(service,
                                                    LatLonToUTM)
        gp = GeoPoint()
        if in_degrees:
            gp.latitude = lat
            gp.longitude = lon
        else:
            gp.latitude = lat
            gp.longitude = lon
        gp.altitude = z
        utm_res = latlontoutm_service(gp)

        return (utm_res.utm_point.x, utm_res.utm_point.y)
    except rospy.service.ServiceException:
        rospy.logerr_throttle_identical(5, "LatLon to UTM service failed! namespace:{}".format(service))
        return (None, None)
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import sys
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import threading
import time
import math
bridge = CvBridge()

picture = np.zeros((240, 320, 3), dtype=np.uint8)
vx = 0
az = 0
damage = 0


class Client():
    def __init__(self):
        ic = Sample_info()
        rospy.init_node('image_listener')
        rospy.Rate(10)
        self.maxSteps = 100000
        self.S = ServerState()
        self.R = DriverAction()
        control = threading.Thread(target=self.R.apply_action)
        control.setDaemon(True)
        self.spin = threading.Thread(target=rospy.spin)
        self.spin.setDaemon(True)
        self.spin.start()
        control.start()
        time.sleep(0.5)

    def respond_to_server(self):  # control information transmit
        try:
            if self.R.d['meta']:
                sys.exit(-1)
            # self.R.apply_action()
        except:
            print "Error in Respond_to_server "
            sys.exit(-1)

    def get_servers_input(self):
        try:
            # Receive robot data
            self.S.get_state()
        except:
            print "Error in get_servers_input"


def clip(v, lo, hi):
    if v < lo:
        return lo
    elif v > hi:
        return hi
    elif math.isnan(v):
        return 0.0
    else:
        return v


class DriverAction():  # R
    def __init__(self):
        self.actionstr = str()
        # "d" is for data dictionary.
        self.d = {'linearX': 0.05,
                  'angularZ': 0.0,
                  'meta': False}

    def apply_action(self):
        pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        twist = Twist()
        while not rospy.is_shutdown():
            try:
                twist.linear.x = clip(self.d['linearX'], -0.26, 0.26)
                twist.angular.z = clip(self.d['angularZ'], -1.82, 1.82)
                # print twist.linear.x
                pub.publish(twist)
            except:
                exit(-1)


class ServerState(): # S
    'What the server is reporting right now.'
    def __init__(self):
        self.servstr = str()
        self.d = dict()  # .S.d

    def get_state(self):
        global picture, vx, az, damage
        self.d['img'] = picture
        self.d['linearX'] = vx
        self.d['angularZ'] = az
        self.d['damage'] = damage


class Sample_info:
    def __init__(self):
        self.num = 1
        self.temp = 1
        image_topic = "/camera/depth/image_raw"
        # self.image_pub = rospy.Publisher(image_topic, Image, queue_size=1)
        rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1)

    def image_callback(self, data):
        # print("Received an image!")
        try:
            cv2_img = bridge.imgmsg_to_cv2(data, "32FC1")
            cv2_a = cv2_img * 256 / 7
            cv2_a = cv2_a[0:1080, 240:1680]  # [y0:y1, x0:x1]
            global picture, damage
            cv2_b = cv2.resize(cv2_a, (320, 240))
            damage = self.collosion(cv2_b)
            picture = cv2.cvtColor(cv2_b, cv2.COLOR_GRAY2BGR).astype(np.uint8)
        except CvBridgeError, e:
            print(e)
        rospy.Subscriber("/cmd_vel", Twist, self.callback2, queue_size=1)

    def callback2(self, msg):
        global vx, az
        vx = msg.linear.x
        az = msg.angular.z
        # print 'vx, az', vx, az
        # rospy.signal_shutdown('hhh')

    def collosion(self, img):
        num = 0
        i = 80
        j = 50
        while i <= 199:
            while j <= 149:
                if img[j][i] > 10:
                    num = num + 1
                j = j + 1
            j = 50
            i = i + 1
        r = (12000 - num) / 12000.0
        return r

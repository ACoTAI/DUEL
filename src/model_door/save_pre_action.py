import time

import rospy
import tf
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
import threading
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import os

bridge = CvBridge()




class InfoGetter(object):
    def __init__(self):
        self._event = threading.Event()
        self._msg = None

    def __call__(self, msg):
        self._msg = msg
        self._event.set()

    def get_msg(self, timeout=None):
        self._event.wait(timeout)
        return self._msg

def set_start(forward_speed = 0.0):

    pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
    twist = Twist()
    pose = Pose()
    twist.linear.x = forward_speed
    twist.linear.y = 0
    twist.linear.z = 0
    twist.angular.x = 0
    twist.angular.y = 0
    twist.angular.z = 0
    pose.position.x = 0.0
    pose.position.y = 0.0
    pose.position.z = 0.0
    pose.orientation.x = 0
    pose.orientation.y = 0
    pose.orientation.z = 0
    pose.orientation.w = 0
    i = 0
    while True:
        time.sleep(3.5)
        pub.publish(ModelState('turtlebot3_waffle', pose, twist, 'ground_plane'))
        i += 1
        if i == 1:
            break
    print("reset the position")


def data_generator(batch_size=128, transforms=None, shuffle=True):
    img_message = InfoGetter()
    move_cmd = InfoGetter()
    rospy.Subscriber('/camera/depth/image_raw', Image, img_message, queue_size=1)
    rospy.Subscriber('/cmd_vel', Twist, move_cmd, queue_size=1)
    demonum = 1
    map_path = "/home/xxx/catkin_ws/src/duel/expert_data/model_door"
    while demonum <= 1:
        print("This is the " + str(demonum) + " time: ")
        #demo_path = os.path.join(map_path, "demo_" + str(demonum))
#        if not os.path.exists(demo_path):
            #os.makedirs(demo_path)
        count = 0
        thetime = time.time()
        number = 0
        while True:
            count += 1
            if count % 1 == 0:
                try:
                    cv2_img = bridge.imgmsg_to_cv2(img_message.get_msg(), "32FC1")
                    cv2_a = cv2_img * 256 / 7
                    global cv2_b
                    cv2_a = cv2_a[0:1080, 240:1680]
                    cv2_b = cv2.resize(cv2_a, (320, 240))
                    num = 0
                    i = 80
                    j = 50
                    while i <= 199:
                        while j <= 149:
                            if cv2_b[i][j] > 10:
                                num = num + 1
                            j = j + 1
                        j = 50
                        i = i + 1
                    damage = (12000 - num) / 12000.0
                    #cv2.imwrite(os.path.join(demo_path, str(number) + ".png"), cv2_b)
                    f = open(os.path.join(map_path, 'pre_action' + ".txt"), 'a')
                    f.write(str(move_cmd.get_msg().linear.x) + ' ' + str(move_cmd.get_msg().angular.z)  + '\n')
                    number += 1
                except CvBridgeError, e:
                    print(e)
            if time.time() - thetime >= 30.0:
                set_start()
                break
        demonum += 1


if __name__== '__main__':
    print("start!")
    rospy.init_node('image_subscriber', anonymous=True)
    set_start()
    data_generator()

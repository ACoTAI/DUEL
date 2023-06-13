import threading
import numpy as np
import connection
import copy
import collections as col
import os
import time
import math
import rospy
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose


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


class TurtleBotEnv():
    initial_reset = True

    def __init__(self):
        self.time_step = 0

        self.pre_action_0 = np.zeros(2, dtype=np.float32)
        self.pre_action_1 = np.zeros(2, dtype=np.float32)

        self.pre_pose_x = 0.0
        self.pre_pose_y = 0.0

        print("launch gazebo")
        #  os.system("killall gnome-terminal-server")
        os.system("pkill gzclient")
        os.system("pkill gzserver")
        time.sleep(0.5)
        os.system("roslaunch duel model_corridor.launch &")
        time.sleep(6)

    def step(self, u):
        # convert thisAction to the actual torcs actionstr
        client = self.client

        this_action = self.agent_to_robot(u)

        # Apply Action
        action_turtlebot = client.R.d

        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)
        self.pre_action_0 = self.pre_action_1
        self.pre_action_1 = np.array([action_turtlebot['linearX'] / 0.26,
                                      action_turtlebot['angularZ'] / 1.82])

        action_turtlebot['linearX'] = this_action['vx']
        action_turtlebot['angularZ'] = this_action['az']

        client.R.d['linearX'] = action_turtlebot['linearX']
        client.R.d['angularZ'] = action_turtlebot['angularZ']

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into TurtleBot
        # client.respond_to_server()
        # Get the response of TurtleBot
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d  # need to be changed

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observation(obs)

        # Reward setting Here #######################################
        # direction-dependent positive reward
        reward = 0
        done = False

        # goal detection
        # if self.time_step > 400:
        #     cur_pos = InfoGetter()
        #     rospy.Subscriber('/odom', Odometry, cur_pos, queue_size=1)
        #     cur_pos_x = cur_pos.get_msg().pose.pose.position.x
        #     if cur_pos_x >= 6.5:
        #         print ("Successfully reach the goal")
        #         done = True

        # collision detection
        # if (self.time_step > 180) and (obs['damage'] - obs_pre['damage'] > 0) and obs['damage'] >= 0.98:
        #     print("collide")
        #     reward += 1
        #     done = True

        if (self.time_step > 400) and (self.time_step % 5 == 0):
            cur_pos = InfoGetter()
            rospy.Subscriber('/odom', Odometry, cur_pos, queue_size=1)
            cur_pos_x = cur_pos.get_msg().pose.pose.position.x
            cur_pos_y = cur_pos.get_msg().pose.pose.position.y
            step_distance = (cur_pos_x - self.pre_pose_x) ** 2 + (cur_pos_y - self.pre_pose_y) ** 2
            if step_distance <= 0.001:
                print("Collide")
                reward += 1
                done = True
            self.pre_pose_x = cur_pos_x
            self.pre_pose_y = cur_pos_y

        self.time_step += 1

        return self.get_obs(), reward, done, {}

    def reset(self, relaunch=False):
        print("Initialization environment")

        self.time_step = 0

        if self.initial_reset is not True:
            # self.client.R.d['meta'] = True
            # self.client.respond_to_server()

            # TENTATIVE. Restarting Gazebo every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_gazebo()
                print("### Gazebo is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = connection.Client()
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from TurtleBot

        obs = client.S.d  # Get the current full-observation from TurtleBot
        self.observation = self.make_observation(obs)

        self.initial_reset = False
        return self.get_obs()

    def end(self):
        os.system("killall gnome-terminal-server")
        os.system("pkill gzclient")
        os.system("pkill gzserver")

    def get_obs(self):
        return self.observation

    def reset_gazebo(self):
        self.time_step = 0
        pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        twist = Twist()
        pose = Pose()
        twist.linear.x = 0
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
            time.sleep(0.5)
            pub.publish(ModelState('turtlebot3_waffle', pose, twist, 'ground_plane'))
            i += 1
            if i == 1:
                break
        print("Reset the position")

    def agent_to_robot(self, u):
        robot_action = {'vx': u[0] * 0.26, 'az': u[1] * 1.82}
        return robot_action

    def make_observation(self, raw_obs):  # return namedtuple
        names = ['img', 'linearX', 'angularZ', 'damage', 'pre_action_0', 'pre_action_1']
        Observation = col.namedtuple('Observaion', names)

        return Observation(img=raw_obs['img'],
                           linearX=np.array(raw_obs['linearX'], dtype=np.float32) / 0.26,
                           angularZ=np.array(raw_obs['angularZ'], dtype=np.float32) / 1.82,
                           damage=np.array(raw_obs['damage'], dtype=np.float32),
                           pre_action_0=self.pre_action_0,
                           pre_action_1=self.pre_action_1)

import threading
import numpy as np
from nav_msgs.msg import Odometry
import connection
import copy
import collections as col
import os
import time
import ctypes
import inspect
import rospy
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
import math


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
    terminal_judge_start = 300  # If after 100 timestep still no progress, terminated
    termination_limit_stuck_cnt = 70  # If radius smaller than 0.15m more than 50 timestep, terminated
    default_speed = 50

    initial_reset = True

    def __init__(self):
        self.time_step = 0
        self.stuck_cnt = 0
        self.slow = 0

        self.pre_action_0 = np.zeros(2, dtype=np.float32)
        self.pre_action_1 = np.zeros(2, dtype=np.float32)

        self.pre_pose_x = 0.0
        self.pre_pose_y = 0.0

        print("launch gazebo")
        # os.system("killall gnome-terminal-server")
        os.system("pkill gzclient")
        os.system("pkill gzserver")
        time.sleep(0.5)
        os.system("roslaunch duel model_door.launch &")
        time.sleep(6)
        # os.system("gnome-terminal -e 'roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch'")
        # time.sleep(1)

        high = np.array([255, 1, 1, 1])
        low = np.array([0, -1, -1, 0])

    def step(self, u):
        # convert thisAction to the actual turtlebot actionstr
        client = self.client

        this_action = self.agent_to_robot(u)

        # Apply Action
        action_turtlebot = client.R.d

        # Save the privious full-obs from gazebo for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)
        self.pre_action_0 = self.pre_action_1
        self.pre_action_1 = np.array([action_turtlebot['linearX']/0.26,
                                      action_turtlebot['angularZ']/1.82])

        action_turtlebot['linearX'] = this_action['vx']
        action_turtlebot['angularZ'] = this_action['az']

        client.R.d['linearX'] = action_turtlebot['linearX']
        client.R.d['angularZ'] = action_turtlebot['angularZ']

        # Get the response of TurtleBot
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d  # need to be changed

        # Make an obsevation from a raw observation vector from TurtleBot
        self.observation = self.make_observation(obs)

        # Reward setting Here
        # direction-dependent positive reward
        reward = 0

        # collision detection
        if (obs['damage'] - obs_pre['damage'] > 0) and obs['damage'] > 0.95:
            print("collide")
            # reward = (self.time_step - 300)*400 / 300
            reward = -200
            episode_terminate = True
            client.R.d['meta'] = True

#################################################################
        if (self.time_step > 100) and (self.time_step % 5 == 0):
            cur_pos = InfoGetter()
            rospy.Subscriber('/odom', Odometry, cur_pos, queue_size=1)
            cur_pos_x = cur_pos.get_msg().pose.pose.position.x
            cur_pos_y = cur_pos.get_msg().pose.pose.position.y
            step_distance = math.sqrt((cur_pos_x - self.pre_pose_x) ** 2 + (cur_pos_y - self.pre_pose_y) ** 2)
            if step_distance <= 0.0005:
                print("Stuck in one place")
                reward = -200
                episode_terminate = True
                client.R.d['meta'] = True
            self.pre_pose_x = cur_pos_x
            self.pre_pose_y = cur_pos_y
#################################################################

        '''
        if obs['damage'] == 1:
            print("collide")
            reward = -200
            episode_terminate = True
            client.R.d['meta'] = True
        '''

        # Episode is terminated if the agent stuck for a long time
        if client.R.d['angularZ'] != 0:
            radius = client.R.d['linearX'] / client.R.d['angularZ'] - 0.1435
            if radius < 0.15:
                self.stuck_cnt += 1
            else:
                self.stuck_cnt = 0

        if self.stuck_cnt > self.termination_limit_stuck_cnt:
            print("No progress, circle")
            reward = -200
            episode_terminate = True
            client.R.d['meta'] = True

        if self.terminal_judge_start < self.time_step:
            print("No progress, time out")
            #reward = -200
            episode_terminate = True
            client.R.d['meta'] = True

        # Episode is terminated if the agent runs backward
        if client.R.d['linearX'] < 0:
            print("Run backward")
            reward = -200
            episode_terminate = True
            client.R.d['meta'] = True

        # Episode is terminated if the agent runs too slow
        if abs(client.R.d['linearX']) < 0.03:
            self.slow += 1
        else:
            self.slow = 0

        if self.slow > 20:
            print("too slow")
            reward = -200
            episode_terminate = True
            client.R.d['meta'] = True

        # if client.R.d['meta'] is True:  # Send a reset signal
            # client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, client.R.d['meta'], {}

    def reset(self, relaunch=False):
        print("Reset")

        self.time_step = 0
        self.stuck_cnt = 0
        self.slow = 0
        self.pre_action_0 = np.zeros(2, dtype=np.float32)
        self.pre_action_1 = np.zeros(2, dtype=np.float32)

        if self.initial_reset is not True:
            # TENTATIVE. Restarting Gazebo every episode suffers the memory leak bug!

            # stop_thread(self.client.control)
            self.client.R.close = True
            stop_thread(self.client.spin)
            del self.client
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
        print("relaunch gazebo")
        os.system("killall gnome-terminal-server")
        os.system("pkill gzclient")
        os.system("pkill gzserver")
        time.sleep(0.5)
        os.system("roslaunch duel model_door.launch &")
        time.sleep(6)

    # def reset_gazebo(self):
    #     print("relaunch gazebo")
    #     os.system("killall gnome-terminal-server")
    #     os.system("pkill gzclient")
    #     os.system("pkill gzserver")
    #     time.sleep(0.5)
    #     os.system("roslaunch duel model_corridor_1.launch &")
    #     pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
    #     rospy.wait_for_service('/gazebo/set_model_state')
    #     # set_state = rospy.ServiceProxy('/gazebo/set_model_state', ModelState)
    #     twist = Twist()
    #     pose = Pose()
    #     twist.linear.x = 0
    #     twist.linear.y = 0
    #     twist.linear.z = 0
    #     twist.angular.x = 0
    #     twist.angular.y = 0
    #     twist.angular.z = 0
    #     pose.position.x = 0
    #     pose.position.y = 0
    #     pose.position.z = 0
    #     pose.orientation.x = 0
    #     pose.orientation.y = 0
    #     pose.orientation.z = 0
    #     pose.orientation.w = 0
    #     time.sleep(1.0)
    #     # set_state('turtlebot3_waffle', pose, twist, 'ground_plane')
    #     pub.publish(ModelState('turtlebot3_waffle', pose, twist, 'ground_plane'))
    #     # os.system("gnome-terminal -e 'roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch'")
    #     # time.sleep(1)


    def agent_to_robot(self, u):
        robot_action = {'vx': u[0]*0.26, 'az': u[1]*1.82}
        return robot_action

    def make_observation(self, raw_obs):  # return namedtuple
        names = ['img', 'linearX', 'angularZ', 'damage', 'pre_action_0', 'pre_action_1']
        Observation = col.namedtuple('Observaion', names)

        return Observation(img=raw_obs['img'],
                           linearX=np.array(raw_obs['linearX'], dtype=np.float32)/0.26,
                           angularZ=np.array(raw_obs['angularZ'], dtype=np.float32)/1.82,
                           damage=np.array(raw_obs['damage'], dtype=np.float32),
                           pre_action_0=self.pre_action_0,
                           pre_action_1=self.pre_action_1)


def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)
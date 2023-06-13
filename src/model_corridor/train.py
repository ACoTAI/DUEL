from gym_turtlebot import TurtleBotEnv
import numpy as np
from models import TRPOAgent
import argparse
import time
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
#import tensorflow as tf
import json
import os

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def playGame(finetune=0):
    demo_dir = "/home/xxx/catkin_ws/src/duel/expert_data/model_corridor/"
    param_dir = "/home/xxx/catkin_ws/src/duel/params/"
    pre_actions_path = "/home/xxx/catkin_ws/src/duel/expert_data/model_corridor/pre_actions.npz"
    feat_dim = [8, 10, 1024]
    img_dim = [60, 80, 3]
    aux_dim = 7
    encode_dim = 2
    action_dim = 2

    np.random.seed(1024)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # config = tf.ConfigProto(gpu_options=gpu_options)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # from keras import backend as K
    # K.set_session(sess)
    tf.keras.backend.set_session(sess)

    # initialize the env
    env = TurtleBotEnv()

    # define the model
    pre_actions = np.load(pre_actions_path)["actions"]
    agent = TRPOAgent(env, sess, feat_dim, aux_dim, encode_dim, action_dim, img_dim, pre_actions)

    # Load expert (state, action) pairs
    demo = np.load(demo_dir + "model_corridor.npz")

    # Now load the weight
    print("Now we load the weight")
    try:
        if finetune:
            agent.generator.load_weights(
                param_dir + "params_0/model_corridor/generator_model_201.h5")
            agent.discriminator.load_weights(
                param_dir + "params_0/model_corridor/discriminator_model_201.h5")
            agent.baseline.model.load_weights(
                param_dir + "params_0/model_corridor/baseline_model_201.h5")
            agent.posterior.load_weights(
                param_dir + "params_0/model_corridor/posterior_model_201.h5")
            agent.posterior_target.load_weights(
                param_dir + "params_0/model_corridor/posterior_target_model_201.h5")
            print("load duel weight")
        else:
            agent.generator.load_weights(
                param_dir + "params_bc/model_corridor/generator_bc_model_1.h5")             #80
            print("load BC weight")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("TurtleBot Experiment Start.")
    agent.learn(demo)

    print("Finish.")


if __name__ == "__main__":
    playGame()

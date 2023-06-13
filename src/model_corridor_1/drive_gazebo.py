from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
import numpy as np
# import tensorflow as tf
from gym_turtlebot_test import TurtleBotEnv
from models import Generator
from keras.models import Model
import time
import csv
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

code = 1

feat_dim = [8, 10, 1024]
aux_dim = 7
encode_dim = 2
action_dim = 2
param_path = "/home/xxx/catkin_ws/src/duel/params/params_0/model_corridor_1/generator_model_201.h5" # lalala
pre_actions_path = "/home/xxx/catkin_ws/src/duel/expert_data/model_corridor_1/pre_actions.npz"

MAX_STEP_LIMIT = 280
MIN_STEP_LIMIT = 100
PRE_STEP = 20

epoch = 5


def clip(v, lo, hi):
    if v < lo: return lo
    elif v > hi: return hi
    else: return v


def get_state(ob, aux_dim, feat_extractor):
    img = ob.img
    img = cv2.resize(img, (160, 120))
    x = np.expand_dims(img, axis=0).astype(np.float32)
    x = preprocess_input(x)
    feat = feat_extractor.predict(x)
    aux = np.zeros(aux_dim, dtype=np.float32)
    aux[0] = ob.damage
    aux[1] = ob.linearX
    aux[2] = ob.angularZ
    aux[3:5] = ob.pre_action_0
    aux[5:7] = ob.pre_action_1

    return feat, np.expand_dims(aux, axis=0)


def test():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # from keras import backend as K
    # K.set_session(sess)
    tf.keras.backend.set_session(sess)


    generator = Generator(sess, feat_dim, aux_dim, encode_dim, action_dim)
    base_model = ResNet50(weights='imagenet', include_top=False)
    feat_extractor = Model(
        input=base_model.input,
        output=base_model.get_layer('activation_40').output
    )

    try:
        generator.model.load_weights(param_path)
        print("Weight load successfully")
    except:
        print("cannot find weight")

    env = TurtleBotEnv()

    print("Start driving ...")
    ob = env.reset(relaunch=False) # need to be changed
    feat, aux = get_state(ob, aux_dim, feat_extractor)

    encode = np.zeros((1, encode_dim), dtype=np.float32)
    encode[0, code] = 1
    # print "Encode:", encode[0]

    pre_actions = np.load(pre_actions_path)["actions"]

    rewards = 0

    for r in xrange(epoch):
        print "This is the", r + 1, "round:"
        time_start = time.time()
        steps = MAX_STEP_LIMIT
        for i in xrange(MAX_STEP_LIMIT):
            if i < PRE_STEP:
                action = pre_actions[i]
            else:
                action = generator.model.predict([feat, aux, encode])[0]

            ob, reward, done, _ = env.step(action)
            rewards += reward
            feat, aux = get_state(ob, aux_dim, feat_extractor)

            if done:
                steps = i
                break

            print "Start deciding ..."

            csv_name = 'vel_cmd_m1_left' + str(r + 1)
            l_x = action[0] * 0.26
            a_z = action[1] * 1.82
            with open(csv_name, 'a') as csvfile:
                fieldnames = ['step', 'l_x', 'a_z']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'step': i, 'l_x': l_x, 'a_z': a_z})

            print "Step:", i + 1, "Damage:", ob.damage.item(), \
                "Action: %.6f %.6f " % (l_x, a_z)

        print "Time: %f s" % (time.time() - time_start)
        print "Steps: %d" % steps
        env.reset_gazebo()

    acc = (1 - float(rewards) / epoch) * 100
    print("Finish.")
    print "There are %d collisions" % rewards
    print "The success rate of obstacle avoidance is %.2f %%" % acc


if __name__ == "__main__":
    test()
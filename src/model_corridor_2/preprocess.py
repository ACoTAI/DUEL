# from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
import numpy as np
import time
import cv2


def collect_demo(path, num_patch):

    # initialize variable
    auxs_tmp = 0
    actions_tmp = 0
    imgs_tmp = 0
    auxs = 0
    actions = 0
    imgs = 0

    for i in xrange(num_patch):
        path_patch = path + str(i+1) + "/"
        file_name = path_patch + "speed.txt"
        raw = open(file_name, 'r').readlines()
        pa = np.zeros(4, dtype=np.float32)

        print "Loading patch %d ..." % (i + 1)
        #print(len(raw))
        for j in xrange(0, len(raw)):
            data = np.array(raw[j].strip().split(" ")).astype(np.float32)
            aux = np.expand_dims(
                np.array([data[3], data[1]/0.26, data[2]/1.82, pa[0], pa[1], pa[2], pa[3]]),
                axis=0).astype(np.float32)
            data[1] = data[1]/0.26
            data[2] = data[2] / 1.82
            action = np.expand_dims(data[1:3], axis=0).astype(np.float32)
            pa[0:2] = pa[2:4]  # pre_action_0
            pa[2:4] = action[:]  # pre_action_1

            img_path = path_patch + str(j) + ".png"
            # img = image.load_img(img_path, color_mode='grayscale')
            img = image.load_img(img_path)
            img = image.img_to_array(img)
            img = cv2.resize(img, (160, 120))
            img = np.expand_dims(img, axis=0).astype(np.uint8)

            if j == 0:
                auxs_tmp = aux
                actions_tmp = action
                imgs_tmp = img
            else:
                auxs_tmp = np.concatenate((auxs_tmp, aux), axis=0)
                actions_tmp = np.concatenate((actions_tmp, action), axis=0)
                imgs_tmp = np.concatenate((imgs_tmp, img), axis=0)

        if i == 0:
            auxs = auxs_tmp
            actions = actions_tmp
            imgs = imgs_tmp
        else:
            auxs = np.concatenate((auxs, auxs_tmp), axis=0)
            actions = np.concatenate((actions, actions_tmp), axis=0)
            imgs = np.concatenate((imgs, imgs_tmp), axis=0)

        print "Current total:", imgs.shape, auxs.shape, actions.shape

    print "Images:", imgs.shape, "Auxs:", auxs.shape, "Actions:", actions.shape

    return imgs, auxs, actions


def main():
    num_patch = 100
    demo_path = "/home/xxx/catkin_ws/src/duel/expert_data/model_corridor_2/demo_"

    imgs, auxs, actions = collect_demo(demo_path, num_patch)

    np.savez_compressed("/home/xxx/catkin_ws/src/duel/expert_data/model_corridor_2/model_corridor_2",
                        imgs=imgs, auxs=auxs, actions=actions)
    print "Finished."


if __name__ == "__main__":
    main()

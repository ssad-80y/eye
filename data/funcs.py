
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFilter
import random
import torch
from torchvision import transforms as trs
from scipy import interpolate as intr


class Augm():
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    @staticmethod
    def rgb_to_gray(img):
        grayImage = np.zeros(img.shape)
        R = np.array(img[:, :, 0])
        G = np.array(img[:, :, 1])
        B = np.array(img[:, :, 2])

        R = (R * .299)
        G = (G * .587)
        B = (B * .114)

        Avg = (R + G + B)
        grayImage = img

        for i in range(3):
            grayImage[:, :, i] = Avg

        return grayImage

    @staticmethod
    def augm(img, left, right, left_cl, right_cl, norm=320):
        # pdb.set_trace()
        y, x = img.shape
        x = float(x)
        y=float(y)

        horizontal_value = 0.0
        vertical_value = 0.0

        horizontal_value1 = 0.0
        vertical_value1 = 0.0

        i_o = random.randint(0,1)

        if i_o:
            horizontal_value = random.randint(0, int(x/2))
            vertical_value = random.randint(0, y/2)
        else:
            horizontal_value1 = random.randint(0, int(x/2))
            vertical_value1 = random.randint(0, y / 2)


        horizontal_value = int(horizontal_value)
        horizontal_value1 = int(horizontal_value1)
        vertical_value=int(vertical_value)
        vertical_value1=int(vertical_value1)
        x = int(x)
        y = int(y)

        img = img[:, horizontal_value:x-horizontal_value1]
        img = img[vertical_value:y-vertical_value1, :]
        img = np.pad(img, ((vertical_value1, vertical_value),(horizontal_value1, horizontal_value)), 'linear_ramp' )

        left1 = left.copy()
        right1 = right.copy()

        if i_o:
            left1[0] = 0.0 if ((left[0]-vertical_value)<0.0 or (left[1]-horizontal_value)<0.0) else left[0]-vertical_value
            left1[1] = 0.0 if ((left[1]-horizontal_value)<0.0 or (left[0]-vertical_value)<0.0) else left[1]-horizontal_value

            right1[0] = 0.0 if ((right[0] - vertical_value) < 0.0 or (right[1] - horizontal_value) < 0.0) else right[0] - vertical_value
            right1[1] = 0.0 if ((right[1] - horizontal_value) < 0.0 or (right[0] - vertical_value) < 0.0) else right[1] - horizontal_value


        else:
            left1[0]= 0.0 if ((left[0] + vertical_value1) > y or (left[0] + horizontal_value1) > x) else left[0] + vertical_value1
            left1[1] = 0.0 if ((left[1] + horizontal_value1) > x or (left[0] + vertical_value1) > y) else left[1] + horizontal_value1

            right1[0]= 0.0 if ((right[0] + vertical_value1) > y or (right[1] + horizontal_value1) > x) else right[0] + vertical_value1
            right1[1] = 0.0 if ((right[1] + horizontal_value1) > x or (right[0] + vertical_value1) > y) else right[1] + horizontal_value1


        return img, left1, right1, right_cl, left_cl




    def __init__(self):

        pass



def sh(kek):
    plt.imshow(kek)
    plt.show()
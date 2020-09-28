import random

import numpy as np
import torch
from PIL import Image

from data.base_dataset import BaseDataset
from data.funcs import Augm as A




class AlignedDataset(BaseDataset):
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

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        # test_1k.txt
        # train_5k.txt


        self.path1 = ''
        self.path2 = "path"
        with open('data/train.txt') as lines1:
            name_file = [row.strip() for row in lines1]
        self.path = name_file
        print(len(self.path))
        self.list = ['median', 'mean', 'edge', 'maximum']
    #     y





    def __getitem__(self, index):

        self.canvas = np.ones((240, 320))
        self.image_name = self.path[index]
        name = self.image_name
        with open(self.path1 + name + '.txt') as lines2:
            name_file1 = [row1.strip() for row1 in lines2]
        tags = name_file1

        self.im = Image.open(self.path1 + self.image_name + '.jpg')


        self.canvas = self.rgb_to_gray(np.array(self.im))[:, :, 0]


        self.canvas = self.canvas / 255.0

        left_cl = 1 if tags[2] == '0' else 0  # left
        right_cl = 1 if tags[5] == '0' else 0  # right



        crds_left = np.ones((3))
        crds_right = np.ones((3))


        crds_left[0] = (int(tags[0]))
        crds_left[1] = (int(tags[1]))
        crds_left[2] = int(tags[6]) if int(tags[0]) != 0 and int(tags[1]) != 0 else 0.0


        crds_right[0] = (int(tags[3]))
        crds_right[1] = (int(tags[4]))
        crds_right[2] = int(tags[6]) if int(tags[3]) != 0 and int(tags[4]) != 0 else 0.0

        self.canvas, crds_left, crds_right, left_cl, right_cl = A.augm(self.canvas, crds_left, crds_right, left_cl,
                                                                       right_cl)

        crds_left[:2] = 0.0 if min(crds_left) == 0 else crds_left[:2]
        crds_right[:2] = 0.0 if min(crds_right) == 0 else crds_right[:2]



        crds_left[2] = 0 if crds_left[0] == 0.0 or crds_left[1] == 0.0 else crds_left[2]

        crds_right[2] = 0 if crds_right[0] == 0.0 or crds_right[1] == 0.0 else crds_right[2]

        crds_right /= 320.0
        crds_left /= 320.0

        left_cl = 2 if crds_left[0] == 0.0 or crds_left[1] == 0.0 else left_cl
        right_cl = 2 if crds_right[0] == 0.0 or crds_right[1] == 0.0 else right_cl


        GT = np.ones((2, 4))


        GT[0, :1] = left_cl
        GT[0, 1:] = crds_left.reshape(1, 3)

        GT[1, :1] = right_cl
        GT[1, 1:] = crds_right.reshape(1, 3)

        Input = torch.from_numpy(self.canvas).float()

        GT = torch.from_numpy(GT).float()


        return {'Input': Input, 'GT': GT, 'Name': name[1]}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.path)

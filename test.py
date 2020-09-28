"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

import cv2
from PIL import Image as im
import pdb
import time
import scipy.misc
import torch
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as rt



def rgb_to_gray(img):
        grayImage = np.zeros(img.shape)
        R = np.array(img[:, :, 0])
        G = np.array(img[:, :, 1])
        B = np.array(img[:, :, 2])

        R = (R *.299)
        G = (G *.587)
        B = (B *.114)

        Avg = (R+G+B)
        grayImage = img

        for i in range(3):
           grayImage[:,:,i] = Avg

        return grayImage

sigmoid = lambda x: 1/(1 + np.exp(-x))

if __name__ == '__main__':
    # opt = TestOptions().parse()  # get test options
    # # hard-code some parameters for test
    # opt.num_threads = 3   # test code only supports num_threads = 1
    # opt.batch_size = 1    # test code only supports batch_size = 1
    # opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    # opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    # opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    #
    # model = create_model(opt)      # create a model given opt.model and other options
    # model.setup(opt)               # regular setup: load and print networks; create schedulers

    



    capture = cv2.VideoCapture(1)
    kek = capture.get(cv2.CAP_PROP_FPS)
    capture.set(cv2.CAP_PROP_FPS, 120)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)


    kek1 = capture.get(cv2.CAP_PROP_FPS)
    print(kek)
    print(kek1)
    name = "video"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    print(type(fourcc))
    out = cv2.VideoWriter(name, fourcc, 120, (320, 240))
    n=0
    first = time.time()
    sess = rt.InferenceSession("kek.onnx")
    input_name = sess.get_inputs()[0].name


    while(1):
    # while(1):

        ret, frame = capture.read()
        # pdb.set_trace()/
        # pdb.set_trace()
        frame1 = rgb_to_gray(frame)[:,:,0]
        frame1 = frame1.reshape(1,1,240, 320)
        frame1 = frame1.astype(np.float32)/255.0
        # frame1 = frame1.numpy()

        pred_onx = sess.run(None, {input_name: frame1})[0]


        # pdb.set_trace()
        # model.set_input(frame1)  # unpack data from data loader
        # cl1, cr1, lst1, rst1 = model.test()

        # pdb.set_trace()

        lst1 = np.argmax(pred_onx[:,:3])
        rst1 = np.argmax(pred_onx[:,6:9])


        img = frame

        cv2.putText(img, str(lst1), (2,20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
        cv2.putText(img, str(rst1), (290,20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0, 0, 255), 1)

        
        cv2.imshow('video',img )


        out.write(img)
        n+=1
        if cv2.waitKey(1) == 27:
            break


    cv2.destroyAllWindows()
    print(time.time()-first)
    print(n)
    input()
    capture.release()


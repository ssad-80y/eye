import torch
from .base_model import BaseModel
from . import networks
import pdb
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import cv2





class Pix2PixModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):


        if is_train:

            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)

        self.loss_names = ['cl_left', 'cl_right', 'crd']

        self.visual_names = ['gt', 'fake_B']

        if self.isTrain:
            self.model_names = ['G']

        else:  # during test time, only load G
            self.model_names = ['G']

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


        
        

        

        if self.isTrain:


            self.criterionL1 = torch.nn.L1Loss()
            self.loss = torch.nn.CrossEntropyLoss()

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.001)
            
            self.optimizers.append(self.optimizer_G)
        self.i = 0
        self.acc = 0.0
            

    def set_input(self, input):

        self.Input = input['Input'].to(self.device)
        self.gt = input['GT'].to(self.device).unsqueeze(1)
        self.name = input['Name']





        


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        self.fake_B = self.netG(self.Input.unsqueeze(1))
        # pdb.set_trace()

        # self.sfm_left = F.softmax(self.fake_B[:,:2], dim = 1)
        # self.sfm_right =  F.softmax(self.fake_B[:,5:7], dim = 1)
        self.sfm_left = (self.fake_B[:,:3])
        self.sfm_right = (self.fake_B[:,6:9])

        self.crd_left = torch.sigmoid(self.fake_B[:,3:6])
        self.crd_right = torch.sigmoid(self.fake_B[:,9:])


        





        

    
    def backward_G(self):

        self.loss_cl_left = self.loss(self.sfm_left.reshape(-1,3,1,1),  self.gt[:,:,0,0].reshape(-1,1,1).type(torch.cuda.LongTensor))
        self.loss_cl_right = self.loss(self.sfm_right.reshape(-1,3,1,1),  self.gt[:,:,1,0].reshape(-1,1,1).type(torch.cuda.LongTensor))

        

        self.loss_crd = self.criterionL1(self.crd_left,  self.gt[:,:,0, 1:].squeeze(1).unsqueeze(2).unsqueeze(3))
        self.loss_crd += self.criterionL1(self.crd_right,  self.gt[:,:,1, 1:].squeeze(1).unsqueeze(2).unsqueeze(3))

        self.loss_G =  self.loss_crd + self.loss_cl_left+self.loss_cl_right
        # print(self.gt[1,0,:2])
       # print(self.loss_G)


        # combine loss and calculate gradients
        
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_G.zero_grad()
        self.optimizer_G.param_groups[0]['lr'] = 0.001
        self.backward_G()
        self.optimizer_G.step()

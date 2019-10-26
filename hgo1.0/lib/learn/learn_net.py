# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from net.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from net.backbone import build_backbone
from net.ASPP import ASPP
# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------
import torch
import argparse
import os
import sys
import cv2
import time

class Configuration():
	def __init__(self):
		self.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname("__file__"),'..','..'))
		self.EXP_NAME = 'deeplabv3+voc'

		self.DATA_NAME = 'VOC2012'
		self.DATA_AUG = True
		self.DATA_WORKERS = 8
		self.DATA_RESCALE = 512
		self.DATA_RANDOMCROP = 512
		self.DATA_RANDOMROTATION = 0
		self.DATA_RANDOMSCALE = 2
		self.DATA_RANDOM_H = 10
		self.DATA_RANDOM_S = 10
		self.DATA_RANDOM_V = 10
		self.DATA_RANDOMFLIP = 0.5
		
		self.MODEL_NAME = 'deeplabv3plus'
		self.MODEL_BACKBONE = 'res101_atrous'
		self.MODEL_OUTPUT_STRIDE = 16
		self.MODEL_ASPP_OUTDIM = 256
		self.MODEL_SHORTCUT_DIM = 48
		self.MODEL_SHORTCUT_KERNEL = 1
		self.MODEL_NUM_CLASSES = 21
		self.MODEL_SAVE_DIR = os.path.join(self.ROOT_DIR,'model',self.EXP_NAME)

		self.TRAIN_LR = 0.007
		self.TRAIN_LR_GAMMA = 0.1
		self.TRAIN_MOMENTUM = 0.9
		self.TRAIN_WEIGHT_DECAY = 0.00004
		self.TRAIN_BN_MOM = 0.0003
		self.TRAIN_POWER = 0.9
		self.TRAIN_GPUS = 2
		self.TRAIN_GPU=1,2
		self.TRAIN_BATCHES = 1
		self.TRAIN_SHUFFLE = True
		self.TRAIN_MINEPOCH = 0	
		self.TRAIN_EPOCHS = 46
		self.TRAIN_LOSS_LAMBDA = 0
		self.TRAIN_TBLOG = True
		self.TRAIN_CKPT = None#os.path.join(self.ROOT_DIR,'model/deeplabv3+voc/deeplabv3plus_xception_VOC2012_itr0.pth')

		self.LOG_DIR = os.path.join(self.ROOT_DIR,'log',self.EXP_NAME)

		self.TEST_MULTISCALE = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
		self.TEST_FLIP = True
		self.TEST_CKPT = os.path.join(self.ROOT_DIR,'model/deeplabv3+voc/deeplabv3plus_res101_atrous_VOC2012_epoch46_all.pth')
		self.TEST_GPUS = 4
		self.TEST_BATCHES = 16		

		self.__check()
		self.__add_path(os.path.join(self.ROOT_DIR, 'lib'))
		
	def __check(self):
		if not torch.cuda.is_available():
			raise ValueError('config.py: cuda is not avalable')
		if self.TRAIN_GPUS == 0:
			raise ValueError('config.py: the number of GPU is 0')
		#if self.TRAIN_GPUS != torch.cuda.device_count():
		#	raise ValueError('config.py: GPU number is not matched')
		if not os.path.isdir(self.LOG_DIR):
			os.makedirs(self.LOG_DIR)
		if not os.path.isdir(self.MODEL_SAVE_DIR):
			os.makedirs(self.MODEL_SAVE_DIR)

	def __add_path(self, path):
		if path not in sys.path:
			sys.path.insert(0, path)





class deeplabv3plus(nn.Module):
	def __init__(self, cfg):
		super(deeplabv3plus, self).__init__()
		self.backbone = None		
		self.backbone_layers = None
		input_channel = 2048		
		self.aspp = ASPP(dim_in=input_channel, 
				dim_out=cfg.MODEL_ASPP_OUTDIM, 
				rate=16//cfg.MODEL_OUTPUT_STRIDE,
				bn_mom = cfg.TRAIN_BN_MOM)
		self.dropout1 = nn.Dropout(0.5)
		self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
		self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE//4)

		indim = 256
		self.shortcut_conv = nn.Sequential(
				nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL, 1, padding=cfg.MODEL_SHORTCUT_KERNEL//2,bias=True),
				SynchronizedBatchNorm2d(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),		
		)		
		self.cat_conv = nn.Sequential(
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM+cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
				SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
				nn.ReLU(inplace=True),
				nn.Dropout(0.1),
		)
		self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, SynchronizedBatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
		self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
		self.backbone_layers = self.backbone.get_layers()

	def forward(self, x):
		x_bottom = self.backbone(x)
		layers = self.backbone.get_layers()
		feature_aspp = self.aspp(layers[-1])
		feature_aspp = self.dropout1(feature_aspp)
		feature_aspp = self.upsample_sub(feature_aspp)

		feature_shallow = self.shortcut_conv(layers[0])
		feature_cat = torch.cat([feature_aspp,feature_shallow],1)
		result = self.cat_conv(feature_cat) 
		result = self.cls_conv(result)
		result = self.upsample4(result)
		return result

if __name__=='__main__':
    input_1=torch.randn(16,3,512,512)
    cfg = Configuration() 
    net = deeplabv3plus(cfg)
    out=net(input_1)
    print('input',input_1.shape)
    print('out',out.shape)
    # print(net)
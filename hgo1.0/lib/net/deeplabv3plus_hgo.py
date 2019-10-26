# ----------------------------------------
# Written by Ruijie Zhang
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from net.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from net.backbone import build_backbone
from net.ASPP import ASPP
from net.SPP import spatial_pyramid_pool

# def ten_mul(img,mask):     
#         img_channel=torch.split(img,1,dim=1)
#         l=[]
#         for i in img_channel:
#                 i=i.type(torch.FloatTensor)
#                 mask=mask.type(torch.FloatTensor)

#                 out=i*mask
#                 l.append(out)

#         result=l[0]
#         for i in range(1,len(l)):
      
#                 result=torch.cat((result,l[i]),1)
#         return result

class deeplabv3plus(nn.Module):
	def __init__(self, cfg,test=False):
		super(deeplabv3plus, self).__init__()
		if test:
			self.batch_size=cfg.TEST_BATCHES
		else:
			self.batch_size=cfg.TRAIN_BATCHES
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
		self.backbone1 = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
		self.backbone2 = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
		self.backbone_layers1 = self.backbone1.get_layers()
		self.backbone_layers2 = self.backbone2.get_layers()
		self.tc_conv = nn.Sequential(
            nn.Conv2d(6, 96, 7,stride=3),  # 96*20*20
            nn.ReLU(inplace=True), # 96*20*20
            nn.MaxPool2d(2),  # 96*10*10
            nn.Conv2d(96, 256, 5), # 192*6*6
            nn.ReLU(),  # 192*6*6
            nn.MaxPool2d(2),  # 192*3*3
            nn.Conv2d(256, 512, 3),
            nn.ReLU()  # 256*1*1           
        )
		self.liner = nn.Sequential(            
            nn.Linear(41472,256),
            nn.ReLU()           
        )
		self.out = nn.Linear(256, 1)
	def forward(self, x1, x2):
		# print('forward',x1.shape,x2.shape)
		x1_bottom = self.backbone1(x1)
		x2_bottom = self.backbone2(x2)

		layers_1 = self.backbone1.get_layers()
		layers_2 = self.backbone2.get_layers()

		feature_aspp1 = self.aspp(layers_1[-1])
		feature_aspp2 = self.aspp(layers_2[-1])

		feature_aspp1 = self.dropout1(feature_aspp1)
		feature_aspp2 = self.dropout1(feature_aspp2)

		feature_aspp1 = self.upsample_sub(feature_aspp1)
		feature_aspp2 = self.upsample_sub(feature_aspp2)

		feature_shallow1 = self.shortcut_conv(layers_1[0])
		feature_shallow2= self.shortcut_conv(layers_2[0])

		feature_cat1 = torch.cat([feature_aspp1,feature_shallow1],1)
		feature_cat2 = torch.cat([feature_aspp2,feature_shallow2],1)

		result1 = self.cat_conv(feature_cat1) 
		result1 = self.cls_conv(result1)
		result1 = self.upsample4(result1)

		result2 = self.cat_conv(feature_cat2) 
		result2 = self.cls_conv(result2)
		result2 = self.upsample4(result2)

		# if thre>500:
		# 	result2=result2.detach()
		# 	result1=result1.detach()
		# print('forward2',result1.shape,result2.shape)
		twoc_ipt1 = self.ten_mul(x1,result1)
		twoc_ipt2 = self.ten_mul(x2,result2)
		print('forward3',twoc_ipt1.shape,twoc_ipt2.shape)
		twoc_input = torch.cat((twoc_ipt1,twoc_ipt2),1)
		# print('forward4',twoc_input.shape)
		twoc_input=twoc_input.cuda()
		# print('2cccc,',twoc_input.shape)

		# output = self.forward_2c(twoc_input)
		output_num = [8,4,1]
		# print('twoc_input',twoc_input.shape)
		x = self.tc_conv(twoc_input)   
		# print('forward5',x.shape)
		x_spp=spatial_pyramid_pool(x, int(x.shape[0]), [int(x.size(2)),int(x.size(3))], output_num)
		print('forward6',x_spp.shape)
		# print('debug',x_spp.shape)
		x = self.liner(x_spp)    
		# print('forward7',x.shape) 
		out = self.out(x)
		# print('forward8',out.shape)
		return result1,result2,out

	# def forward_2c(self,x):
	# 	output_num = [8,4,1]
	# 	x = self.tc_conv(x)   

	# 	x_spp=spatial_pyramid_pool(x, self.batch_size, [int(x.size(2)),int(x.size(3))], output_num)
	# 	x = self.liner(x)     
	# 	x = self.out(x)
 
	# 	return x

	# def forward(self,x1,x2):
	# 	out1,out2 = self.forward_cls(x1,x2)
	# 	mask1=out1[0][0]
	# 	mask2=out2[0][0]
	# 	twoc_ipt1=ten_mul(x1,mask1)
	# 	twoc_ipt2=ten_mul(x2,mask2)
	# 	twoc_input=torch.cat((twoc_ipt1,twoc_ipt2),1)
	# 	result = self.forward_2c(twoc_input)
		
	# 	return out1,out2,result
	def ten_mul(self,img,mask_all):  
		# print('ten_mul',img.shape,mask_all.shape) 
		mask=torch.min(mask_all,1)[1]  
		mask=torch.unsqueeze(mask,1)
		# print('ten_mul 2',mask.shape) 
		img_channel=torch.split(img,1,dim=1)
		l=[]
		for i in img_channel:
				i=i.type(torch.FloatTensor)
				mask=mask.type(torch.FloatTensor)
				# print('i',i.shape,mask.shape)
				out=i*mask
				# print('out',out.shape)
				l.append(out)

		result=l[0]
		for i in range(1,len(l)):

				result=torch.cat((result,l[i]),1)
		return result
if __name__=='__main__':
	input_1=torch.tensor(3,200,200)
	print('input',input_1.shape)
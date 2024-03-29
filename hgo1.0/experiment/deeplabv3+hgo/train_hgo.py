# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import sys
import numpy as np

from config import cfg
from datasets.generateData import generate_dataset
from net.generateNet import generate_net
import torch.optim as optim
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from net.loss import HgoLoss
from net.sync_batchnorm.replicate import patch_replication_callback
def train_net():
	print('start')
	dataset = generate_dataset(cfg.DATA_NAME, cfg, 'train')
	dataloader = DataLoader(dataset, 
				batch_size=cfg.TRAIN_BATCHES, 
				shuffle=cfg.TRAIN_SHUFFLE, 
				num_workers=cfg.DATA_WORKERS,
				drop_last=True)
	
	net = generate_net(cfg)
	if cfg.TRAIN_TBLOG:
		from tensorboardX import SummaryWriter
		# Set the Tensorboard logger
		tblogger = SummaryWriter(cfg.LOG_DIR)

	
	# os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
	# print('zzz',torch.cuda.current_device())
	print('Use %d GPU'%cfg.TRAIN_GPUS)
	# print('zrj',torch.cuda.device_count())
	device = torch.device('cuda:4')
	if cfg.TRAIN_GPUS > 1:
		net = nn.DataParallel(net,device_ids=[4,7])
		patch_replication_callback(net)
	net.to(device)		

	if cfg.TRAIN_CKPT:
		pretrained_dict = torch.load(cfg.TRAIN_CKPT)
		net_dict = net.state_dict()
		# print(net_dict.keys())
		# input()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape==net_dict[k].shape)}
		net_dict.update(pretrained_dict)
		net.load_state_dict(net_dict)
		# net.load_state_dict(torch.load(cfg.TRAIN_CKPT),False)
	# print('zzz1')
	criterion = nn.CrossEntropyLoss(ignore_index=255)
	hgo_loss= HgoLoss()
	optimizer = optim.SGD(
		params = [
			{'params': get_params(net.module,key='1x'), 'lr': cfg.TRAIN_LR},
			{'params': get_params(net.module,key='10x'), 'lr': 10*cfg.TRAIN_LR}
		],
		momentum=cfg.TRAIN_MOMENTUM
	)
	#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN_LR_MST, gamma=cfg.TRAIN_LR_GAMMA, last_epoch=-1)
	itr = cfg.TRAIN_MINEPOCH * len(dataloader)
	max_itr = cfg.TRAIN_EPOCHS*len(dataloader)
	running_loss = 0.0
	tblogger = SummaryWriter(cfg.LOG_DIR)
	#net.eval()
	for epoch in range(cfg.TRAIN_MINEPOCH, cfg.TRAIN_EPOCHS):
		#scheduler.step()
		#now_lr = scheduler.get_lr()
		# running_loss = 0.0
		for i_batch, sample_batched in enumerate(dataloader):
			now_lr = adjust_lr(optimizer, itr, max_itr)
			inputs_img1_batched, labels_img1_batched = sample_batched['image1'], sample_batched['segmentation1']
			inputs_img2_batched, labels_img2_batched = sample_batched['image2'], sample_batched['segmentation2']
			inputs_label_batched=sample_batched['label']
			# print('input',inputs_img1_batched.shape,inputs_img2_batched.shape)
			optimizer.zero_grad()
			labels_img1_batched = labels_img1_batched.long().to(4)
			labels_img2_batched = labels_img2_batched.long().to(4)
			inputs_label_batched = inputs_label_batched.float().to(4)
			#0foreground_pix = (torch.sum(labels_batched!=0).float()+1)/(cfg.DATA_RESCALE**2*cfg.TRAIN_BATCHES)
			# print('input',inputs_batched.shape)
			predicts_img1_batched, predicts_img2_batched, predicts_label = net(inputs_img1_batched,inputs_img2_batched)
	
			# print('out',predicts_img1_batched.shape,predicts_label.shape)
			predicts_img1_batched = predicts_img1_batched.to(4) 
			predicts_img2_batched = predicts_img2_batched.to(4) 
			predicts_label = predicts_label.float().to(4) 
			loss_img1 = criterion(predicts_img1_batched, labels_img1_batched,)
			loss_img2 = criterion(predicts_img2_batched, labels_img2_batched,)
			# print('inputs_label_batched',inputs_label_batched.shape)
			loss_2c=hgo_loss(predicts_label,inputs_label_batched)
			# print('zzzhrj',predicts_batched.shape,labels_batched.shape)
			loss_1=0.5*(loss_img1+loss_img2)
			loss=0.7*loss_2c+0.3*loss_1
			loss.backward()
			optimizer.step()
			# print('zrj',loss.item(),running_loss)
			running_loss +=  loss.item()
			# print('zrj',type(loss),running_loss) 
			print('epoch:%d/%d\tbatch:%d/%d\titr:%d\tlr:%g\tloss:%g\tloss1:%g\tloss2:%g ' % 
				(epoch, cfg.TRAIN_EPOCHS, i_batch, dataset.__len__()//cfg.TRAIN_BATCHES,
				itr+1, now_lr, running_loss,loss_1,loss_2c))
			# if cfg.TRAIN_TBLOG and itr%100 == 0:
			# 	#inputs = np.array((inputs_batched[0]*128+128).numpy().transpose((1,2,0)),dtype=np.uint8)
			# 	#inputs = inputs_batched.numpy()[0]
			# 	inputs = inputs_batched.numpy()[0]/2.0 + 0.5
			# 	labels = labels_batched[0].cpu().numpy()
			# 	labels_color = dataset.label2colormap(labels).transpose((2,0,1))
			# 	predicts = torch.argmax(predicts_batched[0],dim=0).cpu().numpy()
			# 	predicts_color = dataset.label2colormap(predicts).transpose((2,0,1))
			# 	pix_acc = np.sum(labels==predicts)/(cfg.DATA_RESCALE**2)

			# 	tblogger.add_scalar('loss', running_loss, itr)
			# 	tblogger.add_scalar('loss_1', loss_1, itr)
			# 	tblogger.add_scalar('loss_2', loss_2c, itr)
			# 	tblogger.add_scalar('lr', now_lr, itr)
			# 	tblogger.add_scalar('pixel acc', pix_acc, itr)
			# 	tblogger.add_image('Input', inputs, itr)
			# 	tblogger.add_image('Label', labels_color, itr)
			# 	tblogger.add_image('Output', predicts_color, itr)
			running_loss = 0.0
			
			if itr % 60 == 0:
				save_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_%s_%s_itr%d.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,itr))
				torch.save(net.state_dict(), save_path)
				print('%s has been saved'%save_path)

			itr += 1
		
	save_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_%s_%s_epoch%d_all.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,cfg.TRAIN_EPOCHS))		
	torch.save(net.state_dict(),save_path)
	if cfg.TRAIN_TBLOG:
		tblogger.close()
	print('%s has been saved'%save_path)

def adjust_lr(optimizer, itr, max_itr):
	now_lr = cfg.TRAIN_LR * (1 - itr/(max_itr+1)) ** cfg.TRAIN_POWER
	optimizer.param_groups[0]['lr'] = now_lr
	optimizer.param_groups[1]['lr'] = 10*now_lr
	return now_lr

def get_params(model, key):
	for m in model.named_modules():
		if key == '1x':
			if 'backbone' in m[0] and isinstance(m[1], nn.Conv2d):
				for p in m[1].parameters():
					yield p
		elif key == '10x':
			if 'backbone' not in m[0] and isinstance(m[1], nn.Conv2d):
				for p in m[1].parameters():
					yield p
if __name__ == '__main__':
	train_net()



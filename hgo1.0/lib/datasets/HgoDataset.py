# ----------------------------------------
# Written by Ruijie Zhang
# ----------------------------------------

from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
import multiprocessing
from skimage import io
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from datasets.transform import *

class HgoDataset(Dataset):
    def __init__(self, dataset_name, cfg, period):
        self.dataset_name = dataset_name
        self.root_dir = os.path.join(cfg.ROOT_DIR,'data','HgoNet_data')
        # self.dataset_dir = os.path.join(self.root_dir,dataset_name)
        self.rst_dir = os.path.join(self.root_dir,'results',dataset_name,'Segmentation')
        # self.eval_dir = os.path.join(self.root_dir,'eval_result',dataset_name,'Segmentation')
        self.period = period
        if 'train' in self.period:  
            self.img_dir = os.path.join(self.root_dir, 'JPEGImages')
        else:
            self.img_dir = os.path.join(self.root_dir, 'real_test_images')
        
        self.seg_dir = os.path.join(self.root_dir, 'SegmentationClass')
        self.set_dir = os.path.join(self.root_dir, 'ImageSets')
        file_name = self.set_dir+'/'+period+'.txt'
       
        df = pd.read_csv(file_name, names=['file_img1','file_img2','label'],sep=',')
        # print(df)
        self.name_list_img1 = df['file_img1'].values
        self.name_list_img2 = df['file_img2'].values
        self.label_list= df['label'].values #同一个索引在label list中可以确认
        self.totensor = ToTensor()
        self.cfg = cfg
	
        if dataset_name == 'hgo2019':
            self.categories = ['door']  #1    
            

            self.num_categories = len(self.categories)
            assert(self.num_categories+1 == self.cfg.MODEL_NUM_CLASSES)
            self.cmap = self.__colormap(len(self.categories)+1)


        if cfg.DATA_RESCALE > 0:
            self.rescale = Rescale(cfg.DATA_RESCALE,fix=False)
            #self.centerlize = Centerlize(cfg.DATA_RESCALE)
        if 'train' in self.period:  

            # if cfg.DATA_RANDOMCROP > 0:
            #     self.randomcrop = RandomCrop(cfg.DATA_RANDOMCROP)
            # if cfg.DATA_RANDOMROTATION > 0:
            #     self.randomrotation = RandomRotation(cfg.DATA_RANDOMROTATION)
            # if cfg.DATA_RANDOMSCALE != 1:
            #     self.randomscale = RandomScale(cfg.DATA_RANDOMSCALE)
            if cfg.DATA_RANDOMFLIP > 0:
                self.randomflip = RandomFlip(cfg.DATA_RANDOMFLIP)
            # if cfg.DATA_RANDOM_H > 0 or cfg.DATA_RANDOM_S > 0 or cfg.DATA_RANDOM_V > 0:
            #     self.randomhsv = RandomHSV(cfg.DATA_RANDOM_H, cfg.DATA_RANDOM_S, cfg.DATA_RANDOM_V)
        else:
            self.multiscale = Multiscale(self.cfg.TEST_MULTISCALE)

    def __len__(self):
        return len(self.name_list_img1)

    def __getitem__(self, idx):
        name_img1 = self.name_list_img1[idx]
        name_img2 = self.name_list_img2[idx]
        label = self.label_list[idx]# 0 is changed 1 is unchanged
        # print('img1_file',self.img_dir + '/' + name_img1 + '.jpg')
        # print('img2_file',self.img_dir + '/' + name_img2 + '.jpg')
        img1_file = self.img_dir + '/' + name_img1 + '.jpg'
        
        img2_file = self.img_dir + '/' + name_img2 + '.jpg'
        

        image1 = cv2.imread(img1_file)        
        image2 = cv2.imread(img2_file)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        # print('name_img1',idx,name_img1)
        # print(image1.shape)
        # print('name_img2',idx,name_img2)
        # print(image2.shape)
        # image = np.array(io.imread(img_file),dtype=np.uint8)
        r,c,_ = image1.shape
        sample = {'image1': image1,'image2': image2,'label': label, 'name_img1': name_img1,'name_img2': name_img2, 'row': r, 'col': c}

        
        if 'train' in self.period:
            seg_file1 = self.seg_dir + '/' + name_img1 + '.png'
            seg_file2 = self.seg_dir + '/' + name_img2 + '.png'


            segmentation1 = np.array(Image.open(seg_file1))
            segmentation2 = np.array(Image.open(seg_file2))

            sample['segmentation1'] = segmentation1
            sample['segmentation2'] = segmentation2
            
            # if self.cfg.DATA_RANDOM_H>0 or self.cfg.DATA_RANDOM_S>0 or self.cfg.DATA_RANDOM_V>0:
            #     sample = self.randomhsv(sample)
            if self.cfg.DATA_RANDOMFLIP > 0:
                sample = self.randomflip(sample)
            # if self.cfg.DATA_RANDOMROTATION > 0:
            #     sample = self.randomrotation(sample)
            # if self.cfg.DATA_RANDOMSCALE != 1:
            #     sample = self.randomscale(sample)
            # if self.cfg.DATA_RANDOMCROP > 0:
            #     sample = self.randomcrop(sample)
            if self.cfg.DATA_RESCALE > 0:
                #sample = self.centerlize(sample)
                sample = self.rescale(sample)
        else:
            if self.cfg.DATA_RESCALE > 0:
                # print('?????????')
                sample = self.rescale(sample)
            sample = self.multiscale(sample)
        # print('name_img1',idx,name_img1)
        # print(image1.shape)
        # print('name_img2',idx,name_img2)
        # print(image2.shape)

        if 'segmentation1' in sample.keys():
            sample['mask1'] = sample['segmentation1']==255
            sample['mask2'] = sample['segmentation2']==255
            t1 = sample['segmentation1']
            t2 = sample['segmentation2']
            t1=(t1/255).astype(int)
            t2=(t2/255).astype(int)
            # print(t.shape)
            # t[t >= self.cfg.MODEL_NUM_CLASSES] = 0
            # print('tttt',type(t),self.cfg.MODEL_NUM_CLASSES)
            sample['segmentation1_onehot']=onehot(t1,self.cfg.MODEL_NUM_CLASSES)
            sample['segmentation2_onehot']=onehot(t2,self.cfg.MODEL_NUM_CLASSES)

        sample = self.totensor(sample)

        return sample



    def __colormap(self, N):
        """Get the map from label index to color

        Args:
            N: number of class

            return: a Nx3 matrix

        """
        cmap = np.zeros((N, 3), dtype = np.uint8)

        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

        for i in range(N):
            r = 0
            g = 0
            b = 0
            idx = i
            for j in range(7):
                str_id = uint82bin(idx)
                r = r ^ ( np.uint8(str_id[-1]) << (7-j))
                g = g ^ ( np.uint8(str_id[-2]) << (7-j))
                b = b ^ ( np.uint8(str_id[-3]) << (7-j))
                idx = idx >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        return cmap
    
    def label2colormap(self, label):
        m = label.astype(np.uint8)
        r,c = m.shape
        cmap = np.zeros((r,c,3), dtype=np.uint8)
        cmap[:,:,0] = (m&1)<<7 | (m&8)<<3
        cmap[:,:,1] = (m&2)<<6 | (m&16)<<2
        cmap[:,:,2] = (m&4)<<5
        return cmap
    
    def save_result(self, result_list, model_id):
        """Save test results

        Args:
            result_list(list of dict): [{'name':name1, 'predict':predict_seg1},{...},...]

        """
        i = 1
        folder_path = os.path.join(self.rst_dir,'%s_%s_cls'%(model_id,self.period))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for sample in result_list:
            file_path = os.path.join(folder_path, '%s.png'%sample['name'])
            sample['predict'] = self.label2colormap(sample['predict'])
            # p = self.__coco2voc(sample['predict'])
            cv2.imwrite(file_path, sample['predict'])
            print('[%d/%d] %s saved'%(i,len(result_list),file_path))
            i+=1

    
    def do_python_eval(self, model_id):
        predict_folder = os.path.join(self.rst_dir,'%s_%s_cls'%(model_id,self.period))
        gt_folder = self.seg_dir
        TP = []
        P = []
        T = []
        for i in range(self.cfg.MODEL_NUM_CLASSES):
            TP.append(multiprocessing.Value('i', 0, lock=True))
            P.append(multiprocessing.Value('i', 0, lock=True))
            T.append(multiprocessing.Value('i', 0, lock=True))
        
        def compare(start,step,TP,P,T):
            for idx in range(start,len(self.name_list),step):
                print('%d/%d'%(idx,len(self.name_list)))
                name = self.name_list[idx]
                predict_file = os.path.join(predict_folder,'%s.png'%name)
                gt_file = os.path.join(gt_folder,'%s.png'%name)
                predict = np.array(Image.open(predict_file)) #cv2.imread(predict_file)
                gt = np.array(Image.open(gt_file))
                cal = gt<255
                mask = (predict==gt) * cal
          
                for i in range(self.cfg.MODEL_NUM_CLASSES):
                    P[i].acquire()
                    P[i].value += np.sum((predict==i)*cal)
                    
                    P[i].release()
                    T[i].acquire()
                    T[i].value += np.sum((gt==i)*cal)
                    T[i].release()
                    TP[i].acquire()
                    TP[i].value += np.sum((gt==i)*mask)
                    TP[i].release()
        p_list = []
        for i in range(8):
            p = multiprocessing.Process(target=compare, args=(i,8,TP,P,T))
            p.start()
            p_list.append(p)
        for p in p_list:
            p.join()
        IoU = []
        for i in range(self.cfg.MODEL_NUM_CLASSES):
            IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
        for i in range(self.cfg.MODEL_NUM_CLASSES):
            if i == 0:
                print('%11s:%7.3f%%'%('backbound',IoU[i]*100),end='\t')
            else:
                if i%2 != 1:
                    print('%11s:%7.3f%%'%(self.categories[i-1],IoU[i]*100),end='\t')
                else:
                    print('%11s:%7.3f%%'%(self.categories[i-1],IoU[i]*100))
                    
        miou = np.mean(np.array(IoU))
        print('\n======================================================')
        print('%11s:%7.3f%%'%('mIoU',miou*100))    


    def __coco2voc(self, m):
        r,c = m.shape
        result = np.zeros((r,c),dtype=np.uint8)
        for i in range(0,21):
            for j in self.coco2voc[i]:
                result[m==j] = i
        return result




import torch
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import choice as npc
import numpy as np
import time
import random
import torchvision.datasets as dset
from PIL import Image
import json
import cv2

def get_data(data_json_file,img_path):
    #Return the data contain train and test, and their labels.
    data = []
	
    with open(data_json_file) as f:
        datastr = json.load(f)
        #print(datastr)
    for i in datastr:
        #print(i["image02"])
        d = {'image01': img_path+i["image01"], 'image02': img_path+i["image02"], 'label': i["label"]}
        
        data.append(d)
        # print(data)
    return data

class OmniglotTrain(Dataset):

    def __init__(self, jsonPath,img_path, transform=None):
        super(OmniglotTrain, self).__init__()
        np.random.seed(0)
        # self.dataset = dataset
        self.transform = transform
        
        self.datas, self.num_img = self.loadToMem(jsonPath,img_path)

    def loadToMem(self, jsonPath,img_path):
        print("begin loading training dataset to memory")
        data = {}
        agrees = [0, 90, 180, 270]
        idx = 0
        for agree in agrees:
            data_all = get_data(jsonPath,img_path)
            num_data=len(data_all)


        print("finish loading %s images training dataset to memory" %num_data)
        return data_all,num_data

    def __len__(self):
        # print(self.num_img)
        return  6400

    def __getitem__(self,index):
        # image1 = random.choice(self.dataset.imgs)
        label = None
        img1 = None
        img2 = None

        tmp_set=random.choice(self.datas)
       
        image1_ori=Image.open(tmp_set['image01'])
        image2_ori=Image.open(tmp_set['image02'])
        image1=image1_ori.resize((600,540))
        image2=image2_ori.resize((600,540))
        
        label=float(tmp_set['label'])

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
           
        image=torch.cat((image1,image2),0)

        return image, torch.from_numpy(np.array([label], dtype=np.float32))# all transform to tensor


class OmniglotTest(Dataset):

    def __init__(self, jsonPath,img_path, transform=None, times=200, way=20):
        np.random.seed(1)
        super(OmniglotTest, self).__init__()
        self.transform = transform
     
   
        self.datas, self.num_imgs = self.loadToMem(jsonPath,img_path)
       

    def loadToMem(self, jsonPath,img_path):
        print("begin loading test dataset to memory")
       
        

   
        data_all = get_data(jsonPath,img_path)
        num_data=len(data_all)
        print("finish loading %d images test dataset to memory"%num_data)
        return data_all, num_data

    def __len__(self):
        return self.num_imgs
    
    


    def getimg(self,index,save=False):
        # image1 = random.choice(self.dataset.imgs)
        label = None
        img1 = None
        img2 = None
        # get image from same class
        tmp_set=self.datas[index]
        image10=Image.open(tmp_set['image01'])
        image20=Image.open(tmp_set['image02'])
        image1=image10.resize((600,540))
        image2=image20.resize((600,540))
        if save:
            image10.save('/home/zhrj/Change_Dection/Sim_Train_Code_zrj/train_code_pse_spp/wrong/'+tmp_set['image01'].split('/')[-1])
            image20.save('/home/zhrj/Change_Dection/Sim_Train_Code_zrj/train_code_pse_spp/wrong/'+tmp_set['image02'].split('/')[-1])
        label=float(tmp_set['label'])

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        image=torch.cat((image1,image2),0)
        
        image=image.unsqueeze(0)
        
        # print('test zrj',image.size(),self.num_img)
        return image , torch.from_numpy(np.array([label], dtype=np.float32))

    

    # def getimg(self):


    # def __getitem__(self, index):
    #     idx = index % self.way
    #     label = None
    #     # generate image pair from same class
    #     if idx == 0:
    #         self.c1 = random.randint(0, self.num_classes - 1)
    #         self.img1 = random.choice(self.datas[self.c1])
    #         img2 = random.choice(self.datas[self.c1])
    #     # generate image pair from different class
    #     else:
    #         c2 = random.randint(0, self.num_classes - 1)
    #         while self.c1 == c2:
    #             c2 = random.randint(0, self.num_classes - 1)
    #         img2 = random.choice(self.datas[c2])

    #     if self.transform:
    #         img1 = self.transform(self.img1)
    #         img2 = self.transform(img2)
    #     return img1, img2


# test
if __name__=='__main__':
    omniglotTrain = OmniglotTrain('./images_background', 30000*8)
    print(omniglotTrain)

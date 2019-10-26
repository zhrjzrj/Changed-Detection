import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from spp_layer import spatial_pyramid_pool
# from Encoding import load_feature


class MyModule(nn.Module):

    def __init__(self,batch_size):
        super(MyModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 96, 7,stride=3),  # 96*20*20
            nn.ReLU(inplace=True), # 96*20*20
            nn.MaxPool2d(2),  # 96*10*10
            nn.Conv2d(96, 192, 5), # 192*6*6
            nn.ReLU(),  # 192*6*6
            nn.MaxPool2d(2),  # 192*3*3
            nn.Conv2d(192, 256, 3),
            nn.ReLU()  # 256*1*1           
        )
 
        self.batch_size=batch_size
        
        self.liner = nn.Sequential(            
            nn.Linear(20736,256),
            nn.ReLU()           
        )
        self.batch_size=batch_size
        self.out = nn.Linear(256, 1)

    def forward(self, x):

        x = self.conv1(x)
        output_num = [8,4,1]
        x_spp=spatial_pyramid_pool(x, self.batch_size, [int(x.size(2)),int(x.size(3))], output_num)
        x = self.liner(x_spp)
        x = self.out(x)

        return x
   


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss,self).__init__()
        # return

    def forward(self,output,label):
        # print('zzzzz',output.size(),label.size())
        zero=torch.zeros_like(output)
        tem_loss=1-output*label
        tmp=torch.cat((tem_loss,zero),1)
        max_tmp=torch.max(tmp,1)
        loss=torch.mean(max_tmp[0])
        return loss
       

   

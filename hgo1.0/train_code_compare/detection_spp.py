import torch
import pickle
import torchvision
from torchvision import transforms
import torchvision.datasets as dset
from torchvision import transforms
from mydataset_compare import OmniglotTrain, OmniglotTest
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model_2channel_compare import *
import time
import numpy as np
import gflags
import sys
from collections import deque
import os



Flags = gflags.FLAGS
gflags.DEFINE_string("test2_json_path", "/home/zhrj/Change_Dection/Sim_Train_Code_zrj/real_test/real_test.json", "training folder")
gflags.DEFINE_string("test2_img_path", "/home/zhrj/Change_Dection/Sim_Train_Code_zrj/real_test/img_cut/", 'path of testing folder')
gflags.DEFINE_string("test_json_path", "/home/zhrj/Change_Dection/Sim_Train_Code_zrj/Sim_upd0504_ch_Test.json", "training folder")
gflags.DEFINE_string("test_img_path", "/home/zhrj/Change_Dection/SimData01_update0504/", 'path of testing folder')
# gflags.DEFINE_string("model_path", "/home/zhrj/Change_Dection/Sim_Train_Code_zrj/model_spp/model-inter-25.pt", 'path of testing folder')
gflags.DEFINE_string("batch_size", 1, "number of batch size")
# gflags.DEFINE_string("train_json_path", "/home/zhrj/Change_Dection/Sim_Train_Code_zrj/Sim_upd0504_ch_Train.json", "training folder")
# gflags.DEFINE_string("train_img_path", "/home/zhrj/Change_Dection/SimData01_update0504/", 'path of testing folder')
gflags.DEFINE_integer("workers", 4, "number of dataLoader workers")
os.environ["CUDA_VISIBLE_DEVICES"] ='5'
def main(model_path,dis):
   
    data_transforms = transforms.Compose([
        # transforms.RandomAffine(15),
        transforms.ToTensor()
    ])


    testSet = OmniglotTest(Flags.test_json_path, Flags.test_img_path,transform=data_transforms)
  
    # torch.load(Flags.model_path)
    batch_size=Flags.batch_size
    net = MyModule(batch_size)
    print('load...')
    net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(model_path).items()})
    # weight=torch.load(model_path)
    # weight = nn.DataParallel(weight)
    # cudnn.benchmark = True
    # print('weight',weight)
    # input()
    
    print('get')
    

    wrong=0
    right=0
    result=0
    changed=0
    unchanged=0
    changed_pre=0.0
    unchanged_pre=0.0
    
    num=testSet.__len__()
    changed_right=0
    changed_wrong=0
    unchanged_right=0
    unchanged_wrong=0
    precision=0.0
    c_num=0
    for i in range(num):
        img=testSet.getimg(i)[0]
        
        label=testSet.getimg(i)[1]
        # print(img_diff.size(),img.size())
        detection=net.forward(img).data.cpu().numpy()
        # print(detection,label)
        # input()

        # print(label)
        if label<0:
            c_num+=1
            if detection<0:
                changed_right+=1
            else:
                changed_wrong+=1
            
        else:
            if detection>0:
                unchanged_right+=1
            else:
                unchanged_wrong+=1

  
    print('call back',changed_right,unchanged_right,changed_right/c_num)
        # if detection<dis:
        #     result=-1
            
        # else:
        #     result=1
        # if result==label:
            
        #     right+=1
        #     if result==-1:
        #         changed_pre+=1# 有变化的检测正确
        #     else:
        #         unchanged_pre+=1
        # else:
        #     wrong+=1
        #     img_error=testSet.getimg(i,save=True)[0]
        # if label==-1:#有变化的数量
        #     changed+=1
        # else:
        #     unchanged+=1
    



    # precison=right/num*100
    # ch_pre=changed_pre/changed*100
    # unch_pre=unchanged_pre/unchanged*100

    # ch_pre=round(ch_pre,2)
    # unch_pre=round(unch_pre,2)
    # precison=round(precison,2)
    
    
    # print('------------basic information----------------')
    # print('model',model_path)
    # print('num of data ',testSet.__len__())
    # print('num of wrong detection ',wrong)
    # print('num of right detection ',right)
    # print('changed img',changed,changed_pre)
    # print('unchanged img',unchanged,unchanged_pre)

    # print('------------the detection result--------------')
    # print('dis ',dis)
    # print('changed precision %s'%str(ch_pre)+'%')
    # print('unchanged precision %s'%str(unch_pre)+'%')
    # print('precision  %s '%str(precison)+'%')
    
if __name__=='__main__':
    # k=300
    # for _ in range(10):
    #     model_path="/home/zhrj/Change_Dection/Sim_Train_Code_zrj/train_code_pse_spp/model_spp/model-inter-"+str(k)+".pth"
    #     print(model_path,0)
    #     main(model_path,0)
    #     k+=5
    model_path="/home/zhrj/HgoNet/train_code_compare/model/model-inter-0.pth"
    # # # print('model:',model_path)
    # # # main(model_path)
    dis=0
    main(model_path,dis)
    # for _ in range(20):
    #     main(model_path,dis)
    #     dis+=1

    # test_data=testSet.getimg(1)[0]
    # print(testSet.__len__())
    # input()
    # test_data = Variable(test_data)
    # output = net.forward(test_data).data.cpu().numpy()
    # print(output)
    # if output<0:
    #     print('-1')
    # else:
    #     print('1')
    # input()
        # pred = np.argmax(output)

        # if pred == 0:
        #     right += 1
        # else: error += 1
    # print('*'*70)
    # print('[%d]\tTest set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f'%(batch_id, right, error, right*1.0/(right+error)))
    # print('*'*70)
    #         queue.append(right*1.0/(right+error))
    
    
    
    # img1=Image.open('/home/zhrj/Change_Dection/Sim_Train_Code_zrj/test1.jpg').convert('L')
    # img2=Image.open('/home/zhrj/Change_Dection/Sim_Train_Code_zrj/test2.jpg').convert('L')

    # print(the_model)

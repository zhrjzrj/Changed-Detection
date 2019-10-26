import torch
import pickle
import torchvision
from torchvision import transforms
import torchvision.datasets as dset
from torchvision import transforms
from mydataset_compare import OmniglotTrain
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


if __name__ == '__main__':

    Flags = gflags.FLAGS
    gflags.DEFINE_bool("cuda", True, "use cuda")
    # gflags.DEFINE_string("train_json_path", "/home/zhrj/Change_Dection/Sim_Train_Code_zrj/Sim_update_rotate_ch_Train.json", "training folder")
    
    gflags.DEFINE_string("train_json_path", "/home/zrj/HgoNet/data/HgoNet_data/ImageSets/train.json", "training folder")
    gflags.DEFINE_string("train_img_path", "/home/zrj/HgoNet/data/HgoNet_data/JPEGImages/", 'path of testing folder')

   
    gflags.DEFINE_integer("workers", 4, "number of dataLoader workers")
    gflags.DEFINE_integer("batch_size", 16, "number of batch size")
    gflags.DEFINE_float("lr", 0.01, "learning rate")
    gflags.DEFINE_integer("show_every", 10, "show result after each show_every iter.")
    gflags.DEFINE_integer("save_every", 5, "save model after each save_every iter.")
    gflags.DEFINE_string("epoch", 2000, "num of epoch")
    gflags.DEFINE_integer("max_iter", 50000, "number of iterations before stopping")
    gflags.DEFINE_string("model_path", "/home/zrj/train_code_compare/model", "path to store model")
    gflags.DEFINE_string("gpu_ids", "7", "gpu ids used to train")
    # gflags.DEFINE_string("premodel_path", "/home/zhrj/Change_Dection/Sim_Train_Code_zrj/train_code_pse/model_spp/model-inter-100.pth", 'path of testing folder')
    Flags(sys.argv)

    data_transforms = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor()
    ])
    


    os.environ["CUDA_VISIBLE_DEVICES"] = Flags.gpu_ids
    print("use gpu:", Flags.gpu_ids, "to train.")

    trainSet = OmniglotTrain(Flags.train_json_path, Flags.train_img_path,transform=data_transforms)


    trainLoader = DataLoader(trainSet, batch_size=Flags.batch_size, shuffle=False, num_workers=Flags.workers)

    loss_fn = Myloss()
    net = MyModule(batch_size=Flags.batch_size)
    # net.load_state_dict(torch.load(Flags.premodel_path))
    # net.loa/d_state_dict({k.replace('module.',''):v for k,v in torch.load(Flags.premodel_path).items()})
    # multi gpu
    if len(Flags.gpu_ids.split(",")) > 1:
        net = torch.nn.DataParallel(net)

    if Flags.cuda:
        net.cuda()

    net.train()

    optimizer = torch.optim.SGD(net.parameters(),lr = Flags.lr,momentum=0.9 )
    optimizer.zero_grad()

    train_loss = []
    loss_val = 0
    time_start = time.time()
    
    # print(trainLoader)
    # input()
    for  epoch in range(Flags.epoch):
        # print('net',net)
        for batch_id,(img,  label) in enumerate(trainLoader, 1):
            # print((img.size(), label))
            # input()
            if batch_id > Flags.max_iter:
                break
            if Flags.cuda:
                img, label = Variable(img.cuda()), Variable(label.cuda())
            else:
                img, label = Variable(img), Variable(label)
            optimizer.zero_grad()
            # print(img[0][0][270][300])
            output = net.forward(img)
            # print(output.size(),label.size())
            # input()loss_val/Flags.show_every
            # print(output)
            # print(label)
            loss = loss_fn(output, label)
            # print('loss',loss)
            loss_val += loss.item()
            loss.backward()
            optimizer.step()
            if batch_id % Flags.show_every == 0 :
                print('[%d]epoch\t [%d]loss: %.5f \ttime lapsed:%.2f s'%(epoch,batch_id, loss_val/Flags.show_every, time.time() - time_start))
                loss_val = 0
                time_start = time.time()
        if epoch % Flags.save_every == 0:
            # net.state_dict(net.state_dict(), Flags.model_path + '/model-inter-' + str(epoch) + ".pth")
            torch.save(net.state_dict(), Flags.model_path + '/model-inter-' + str(epoch) + ".pth")

            train_loss.append(loss_val)

    with open('train_loss', 'wb') as f:
        pickle.dump(train_loss, f)



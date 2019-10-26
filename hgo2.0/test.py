from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOC_ROOT, VOC_CLASSES as labelmap
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
import torch.utils.data as data
from ssd import build_ssd
import cv2

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_COCO_20000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()
# torch.cuda.set_device(4)
if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    
    filename = save_folder+'test_box.txt'
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img1,img2 ,name1,name2= testset.pull_image(i)
        print('img1 and img2: ',img1.shape,img2.shape)

        res1=cv2.imread(name1)
        res2=cv2.imread(name2)
    
        x1 = torch.from_numpy(transform(img1)[0]).permute(2, 0, 1)
        x1 = Variable(x1.unsqueeze(0))
        x2 = torch.from_numpy(transform(img2)[0]).permute(2, 0, 1)
        x2 = Variable(x2.unsqueeze(0))
        
        with open(filename, mode='a') as f:
            f.write('\nGROUND TRUTH FOR: '+name1+'  ' +name2+'\n')

        if cuda:
            x1 = x1.cuda()
            x2 = x2.cuda()

        y = net(x1,x2)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img1.shape[1], img1.shape[0],
                             img1.shape[1], img1.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            # print()
            j = 0
            flag=0
            while detections[0, i, j, 0] >= 0.85:
                print('get box')
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS: '+'\n')
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                # print('coords',coords)
                # input()
                with open(filename, mode='a') as f:
                    f.write(str(pred_num)+' label: '+label_name+' score: ' +
                            str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                            

                j += 1
                
                
                img1_box=cv2.rectangle(res1,(int(coords[0]),int(coords[2])),(int(coords[1]),int(coords[3])),(0,255,0),3)
                img2_box=cv2.rectangle(res2,(int(coords[0]),int(coords[2])),(int(coords[1]),int(coords[3])),(0,255,0),3)
                flag=1
            if flag:
                save1=cv2.imwrite('/home/zrj/Object_detection/hgo3.0/result_1021/'+str(i)+'_'+name1.split('/')[-1],img1_box)
                save2=cv2.imwrite('/home/zrj/Object_detection/hgo3.0/result_1021/'+str(i)+'_'+name2.split('/')[-1],img2_box)

def test_voc():
    # load net
    num_classes = 1 + 1 # +1 background
    print(num_classes)
    net = build_ssd('test', 300, num_classes) # initialize SSD

    model='/home/zrj/Object_detection/hgo3.0/weights/ssd3001020_COCO_5000.pth'
    net.load_state_dict(torch.load(model))

    net.eval()
    print('Finished loading model!')
 
    testset = VOCDetection(args.voc_root, [('2007', 'test')], None, VOCAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True


    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(500, (0,0,0)),
             thresh=args.visual_threshold)

if __name__ == '__main__':

    test_voc()

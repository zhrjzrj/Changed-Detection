"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import HOME
import os.path as osp
import os
from torchvision import transforms
import sys
import torch
import torch.utils.data as data
import cv2
import pandas as pd
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (  # always index 0
    'change')
# torch.cuda.set_device(4)
# note: if you used our download scripts, this should be right
VOC_ROOT = osp.join(HOME, "data/VOCdevkit/")


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target1, target2):
        # print('width,height',width, height)
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        # print('tar1,',target1)
        # print('tar2,',target2)
        
        bndbox1=self.getann(target1)
        bndbox2=self.getann(target2)
        if len(bndbox1)>0:
            res +=bndbox1
        if len(bndbox2)>0:
            res +=bndbox2
       
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]
    def getann(self,ann):
        
        res=[]
        
        if ann==500:
            return res
        # if ann2==500:
        for obj in ann.iter('size'):
            height=int(obj.find('height').text)
            width=int(obj.find('width').text)
        for obj in ann.iter('object'):
            name=obj.find('name').text
            bbox=obj.find('bndbox')
            pts=['xmin','ymin','xmax','ymax']
            bndbox = []
            for i,pt in enumerate(pts):
                cur_pt=int(bbox.find(pt).text)-1
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(float(cur_pt))
            bndbox.append(0)#1 to changed
            res += [bndbox]

        return res

class VOCDetection(data.Dataset):


    def __init__(self, root,
                 image_sets=[('2007', 'trainval')],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
    

        self.set_dir='/home/zrj/HgoNet3.0/data/cap_body'
        self.test_dir='/home/zrj/Object_detection/data/HgoNet_data/'
        self._imgpath=osp.join(self.set_dir,'JPEGImages','%s.jpg')
        self._testimgpath=osp.join(self.test_dir,'test_detection','%s.jpg')
        self._annopath=osp.join(self.set_dir,'Annotations','%s.xml')
        file_name = self.set_dir+'/'+'ImageSets'+'/'+'train.txt'
        df = pd.read_csv(file_name, names=['file_img1','file_img2','label'],sep=',')
        self.name_list_img1 = df['file_img1'].values
        self.name_list_img2 = df['file_img2'].values
        self.totensor=transforms.ToTensor()
        self.test_txt='/home/zrj/Object_detection/data/HgoNet_data/ImageSets/test_detection.txt'
        df_t = pd.read_csv(self.test_txt, names=['file_img1','file_img2','label'],sep=',')
        self.test_img1_list=df_t['file_img1'].values
        self.test_img2_list=df_t['file_img2'].values
    def __getitem__(self, index):
        img1_id=self.name_list_img1[index]
        img2_id=self.name_list_img2[index]

        if os.path.exists(self._annopath % img1_id):
            tmp_target1=ET.parse(self._annopath % img1_id).getroot()
        else:
            tmp_target1=500

        if os.path.exists(self._annopath % img2_id):
            tmp_target2=ET.parse(self._annopath % img2_id).getroot()
        else:
            tmp_target2=500

        img1=cv2.imread(self._imgpath % img1_id)
        img2=cv2.imread(self._imgpath % img2_id)
        img1=cv2.resize(img1, (500,500))
        img2=cv2.resize(img2, (500,500))
        # print('img1',img1.shape)
        # print('img2',img1.shape)
        height1, width1, channels1 = img1.shape
        height2, width2, channels2 = img2.shape

        if self.target_transform is not None:
            target = self.target_transform(tmp_target1, tmp_target2)
        # print('tar',len(target))
        

        # print('targggg',target)
        if self.transform is not None:
            target = np.array(target)
            tmp=[]
            # print('target',target.shape,img1.shape,img1_id,img2_id)
            # print()
            img1, boxes, labels = self.transform(img1, target[:, :4], target[:, 4])
            img2, _, _ = self.transform(img2,tmp, tmp)
            # to rgb
            img1 = img1[:, :, (2, 1, 0)]
            img2 = img2[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            img1=torch.from_numpy(img1).permute(2, 0, 1) 
            img2=torch.from_numpy(img2).permute(2, 0, 1)
        # print('???????????',img1.shape,img2.shape,type(target))
        return img1, img2,target

        # im1,im2, gt, h, w = self.pull_item(index)
        
        # return im1,im2,gt

    def __len__(self):
        return len(self.name_list_img1)








  
    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        print('index',index)
        # print('get img',self._testimgpath % img1_id,self._testimgpath % img2_id)
        img1_id=self.test_img1_list[index]
        img2_id=self.test_img2_list[index]
        print('get img',self._testimgpath % img1_id,self._testimgpath % img2_id)
        img1=cv2.imread(self._testimgpath % img1_id)
        img2=cv2.imread(self._testimgpath % img2_id)
        name1=self._testimgpath % img1_id
        name2=self._testimgpath % img2_id
        return img1,img2,name1,name2

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

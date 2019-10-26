# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import cv2
import numpy as np
import torch

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, is_continuous=False,fix=False):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size,output_size)
        else:
            self.output_size = output_size
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST
        self.fix = fix

    def __call__(self, sample):
        image1 = sample['image1']
        image2 = sample['image2']
        h, w = image1.shape[:2]
        if self.output_size == (h,w):
            return sample
            
        if self.fix:
            h_rate = self.output_size[0]/h
            w_rate = self.output_size[1]/w
            min_rate = h_rate if h_rate < w_rate else w_rate
            new_h = h * min_rate
            new_w = w * min_rate
        else: 
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img1 = cv2.resize(image1, dsize=(new_w,new_h), interpolation=cv2.INTER_CUBIC)
        img2 = cv2.resize(image2, dsize=(new_w,new_h), interpolation=cv2.INTER_CUBIC)

        
        top = (self.output_size[0] - new_h)//2
        bottom = self.output_size[0] - new_h - top
        left = (self.output_size[1] - new_w)//2
        right = self.output_size[1] - new_w - left
        if self.fix:
            img1 = cv2.copyMakeBorder(img1,top,bottom,left,right, cv2.BORDER_CONSTANT, value=[0,0,0]) 
            img2 = cv2.copyMakeBorder(img2,top,bottom,left,right, cv2.BORDER_CONSTANT, value=[0,0,0])   

        if 'segmentation1' in sample.keys():
            segmentation1 = sample['segmentation1'] 
            segmentation2 = sample['segmentation1'] 
            seg1 = cv2.resize(segmentation1, dsize=(new_w,new_h), interpolation=self.seg_interpolation)
            seg2 = cv2.resize(segmentation2, dsize=(new_w,new_h), interpolation=self.seg_interpolation)
            if self.fix:
                seg1 = cv2.copyMakeBorder(seg1,top,bottom,left,right, cv2.BORDER_CONSTANT, value=[0])
                seg2 = cv2.copyMakeBorder(seg2,top,bottom,left,right, cv2.BORDER_CONSTANT, value=[0])
            sample['segmentation1'] = seg1
            sample['segmentation2'] = seg2
        sample['image1'] = img1
        sample['image2'] = img2
        return sample

class Centerlize(object):
    def __init__(self, output_size, is_continuous=False):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        if self.output_size == (h,w):
            return sample

        if isinstance(self.output_size, int):
            new_h = self.output_size
            new_w = self.output_size
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        
        top = (new_h - h) // 2  
        bottom = new_h - h - top
        left = (new_w - w) // 2
        right = new_w - w -left
        img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])   
        if 'segmentation' in sample.keys():
            segmentation = sample['segmentation'] 
            seg=cv2.copyMakeBorder(segmentation,top,bottom,left,right,cv2.BORDER_CONSTANT,value=[0])
            sample['segmentation'] = seg
        sample['image'] = img
        
        return sample
                     
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        new_h = h if new_h >= h else new_h
        new_w = w if new_w >= w else new_w

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        image = image[top: top + new_h,
                      left: left + new_w]

        segmentation = segmentation[top: top + new_h,
                      left: left + new_w]
        sample['image'] = image
        sample['segmentation'] = segmentation
        return sample
class RandomHSV(object):
    """Generate randomly the image in hsv space."""
    def __init__(self, h_r, s_r, v_r):
        self.h_r = h_r
        self.s_r = s_r
        self.v_r = v_r

    def __call__(self, sample):
        image = sample['image']
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h = hsv[:,:,0].astype(np.int32)
        s = hsv[:,:,1].astype(np.int32)
        v = hsv[:,:,2].astype(np.int32)
        delta_h = np.random.randint(-self.h_r,self.h_r)
        delta_s = np.random.randint(-self.s_r,self.s_r)
        delta_v = np.random.randint(-self.v_r,self.v_r)
        h = (h + delta_h)%180
        s = s + delta_s
        s[s>255] = 255
        s[s<0] = 0
        v = v + delta_v
        v[v>255] = 255
        v[v<0] = 0
        hsv = np.stack([h,s,v], axis=-1).astype(np.uint8)	
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.uint8)
        sample['image'] = image
        return sample

class RandomFlip(object):
    """Randomly flip image"""
    def __init__(self, threshold):
        self.flip_t = threshold
    def __call__(self, sample):
        image1, image2, segmentation1,segmentation2 = sample['image1'], sample['image2'], sample['segmentation1'], sample['segmentation2']
        if np.random.rand() < self.flip_t:
            image1_flip = np.flip(image1, axis=1)
            image2_flip = np.flip(image2, axis=1)
            segmentation1_flip = np.flip(segmentation1, axis=1)
            segmentation2_flip = np.flip(segmentation2, axis=1)
            sample['image1'] = image1_flip
            sample['image2'] = image2_flip
            sample['segmentation1'] = segmentation1_flip
            sample['segmentation2'] = segmentation2_flip
        return sample

class RandomRotation(object):
    """Randomly rotate image"""
    def __init__(self, angle_r, is_continuous=False):
        self.angle_r = angle_r
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST

    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']
        row, col, _ = image.shape
        rand_angle = np.random.randint(-self.angle_r, self.angle_r) if self.angle_r != 0 else 0
        m = cv2.getRotationMatrix2D(center=(col/2, row/2), angle=rand_angle, scale=1)
        new_image = cv2.warpAffine(image, m, (col,row), flags=cv2.INTER_CUBIC, borderValue=0)
        new_segmentation = cv2.warpAffine(segmentation, m, (col,row), flags=self.seg_interpolation, borderValue=0)
        sample['image'] = new_image
        sample['segmentation'] = new_segmentation
        return sample

class RandomScale(object):
    """Randomly scale image"""
    def __init__(self, scale_r, is_continuous=False):
        self.scale_r = scale_r
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST

    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']
        row, col, _ = image.shape
        rand_scale = np.random.rand()*(self.scale_r - 1/self.scale_r) + 1/self.scale_r
        img = cv2.resize(image, None, fx=rand_scale, fy=rand_scale, interpolation=cv2.INTER_CUBIC)
        seg = cv2.resize(segmentation, None, fx=rand_scale, fy=rand_scale, interpolation=self.seg_interpolation)
        sample['image'] = img
        sample['segmentation'] = seg
        return sample

class Multiscale(object):
    def __init__(self, rate_list):
        self.rate_list = rate_list

    def __call__(self, sample):
        image1 = sample['image1']
        image2 = sample['image2']
        row, col, _ = image1.shape
        image_multiscale = []
        for rate in self.rate_list:
            rescaled_image1 = cv2.resize(image1, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
            rescaled_image2 = cv2.resize(image2, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
            sample['image1_%f'%rate] = rescaled_image1
            sample['image2_%f'%rate] = rescaled_image2
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        key_list = sample.keys()
        for key in key_list:
            if 'image1' in key:
                image1 = sample[key]
                # swap color axis because
                # numpy image: H x W x C
                # torch image: C X H X W
                image1 = image1.transpose((2,0,1))
                sample[key] = torch.from_numpy(image1.astype(np.float32)/255.0)

            elif 'image2' in key:
                image2 = sample[key]       
                image2 = image2.transpose((2,0,1))
                sample[key] = torch.from_numpy(image2.astype(np.float32)/255.0)
                
            elif 'segmentation1' == key:
                segmentation1 = sample['segmentation1']
                sample['segmentation1'] = torch.from_numpy(segmentation1.astype(np.float32))
            elif 'segmentation1' == key:
                segmentation2 = sample['segmentation2']
                sample['segmentation2'] = torch.from_numpy(segmentation2.astype(np.float32))

            elif 'segmentation1_onehot' == key:
                onehot1 = sample['segmentation1_onehot'].transpose((2,0,1))
                sample['segmentation1_onehot'] = torch.from_numpy(onehot1.astype(np.float32))
            elif 'segmentation2_onehot' == key:
                onehot2 = sample['segmentation2_onehot'].transpose((2,0,1))
                sample['segmentation2_onehot'] = torch.from_numpy(onehot2.astype(np.float32))  

            elif 'mask2' == key:
                mask1 = sample['mask1']
                sample['mask1'] = torch.from_numpy(mask1.astype(np.float32))
            elif 'mask2' == key:
                mask2 = sample['mask2']
                sample['mask2'] = torch.from_numpy(mask2.astype(np.float32))

        return sample

def onehot(label, num):
    # print('onehot',type(label),label.shape,type(num),num)
    m = label
    # print(num,type(m))
    one_hot = np.eye(num)[m]
    # print('???',one_hot.shape)
    return one_hot

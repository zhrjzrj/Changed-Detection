# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp

# torch.cuda.set_device(4)
class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos #3
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']
        # print('================================================================================================self.variance',self.variance)

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        # print('loss compute targets',len(targets))
        # print('loss compute targets 2',targets)
        # input()
        loc_data, conf_data, priors = predictions
        # print('input',loc_data.shape,conf_data.shape,priors.shape)
        #位置，分类，和预设框
        #batch size的数量
        num = loc_data.size(0)
        #框的数量
        priors = priors[:loc_data.size(1), :]
        #取一下预设框，没看懂。产生预设框也没有多余的部分啊
        num_priors = (priors.size(0))
        #预设框的数量
        num_classes = self.num_classes
        #类的数量

        #将预设框和gt框匹配
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        #truth是gt，坐标和label（最后一位）
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            # print('truths',truths.shape)
            # print('defaults',defaults.shape)
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()


        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        
        
        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)
        
        #对置信度>0的部分统计，看看有几个能用的预测框是positive的框
        
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        
        #将置信度>0的预测框取出来放在loc-p中，以及这些预测框对应的gt框放在loc-t中
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        # print('loc_p',/loc_p.shape)
        
        # print('loc_t',loc_t.shape)
        # input()
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
        # print('loss_l',loss_l)
        # Compute max conf across batch for hard negative mining
        #先对预测的置信度重分一下
        
        batch_conf = conf_data.view(-1, self.num_classes)
        
        a=conf_t.view(-1, 1)
        
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        
        #pos是confi＞0的部分的idx
        loss_c = loss_c.view(num, -1)
        
        loss_c[pos] = 0  # filter out pos boxes for now
        
        # print('loss_c',loss_c.shape)
        
        #两次sort可以输出loss_c的从大到小的排序映射，即给loss_c贴了顺序标签
        _, loss_idx = loss_c.sort(1, descending=True)
        # print('loss_idx',loss_idx.shape)
        _, idx_rank = loss_idx.sort(1)
        # print('idx_rank',idx_rank.shape)
        #pos和neg的数量控制一下，控制在1：3
        num_pos = pos.long().sum(1, keepdim=True)
        # print('num_pos',pos.size(1)-1)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        # print('num_neg',num_neg)
        #idx_rank表示loss_c的排序映射
        num_neg.expand_as(idx_rank)
        
        neg = idx_rank < num_neg.expand_as(idx_rank)
        #只选top neg数量的框的id
        
        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # print('pos_idx,neg_idx',pos_idx.shape,neg_idx.shape)
        
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        # print('conf_p',conf_p.shape)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        # print('targets_weighted',targets_weighted.shape)
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
        # print('loss_c',loss_c)
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        # print('N',N)
        loss_l /= N
        loss_c /= N
        
        return loss_l, loss_c

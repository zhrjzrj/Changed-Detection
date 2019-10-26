#=========================================
# Written by Ruijie Zhang
#=========================================
import torch
import torch.nn as nn

class HgoLoss(nn.Module):
    def __init__(self):
        super(HgoLoss,self).__init__()
        

    def forward(self,output,label_o):
        # print('loss',output.shape,label.shape)
        label=label_o.unsqueeze(1)
        # print('loss',output.shape,label.shape)
        zero=torch.zeros_like(output)
        tem_loss=1-output*label
        tmp=torch.cat((tem_loss,zero),1)
        max_tmp=torch.max(tmp,1)
        loss=torch.mean(max_tmp[0])
        return loss
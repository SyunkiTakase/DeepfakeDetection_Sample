import torch
from torch import nn
import torchvision
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
from utils.sam import SAM
import timm
from timm.models import create_model
import torchvision.models as models


class Detector(nn.Module):

    def __init__(self):
        super(Detector, self).__init__()
        self.net=models.resnet50(pretrained=True)
        self.net.fc = torch.nn.Linear(self.net.fc.in_features, 2)
        # self.net=create_model('resnet50.tv_in1k',pretrained=True,num_class=2)
        self.cel=nn.CrossEntropyLoss()
        self.optimizer=SAM(self.parameters(),torch.optim.SGD,lr=0.001, momentum=0.9)
        
        

    def forward(self,x):
        x=self.net(x)
        return x
    
    def training_step(self,x,target):
        for i in range(2):
            pred_cls=self(x)
            if i==0:
                pred_first=pred_cls
            loss_cls=self.cel(pred_cls,target)
            loss=loss_cls
            self.optimizer.zero_grad()
            loss.backward()
            if i==0:
                self.optimizer.first_step(zero_grad=True)
            else:
                self.optimizer.second_step(zero_grad=True)
        
        return pred_first
    
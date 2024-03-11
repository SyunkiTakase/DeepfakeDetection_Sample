import torch
from torch import nn
import torchvision
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
import timm
from timm.models import create_model
import torchvision.models as models


class Detector(nn.Module):

    def __init__(self):
        super(Detector, self).__init__()
        self.net=create_model('xception',pretrained=True,num_classes=2)
        self.cel=nn.CrossEntropyLoss()        
        

    def forward(self,x):
        x=self.net(x)
        return x
    

    
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
        self.net=models.resnet50(pretrained=True)
        self.net.fc = torch.nn.Linear(self.net.fc.in_features, 2)
        # self.net=create_model('resnet50.tv_in1k',pretrained=True,num_class=2)
        

    def forward(self,x):
        x=self.net(x)
        return x
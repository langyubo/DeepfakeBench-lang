'''
Description: 
Author: lang_yubo
Date: 2023-04-13 22:33:57
LastEditTime: 2023-05-04 10:17:47
LastEditors: lang_yubo
'''
import logging

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import models

from functools import partial
from math import exp
import timm
from timm.loss import LabelSmoothingCrossEntropy # This is better than normal nn.CrossEntropyLoss
from timm.models.vision_transformer import VisionTransformer, _cfg

from utils.registry import BACKBONE

logger = logging.getLogger(__name__)


@BACKBONE.register_module(module_name="vit")
class ViT(nn.Module):
    def __init__(self, cfftrans_config):
        """Vision Transformer with depth supervision

        Args:
            num_class (_type_): 分类数量
            output_img_results (bool, optional): 是否输出模型的图像结果. Defaults to False.
        """        
        super(ViT, self).__init__()
        self.num_class = cfftrans_config["num_classes"]
        self.embed_dim = cfftrans_config["embed_dim"]
        self.model_type = cfftrans_config['model_type']  # deit_tiny(embed_dim:192,depth=12), deit_base(embed_dim:768,depth=12), cait(embed_dim:384,depth=24)
        self.pretrained = cfftrans_config['backbone_pretrained']

        if self.model_type == 'deit_tiny':
            self.embed_dim = 192
            self.model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=self.pretrained)
        elif self.model_type == 'deit_base':
            self.embed_dim = 768
            self.model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=self.pretrained)
        elif self.model_type == 'cait':
            self.embed_dim = 384
            self.model = torch.hub.load('facebookresearch/deit:main', 'cait_S24_224', pretrained=self.pretrained)
        else:
            raise NotImplementedError('Vit model {} is not implemented, you can only set Vit in [deit_tiny, deit_base, cait]'.format(self.model_type))
        
        if self.pretrained:
                logger.info('Load pretrained model successfully!')

        input_dim = self.model.head.in_features
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))    #移除原模型最后一层lc

        self.relu = nn.ReLU(inplace=True)
        self.last_linear = nn.Linear(input_dim, self.num_class)

    def features(self, input):
        x = self.model(input)   
        return x   # features: 192 dim  (bs,embed_dim=192)  

    def classifier(self,features):
        # x = self.relu(features)
        x = features[:,0,:]
        x = x.view(x.size(0), -1)
        out = self.last_linear(x)
        return out

    def forward(self, input):
        x = self.features(input)
        out = self.classifier(x)
        return out, x
    

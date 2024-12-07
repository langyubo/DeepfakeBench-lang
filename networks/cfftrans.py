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


def get_deit_model(num_class):
    model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)

    for param in model.parameters(): #freeze model
        param.requires_grad = False

    n_inputs = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(n_inputs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_class)
    )
    return model
    # model = model.to(device)
    # print(model)

class TransAD(nn.Module):   # TransAD(anti deepfakes)
    def __init__(self, num_class):
        super(TransAD, self).__init__()
        self.num_class = num_class
        self.model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
        self.forward_features = self.model.forward_features

        n_inputs = self.model.head.in_features
        self.head = nn.Sequential(
        nn.Linear(n_inputs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, self.num_class)
        )

        self.patch_fc = nn.Linear(192, 2)
        self.dp = nn.Dropout(0.6)

    def forward(self, x):
        features = self.forward_features(x)    #bs 197((224/16)*(224/16)+cls token),192
        out = self.head(features[:,0])   
        patch_feature = features[:,1:,:]  # bs, 196,192
        patch_out = self.patch_fc(self.dp(patch_feature.contiguous().view(-1, 192)))
        return out, patch_out


class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB( self.convB( self.leakyreluA(self.convA( torch.cat([up_x, concat_with], dim=1) ) ) )  )


class UpSample_v2(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample_v2, self).__init__()
        self.pconvA = nn.Conv2d(skip_input, output_features, kernel_size=1)
        self.depthWiseConv = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1,groups=output_features)
        self.layerNorm = nn.LayerNorm((28,28))
        self.pconvB = nn.Conv2d(output_features, output_features, kernel_size=1)
        self.gelu = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)  #bs,c,h,w
        return self.gelu(self.pconvB(self.layerNorm( self.depthWiseConv(self.pconvA( torch.cat([up_x, concat_with], dim=1) ) ) ) ) )


class UpSample_only(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample_only, self).__init__()
        self.pconvA = nn.Conv2d(skip_input, output_features, kernel_size=1)
        self.depthWiseConv = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1,groups=output_features)
        self.layerNorm = nn.LayerNorm((28,28))
        self.relu = nn.LeakyReLU(0.2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.eca_layer = eca_layer(192)   #channel attention

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)  #bs,c,h,w
        #avgpool之后的尺寸为 [bs, 192, 1]
        return self.relu(self.layerNorm( self.depthWiseConv(self.pconvA( torch.cat([up_x, concat_with], dim=1) ) ) ) ) , self.avgpool(self.eca_layer(up_x+concat_with)).squeeze()

class UpSample_v3(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample_v3, self).__init__()
        self.pconvA = nn.Conv2d(skip_input, output_features, kernel_size=1)
        self.depthWiseConv = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1,groups=output_features)
        self.layerNorm = nn.LayerNorm((28,28))
        self.relu = nn.LeakyReLU(0.2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)  #bs,c,h,w
        #avgpool之后的尺寸为 [bs, 192, 1]
        return self.relu(self.layerNorm( self.depthWiseConv(self.pconvA( torch.cat([up_x, concat_with], dim=1) ) ) ) )



class SpatialAttention(nn.Module):
    """空间注意力
    Args:
        nn (_type_): _description_
    """    
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x.size() 30,40,50,30
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 30,1,50,30
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 30,1,50,30
        return self.sigmoid(x)  # 30,1,50,30


class eca_layer(nn.Module):  #通道注意力
    """Constructs a ECA module.
        ECA-Net: Efficient Channel Attention
        https://github.com/BangguWu/ECANet/blob/b332f6b3e6e2afe8a3287dc8ee8440a0fbec74c4/models/eca_module.py#L5
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)




# https://github.com/Mikoto10032/AutomaticWeightedLoss
# A PyTorch implementation of Liebel L, Körner M. Auxiliary tasks in multi-task learning[J]. arXiv preprint arXiv:1805.06334, 2018.
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        # params = torch.tensor([0.75,0.5,0.2], requires_grad = True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum



class CrossLayerFeatureFusion(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super(CrossLayerFeatureFusion, self).__init__()
        
        self.linear_q = nn.Linear(d_model, dim_feedforward)
        self.linear_k = nn.Linear(d_model, dim_feedforward)
        self.linear_v = nn.Linear(d_model, dim_feedforward)

        self.self_attn = nn.MultiheadAttention(dim_feedforward, nhead)
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(dim_feedforward, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(dim_feedforward)
        self.norm2 = nn.LayerNorm(dim_feedforward)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, lower_feature, upper_feature):
        q = self.linear_q(lower_feature)
        k = self.linear_k(lower_feature)
        v = self.linear_v(upper_feature)
        
        attn_output, _ = self.self_attn(q, k, v)  
    
        out_feature = v + attn_output
        out_feature = self.norm1(out_feature)

        mlp_out_feature = self.dropout(self.linear1(out_feature))
        out_feature = out_feature + mlp_out_feature
        out_feature = self.norm2(out_feature)
        out_feature = self.linear2(out_feature)

        return out_feature


@BACKBONE.register_module(module_name="cfftrans")
class CffTrans(nn.Module):   # TransAD(anti deepfakes)
    def __init__(self, cfftrans_config):
        """Vision Transformer with depth supervision

        Args:
            num_class (_type_): 分类数量
            output_img_results (bool, optional): 是否输出模型的图像结果. Defaults to False.
        """        
        super(CffTrans, self).__init__()

        self.with_MHA = cfftrans_config["with_MHA"]
        self.with_depth = cfftrans_config["with_depth"]
        self.output_img_results = cfftrans_config["output_img_results"]
        self.num_class = cfftrans_config["num_classes"]
        self.alpha = cfftrans_config["alpha"]   # The Ratio Between Binary and Depth Supervisions
        self.omega = cfftrans_config["omega"]  # The Depth Supervisions 的放大倍率
        self.embed_dim = cfftrans_config["embed_dim"]
        self.embed_len = cfftrans_config["embed_len"]

        self.model = VisionTransformer(
            patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        # load checkpoint (这部分功能转移到建立detector的代码里了)
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
        #     map_location="cpu", check_hash=True
        # )
        # self.model.load_state_dict(checkpoint["model"])

        self.patch_embed = self.model.patch_embed
        self.pos_drop = self.model.pos_drop
        self.norm_pre = self.model.norm_pre
        self.transformer_encoders = self.model.blocks[:12]

        self.spa1 = SpatialAttention()
        self.spa2 = SpatialAttention()
        self.upsample = UpSample_only(384, 64)
        self.conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
        
        self.gap = nn.AdaptiveAvgPool1d(1)

        n_inputs = self.model.head.in_features
        
        if self.with_depth:
            self.clssification_head = nn.Sequential(
            nn.Linear(n_inputs+196, n_inputs),
            nn.BatchNorm1d(n_inputs),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(n_inputs, 2),
            )
        else:
            self.clssification_head = nn.Sequential(
            nn.Linear(n_inputs, n_inputs),
            nn.BatchNorm1d(n_inputs),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(n_inputs, 2),
            )
        self.patch_fc = nn.Linear(192, 2)
        self.patch_relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.6)

        self.conv1d = nn.Conv1d(192*3, 8, 1)
        
        self.cross_layer_fusion = CrossLayerFeatureFusion(192,4,256)
        self.depth_head = nn.Sequential(
        nn.Conv2d(192, 64, 3, 1, 1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        nn.Dropout(0.2),
        nn.Conv2d(64, 1, 1),
        nn.Sigmoid()
        )

    def _pos_embed(self, x):
        x = torch.cat((self.model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.model.pos_embed
        return self.pos_drop(x)
    
    def features_and_classifier(self, input):
        bs = input.shape[0]  # batch size
        x = self.patch_embed(input)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        # for i in range(len(self.model.blocks)):
        for i in range(12):
            x = self.transformer_encoders[i](x)
            if i == 0:
                cls_token0 = x[:,0]
                feature0 = x[:,1:,:]  #bs, 196, 192   except cls_token of layer0's output
            if i == 6:
                cls_token5 = x[:,0]
                feature5 = x[:,1:,:]  #bs, 196, 192   except cls_token of layer6's output
        feature12 = x[:,1:,:]         #bs, 196, 192   except cls_token of layer12's output

        # calculate cross layer features
        if self.with_MHA:
            cff_features = self.cross_layer_fusion(feature0, feature5)
            cff_features = self.cross_layer_fusion(cff_features, feature12) # bs, 196, 192
        else:
            cff_features = feature12

        depth_feature = cff_features

        middle_cls_feature = self.gap(cff_features).squeeze() #bs, 196
        cls_feature  = x[:,0,:]                                     #bs 197((224/16)*(224/16)+cls token),192    

        if self.with_depth:
            mid_feature = torch.cat([cls_feature, middle_cls_feature], dim=1)
            for i in range(len(self.clssification_head)):
                mid_feature = self.clssification_head[i](mid_feature)
                if i == 2:
                    out_features = mid_feature
            out = mid_feature
        else:

            mid_feature = cls_feature
            for i in range(len(self.clssification_head)):
                mid_feature = self.clssification_head[i](mid_feature)
                if i == 2:
                    out_features = mid_feature
            out = mid_feature

        return depth_feature, out_features, out   #用于深度图估计的features， 网络输出features， 分类结果

    def depth_map(self, depth_feature_input):
        bs = depth_feature_input.shape[0]  # batch size
        depth_feature_input = depth_feature_input.transpose(1,2).view(bs, -1, 14, 14)
        depth_feature_input = F.interpolate(depth_feature_input, size=[14*2, 14*2], mode='bilinear', align_corners=True)  #bs, 192, 28, 28

        dm = self.depth_head(depth_feature_input).squeeze()  #bs, 28, 28 
        return dm

    def forward(self, input):
        depth_feature, out_features, out = self.features_and_classifier(input)
        depth_map = self.depth_map(depth_feature)
        return out_features, out, depth_feature, depth_map
        


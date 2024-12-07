'''
# author: Yubo Lang
# email: lang_yubo@163.com
# date: 2024-0807
# description: Class for the CFF trans

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation

Reference:
@inproceedings{rossler2019faceforensics++,
  title={Faceforensics++: Learning to detect manipulated facial images},
  author={Rossler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Thies, Justus and Nie{\ss}ner, Matthias},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={1--11},
  year={2019}
}
'''

import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='cfftrans')
class CfftransDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        self.alpha = config['backbone_config']['alpha']
        self.omega = config['backbone_config']['omega']
        
    def build_backbone(self, config):
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)
        # if donot load the pretrained weights, fail to get good results
        state_dict = torch.load(config['pretrained'])
        backbone.model.load_state_dict(state_dict["model"])
        logger.info('Load pretrained model successfully!')
        return backbone
    
    def build_loss(self, config):
        cls_loss_class = LOSSFUNC[config['loss_func']['cls_loss']]
        depth_loss_class = LOSSFUNC[config['loss_func']['depth_loss']]
        cls_loss_func = cls_loss_class()
        dep_loss_func = depth_loss_class()

        loss_func = {
            'cls': cls_loss_func, 
            'dep': dep_loss_func,
        }
        return loss_func
    
    def outputs(self, data_dict: dict) -> torch.tensor:
        return self.backbone.forward(data_dict['image']) #32,3,256,256
    
    def features(self, data_dict: dict) -> torch.tensor:
        return self.backbone.forward(data_dict['image']) #32,3,256,256

    def classifier(self, data_dict: torch.tensor) -> torch.tensor:
        return self.backbone.forward(data_dict['image']) #32,3,256,256

    def get_losses(self, data_dict: dict, pred_dict: dict, inference=False) -> dict:
        if not inference:
            return self.get_train_losses(data_dict, pred_dict)
        else:  # inference
            return self.get_test_losses(data_dict, pred_dict)
    
    def get_train_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        # get label
        label = data_dict['label']
        label_depth = data_dict['depth_map']
        # get pred
        pred = pred_dict['cls']
        pred_depth = pred_dict['pred_depth']

        # 1. classification loss for common features
        loss_cls = self.loss_func['cls'](pred, label)
        
        # 2. depth map loss
        loss_depth = self.loss_func['dep'](pred_depth, label_depth)

        # 3. total loss
        loss = self.alpha*loss_cls + self.omega*(1-self.alpha)*loss_depth
        loss_dict = {
            'overall': loss,
            'cls': loss_cls,
            'depth': loss_depth,
        }
        return loss_dict
    
    def get_test_losses(self, data_dict: dict, pred_dict: dict) -> dict:
            # get label
            label = data_dict['label']
            # get pred
            pred = pred_dict['cls']
            # for test mode, only classification loss for common features
            loss = self.loss_func['cls'](pred, label)
            loss_dict = {'common': loss}
            return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict
    
    def get_test_metrics(self):
        y_pred = np.concatenate(self.prob)
        y_true = np.concatenate(self.label)
        # auc
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # eer
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # ap
        ap = metrics.average_precision_score(y_true,y_pred)
        # acc
        acc = self.correct / self.total
        # reset the prob and label
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        return {'acc':acc, 'auc':auc, 'eer':eer, 'ap':ap, 'pred':y_pred, 'label':y_true}

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features and the prediction by backbone
        out_features, out_cls, depth_features, pred_depth = self.outputs(data_dict)
        # get the probability of the pred
        prob = torch.softmax(out_cls, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': out_cls, 'prob': prob, 'feat': out_features, 'pred_depth':pred_depth}
        if inference:
            self.prob.append(
                pred_dict['prob']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            self.label.append(
                data_dict['label']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            # deal with acc
            _, prediction_class = torch.max(out_cls, 1)
            correct = (prediction_class == data_dict['label']).sum().item()
            self.correct += correct
            self.total += data_dict['label'].size(0)
        return pred_dict


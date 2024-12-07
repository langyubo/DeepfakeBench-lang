"""
eval pretained model.
"""
import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torchvision import transforms as T

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.ff_blend import FFBlendDataset
from dataset.fwa_blend import FWABlendDataset
from dataset.pair_dataset import pairDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from logger import create_logger

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='/home/diode/mnt/DeepfakeBench-main/training/config/detector/xception.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_img_path", type=str,
                    default='/home/diode/mnt/DeepfakeBench-main/test_one_imgs/2.png')
parser.add_argument('--weights_path', type=str, 
                    default='/home/diode/mnt/DeepfakeBench-main/training/pretrained/xception_best.pth')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(img_path, config):
    def to_tensor(img):
        """
        Convert an image to a PyTorch tensor.
        """
        return T.ToTensor()(img)

    def normalize(img, config):
        """
        Normalize an image.
        """
        mean = config['mean']
        std = config['std']
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)
    
    size = config['resolution']
    img = cv2.imread(img_path)
    if img is None: 
        raise ValueError('Loaded image is None: {}'.format(img_path))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    image_trans = np.array(img, dtype=np.uint8)
    image_trans = normalize(to_tensor(image_trans), config)
    image_trans = torch.unsqueeze(image_trans, dim=0)  #扩维成[bs c w h]形式
    return image_trans


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring

def test_one_image(model, img_tensor):
    data_dict = {}
    data_dict['mask'], data_dict['landmark'] = None, None
    label = torch.Tensor([[1]])
    data_dict['image'], data_dict['label'] = img_tensor.to(device), label.to(device)
    
    # model forward without considering gradient computation
    predictions = inference(model, data_dict)

    return predictions

def test_one_dataset(model, data_loader):
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
    
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)

        # deal with the feat, pooling if needed
        if len(predictions['feat'].shape) == 4:
            predictions['feat'] = F.adaptive_avg_pool2d(predictions['feat'], (1, 1)).reshape(predictions['feat'].shape[0], -1)
        predictions['feat'] = predictions['feat'].cpu().numpy()
    
    return predictions
    
def test_epoch(model, test_data_loaders):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        # compute loss for each dataset
        predictions = test_one_dataset(model, test_data_loaders[key])
        
        # compute metric for each dataset
        metric_one_dataset = model.get_test_metrics()
        metrics_all_datasets[key] = metric_one_dataset
        
        # info for each dataset
        tqdm.write(f"dataset: {key}")
        for k, v in metric_one_dataset.items():
            tqdm.write(f"{k}: {v}")

    return metrics_all_datasets

def test_epoch_on_oneimg(model, img_tensor):
    # set model to eval mode
    model.eval()

    prob = test_one_image(model, img_tensor)

    return prob

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


# def main():
    # # parse options and load config
    # with open(args.detector_path, 'r') as f:
    #     config = yaml.safe_load(f)

    # weights_path = None
    # # If arguments are provided, they will overwrite the yaml settings
    # if args.test_dataset:
    #     config['test_dataset'] = args.test_dataset
    # if args.weights_path:
    #     config['weights_path'] = args.weights_path
    #     weights_path = args.weights_path
    
    # # init seed
    # init_seed(config)

    # # set cudnn benchmark if needed
    # if config['cudnn']:
    #     cudnn.benchmark = True

    # # prepare the testing data loader
    # test_img_tensor = prepare_testing_data(config)
    
    # # prepare the model (detector)
    # model_class = DETECTOR[config['model_name']]
    # model = model_class(config).to(device)
    # epoch = 0
    # if weights_path:
    #     try:
    #         epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
    #     except:
    #         epoch = 0
    #     ckpt = torch.load(weights_path, map_location=device)
    #     model.load_state_dict(ckpt, strict=True)
    #     print('===> Load checkpoint done!')
    # else:
    #     print('Fail to load the pre-trained weights')
    
    # # start testing
    # best_metric = test_epoch(model, test_data_loaders)
    # print('===> Test Done!')

def main_on_oneimg():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)

    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    # if args.test_dataset:
    #     config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path

    test_imgpath = args.test_img_path

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_img_tensor = prepare_testing_data(test_imgpath, config)
    
    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    epoch = 0
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')
    
    # start testing
    test_prob = test_epoch_on_oneimg(model, test_img_tensor)
    prob = test_prob['prob']
    pred_cls = test_prob['cls']
    print(f'prob = {prob}, cls = {pred_cls}')
    print('===> Test Done!')

if __name__ == '__main__':
    main_on_oneimg()
    # img_path = 'test_one_imgs/1.png'
    # with open(args.detector_path, 'r') as f:
    #     config = yaml.safe_load(f)
    # prepare_testing_data(img_path, config)

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
import dlib
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from PIL import Image, ImageDraw, ImageFont 

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

from skimage import transform as trans

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='/home/DeepfakeBench-main/training/config/detector/xception_with_mask.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_img_path", type=str,
                    default='/home/DeepfakeBench-main/test_one_imgs/2.png')
parser.add_argument('--weights_path', type=str, 
                    default='/home/DeepfakeBench-main/training_logs/xception_with_mask/xception_mask_2024-08-13-14-55-17/test/FaceForensics++/ckpt_best.pth')
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
    img = img_path
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


def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
        """ 
        align and crop the face according to the given bbox and landmarks
        landmark: 5 key points
        """

        M = None
        target_size = [112, 112]
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        if target_size[1] == 112:
            dst[:, 0] += 8.0

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize

        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.
        y_margin = target_size[1] * margin_rate / 2.

        # move
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        # resize
        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        # use skimage tranformation
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        # M: use opencv
        # M = cv2.getAffineTransform(src[[0,1,2],:],dst[[0,1,2],:])

        img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))
        
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
            mask = cv2.resize(mask, (outsize[1], outsize[0]))
            return img, mask
        else:
            return img, None

def get_keypts(image, face, predictor, face_detector):
    # detect the facial landmarks for the selected face
    shape = predictor(image, face)
    
    # select the key points for the eyes, nose, and mouth
    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    
    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

    return pts


def test_on_face_img(face_img):
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

    
    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_img_tensor = prepare_testing_data(face_img, config)
    
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
    pred_mask = test_prob['forgery_mask']
    return [prob, pred_cls, pred_mask]

### 在图像中添加中文（解决opencv添加中文乱码的问题）
def cv2_img_add_text(img, text_str, left, top, text_color, text_size):    
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("msjh.ttc", text_size, encoding="utf-8")
    draw.text((left, top), text_str, text_color, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def crop_faces(input_img_path, number_of_faces, face_detector, predictor):
    face_img = cv2.imread(input_img_path)
    if face_img is None: 
        raise ValueError('Loaded image is None: {}'.format(input_img_path))
    
    img_name = input_img_path.split('/')[-1]

    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    # Detect with dlib
    faces = face_detector(face_img, 1)
    
    if len(faces)>number_of_faces:
        # For now only take the biggest face
        areas = [rect.width() * rect.height() for rect in faces]
        largest_indices = np.argsort(areas)[::-1][:number_of_faces]
        largest_faces = [faces[i] for i in largest_indices]
        faces = largest_faces
    
    face_id = 0
    for face in faces:
        # Get the landmarks/parts for the face in box d only with the five key points
        landmarks = get_keypts(face_img, face, predictor, face_detector)

        res = 256
        mask = None
        # Align and crop the face
        cropped_face, mask_face = img_align_crop(face_img, landmarks, outsize=(res, res), mask=mask)
        # 进推理***********得到推理结果
        [pred_cls, pred, pred_mask] = test_on_face_img(cropped_face)
        # ###############################

        # annotation
        # unpack the position object
        startX = int(face.left())
        startY = int(face.top())
        endX = int(face.right())
        endY = int(face.bottom())
        # draw the bounding box from the correlation object tracker
        cv2.rectangle(face_img, (startX, startY), (endX, endY), (0, 255, 0), thickness = 2)
        
        #文字提示
        # cv2_img_add_text(face_img, 'Deepfake Probability：', startX, startY-5, (255,0,255), 1)
        Softmax_fn=nn.Softmax(dim=-1)
        pred_probability =Softmax_fn(pred)
        cv2.putText(face_img, '                   '+str(round(pred_probability.cpu().numpy()[0][0]*100,2)) +'%,  '+ str(round(pred_probability.cpu().numpy()[0][1]*100,2)) + '%', 
                    (startX, startY-15),                      #坐标
                    cv2.FONT_HERSHEY_SIMPLEX,              #字体
                    0.5,                                     #字号
                    (255,0,255),                           #颜色
                    1)                                     #字的线宽
        cv2.putText(face_img, 'Classification Score: Real,    Fake', 
                    (startX, startY-30),                      #坐标
                    cv2.FONT_HERSHEY_SIMPLEX,              #字体
                    0.5,                                     #字号
                    (255,0,255),                           #颜色
                    1)                                     #字的线宽
        deepfake_probability = round(pred_cls.cpu().numpy()[0]*100, 2) 
        cv2.putText(face_img, 'Deepfake Probability: '+ str(deepfake_probability)+'%', 
                    (startX, startY-45),                      #坐标
                    cv2.FONT_HERSHEY_SIMPLEX,              #字体
                    0.5,                                     #字号
                    (255,0,255),                           #颜色
                    1)                                     #字的线宽
        alpha = 0.3
        pred_mask_img = pred_mask.cpu().numpy()*255
        pred_mask_img = pred_mask_img.astype('uint8')
        other_ch = np.zeros_like(pred_mask_img)
        pred_mask_img_rgb = np.dstack((pred_mask_img, other_ch, other_ch))
        # dst = np.zeros_like(cropped_face)
        # cv2.addWeighted(cropped_face, 1 - alpha, pred_mask_img_rgb, alpha, 0, dst, cv2.CV_32F)
        dst = cv2.addWeighted(cropped_face, 1 - alpha, pred_mask_img_rgb, alpha, 0)
        dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'test_one_imgs/news/{img_name}-{face_id}-result_mask.jpg',dst)
        face_id += 1 
            
    face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'/home/DeepfakeBench-main/test_one_imgs/news/{img_name}-result.jpg',face_img)
    


def test_on_face_img(face_img):
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

    
    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_img_tensor = prepare_testing_data(face_img, config)
    
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
    pred_mask = test_prob['forgery_mask']
    return [prob, pred_cls, pred_mask]


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
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = 'preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat'
    ## Check if predictor path exists
    if not os.path.exists(predictor_path):
        raise ValueError('predictor_path is None: {}'.format(predictor_path))
    face_predictor = dlib.shape_predictor(predictor_path)

    input_img_path = 'test_one_imgs/snap12.png'
    crop_faces(input_img_path, 3, face_detector, face_predictor)

    # main_on_oneimg()
    
    
    # img_path = 'test_one_imgs/1.png'
    # with open(args.detector_path, 'r') as f:
    #     config = yaml.safe_load(f)
    # prepare_testing_data(img_path, config)

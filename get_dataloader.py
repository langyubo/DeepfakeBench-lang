'''
Description:  创建dataloader
Author: lang_yubo
Date: 2022-12-07 15:37:46
LastEditTime: 2023-01-05 00:51:51
LastEditors: lang_yubo
'''

import torch
import torchvision
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import albumentations as A
from PIL import Image
from matplotlib import pyplot as plt
import os
import random


class transformA(object):  #数据增强类，用于对face image、face depth和mask进行数据增强
    def __init__(self, scale_size=256, output_size=224, is_train=False):
        assert isinstance(scale_size, (int, tuple))
        assert isinstance(output_size, (int, tuple))
        
        self.scale_size = scale_size
        self.is_train = is_train
                
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2 # tuple
            self.output_size = output_size
            
    def __call__(self, sample):
        imidx, image, depth = sample['imidx'], sample['image'],sample['depth']
        
        # crop size
        h, w = image.shape[0:2]
        new_h, new_w = self.output_size
        
        # 为了显示效果好 先不做normalize
        aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=self.scale_size, width=self.scale_size, interpolation=3, always_apply=False, p=1),
            A.RandomCrop(height=new_h, width=new_w, p=1),
            ])

        
        aug2 = A.Compose([
                A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit= 0.1,
                                     val_shift_limit=0.1, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.4,
                                           contrast_limit=0.2, p=0.8),
                ],p=0.3),
                A.OneOf([
                    A.GaussNoise(),  # 将高斯噪声添加到输入图像
                    A.GaussNoise(),  # 将高斯噪声应用于输入图像。
                ], p=0.2),  # 应用选定变换的概率
                A.OneOf([
                    A.MotionBlur(p=0.2),  # 使用随机大小的内核将运动模糊应用于输入图像。
                    A.MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
                    A.Blur(blur_limit=3, p=0.1),  # 使用随机大小的内核模糊输入图像。
                ], p=0.2),
                A.ToGray(p=0.05),  # 随机明亮对比度
                A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                )
                ])

        aug_eval = A.Compose([
                    A.Resize(height=self.output_size[0], width=self.output_size[1], interpolation=3, always_apply=False, p=1),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],)
                ])

        # depth图resize操作， 需要把depthresize成网络的输出大小
        depth_size = 28  # 112
        depth_resize_trans = A.Compose([A.Resize(height=depth_size, width=depth_size, interpolation=3, always_apply=False, p=1)])

        if self.is_train:   # 训练阶段，全部数据增强
            augmented = aug(image=image, mask=depth)  #随机翻转，随机裁剪 这俩操作对faceimage和depth同时进行

            image = augmented['image']
            depth = augmented['mask']       
        
            augmented2 = aug2(image=image)  #高斯噪声、亮度变换、对比度变换、标准化 只对face image做变换
            image = augmented2['image']
            
            #resize depth
            resize_depth = depth_resize_trans(image = depth)['image']


            image = torchvision.transforms.ToTensor()(image)
            resize_depth = torchvision.transforms.ToTensor()(resize_depth/255)
            
            return {'imidx':torch.from_numpy(imidx),'image':image, 'depth':resize_depth}   
        else:   #测试阶段，无数据增强 只有resize和归一化
            augmented = aug_eval(image=image)  #对测试数据做裁切和标准化
            image = augmented['image']
            
            #resize label
            resize_depth = depth_resize_trans(image = depth)['image']

            image = torchvision.transforms.ToTensor()(image)
            resize_depth = torchvision.transforms.ToTensor()(resize_depth/255)
            
            return {'imidx':torch.from_numpy(imidx),'image':image, 'depth':resize_depth}   


class LoadData(Dataset):
    def __init__(self, with_mask, txt_path, transform=None):
        super().__init__()
        self.imgs_info = self.get_images(txt_path)
        self.transform = transform
        self.with_mask = with_mask
        self.with_patch_label = False
        # 定义patch embedding 用来计算每个patch的label
        self.patch_embedding = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=16, stride=16, bias=False)
        pe_weight = torch.ones(1,1,16,16) 
        self.patch_embedding.weight = nn.Parameter(pe_weight, requires_grad=False)
        self.patch_label_threshold = 16*16*0.5   #像素大于这个数的就被置为假patch

    def __getitem__(self, index):
        imidx = np.array([index])
        img_path, label = self.imgs_info[index]
        if label == '1':
            label='0'
        else:
            label='1'

        img = Image.open(img_path)
        img = img.convert('RGB')
        img = np.array(img)

        label = int(label)
        
        
        if label == 0: #真实图像  有完整的depth 但是没有mask
            depth_path = img_path.replace('_faces','_faces_depth')
            face_depth3 = Image.open(depth_path)
            face_mask = Image.new('1', img.shape[0:2])
        # 假图像时，创建mask加上face depth图的结果
        else:  #篡改图像，depth需要乘上mask， 有完整的mask
            depth_path = img_path.replace('_faces','_faces_depth')
            face_depth3 = Image.open(depth_path)
            face_depth3 = face_depth3.convert('L')

            if self.with_mask:
                face_mask = Image.open(img_path.replace('_faces','_faces_mask').replace('.jpg','.png'))  #读取mask图像，之后将图像模式转为二值模式
                face_mask = face_mask.convert('1')

                masked_depth = np.array(face_depth3) * (1-np.array(face_mask))
                face_depth3 = Image.fromarray(np.uint8(masked_depth))
            else:
                face_mask = Image.new('1', img.shape[0:2])
        # 假图像时，创建单纯的全0face depth图
        # else:  #篡改图像，depth需要乘上mask， 有完整的mask
        #     face_depth3 = Image.new('RGB', img.shape[0:2])   #新建一个全0图像  ??????  此处需要进一步操作
        #     face_mask = Image.open(img_path.replace('_faces','_faces_mask').replace('.jpg','.png'))  #读取mask图像，之后将图像模式转为二值模式
        #     face_mask = face_mask.convert('1')
        img2tensor = transforms.ToTensor()
        face_mask = img2tensor(face_mask)  #[1,224,224]
        if self.with_patch_label:
            # 计算mask经过patch embedding之后 每个patch的label，用来之后的loss    
            face_mask_patch = self.patch_embedding(face_mask) #[1,224*16,224/16]=[1,14,14]
            face_mask_patch = face_mask_patch.flatten(1).squeeze().numpy() #[14*14=196]
            face_mask_patch_label = np.array([0 if x>self.patch_label_threshold else 1 for x in face_mask_patch])  # 0代表假的patch  1代表真的patch
        else:
            face_mask_patch_label = np.array([0]*196)



        face_depth = np.array(face_depth3.convert('L'))

        sample = {'imidx': imidx, 'image': img, 'depth': face_depth}

        if self.transform:
            sample = self.transform(sample)

        sample['label']=label
        sample['patch_label'] = face_mask_patch_label  
        if self.with_mask:
            sample['mask'] = face_mask
        else:
            sample['mask'] = img2tensor(face_depth3.convert('L'))

        sample['landmark'] = torch.Tensor([0]*10)
        return sample
        
    def __len__(self):
        return len(self.imgs_info)
    
        
    def get_images(self, txt_path):
        with open(txt_path, 'r') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x:x.strip().split('\t'), imgs_info))
        return imgs_info


def getTrainingTestingData(train_txt_path, val_txt_path, test_txt_path, batch_size, with_mask=True):
    '''
    description: 生成dataloader 每个enumerate的dataloader返回一个batch的sample
    return {dict} sample: sample中包含 {'imidx': imidx, 'image': img, 'depth': face_depth, 'label':img label, 'patch_label':patch label, 'mask':face mask}
    '''    
    # 构建数据加载迭代器
    # single_img 数据迭代器
    train_transforms = transformA(is_train=True)
    val_transforms = transformA()
    train_dataset = LoadData(with_mask, txt_path=train_txt_path, transform=train_transforms)
    train_dataloader = DataLoader(train_dataset, 
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4,
                                drop_last=True) 

    val_dataset = LoadData(with_mask, txt_path=val_txt_path, transform=val_transforms)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4,
                                drop_last=True) 


    test_dataset = LoadData(with_mask, txt_path=test_txt_path, transform=val_transforms)
    test_dataloader = DataLoader(test_dataset, 
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4,
                                drop_last=True) 

    return train_dataloader, val_dataloader, test_dataloader


def getTestingDataLoader(test_txt_path, config, with_mask = True, iuput_size = 224, is_train=False):
    """get test dataloader

    Args:
        test_txt_path (str): test数据的txt地址
        batch_size (int): batchsize

    Returns:
        dataloader: test数据的dataloader
    """    
    
    val_transforms = transformA(output_size=iuput_size, is_train=is_train)
    test_dataset = LoadData(with_mask, txt_path=test_txt_path, transform=val_transforms)
    test_dataloader = DataLoader(test_dataset, 
                                batch_size=config['test_batchSize'],
                                shuffle=True,
                                num_workers=4,
                                drop_last=True) 

    return test_dataloader


if __name__ == '__main__':
    train_txt_path = 'DataSets/CelebDF_train.txt'
    val_txt_path = 'DataSets/CelebDF_val.txt'
    test_txt_path = 'DataSets/CelebDF_test.txt'

    train_dataloader, val_dataloader, test_dataloader = getTrainingTestingData(train_txt_path, val_txt_path, test_txt_path, batch_size=16, with_mask=True)
    for iteration, batch_data in enumerate(train_dataloader):
            # images, labels = batch_data['image'], batch_data['label']
            # images, labels = images.to(device), labels.to(device)

            # mask_patch_label = data['patch_label'].reshape(-1).to(device)
            # face_depth = data['depth'].float().to(device)
            for key, value in batch_data.items():
                    print(key)
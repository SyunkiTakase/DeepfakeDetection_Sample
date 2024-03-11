import os
from re import A
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets,transforms,models,utils
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import sys
import random
import shutil
from model import Detector
import argparse
from datetime import datetime
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
from preprocess import extract_face
import warnings
import cv2
from sklearn import metrics
import glob
import json
import datetime


def ff_real_reader(json_load, args): # Real画像を読み込みラベルを付与
    data_path = '/home/tr22008/deepfake_detection/SelfBlendedImages-master/data/FaceForensics++/'
    data_path = data_path + 'original_sequences/youtube/raw/frames2/'

    name_list = []

    for k in json_load:
        k1 = str(k[0])
        k2 = str(k[1])

        name_list.append(k1)
        name_list.append(k2)

    x = len(name_list)
    print('real:',x)  
    real_list = []
    for i in range(x):
        data_path2 = data_path + str(name_list[i])
        #print(data_path2)
        real = glob.glob(data_path2 + '/*.png')
        for j in range(len(real)):
            real_list.append(real[j])

    real_label = [0] * len(real_list)
    
    return real_list, real_label

def ff_fake_reader(json_load, args): # Fake画像を読み込みラベルを付与
    data_path = '/home/tr22008/deepfake_detection/SelfBlendedImages-master/data/FaceForensics++/'
    data_path = data_path + 'manipulated_sequences/' + str(args.data_type) + '/youtube/raw/frames/'

    name_list = []

    for k in json_load:
        k1 = str(k[0]) + '_' + str(k[1])
        k2 = str(k[1]) + '_' + str(k[0])

        name_list.append(k1)
        name_list.append(k2)

    x = len(name_list)
    print('fake:',x)
    fake_list = []
    for i in range(x):
        data_path2 = data_path + str(name_list[i])
        fake = glob.glob(data_path2 + '/*.png')
        for j in range(len(fake)):
            fake_list.append(fake[j])    

    fake_label = [1] * len(fake_list)

    return fake_list, fake_label

def dfdcp_real_reader(json_load, args): # Real画像を読み込みラベルを付与
    data_path = '/media/tr22008/hdd/DFDCP/frames/'
    length = len(json_load)
    keys_list = list(json_load.keys())

    name_list = []
    set_list = []
    label_list = []

    for i in json_load.values():
        set_j = str(i['set'])
        label_j = str(i['label'])
        set_list.append(set_j)
        label_list.append(label_j)

    for j in range(length):
        s = set_list[j]
        t = label_list[j]

        if s == 'test' and t == 'real':
            locs = max([i for i, x in enumerate(keys_list[j]) if x == '/'])
            locs = locs + 1    
            path_key = keys_list[j][locs:]
            name_list.append(path_key)

    x = len(name_list)
    print('real:',x)
    real_list = []
    for i in range(x):
        data_path2 = data_path + str(name_list[i])
        real = glob.glob(data_path2 + '/*.png')
        for j in range(len(real)):
            real_list.append(real[j])    

    real_label = [0] * len(real_list)

    return real_list, real_label

def dfdcp_fake_reader(json_load, args): # Fake画像を読み込みラベルを付与
    data_path = '/media/tr22008/hdd/DFDCP/frames/'
    length = len(json_load)
    keys_list = list(json_load.keys())

    name_list = []
    set_list = []
    label_list = []

    for i in json_load.values():
        set_j = str(i['set'])
        label_j = str(i['label'])
        set_list.append(set_j)
        label_list.append(label_j)

    for j in range(length):
        s = set_list[j]
        t = label_list[j]

        if s == 'test' and t == 'fake':
            locs = max([i for i, x in enumerate(keys_list[j]) if x == '/'])
            locs = locs + 1            
            path_key = keys_list[j][locs:]
            name_list.append(path_key)

    x = len(name_list)
    print('fake:',x)
    fake_list = []
    for i in range(x):
        data_path2 = data_path + str(name_list[i])
        fake = glob.glob(data_path2 + '/*.png')
        for j in range(len(fake)):
            fake_list.append(fake[j])    

    fake_label = [1] * len(fake_list)

    return fake_list, fake_label
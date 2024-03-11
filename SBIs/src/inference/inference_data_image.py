import os
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

warnings.filterwarnings('ignore')

save_path = '/home/tr22008/deepfake_detection/SelfBlendedImages-master/result/'
data_path = '/home/tr22008/deepfake_detection/SelfBlendedImages-master/data/FF_data/'

def main(args):
    
    save_path = save_path + str(args.data_type) + 'AUC'

    model=Detector()
    model=model.to(device)
    cnn_sd=torch.load(args.weight_name)["model"]
    model.load_state_dict(cnn_sd)
    model.eval()

    real_list =  glob.glob(data_path + "/original/test/*.png")
    real_label = [0] * len(real_list)
    print(len(real_label))

    fake_list = glob.glob(data_path + str(args.data_type) + "/test/*.png")
    fake_label = [1] * len(fake_list)
    print(len(fake_label))

    face_list = fake_list + real_list
    print(len(face_list))
    label_list = fake_label + real_label
    print(len(label_list))
    pred_list = []
    for i, face_img in enumerate(face_list):
        #print('############')
        #print('i:',i)
    # load bgr
        #print(face_img)
        try:
            face_img = cv2.imread(face_img)
            frame = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            #print(frame.shape)
            label = label_list

        except:
            tqdm.write(f'Fail loading: {face_img}')
            continue


        face_detector = get_model("resnet50_2020-07-20", max_size=max(frame.shape),device=device)
        face_detector.eval()

        face=extract_face(frame,face_detector)

        with torch.no_grad():
            img=torch.tensor(face).to(device).float()/255
            # torchvision.utils.save_image(img, f'test.png', nrow=8, normalize=False, range=(0, 1))
            pred=model(img).softmax(1)[:,1].cpu().data.numpy().tolist()
        
        #print('pred:',pred)
        for j in range(1):
            pred2 = float(pred[j])
        #print('pred2:',pred2)
        if pred2 < 0.5:
            print('i:',i)
            print(f'real: {max(pred):.4f}')
            #print(face_list[i])
            pred_list.append(0)
            #print('pred_list:',len(pred_list))
        else :
            print('i:',i)
            print(f'fake: {max(pred):.4f}')
            #print(face_list[i])
            pred_list.append(1)
            #print('pred_list:',len(pred_list))

        
        #print(type(pred2))
        """
        if pred2 < 0.5:
            print(f'real: {max(pred):.4f}')
            print(face_list[i])
        else :
            print(f'fake: {max(pred):.4f}')
            print(face_list[i])
        """
    print(len(label))
    print(len(pred_list))

    auc = metrics.roc_auc_score(label, pred_list)
    precision, recall, thresholds = metrics.precision_recall_curve(label, pred_list)
    
    
    f, ax = plt.subplots(1, 1)
    ax.plot(recall, precision)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.fill_between(recall,precision, facecolor='b', alpha=0.3)
    f.savefig(save_path + str(auc) + '.png')

    print('AUC:',auc)


if __name__=='__main__':

    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    parser=argparse.ArgumentParser()
    parser.add_argument('-w',dest='weight_name',type=str)
    parser.add_argument('-dt',dest='data_type',type=str)

    args=parser.parse_args()

    main(args)

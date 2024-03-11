import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms,models,utils
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import sys
import random
import shutil
#from model import Detector
# from model2 import Detector
from model3 import Detector
import argparse
from datetime import datetime
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
from preprocess import extract_frames
from datasets import *
from sklearn import metrics
import warnings
import datetime
warnings.filterwarnings('ignore')

def date_times():
    dt_now = datetime.datetime.now()
    year =  dt_now.year
    month = dt_now.month
    day = dt_now.day
    hour = dt_now.hour
    min = dt_now.minute
    
    dt = str(year) + '_' + str(month) + '_' + str(day) + '_' + str(hour) + '_' + str(min)
    print('{0}/{1}/{2}/{3}:{4}'.format(year,month,day,hour,min))

    return dt

def metrics_calculation(dt, label, pred_list, save_path): # AUCの計算
    auc = metrics.roc_auc_score(label, pred_list) # AUCを計算
    precision, recall, thresholds = metrics.precision_recall_curve(label, pred_list) # PrecisionとRecallを計算

    f, ax = plt.subplots(1, 1)
    ax.plot(recall, precision)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.fill_between(recall,precision, facecolor='b', alpha=0.3)
    f.savefig(save_path + dt + '_' + 'AUC:' + str(auc) + '.png')

    return auc

def main(args):
    dt = date_times()

    save_path = '/home/tr22008/deepfake_detection/SelfBlendedImages-master/output/sbi_base3_12_25_13_37_53/AUC/'

    model=Detector()
    model=model.to(device)
    cnn_sd=torch.load(args.weight_name)["model"]
    model.load_state_dict(cnn_sd)
    model.eval()

    face_detector = get_model("resnet50_2020-07-20", max_size=2048,device=device)
    face_detector.eval()

    if args.dataset == 'FFIW':
        video_list,target_list=init_ffiw()
    elif args.dataset == 'FF':
        video_list,target_list=init_ff(args.fake_type)
    elif args.dataset == 'DFD':
        video_list,target_list=init_dfd()
    elif args.dataset == 'DFDC':
        video_list,target_list=init_dfdc()
    elif args.dataset == 'DFDCP':
        video_list,target_list=init_dfdcp()
    elif args.dataset == 'CDF':
        video_list,target_list=init_cdf()
    else:
        NotImplementedError

    output_list=[]
    for filename in tqdm(video_list):
        try:
            face_list,idx_list=extract_frames(filename,args.n_frames,face_detector)

            with torch.no_grad():
                img=torch.tensor(face_list).to(device).float()/255
                pred=model(img).softmax(1)[:,1]
                # print(img.shape)
                
            pred_list=[]
            idx_img=-1
            for i in range(len(pred)):
                if idx_list[i]!=idx_img:
                    pred_list.append([])
                    idx_img=idx_list[i]
                pred_list[-1].append(pred[i].item())
            pred_res=np.zeros(len(pred_list))
            for i in range(len(pred_res)):
                pred_res[i]=max(pred_list[i])
            pred=pred_res.mean()
        except Exception as e:
            print(e)
            pred=0.5
        output_list.append(pred)

    auc = metrics_calculation(dt, target_list, output_list, save_path)

    print(f'{args.dataset}| AUC: {auc:.4f}')


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
    parser.add_argument('-d',dest='dataset',type=str)
    parser.add_argument('-n',dest='n_frames',default=32,type=int)
    parser.add_argument('-ft',dest='fake_type',type=str)
    args=parser.parse_args()

    main(args)

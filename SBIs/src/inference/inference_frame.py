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
import json
import datetime
from data_utils import ff_fake_reader, ff_real_reader, dfdcp_fake_reader, dfdcp_real_reader

dt_now = datetime.datetime.now()
year =  dt_now.year
month = dt_now.month
day = dt_now.day
hour = dt_now.hour
min = dt_now.minute
dt = str(year) + str(month) + str(day) + str(hour) + str(min)
print(dt)

warnings.filterwarnings('ignore')

def metrics_calculation(label, pred_list, save_path): # AUCの計算
	auc = metrics.roc_auc_score(label, pred_list) # AUCを計算
	precision, recall, thresholds = metrics.precision_recall_curve(label, pred_list) # PrecisionとRecallを計算
	
	f, ax = plt.subplots(1, 1)
	ax.plot(recall, precision)
	ax.set_xlabel('recall')
	ax.set_ylabel('precision')
	ax.set_xlim([0, 1])
	ax.set_ylim([0, 1])
	ax.fill_between(recall,precision, facecolor='b', alpha=0.3)
	f.savefig(save_path + str(auc) +  dt + '.png')

	print('AUC:',auc)

def main(args, save_path):
	if args.data_type == 'Deepfakes' or args.data_type == 'FaceSwap' or args.data_type == 'Face2Face' or args.data_type == 'NeuralTextures':
		json_open = open('/home/tr22008/deepfake_detection/SelfBlendedImages-master/data/FaceForensics++/test.json', 'r') # jsonファイル読み込み
		json_load = json.load(json_open)
		#print(len(json_load))
		real_list, real_label = ff_real_reader(json_load, args) # Real画像とラベル
		fake_list, fake_label = ff_fake_reader(json_load, args) # Fake画像とラベル

	elif args.data_type == 'DFDCP':
		json_open = open('/media/tr22008/hdd/DFDCP/dataset2.json', 'r') # jsonファイル読み込み
		json_load = json.load(json_open)
		#print(len(json_load))
		real_list, real_label = dfdcp_real_reader(json_load, args) # Real画像とラベル
		fake_list, fake_label = dfdcp_fake_reader(json_load, args) # Fake画像とラベル

	save_path = save_path + str(args.data_type) + 'AUC'

	model=Detector()
	model=model.to(device)
	cnn_sd=torch.load(args.weight_name)["model"]
	model.load_state_dict(cnn_sd)
	model.eval()

	print('Real images {0}/ labels {1}'.format(len(real_list),len(real_label)))
	print('Fake images {0}/ labels {1}'.format(len(fake_list),len(fake_label)))

	face_list = fake_list + real_list # Fake画像とReal画像
	label_list = fake_label + real_label # Fake画像とReal画像のラベル

	print('All images {0}/ labels {1}'.format(len(face_list),len(label_list)))


	pred_list = []
	for i, face_img in enumerate(face_list):
		try:
			face_img = cv2.imread(face_img)
			frame = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
			label = label_list

		except:
			tqdm.write(f'Fail loading: {face_img}')
			continue

		face_detector = get_model("resnet50_2020-07-20", max_size=max(frame.shape),device=device)
		face_detector.eval()

		face=extract_face(frame,face_detector) # 380×380にクロップ

		try:
			with torch.no_grad():
				img=torch.tensor(face).to(device).float()/255
				pred=model(img).softmax(1)[:,1].cpu().data.numpy().tolist() # Fakeの確率
		
			for j in range(1):
				pred2 = float(pred[j])

			if pred2 < 0.5:
				print('i:',i)
				print(f'real: {max(pred):.4f}')
				pred_list.append(0)
			else :
				print('i:',i)
				print(f'fake: {max(pred):.4f}')
				pred_list.append(1)

		except AssertionError:
			pred_list.append(1)
			continue

	print(len(label))
	print(len(pred_list))
	metrics_calculation(label, pred_list, save_path) # AUCの計算

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

	save_path = '/home/tr22008/deepfake_detection/SelfBlendedImages-master/results/'

	main(args, save_path)
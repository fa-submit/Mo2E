import argparse
import pandas as pd
from models.get_model import get_arch
from utils.get_loaders import get_test_cls_loader
from utils.evaluation import evaluate_multi_cls

from utils.model_saving_loading import load_model
from tqdm import trange
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score, confusion_matrix, accuracy_score
from imblearn.metrics import geometric_mean_score

import os.path as osp
import os
import sys
import numpy as np
import random
import torch

def set_seeds(seed_value, use_cuda):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False

data_path = 'data'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# reproducibility
seed_value = 0
set_seeds(seed_value)

results_path = 'results/'

data_path = '../endotract/data/images/'
csv_val = '../endotract/data/val_endo1.csv'
model_name = 'bit_resnext50_1'
#load_path = args.load_path

load_path1 = "experiments/endo/F1/mxp_2e-1_instance_instance/bit_resnext50_1/epoch_22_K_90.06_mAUC_95.90_MCC_89.86"
load_path2 =  "experiments/endo/F1/mxp_2e-1_instance_class_new/bit_resnext50_1/epoch_25_K_89.84_mAUC_95.11_MCC_90.47"


model1, mean, std = get_arch(model_name, n_classes=23)
model2, mean, std = get_arch(model_name, n_classes=23)
model1, stats = load_model(model1, load_path1, device='cpu')
model2, stats = load_model(model2, load_path2, device='cpu')
model1 = model1.to(device)
model2 = model2.to(device)

if model_name = 'mobilenetV2':
    s1 = torch.linalg.matrix_norm(model1.classifier.weight)
    s2 = torch.linalg.matrix_norm(model2.classifier.weight)
    s1 = s1/(s1+s2)
    s2 = s2/(s1+s2)

if model_name = 'efficientnet_b4':
    s1 = torch.linalg.matrix_norm(model1.classifier.fc.weight)
    s2 = torch.linalg.matrix_norm(model2.classifier.fc.weight)
    s1 = s1/(s1+s2)
    s2 = s2/(s1+s2)

elif model_name == 'bit_resnext50_1':
    s1 = torch.linalg.matrix_norm(model1.head[3].weight)
    s2 = torch.linalg.matrix_norm(model2.head[3].weight)
    s1 = s1/(s1 + s2)
    s2 = s2/(s2 + s1)


print('* Creating Val Dataloaders, batch size = {:d}'.format(1))
val_loader = get_test_cls_loader(csv_path_test=csv_val, data_path=data_path, batch_size=1, mean=mean, std=std, tg_size=(512,512), test=False)

probs_all = []
preds_all = []

probs_all1 = []
preds_all1 = []

probs_all2 = []
preds_all2 = []

probs_all3 = []
preds_all3 = []

labels_all = []

many = [ 2,3,4,5,6,12,13, 15,19,22]
median = [7,  8, 11, 17,21, 14]
head = [ 2,3,4,5,6,12,13, 15,19,22,7,  8, 11, 17,21, 14]
few = [1,9,10,16,18,20,0]

cl_lbl = torch.tensor([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]]).to(device)
map_cc = torch.tensor([[0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,1,0,0,1]]).to(device)
map_ci = torch.tensor([[0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0]]).to(device)

map_cci = map_cc + map_ci

map_ii = torch.tensor([[1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,1,0,1,0,0]]).to(device)
for i_batch, (inputs, labels, _) in enumerate(val_loader):
    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

    logits2 = model2(inputs)
    logits2 =logits2 *  map_cci
    logits1 = model1(inputs)
    logits1 =logits1 *  map_ii


    logits = ( s2 * logits2 + s1 *logits1)


    probs = torch.nn.Softmax(dim=1)(logits)
    _, preds = torch.max(probs, 1)
    

    
    probs_all.extend(probs.detach().cpu().numpy())
    preds_all.extend(preds.detach().cpu().numpy())
    
    labels_all.extend(labels.detach().cpu().numpy())
    

    
p = np.stack(preds_all) 
pb = np.stack(probs_all) 
lbl = np.stack(labels_all)


print_conf = True
text_file = osp.join(results_path, 'performance_val.txt')

vl_auc, vl_k, vl_mcc, vl_f1, vl_bacc, vl_auc_all, vl_f1_all = evaluate_multi_cls(lbl, p, pb, print_conf=False, text_file=text_file)

print('Val- MCC: {:.2f} - mAUC: {:.2f}  - BalAcc: {:.2f} - F1: {:.2f}'.format(100*vl_mcc, 100*vl_auc, 100*vl_bacc, 100*vl_f1))
#print(vl_f1_all)
print("GM", geometric_mean_score(lbl, p, average='macro') )
print("vl_f1_all:",vl_f1_all)

matrix = confusion_matrix(lbl, p)
acc_all = matrix.diagonal()/matrix.sum(axis=1)
#acc_all = acc_all.tolist()
print("acc_all:", acc_all)
vl_f1_all = np.array(vl_f1_all)
print("acc_all:", acc_all.tolist())
print("many acc:", acc_all[many].mean())
print("many f1:",vl_f1_all[many].mean() )
print("median acc:", acc_all[median].mean())
print("median f1:", vl_f1_all[median].mean())
print("few acc:", acc_all[few].mean())
print("few f1", vl_f1_all[few].mean())
print("head acc:", acc_all[head].mean())
print("head f1", vl_f1_all[head].mean())


print(100*vl_mcc, 100*vl_bacc, 100*vl_f1, geometric_mean_score(lbl, p, average='macro'), acc_all[many].mean(),vl_f1_all[many].mean(),acc_all[median].mean(), vl_f1_all[median].mean(),acc_all[few].mean(),vl_f1_all[few].mean(), acc_all[head].mean(),vl_f1_all[head].mean() )

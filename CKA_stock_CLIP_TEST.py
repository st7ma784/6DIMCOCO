from pytest import Cache
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
import clip


import timeit
from torch_cka import CKA

# Dataset=torch.utils.data.TensorDataset(torch.rand(110,3,227,227))
# dataloader = DataLoader(Dataset,
#                             batch_size=10, # according to your device memory
#                             shuffle=False,
#                             pin_memory=True,

#                             drop_last=True
#                             )  # Don't forget to seed your dataloader
    

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from BuildSpainDataSet import COCODataModule
import sys,os
#find optional dir as first arg
if len(sys.argv)>1:
    dir=sys.argv[1]
else:
    dir="."
data=COCODataModule(Cache_dir=dir,annotations=os.path.join(dir,"annotations"),batch_size=10)
data.setup()
dataloader=data.val_dataloader()

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model1 = resnet18.to(device,non_blocking=True).eval()  # Or any neural network of your choice
# model2 = resnet34.to(device,non_blocking=True).eval()  # Or any neural network of your choice
model,_=clip.load("ViT-L/14",device=device)
altmodel,_=clip.load("ViT-L/14",device=device)
model2=model.visual.eval()
model1=altmodel.visual.eval()

cka=CKA(model1,model2,model1_name="ResNet18",model2_name="CLIP",device=device)
cka.model1_info['Dataset'] = dataloader.dataset.__repr__().split('\n')[0]
cka.model2_info['Dataset'] = dataloader.dataset.__repr__().split('\n')[0]
cka.compare(dataloader,dataloader)
cka.plot_results("CKAResCLIPStock.png","Comparison of 2 stock clip models using base implementation")
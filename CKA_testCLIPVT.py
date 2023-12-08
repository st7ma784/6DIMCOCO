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
from itertools import islice


class myCKA(CKA):

    def _log_layer(self,
                   model: str,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):

        if model == "model1":
            self.model1_features[name] = out

        elif model == "model2":
            self.model2_features[name] = out

        else:
            raise RuntimeError("Unknown model name for _log_layer.")

    def _insert_hooks(self):
        # Model 1
        for name, layer in self.model1.named_modules():
            #we need to only include layers that end with mlp or include "ln"
        # print(name)
            if self.model1_layers is not None:
                if name in self.model1_layers:
                    self.model1_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model1", name))
            else:
                
                self.model1_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model1", name))

        # Model 2
        for name, layer in self.model2.named_modules():

            if self.model2_layers is not None:
                if name in self.model2_layers:
                    self.model2_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model2", name))
            else:

                    
                self.model2_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model2", name))




def batch_test_method(methodA,methodB=None,convertOO=False,permute=True,dataloader=None):

    #same as below but we assume methodA and methodB take a batch of inputs
    print("Testing method",methodA.__name__)
    if convertOO:
        #we assume these methods are imported from a class and the first arg is self.
        _methodA=lambda K,L:methodA(None,K,L)
        if methodB is not None:
            methodC=lambda K,L:methodB(None,K)
    else:
        _methodA=methodA
        if methodB is None:
            methodC=methodA
        else:
            methodC=lambda K,L:methodB(K)
    # resnet18 = models.resnet18(pretrained=True)
    # resnet34 = models.resnet34(pretrained=True)

    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model1 = resnet18.to(device,non_blocking=True).eval()  # Or any neural network of your choice
    # model2 = resnet34.to(device,non_blocking=True).eval()  # Or any neural network of your choice
    model,_=clip.load("ViT-L/14",device=device)
 
    model2=model.visual.eval()
    model1=model.transformer.eval()
    
    cka=myCKA(model1,model2,model1_name="ResNet18",model2_name="CLIP",device=device)
    cka.model1_info['Dataset'] = dataloader.dataset.__repr__().split('\n')[0]
    cka.model2_info['Dataset'] = dataloader.dataset.__repr__().split('\n')[0]

    N = len(cka.model1_layers) if cka.model1_layers is not None else len(list(cka.model1.modules()))
    M = len(cka.model2_layers) if cka.model2_layers is not None else len(list(cka.model2.modules()))
    N=86
    M=170
    cka.m1_matrix=torch.zeros((N,M),device=device)
    cka.m2_matrix=torch.zeros((M),device=device)
    cka.hsic_matrix=torch.zeros((N),device=device)

    with torch.no_grad():
        #only do first 10 batches
        for x1 in tqdm(dataloader):
        

            i=x1[0].to(device,non_blocking=True,dtype=torch.half)
            text=x1[1].to(device,non_blocking=True)[:,0]
            
            cka.model2_features = {}
            cka.model1_features = {}
            print(text.shape)
             #shape is B x  77 

            EOT_index=text.argmax(dim=1) #shape is B
            model2(i)
            #model2(i)
            model.encode_text(text)
            features,features2=[],[]
            for _, feat1 in cka.model1_features.items():
                #shuffle based on feat[1]
                #1 find shape
                #2 make a set of random indices
                #3 permute based on those indices
                #4 continue as normal
                feat1=feat1[0]
                # print(feat1)
                if feat1.shape[0]==10:
                    if len(feat1.shape)==3: #B,seq,hidden
                        if feat1.shape[1]==77:
                            feat1=feat1[torch.arange(feat1.shape[0]),EOT_index]
                        elif feat1.shape[2]==77:
                            feat1=feat1[torch.arange(feat1.shape[0]),:,EOT_index]
                        X = feat1.flatten(1)
                    X = feat1.flatten(1)
                                
                    features.append((X @ X.t()).fill_diagonal_(0))
            cka.model1_features = {}

            for _,feat2 in cka.model2_features.items():
                #print(feat2)
                #if clip ... we need to do something different.
                feat2=feat2[0]
                if feat2.shape[0]==10:
                    Y = feat2.flatten(1)

                    features2.append((Y @ Y.t()).fill_diagonal_(0))
            cka.model2_features = {}
            
            cka.m2_matrix=torch.add(cka.m2_matrix, methodC(torch.stack(features2),torch.stack(features2)))#//(10 * (10 - 3))
            
            cka.hsic_matrix=torch.add(cka.hsic_matrix, methodC(torch.stack(features),torch.stack(features)))#/(10 * (10 - 3))
            
            cka.m1_matrix =torch.add(cka.m1_matrix, _methodA(torch.stack(features),torch.stack(features2)))#/(10 * (10 - 3))

#end of profiling

    print(cka.m1_matrix.shape)
    print(cka.m2_matrix.shape)
    print(cka.hsic_matrix.shape)

    # print(cka.hsic_matrix)
    # print(cka.m1_matrix)
    # print(cka.m2_matrix)
    RESULTS=torch.div(cka.m1_matrix, torch.mul(torch.sqrt(cka.hsic_matrix).unsqueeze(1),torch.sqrt(cka.m2_matrix).unsqueeze(0)))
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # print(prof.key_averages().table(sort_by="self_cude_memory_usage", row_limit=10))
    #print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    #scrape from the table the time totals at the end of the table
    # total_CPU_time=prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1).split('\n')[-3]
    # print(total_CPU_time)
    # total_CPU_time=total_CPU_time.split(' ')[-1]
    # total_CUDA_time=prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1).split('\n')[-2].split(' ')[-1]
    # total_CPU_time= float(total_CPU_time[:-2]) if total_CPU_time[-2:]=="ms" else float(total_CPU_time[:-1])*1000
    # total_CUDA_time= float(total_CUDA_time[:-2]) if total_CUDA_time[-2:]=="ms" else float(total_CUDA_time[:-1])*1000
    # total_time=total_CPU_time+total_CUDA_time
    #plot the results with plt magma colormap
    import matplotlib.pyplot as plt
    import numpy as np
    plt.imshow(RESULTS.cpu().numpy(),cmap="magma") 
    plt.savefig("results_CLIPVTselEOT.png")


if __name__ == "__main__":

    from model.trainclip_cka_base import LightningCLIPModule

    # In the clip model, we get a input of shape L,B,B where L is the number of layers, B is the batch size
    # this is the same as doing the BMM of LBF and LBF.permute(0,2,1)  to get LBB

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

    methoda=LightningCLIPModule.batch_HSIC3
    methodb=LightningCLIPModule.batch_HSIC2
    #time the new method
    print("Timing new batched method")
    starttimer=timeit.default_timer()
    batch_test_method(methoda,methodb,convertOO=True,permute=False,dataloader=dataloader)
    print("New batched method took",timeit.default_timer()-starttimer)

    # print("Timing new batched method")
    # starttimer=timeit.default_timer()
    # batch_test_method(methoda,methodb,convertOO=True,permute=True,dataloader=dataloader)
    # print("New batched method took",timeit.default_timer()-starttimer)
    # # edit this to also check the CKA methods in the torch_cka.py file and the model.trainclip_cka_base.py file
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
from warnings import warn
from mpl_toolkits import axes_grid1
import matplotlib.pyplot as plt
def add_colorbar(im, aspect=10, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

import timeit

# A streamlined version of https://github.com/AntixK/PyTorch-Model-Compare}

def ORRIG_HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.
        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(self.device)
        result = torch.add(torch.trace(K @ L), (((ones.t() @ K @ ones @ ones.t() @ L @ ones) / (N - 1) ) - ((ones.t() @ K @ L @ ones) * 2 ))/ (N - 2))
        return result
def ORIG_HSICA(K, L):
        """
        Computes the unbiased estimate of HSIC metric.
        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        # print("K",K.shape)
        # print("L",L.shape)
        N=K.shape[0]
        return torch.add(torch.trace(K@L),torch.div(torch.sum(K)*torch.sum(L)/(N - 1) - (torch.sum(K@L) * 2 ), (N - 2)))
        
def ORIG_HSIC2(K):
        """
        Computes the unbiased estimate of HSIC metric.
        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N=K.shape[0]
        return torch.add(torch.trace(K@K),torch.div(torch.pow(torch.sum(K),2)/(N - 1) - (torch.sum(K@K) * 2 ), (N - 2)))
        
def _HSIC( K, L):
    """
    Computes the unbiased estimate of HSIC metric.
    Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
    """
    with torch.no_grad():
        N = K.shape[0]
        result = torch.div(torch.sum(torch.sum(K,dim=0))*torch.sum(torch.sum(L,dim=0)),(N - 1) * (N - 2))
        #print("Shape of res",result.shape) #[]
        result = torch.add(result,torch.trace(K @ L)) 
        #print("Shape of res2",result.shape)#[]
        result = torch.sub(result,torch.mul(torch.sum(K,dim=0)@torch.sum(L,dim=1),2 / (N - 2)))
        #print("Shape of res3",result.shape)#[]
        #result= torch.div(result, (N * (N - 3)))
        #print("Shape of res4",result.shape)#[]
        return result
def _HSIC2( K):
    """
    Computes the unbiased estimate of HSIC metric.
    Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
    """
    with torch.no_grad():
        N = K.shape[0]
        # resulta=torch.div(torch.sum(torch.sum(K,dim=0)),N-1)
        # resultb=torch.div(torch.sum(torch.sum(K,dim=0)),N-2)
        # result=torch.mul(resulta,resultb)
        result = torch.div(torch.pow(torch.sum(torch.sum(K,dim=0)),2),(N - 1) * (N - 2))#or sum(K)/N-1  * sum(K)/N-2
        result = torch.add(result,torch.trace(K@K)) 
        result = torch.sub(result,torch.mul(torch.sum(torch.mul(torch.sum(K,dim=-2),torch.sum(K,dim=-1))),2 / (N - 2)))
    
        return result

def _BHSIC2( K):
    """
    Computes the unbiased estimate of HSIC metric.
    Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
    """
    with torch.no_grad():
        #print(K.shape)

        N = K.shape[1]
        result = torch.div(torch.pow(torch.sum(torch.sum(K,dim=-2),dim=-1),2),(N - 1) * (N - 2))#or sum(K)/N-1  * sum(K)/N-2
        result = torch.add(result,torch.sum(torch.diagonal(torch.matmul(K,K),dim1=1,dim2=2),dim=1)) 
        result = torch.sub(result,torch.mul(torch.sum(torch.mul(torch.sum(K,dim=-2),torch.sum(K,dim=-1)),dim=-1),2 / (N - 2)))
        return result

def _BHSIC( K, L):
    """
    Computes the unbiased estimate of HSIC metric.
    Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
    """
    with torch.no_grad():
        N = K.shape[0]

        resa=torch.div(torch.sum(torch.sum(K,dim=-2),dim=-1,keepdim=True),N-1)
        resb=torch.div(torch.sum(torch.sum(L,dim=-2),dim=-1,keepdim=True),N-2)
        result = resa@resb.t()
        result = torch.add(result,torch.sum(torch.diagonal(torch.matmul(K.unsqueeze(1),L),dim1=-2,dim2=-1),dim=-1)) 
        result= torch.sub(result,torch.mul(torch.sum(K,dim=-2)@torch.sum(L,dim=-1).t(),2 / (N - 2)))

        return result
def BatchMethod(method):
     #we have a method that takes 2 arguments, k,L, which are usually size 50,50.
    def BatchMethodWrapper(K,L):
        outputs=[method(K[i],L[i]) for i in range(K.shape[0])]
        return torch.stack(outputs)
    return BatchMethodWrapper
Dataset=torch.utils.data.TensorDataset(torch.rand(150,3,227,227))

def batch_test_method(methodA,methodB=None,convertOO=False):

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
    resnet18 = models.resnet18(pretrained=True)
    resnet34 = models.resnet34(pretrained=True)
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model1 = resnet18.to(device,non_blocking=True).eval()  # Or any neural network of your choice
    model2 = resnet34.to(device,non_blocking=True).eval()  # Or any neural network of your choice

    dataloader = DataLoader(Dataset,
                            batch_size=50, # according to your device memory
                            shuffle=False,
                            pin_memory=True,

                            drop_last=True
                            )  # Don't forget to seed your dataloader
    
    from torch_cka import CKA

    cka=CKA(model1,model2,model1_name="ResNet18",model2_name="ResNet34",device=device)
    cka.model1_info['Dataset'] = dataloader.dataset.__repr__().split('\n')[0]
    cka.model2_info['Dataset'] = dataloader.dataset.__repr__().split('\n')[0]

    N = len(cka.model1_layers) if cka.model1_layers is not None else len(list(cka.model1.modules()))
    M = len(cka.model2_layers) if cka.model2_layers is not None else len(list(cka.model2.modules()))
 
    cka.m1_matrix=torch.zeros((N,M),device=device)
    cka.m2_matrix=torch.zeros((M),device=device)
    cka.hsic_matrix=torch.zeros((N),device=device)

    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:

            for x1 in tqdm(dataloader):
                i=x1[0].to(device,non_blocking=True)
                cka.model2_features = {}
                cka.model1_features = {}

                model1(i)
                model2(i)
                features,features2=[],[]
                for _, feat1 in cka.model1_features.items():
                    X = feat1.flatten(1)
                    features.append((X @ X.t()).fill_diagonal_(0))
                cka.model1_features = {}

                for _,feat2 in cka.model2_features.items():
                    Y = feat2.flatten(1)
                    features2.append((Y @ Y.t()).fill_diagonal_(0))
                cka.model2_features = {}
            
                cka.m2_matrix=torch.add(cka.m2_matrix, methodC(torch.stack(features2),torch.stack(features2)))#//(50 * (50 - 3))
                
                cka.hsic_matrix=torch.add(cka.hsic_matrix, methodC(torch.stack(features),torch.stack(features)))#/(50 * (50 - 3))
                
                cka.m1_matrix =torch.add(cka.m1_matrix, _methodA(torch.stack(features),torch.stack(features2)))#/(50 * (50 - 3))
        
        #end of profiling

        print(cka.m1_matrix.shape)
        print(cka.m2_matrix.shape)
        print(cka.hsic_matrix.shape)

        RESULTS=torch.div(cka.m1_matrix, torch.mul(torch.sqrt(cka.hsic_matrix).unsqueeze(1),torch.sqrt(cka.m2_matrix).unsqueeze(0)))
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        # print(prof.key_averages().table(sort_by="self_cude_memory_usage", row_limit=10))
        #print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
        #scrape from the table the time totals at the end of the table
        total_CPU_time=prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1).split('\n')[-3]
        print(total_CPU_time)
        total_CPU_time=total_CPU_time.split(' ')[-1]
        total_CUDA_time=prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1).split('\n')[-2].split(' ')[-1]
        total_CPU_time= float(total_CPU_time[:-2]) if total_CPU_time[-2:]=="ms" else float(total_CPU_time[:-1])*1000
        total_CUDA_time= float(total_CUDA_time[:-2]) if total_CUDA_time[-2:]=="ms" else float(total_CUDA_time[:-1])*1000
        total_time=total_CPU_time+total_CUDA_time
        import matplotlib.pyplot as plt
        import numpy as np
        plt.imshow(RESULTS.cpu().numpy(),cmap="magma") 
        plt.savefig("results{}-{}took{}.png".format(methodA.__name__,methodB.__name__,total_time))




def test_method(methodA,methodB=None,convertOO=False):
     #takes up to 2 methods, B is optional and is used if theres a method for taking only one arguement
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
    resnet18 = models.resnet18(pretrained=True)
    resnet34 = models.resnet34(pretrained=True)
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model1 = resnet18.to(device,non_blocking=True).eval()  # Or any neural network of your choice
    model2 = resnet34.to(device,non_blocking=True).eval()  # Or any neural network of your choice

    dataloader = DataLoader(Dataset,
                            batch_size=50, # according to your device memory
                            shuffle=False,
                            pin_memory=True,

                            drop_last=True
                            )  # Don't forget to seed your dataloader
    

    #print(resnet18.T_destination)
    cka = CKA(model1, model2,
            model1_name="ResNet18",   # good idea to provide names to avoid confusion
            model2_name="ResNet34",   
            device=device)
    cka.model1_info['Dataset'] = dataloader.dataset.__repr__().split('\n')[0]
    cka.model2_info['Dataset'] = dataloader.dataset.__repr__().split('\n')[0]

    N = len(cka.model1_layers) if cka.model1_layers is not None else len(list(cka.model1.modules()))
    M = len(cka.model2_layers) if cka.model2_layers is not None else len(list(cka.model2.modules()))
 
    cka.m1_matrix=torch.zeros((N,M),device=device)
    cka.m2_matrix=torch.zeros((M),device=device)
    cka.hsic_matrix=torch.zeros((N),device=device)

    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:

            for x1 in tqdm(dataloader):
                i=x1[0].to(device,non_blocking=True)
                cka.model2_features = {}
                cka.model1_features = {}

                model1(i)
                model2(i)
                features,features2=[],[]
                for _, feat1 in cka.model1_features.items():
                    X = feat1.flatten(1)
                    features.append((X @ X.t()).fill_diagonal_(0))
                cka.model1_features = {}

                for _,feat2 in cka.model2_features.items():
                    Y = feat2.flatten(1)
                    features2.append((Y @ Y.t()).fill_diagonal_(0))
                cka.model2_features = {}
                print("m2",cka.m2_matrix.shape)
                cka.m2_matrix=torch.add(cka.m2_matrix, torch.stack([methodC(F,F) for F in features2]))#//(50 * (50 - 3))
                print("m2",cka.m2_matrix.shape)

                cka.hsic_matrix=torch.add(cka.hsic_matrix, torch.stack([methodC(F,F) for F in features]))#/(50 * (50 - 3))
                
                cka.m1_matrix =torch.add(cka.m1_matrix, torch.stack([_methodA(F,G) for F in features2 for G in features]).reshape(N,M))#/(50 * (50 - 3))
        
        #end of profiling

        print(cka.m1_matrix.shape)
        print(cka.m2_matrix.shape)
        print(cka.hsic_matrix.shape)

        RESULTS=torch.div(cka.m1_matrix, torch.mul(torch.sqrt(cka.hsic_matrix).unsqueeze(1),torch.sqrt(cka.m2_matrix).unsqueeze(0)))
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        # print(prof.key_averages().table(sort_by="self_cude_memory_usage", row_limit=10))
        #print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
        #scrape from the table the time totals at the end of the table
        total_CPU_time=prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1).split('\n')[-3]
        print(total_CPU_time)
        total_CPU_time=total_CPU_time.split(' ')[-1]
        total_CUDA_time=prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1).split('\n')[-2].split(' ')[-1]
        total_CPU_time= float(total_CPU_time[:-2]) if total_CPU_time[-2:]=="ms" else float(total_CPU_time[:-1])*1000
        total_CUDA_time= float(total_CUDA_time[:-2]) if total_CUDA_time[-2:]=="ms" else float(total_CUDA_time[:-1])*1000
        total_time=total_CPU_time+total_CUDA_time
        import torchvision.transforms.functional as TF
        img = TF.to_pil_image(RESULTS)
        img.save('results{}-{}took{}.png'.format(methodA.__name__,methodB.__name__,total_time))

if __name__ == "__main__":
    from torch_cka import CKA

    #time the original method
    print("Timing original method")
    starttimer=timeit.default_timer()
    test_method(ORIG_HSICA,ORIG_HSIC2)
    
    print("Original method took",timeit.default_timer()-starttimer)

    #time the new method
    print("Timing new method")
    starttimer=timeit.default_timer()
    test_method(_HSIC,_HSIC2)
    print("New method took",timeit.default_timer()-starttimer)


    from model.trainclip_cka_base import LightningCLIPModule

    # In the clip model, we get a input of shape L,B,B where L is the number of layers, B is the batch size
    # this is the same as doing the BMM of LBF and LBF.permute(0,2,1)  to get LBB

    methoda=LightningCLIPModule.batch_HSIC3
    methodb=LightningCLIPModule.batch_HSIC2
    #time the new method
    print("Timing new batched method")
    starttimer=timeit.default_timer()
    batch_test_method(methoda,methodb,convertOO=True)
    print("New batched method took",timeit.default_timer()-starttimer)
    # edit this to also check the CKA methods in the torch_cka.py file and the model.trainclip_cka_base.py file
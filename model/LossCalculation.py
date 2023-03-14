

import torch
from functools import partial,reduce
'''
This is research code... it is not clean and it is not commented

If you wish to use it for TPUs, I strongly recommend you refactor your code to use this style of function factory.
 Otherwise your runs will be very slow.

'''


def oneminus(args):
    return 1-args
def null(*args):
    return args

def normargs(*args):
    return map(lambda arg: arg/arg.norm(dim=-1, keepdim=True), args)
def logargs(args):
    return torch.log(args)


def get_loss_fn(logitsversion=0,norm=False,log=False):
    baseLogits=calculate_loss
    logfunction=lambda x:x
    normfunction=null
    if norm:
        normfunction=normargs
        # if log:
    #     logfunction=logargs
    if logitsversion==0:
        def baseLogits(*args):
            return logfunction(calculate_loss(*args))
    elif logitsversion==1: 
        def baseLogits(*args):
            return oneminus(logfunction(calculate_loss2(*args)))
    elif logitsversion==2: 
        def baseLogits(*args):
            return oneminus(logfunction(calculate_loss3(*args)))        #one minus here
    elif logitsversion==3:
        #this does not work in any arrangement? 
        def baseLogits(*args):
            return oneminus(logfunction(calculate_loss4(*args)))
    elif logitsversion==4:
        #this does not work in any arrangement either?
        def baseLogits(*args):
            return oneminus(logfunction(calculate_loss5(*args)))
    elif logitsversion==5:
        def baseLogits(*args):
            return oneminus(logfunction(calculate_loss6(*args)))

    def lossfn(*args):
        return baseLogits(*normfunction(*args)) 
    return lossfn



def calculate_lossStock(I, C1):

    #normalize image and text features
    I = I / I.norm(dim=-1, keepdim=True)
    C1 = C1 / C1.norm(dim=-1, keepdim=True)
    #calculate logits
    logits_per_image =  I @ C1.T
    logits_per_text =  C1 @ I.T
    #calculate loss
    return logits_per_image, logits_per_text

def calculate_lossbase(I, C1, C2, C3, C4, C5,norm=True,log=False):

    #normalize image and text features
    I = I / I.norm(dim=-1, keepdim=True)
    C1 = C1 / C1.norm(dim=-1, keepdim=True)
    #calculate logits
    logits_per_image =  I @ C1.T
    #calculate loss
    return logits_per_image

def calculate_loss(  I, C1, C2, C3, C4, C5):
    return torch.einsum("abcz,defz->abcdef",torch.einsum("az,bz,cz->abcz",I,C1,C2),torch.einsum("az,bz,cz->abcz",C3,C4,C5))

    
def calculate_loss2(  I, C1, C2, C3, C4, C5):
    return torch.sum(torch.sqrt(reduce(torch.add,[torch.pow(I,2).view( I.shape[0],1,1,1,1,1,-1),
                                                torch.pow(C1,2).view(1,C1.shape[0],1,1,1,1,-1),
                                                torch.pow(C2,2).view(1,1,C2.shape[0],1,1,1,-1),
                                                torch.pow(C3,2).view(1,1,1,C3.shape[0],1,1,-1),
                                                torch.pow(C4,2).view(1,1,1,1,C4.shape[0],1,-1),
                                                torch.pow(C5,2).view(1,1,1,1,1,C5.shape[0],-1)]).sub_(
                        torch.pow(reduce(torch.add,[I.view( I.shape[0],1,1,1,1,1,-1),
                                                    C1.view(1,C1.shape[0],1,1,1,1,-1),
                                                    C2.view(1,1,C2.shape[0],1,1,1,-1),
                                                    C3.view(1,1,1,C3.shape[0],1,1,-1),
                                                    C4.view(1,1,1,1,C4.shape[0],1,-1),
                                                    C5.view(1,1,1,1,1,C5.shape[0],-1)]),2),alpha=1/6)),dim=-1)
    #results in Nan labels
    
def calculate_loss3( I, C1, C2, C3, C4, C5):
    return torch.sqrt(torch.sum(torch.pow(reduce(torch.add,[torch.pow(I,2).view( I.shape[0],1,1,1,1,1,-1),
                                                torch.pow(C1,2).view(1,C1.shape[0],1,1,1,1,-1),
                                                torch.pow(C2,2).view(1,1,C2.shape[0],1,1,1,-1),
                                                torch.pow(C3,2).view(1,1,1,C3.shape[0],1,1,-1),
                                                torch.pow(C4,2).view(1,1,1,1,C4.shape[0],1,-1),
                                                torch.pow(C5,2).view(1,1,1,1,1,C5.shape[0],-1)]).sub_(
                        torch.pow(reduce(torch.add,[I.view( I.shape[0],1,1,1,1,1,-1),
                                                    C1.view(1,C1.shape[0],1,1,1,1,-1),
                                                    C2.view(1,1,C2.shape[0],1,1,1,-1),
                                                    C3.view(1,1,1,C3.shape[0],1,1,-1),
                                                    C4.view(1,1,1,1,C4.shape[0],1,-1),
                                                    C5.view(1,1,1,1,1,C5.shape[0],-1)]),2),alpha=1/6),2),dim=-1))
    #results in nan labels 
    


def calculate_loss4(I, C1, C2, C3, C4, C5):
    #tested and not working
    return torch.sum(torch.sqrt(reduce(torch.add,[torch.pow(I,2).view( I.shape[0],1,1,1,1,1,-1),
                                                torch.pow(C1,2).view(1,C1.shape[0],1,1,1,1,-1),
                                                torch.pow(C2,2).view(1,1,C2.shape[0],1,1,1,-1),
                                                torch.pow(C3,2).view(1,1,1,C3.shape[0],1,1,-1),
                                                torch.pow(C4,2).view(1,1,1,1,C4.shape[0],1,-1),
                                                torch.pow(C5,2).view(1,1,1,1,1,C5.shape[0],-1)])),dim=-1)
def calculate_loss5(I, C1, C2, C3, C4, C5):
    return torch.sum(reduce(torch.add,[ torch.pow(I,2).view( I.shape[0],1,1,1,1,1,-1),
                                        torch.pow(C1,2).view(1,C1.shape[0],1,1,1,1,-1),
                                        torch.pow(C2,2).view(1,1,C2.shape[0],1,1,1,-1),
                                        torch.pow(C3,2).view(1,1,1,C3.shape[0],1,1,-1),
                                        torch.pow(C4,2).view(1,1,1,1,C4.shape[0],1,-1),
                                        torch.pow(C5,2).view(1,1,1,1,1,C5.shape[0],-1)]).sub_(
                    torch.pow(reduce(torch.add,[    I.view( I.shape[0],1,1,1,1,1,-1),
                                                    C1.view(1,C1.shape[0],1,1,1,1,-1),
                                                    C2.view(1,1,C2.shape[0],1,1,1,-1),
                                                    C3.view(1,1,1,C3.shape[0],1,1,-1),
                                                    C4.view(1,1,1,1,C4.shape[0],1,-1),
                                                    C5.view(1,1,1,1,1,C5.shape[0],-1)]),2),alpha=1/6),dim=-1)

    
def calculate_loss6(I, C1, C2, C3, C4, C5):
    return torch.sqrt(torch.sum(reduce(torch.add,[torch.pow(I,2).view( I.shape[0],1,1,1,1,1,-1),
                                                  torch.pow(C1,2).view(1,C1.shape[0],1,1,1,1,-1),
                                                  torch.pow(C2,2).view(1,1,C2.shape[0],1,1,1,-1),
                                                  torch.pow(C3,2).view(1,1,1,C3.shape[0],1,1,-1),
                                                  torch.pow(C4,2).view(1,1,1,1,C4.shape[0],1,-1),
                                                  torch.pow(C5,2).view(1,1,1,1,1,C5.shape[0],-1)]).sub_(
                            torch.pow(reduce(torch.add,[I.view( I.shape[0],1,1,1,1,1,-1),
                                                        C1.view(1,C1.shape[0],1,1,1,1,-1),
                                                        C2.view(1,1,C2.shape[0],1,1,1,-1),
                                                        C3.view(1,1,1,C3.shape[0],1,1,-1),
                                                        C4.view(1,1,1,1,C4.shape[0],1,-1),
                                                        C5.view(1,1,1,1,1,C5.shape[0],-1)]),2),alpha=1/6),dim=-1))
    

def get_loss_sum(version=False):
    if version:
        return add_loss_sum
    else:
        return add_loss_sum2

from functools import reduce
def add_loss_sum(I=[],T=[]):
    return reduce(torch.add,I)/len(I)+reduce(torch.add,T)/len(T)
def add_loss_sum2(I=[],T=[]):
    return reduce(torch.add,I+T)/(len(I)+len(T))


def get_loss_calc(reduction='sum',mask=None,alpha=None):
    #returns a function that calculates the loss from target labels and logits and masks the output with the mask before reduction
#    is used, the loss is actually the negative of the loss
    
    if mask==None:

        def loss(x,y):
            return torch.nn.functional.cross_entropy(x,y,reduction=reduction)


    else:
        def loss(x,y):
            #l=torch.nn.functional.cross_entropy(x.where(mask,torch.tensor(-100)),y.where(mask,torch.tensor(-100)),ignore_index=-100,reduction="mean")
            #l1=torch.nn.functional.cross_entropy(x.where(mask,torch.tensor(-100)),y.where(mask,torch.tensor(-100)),ignore_index=-100,reduction="sum")
            #l2=torch.nn.functional.cross_entropy(x.where(mask,torch.tensor(-100)),y.where(mask,torch.tensor(-100)),ignore_index=-100,reduction="none")
            #print("function loss: {} \n vs \n {} \n vs \n {}".format(l,l1,l2))
            #mask=mask.to(x.device,non_blocking=True)
            print(x.device)
            print(y.device)
            print(mask.device)
            return torch.nn.functional.cross_entropy(torch.mul(x,mask.to(x.device)),torch.mul(y, mask.to(y.device)))#*mask.shape[0]/mask.sum()
         #negative because when mask is used, the loss is actually the negative of the loss
    
    return loss
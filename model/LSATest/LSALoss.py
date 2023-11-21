from sqlite3 import Row
from numpy import zeros
from sklearn.preprocessing import OneHotEncoder
import torch
import logging
def get_all_LSA_fns():
    #returns list of all other fns in this file that take a tensor as input.
    return {
        "LSAfunction": MyLSA,
        "LSAv3": recursiveLinearSumAssignment_v3,
        "LSAv4":recursiveLinearSumAssignment_v4,
        "LSAstock":outputconversion(linear_sum_assignment),

    }
        #outputconversion(no_for_loop_MyLinearSumAssignment),
        #outputconversion(no_for_loop_triu_MyLinearSumAssignment),
        #outputconversion(no_for_loop_v2_MyLinearSumAssignment),
        #outputconversion(no_for_loop_v2_triu_MyLinearSumAssignment),
        #outputconversion(no_for_loop_v3_MyLinearSumAssignment),
        #outputconversion(no_for_loop_v3_triu_MyLinearSumAssignment),

        #"recursive fn":outputconversion(recursiveLinearSumAssignment), # wasn't working in local test ?
        # "recursive fn2 ":outputconversion(recursiveLinearSumAssignment_v2), # wasn't working in local test ?
        # "recursive fn5":recursiveLinearSumAssignment_v5,


def get_loss_fns():
    return {
        #"LSABaseLoss":loss,     not working with wierd shapes?

        "LSAloss_v1":LSA_loss,# <<<<<<<< 
        "LSAloss_v2":LSA_2loss,
        "LSAloss_v3":LSA_3loss,
        "CombinedLosses_v1":combine_lossesv1,# <<<<<<<
        "CombinedLosses_v2":combine_lossesv2,
        "CombinedLosses_v3":combine_lossesv3,
        "CELoss":base_loss,
    }
def loss(one_hot,x):
    #this only works for 2d SQUARE matrices
    #so we remove rows and columns that are all zeros
    # print(x.shape)
    # print(one_hot.shape)
    locations=one_hot

    # #While the sum of all rows and columns is not equal to 1, we need to keep going
    # while torch.any(torch.sum(locations,dim=0,keepdim=False)!=1) or torch.any(torch.sum(locations,dim=1,keepdim=False)!=1):
    #     try:
    #         dim0index=torch.sum(one_hot,dim=0,keepdim=False)==1
    #         dim1index=torch.sum(one_hot,dim=1,keepdim=False)==1
    #         locations=one_hot[dim1index][:,dim0index]

    #         x= x[dim1index][:,dim0index]
    #     except Exception as e:
    #         print("error")
    #         # print(e)
    #         # print(one_hot)
    #         # print(x)
    #         break

    one_hot=locations.int()
    
    
    (xi,indices)=torch.nonzero(one_hot,as_tuple=True) # not sure why this doesnt work, and seems to require LSA  for maths to worrk! 
    i=torch.arange(indices.shape[0],device=indices.device)
    index=indices.clone()
    counts=torch.zeros_like(indices)
    foundself=torch.zeros_like(indices,dtype=torch.bool)
    while torch.any(torch.logical_not(foundself)) and (torch.sum(foundself)<indices.shape[0]):
        index[:]=indices[index]#### herea 
        counts=counts+torch.logical_not(foundself).int()
        foundself=torch.logical_or(foundself,indices[index]==i)
    values=x*one_hot
    values=values[one_hot==1]
    return torch.sum(counts*values) , ((counts*values.float()).unsqueeze(1))@ ((counts*values).unsqueeze(0).float())



def LSA_loss(indices,Tensor):
    #Take the LSA of the tensor, and then use the LSA to index the tensor
    #assert that indices is 2d and has the same number of elements as the input tensor and is boolean
    # assert indices.dtype==torch.bool
    reward=Tensor*indices.int()
    Cost=Tensor*torch.logical_not(indices.bool()).int()

    output= Cost-reward
    
    return torch.sum(output), output


from functools import reduce
def LSA_2loss(one_hot,x):
    #this only works for 2d SQUARE matrices
    #so we remove rows and columns that are all zeros
    one_hot=one_hot.int()
    #we have to do SOMETHING about empty rows and columns, and rows with duplicate items. 
    #we *could* remove them, but that would change the size of the matrix, and we want to keep the size the same.
    #step 1, replace all empty rows and columns with the logical or of the rows and columns with 2 or more items
    #step 2, remove rows and columns that are still empty.
    sums_of_rows=torch.sum(one_hot,dim=0,keepdim=False)
    sums_of_cols=torch.sum(one_hot,dim=1,keepdim=False)
    #step 1
    one_hot[sums_of_rows==0]=torch.sum(one_hot[sums_of_rows>1],dim=0).bool().int().unsqueeze(0)
    #replace with torch.select instead of indexing

    one_hot[:,sums_of_cols==0]=torch.sum(one_hot[:,sums_of_cols>2],dim=1).bool().int().unsqueeze(1)
    #step 2
    # sums_of_rows=torch.sum(one_hot,dim=0,keepdim=False)
    # sums_of_cols=torch.sum(one_hot,dim=1,keepdim=False)
    # one_hot=one_hot[sums_of_cols>0][:,sums_of_rows>0]
    # x=x[sums_of_cols>0][:,sums_of_rows>0]

    #print(one_hot.shape)
    #now we have a square matrix with no empty rows or columns
    (xi,indices)=torch.nonzero(one_hot,as_tuple=True)
    '''    
     # not sure why this doesnt work, and seems to require LSA  for maths to worrk! 

    index=indices.clone().sort().indices
    print(index)
    counts=torch.zeros_like(indices,dtype=torch.int)
    foundself=torch.zeros_like(indices,dtype=torch.int)
    for i in range(indices.shape[0]):
        print(counts)

        counts =torch.add(counts, 1-foundself)
        foundself=torch.logical_or(foundself.to(dtype=torch.bool),indices[index]==torch.arange(indices.shape[0],device=indices.device)).to(dtype=torch.int)
        index[:]=indices[index]#### herea??
    '''
    ## because our one_hot is now potentially 1 or more per row, we need to do this in a loop
    #we're going to iterate through our rows,
    #for each row, we're going to find the index of the first 1, and then find the index of the next 1, and then the next, and so on
    #step1, 
    rows=torch.zeros_like(one_hot,dtype=torch.bool,device=one_hot.device)
    #fill diagonal with 1s
    counts=torch.zeros(one_hot.shape[0],dtype=torch.int,device=one_hot.device)
    rows.fill_diagonal_(True)
    #step2
    for j,startrow in enumerate(rows):
        original=startrow.clone()
        results=torch.zeros((startrow.shape[0],startrow.shape[0]),dtype=torch.int,device=startrow.device)
        for i in range(startrow.shape[0]):
            startrow=one_hot[startrow].bool()
            #print("retrieved",startrow)

            #this may return multiple columns, so we need to do a logical or of all the columns
            startrow=torch.sum(startrow,dim=0,keepdim=False).bool()
            #print(startrow)
            results[i]=startrow
        #now we have a matrix of 1s and 0s, where each row is a row of the original matrix

        #results is the matrix of locations, 
        #we want to find the index of the first time that the original row is true

        #step 1: only select the colunms(s) of the original row
        #step 2: find the first row that is true
        #step 3: find the index of the first row that is true

        #step1:
        results=results[original]
        #step2:
        results=torch.argmax(results,dim=1)
        #step3:
        counts[j]=torch.sum(results)



        #print(counts)


    values=x*one_hot
    positives=x* one_hot
    positives=torch.abs(positives)*counts #.unsqueeze suggested?
    
    #positives[torch.nonzero(one_hot,as_tuple=True)]=positives[torch.nonzero(one_hot,as_tuple=True)]*counts #.unsqueeze suggested?
    negatives=x * (1-one_hot)
    values=values[one_hot==1]
    return torch.sum(torch.abs(positives)), torch.abs(positives)+negatives

def LSA_3loss(one_hot,x):
    one_hot=one_hot.int()
    locations=one_hot
 
        
    (xi,indices)=torch.nonzero(locations,as_tuple=True) # not sure why this doesnt work, and seems to require LSA  for maths to worrk! 

    index=indices.clone()
    counts=torch.zeros_like(indices)
    foundself=torch.zeros_like(indices)
    for i in range(indices.shape[0]):
        index[:]=indices[index]#### herea?? 
        counts += torch.logical_not(foundself)
        foundself=torch.logical_or(foundself,indices[index]==torch.arange(indices.shape[0],device=indices.device))
    values=x*one_hot
    positives=x* one_hot
    positives[xi,indices]=positives[xi,indices]*counts #.unsqueeze suggested?
    negatives=x * (1-one_hot)
    values=values[one_hot==1]
    return torch.sum(torch.abs(positives))+torch.sum(negatives), torch.abs(positives)+negatives


# Define some LSA Methods for testing
from typing import Callable, final

import torch

from scipy.optimize import linear_sum_assignment

from functools import partial
def outputconversion(func): #converts the output of a function back to 1-hot tensor
    def wrapper(*args, **kwargs):
        func=kwargs.pop("func")
        args=list(args)
        x=args.pop(0)
        output=torch.zeros_like(x)

        x1,y1=func(x.cpu().detach(), *args, **kwargs)
        try:
            output[x1,y1]=1
        except:
            output[y1,x1]=1
        return output
    
    return partial(wrapper,func=func)



CELoss=torch.nn.CrossEntropyLoss(reduction="none",)
def base_loss(indices,Tensor):
    labels=torch.arange(Tensor.shape[0],device=Tensor.device)
    #do CE loss in both directions
    loss1=CELoss(Tensor,labels)
    loss2=CELoss(Tensor.T,labels)
    loss= loss1+loss2
    return loss.mean(), loss1.unsqueeze(1)@loss2.unsqueeze(0)

def combine_lossesv1(indices,Tensor):
    loss,logits=base_loss(indices,Tensor)
    loss2,logits2=LSA_loss(indices,Tensor)
    return loss+loss2,logits+logits2
def combine_lossesv2(indices,Tensor):
    loss,logits=base_loss(indices,Tensor)
    loss2,logits2=LSA_2loss(indices,Tensor)
    return loss+loss2,logits+logits2
def combine_lossesv3(indices,Tensor):
    loss,logits=base_loss(indices,Tensor)
    loss2,logits2=LSA_3loss(indices,Tensor)
    return loss+loss2,logits+logits2



def MyLSA(TruthTensor, maximize=True,lookahead=2):
    '''
    If Maximize is False, I'm trying to minimize the costs. 
    This means that the mask must instead make all the weights far above all the others - 'inf' kind of thing. 
    '''
    #assert truthtensor is 2d and nonzero
    mask=torch.zeros_like(TruthTensor)
    results=torch.zeros_like(TruthTensor,dtype=torch.bool)

    finder=torch.argmax if maximize else torch.argmin
    TruthTensor=TruthTensor-(torch.min(torch.min(TruthTensor)))
    replaceval=-1 if maximize else (torch.max(torch.max(TruthTensor)))
    #add a small amount of noise to the tensor to break ties
    TruthTensor=TruthTensor+torch.randn_like(TruthTensor)*1e-6
    dimsizes=torch.tensor(TruthTensor.shape)
    #select index of the smallest value
    bigdim=torch.argmax(dimsizes).item()   # 0 
    small_dim=1-bigdim          # 1
    
    for i in range(TruthTensor.shape[small_dim]-1): # number of rows
        
        arr=torch.where(mask==1,replaceval,TruthTensor)
        deltas=torch.diff(torch.topk(arr,lookahead,dim=bigdim,largest=maximize).values,n=lookahead-1,dim=bigdim)
        col_index=torch.argmax(torch.abs(deltas),dim=small_dim) #this is the column to grab,  Note this measures step so its not important to do argmin...
        row_index=finder(torch.select(arr,small_dim,col_index))
        torch.select(mask,small_dim,col_index).fill_(1)
        torch.select(mask,bigdim,row_index).fill_(1)

        torch.select(torch.select(results,small_dim,col_index),0,row_index).fill_(True)
        # plt.subplot(1,3,1)
        # plt.imshow(arr.detach().cpu().numpy())
        # plt.subplot(1,3,2)
        # plt.imshow(mask.detach().cpu().numpy())
        # plt.subplot(1,3,3)
        # plt.imshow(results.detach().cpu().numpy())
        # plt.show()


    return torch.logical_or(results,torch.logical_not(mask))


def MyLinearSumAssignment(TruthTensor, maximize=True,lookahead=2):
    return MyLSA(TruthTensor, maximize=maximize,lookahead=lookahead).nonzero(as_tuple=True)

def no_for_loop_triu_MyLinearSumAssignment(rewards:torch.Tensor,maximize=True):
    cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(0,torch.max,torch.sub,torch.argmax))[maximize]
    remove=torch.ones_like(rewards,dtype=torch.bool).triu().unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    Costs=next_highest_fn(weights,dim=0).values
    Locations=comb_fn(rewards,Costs)
    dimsizes=torch.tensor(rewards.shape)
    #select index of the smallest value
    dim=torch.argmax(dimsizes)
    col_index=final_fn(Locations,dim=dim)
    
    return torch.arange(Locations.shape[0],device=Locations.device),col_index

def no_for_loop_MyLinearSumAssignment(rewards:torch.Tensor,maximize=True,tril=False):

    cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(0,torch.max,torch.sub,torch.argmax))[maximize]
    remove=torch.zeros_like(rewards,dtype=torch.bool).fill_diagonal_(1).unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    Costs=next_highest_fn(weights,dim=0).values
    #plt.show(plt.imshow(Costs.cpu().numpy()))
    
    Locations=comb_fn(rewards,Costs)
    dimsizes=torch.tensor(rewards.shape)
    #select index of the smallest value
    dim=torch.argmax(dimsizes)
    col_index=final_fn(Locations,dim=dim)

    return torch.arange(Locations.shape[0],device=Locations.device),col_index
def no_for_loop_v2_MyLinearSumAssignment(rewards:torch.Tensor,maximize=True,tril=False):
    cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(0,torch.max,torch.sub,torch.argmax))[maximize] 
    remove=torch.zeros_like(rewards,dtype=torch.bool).fill_diagonal_(1).unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights1=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    weights2=weights.masked_fill(remove.permute(1,0,2),cost_neg)#.permute(1,2,0)

    Costs=next_highest_fn(weights1,dim=1).values
    Costs2=next_highest_fn(weights2,dim=0).values

    Cost_total=Costs+Costs2
    Locations=rewards - Cost_total/2
    #find the dim with the smallest value
    dimsizes=torch.tensor(rewards.shape)
    #select index of the smallest value
    dim=torch.argmax(dimsizes)
    col_index=final_fn(Locations,dim=dim.item())

    return torch.arange(Locations.shape[0],device=Locations.device),col_index

def no_for_loop_v2_triu_MyLinearSumAssignment(rewards:torch.Tensor,maximize=True,tril=False):

    cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(0,torch.max,torch.sub,torch.argmax))[maximize]
   
    remove=torch.ones_like(rewards,dtype=torch.bool).tril().unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights1=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    weights2=weights.masked_fill(remove.permute(1,0,2),cost_neg)#.permute(1,2,0)
    Costs=next_highest_fn(weights1,dim=1).values 
    Costs2=next_highest_fn(weights2,dim=0).values
    Cost_total=Costs+Costs2
    Locations=rewards - Cost_total/2
    dimsizes=torch.tensor(rewards.shape)
    #select index of the smallest value
    dim=torch.argmax(dimsizes)
    col_index=final_fn(Locations,dim=dim)
    row_index=final_fn(Locations,dim=1-dim)
    #cross check these,
    #convert to 1-hot
    Col_one_hot=torch.nn.functional.one_hot(col_index,num_classes=rewards.shape[dim])
    Row_one_hot=torch.nn.functional.one_hot(row_index,num_classes=rewards.shape[1-dim])
    #cross check these,
    final=torch.logical_and(Col_one_hot,Row_one_hot)
    return torch.nonzero(final,as_tuple=True)


def no_for_loop_v3_MyLinearSumAssignment(rewards:torch.Tensor,maximize=True,tril=False):
    cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(0,torch.max,torch.sub,torch.argmax))[maximize] 
    remove=torch.zeros_like(rewards,dtype=torch.bool).fill_diagonal_(1).unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights1=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    weights2=weights.masked_fill(remove.permute(1,0,2),cost_neg)#.permute(1,2,0)

    Costs=next_highest_fn(weights1,dim=1).values
    Costs2=next_highest_fn(weights2,dim=0).values

    Cost_total=Costs+Costs2
    Locations=rewards - Cost_total/2
    dimsizes=torch.tensor(rewards.shape)
    #select index of the smallest value
    dim=torch.argmax(dimsizes)
    col_index=final_fn(Locations,dim=dim)

    return torch.arange(Locations.shape[0],device=Locations.device),col_index

def no_for_loop_v3_triu_MyLinearSumAssignment(rewards:torch.Tensor,maximize=True,tril=False):

    cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(0,torch.max,torch.sub,torch.argmax))[maximize]
   
    remove=torch.ones_like(rewards,dtype=torch.bool).tril().unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights1=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    weights2=weights.masked_fill(remove.permute(1,0,2),cost_neg)#.permute(1,2,0)
    Costs=next_highest_fn(weights1,dim=1).values 
    Costs2=next_highest_fn(weights2,dim=0).values
    Cost_total=Costs+Costs2
    Locations=rewards - Cost_total/2
    dimsizes=torch.tensor(rewards.shape)
    #select index of the smallest value
    dim=torch.argmax(dimsizes)
    col_index=final_fn(Locations,dim=dim)
    return torch.arange(Locations.shape[0],device=Locations.device),col_index


def reduceLinearSumAssignment(rewards:torch.Tensor,cost_neg:torch.Tensor,next_highest_fn: Callable,remove,dim=1):
    removehw,removehwT=remove
    if dim==0:
        removehw,removehwT=removehwT,removehw

    # rewards is HW, weights is  B(H) H W 
    weights=rewards.unsqueeze(0).repeat(*tuple([rewards.shape[0]]+ [1]*len(rewards.shape)))
    #rewards is shape hw, weights is shape h w w
    weights=weights.masked_fill(removehw,cost_neg)#.permute(1,2,0)
    #draw(weights.cpu())
    Costs=next_highest_fn(weights,dim=dim).values #should not be 0  
    #draw(Costs.cpu())
    #print(Costs.shape)
    weights2=rewards.T.unsqueeze(0).repeat(*tuple([rewards.shape[1]]+ [1]*len(rewards.shape)))

    weights2=weights2.masked_fill(removehwT,cost_neg)#.permute(1,2,0)
    Costs2=next_highest_fn(weights2,dim=dim).values #should not be 0

    Cost_total= torch.add(Costs,Costs2.T)
    return Cost_total

def reduceLinearSumAssignment_vm(rewards:torch.Tensor,cost_neg:torch.Tensor,next_highest_fn: Callable,remove:torch.Tensor):
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights1=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    weights2=weights.masked_fill(remove.permute(1,0,2),cost_neg)#.permute(1,2,0)
    
    Costs=next_highest_fn(weights1,dim=1).values #should not be 0  
    Costs2=next_highest_fn(weights2,dim=0).values #should not be 0

    #Cost_total=Costs+Costs2 # max,min or plus? min max seem to be worse than plus
    Cost_total= torch.add(Costs,Costs2)
    
    return Cost_total
def reduceLinearSumAssignment_v2(rewards:torch.Tensor,maximize=False):
    Topv,topi=rewards.topk(k=2,dim=1,largest=maximize)
    costs=Topv[:,0].unsqueeze(1).repeat(1,rewards.shape[-1])
    #print(costs.shape)
    one_hot=torch.zeros_like(rewards, dtype=torch.bool).scatter_(1,topi[:,0].unsqueeze(1),1)
    #draw(one_hot.to(dtype=torch.float,device="cpu"))
    costs[one_hot]=Topv[:,1]
    #draw(costs.cpu())
    topv2,topi2=rewards.topk(k=2,dim=0,largest=maximize)
    costs2=topv2[0].unsqueeze(0).repeat(rewards.shape[0],1)
    one_hot2 = torch.zeros_like(rewards, dtype=torch.bool).scatter_(0, topi2[0].unsqueeze(0), 1)
    costs2[one_hot2]=topv2[1]
    #draw(costs2.cpu())
    Cost_total= costs2+costs
    #draw(Cost_total.cpu())

    return Cost_total


def reduceLinearSumAssignment_v3(rewards:torch.Tensor,maximize=True):

    #30,32
    TotalCosts= torch.max(rewards,dim=1,keepdim=True).values + torch.max(rewards,dim=0,keepdim=True).values
    #30,32
    diffs= torch.diff(rewards.topk(k=2,dim=1,largest=maximize).values,dim=1)
    #30,1
    diffs2= torch.diff(rewards.topk(k=2,dim=0,largest=maximize).values,dim=0)
    #1,32
    one_hot=rewards==torch.max(rewards,dim=1)
    #30,32
    one_hot=one_hot*diffs
    #30,32
    one_hot2=rewards==torch.max(rewards,dim=0)
    one_hot2=one_hot2 * diffs2
    deltas=one_hot+one_hot2
    totalCosts=TotalCosts+deltas
    return totalCosts


def reduceLinearSumAssignment_v4(rewards:torch.Tensor,maximize=True):

    #30,32
    TotalCosts= torch.max(rewards,dim=1,keepdim=True).values + torch.max(rewards,dim=0,keepdim=True).values
    #30,32
    #diffs= torch.diff(rewards.topk(k=2,dim=1,largest=maximize).values,dim=1)
    #30,1
    diffs2= torch.diff(rewards.topk(k=2,dim=0,largest=maximize).values,dim=0)
    #1,32
    #one_hot=torch.nn.functional.one_hot(torch.argmax(rewards,dim=1),num_classes=rewards.shape[1])
    #30,32
    #one_hot=one_hot*diffs
    #30,32
    one_hot2=torch.nn.functional.one_hot(torch.argmax(rewards,dim=0),num_classes=rewards.shape[0])
    #32,30

    one_hot2=one_hot2.T * diffs2
    #deltas=one_hot+one_hot2
    totalCosts=TotalCosts+one_hot2#deltas
    return totalCosts

def recursiveLinearSumAssignment(rewards:torch.Tensor,maximize=False,factor=0.8):
    cost_neg,next_highest_fn,comb_fn,final_fn=((torch.tensor(float('inf')),torch.min,torch.add,torch.argmin),(torch.tensor(float('-inf')),torch.max,torch.sub,torch.argmax))[maximize] 
    #cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(-1e9,torch.max,torch.sub,torch.argmax))[maximize] 
    #we need to make a mask that holds the diagonal of a H x H matrix repeated B times, and then one with the diagonal of a BxB matrix repeated H times
    # rewards=rewards-torch.min(torch.min(rewards,dim=1,keepdim=True).values,dim=0).values
    #^^ should make no difference.....but always worth checking! 
    #rewards=rewards-  (rewards.min())
    #col_index=None
    dimsizes=torch.tensor(rewards.shape)
    #select index of the smallest value
    bigdim=torch.argmax(dimsizes)
    small_dim=torch.argmin(dimsizes)
    output=torch.zeros_like(rewards,dtype=torch.int8)
    removeHHB=torch.zeros((rewards.shape[small_dim],rewards.shape[small_dim]),dtype=torch.bool,device=rewards.device).fill_diagonal_(1).unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape) + [rewards.shape[bigdim]]))
    removeBBH=torch.zeros((rewards.shape[bigdim],rewards.shape[bigdim]),dtype=torch.bool,device=rewards.device).fill_diagonal_(1).unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[small_dim]]))
    for i in range(10):
        cost=reduceLinearSumAssignment(rewards,cost_neg,next_highest_fn,(removeHHB,removeBBH),dim=bigdim)
        rewards=rewards - (cost/factor)
    #col_index=final_fn(rewards,dim=bigdim)
    #return torch.arange(rewards.shape[0],device=rewards.device),col_index
    # output=(torch.arange(rewards.shape[small_dim],device=rewards.device),col_index) if small_dim==1 else (col_index,torch.arange(rewards.shape[small_dim],device=rewards.device))
    # return output
    #find size+1th value, and mask out all values above or below it it
    cutoff=torch.topk(rewards.flatten(),rewards.shape[small_dim]+1,largest=maximize,sorted=True).values[-1]
    #find all values above or below it, we're going to remove the cutoff value from the tensor, then clamp the tensor to 0,1,
    if maximize:
        rewards=torch.sub(rewards,cutoff)
        rewards=torch.clamp(rewards,min=0,max=1)
    else:
        rewards=torch.sub(cutoff,rewards)
        rewards=torch.clamp(rewards,min=0,max=1)
    return torch.nonzero(rewards,as_tuple=True)

def recursiveLinearSumAssignment_v2(rewards:torch.Tensor,maximize=True,factor=1):
    cost_neg,next_highest_fn,comb_fn,final_fn=((torch.tensor(float('inf')),torch.min,torch.add,torch.argmin),(torch.tensor(float('-inf')),torch.max,torch.sub,torch.argmax))[maximize] 
    #cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(-1e9,torch.max,torch.sub,torch.argmax))[maximize] 
    # remove=torch.zeros_like(rewards,dtype=torch.bool).fill_diagonal_(1).unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    # col_index=None
    dimsizes=torch.tensor(rewards.shape)
    #select index of the smallest value
    bigdim=torch.argmax(dimsizes)
    small_dim=torch.argmin(dimsizes)
    for i in range(min(rewards.shape[-2:])):
        cost2=reduceLinearSumAssignment_v2(rewards,maximize=maximize)
        rewards=rewards- (cost2/factor)# can remove
    col_index=final_fn(rewards,dim=bigdim)
        #return torch.arange(rewards.shape[0],device=rewards.device),col_index
    # logging.warning("small dim"+str(small_dim))
    output=(torch.arange(rewards.shape[small_dim],device=rewards.device),col_index) if small_dim==1 else (col_index,torch.arange(rewards.shape[small_dim],device=rewards.device))
    return output
def recursiveLinearSumAssignment_v5(rewards:torch.Tensor,maximize=True,factor=10):
    #create tensor of ints
    output=torch.zeros_like(rewards,dtype=torch.int8)
    #print("out1")
    #draw(output)
    #cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(-1e9,torch.max,torch.sub,torch.argmax))[maximize] 
    # remove=torch.zeros_like(rewards,dtype=torch.bool).fill_diagonal_(1).unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    # col_index=None
    rewards=rewards.clone()
    dimsizes=torch.tensor(rewards.shape)
    #select index of the smallest value
    bigdim=torch.argmax(dimsizes)
    minmax=torch.min if maximize else torch.max
    small_dim=torch.argmin(dimsizes)
    final=torch.greater_equal if maximize else torch.less_equal
    for i in range(10):
        cost2=reduceLinearSumAssignment_v2(rewards,maximize=maximize)
        rewards=rewards- (cost2/factor)# can remove
    
    #draw(output)
    #draw(rewards)    
    cutoff=minmax(torch.topk(rewards.flatten(),rewards.shape[torch.argmin(dimsizes)],largest=maximize,sorted=True).values)    
    return final(rewards,cutoff)



def recursiveLinearSumAssignment_v3(rewards:torch.Tensor,maximize=True,factor=1):
    # final_fn=torch.argmax if maximize else torch.argmin
    #cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(-1e9,torch.max,torch.sub,torch.argmax))[maximize] 
    #remove=torch.zeros_like(rewards,dtype=torch.bool).fill_diagonal_(1).unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    # col_index=None
    maxmin=torch.max if maximize else torch.min
    # dimsizes=torch.tensor(rewards.shape)
    #select index of the smallest value
    # dim=torch.argmax(dimsizes)
    for i in range(7):
        cost=reduceLinearSumAssignment_v3(rewards,maximize=maximize)
        rewards=rewards-(cost/factor)# can remove
        #draw(rewards.cpu())
        #? why is this suggested??? rewards,_,_=torch.svd(rewards)
        #rewards=rewards ** (rewards-cost/factor) #times here makes it very spiky! 
    # col_index=final_fn(rewards,dim=dim)
    # row_index=final_fn(rewards,dim=1-dim)
    #cross check these,
    #convert to 1-hot
    # Col_one_hot=torch.nn.functional.one_hot(col_index,num_classes=rewards.shape[dim])
    Col_one_hot= (rewards== (maxmin(rewards,dim=0,keepdim=True).values))
    #print(Col_one_hot)
    Row_one_hot= (rewards == (maxmin(rewards,dim=1,keepdim=True).values))
    #cross check these,
    #print(Row_one_hot)
    return torch.logical_or(Col_one_hot,Row_one_hot)
    


def recursiveLinearSumAssignment_v4(rewards:torch.Tensor,maximize=True,factor=1):
    # final_fn=torch.argmax if maximize else torch.argmin
    #cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(-1e9,torch.max,torch.sub,torch.argmax))[maximize] 
    #remove=torch.zeros_like(rewards,dtype=torch.bool).fill_diagonal_(1).unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    #y_values=[]
    # col_index=None
    final=torch.greater_equal if maximize else torch.less_equal
    dimsizes=torch.tensor(rewards.shape)
    #select index of the smallest value
    rewards=rewards- (rewards.min().min())
    minmax=torch.min if maximize else torch.max
    # dim=torch.argmax(dimsizes)
    for _ in range(10):
        cost=reduceLinearSumAssignment_v3(rewards,maximize=maximize)
        rewards=rewards-(cost/factor)# can remove
        #draw(rewards.cpu())
        #? why is this suggested??? rewards,_,_=torch.svd(rewards)
        #rewards=rewards ** (rewards-cost/factor) #times here makes it very spiky! 
        # col_index=final_fn(rewards,dim=dim)
        #x,y=torch.arange(rewards.shape[0],device=rewards.device),col_index
        #y_values.append(col_index)
    cutoff=minmax(torch.topk(rewards.flatten(),rewards.shape[torch.argmin(dimsizes)],largest=maximize,sorted=True).values)    
    return final(rewards,cutoff)



#define main function
if __name__ == "__main__":
    #if this file is run, check we have a backwards gradient for the loss function
    from tqdm import tqdm
    import time,os
    import matplotlib.pyplot as plt
    torch.autograd.set_detect_anomaly(True)
    #for our test we need a custom layer that transposes the input
    class MyLinear(torch.nn.Linear):
        def forward(self,x):
            return super().forward(x.T).T


    os.makedirs("results",exist_ok=True)


    results={}
    for name,loss_fn in get_loss_fns().items():
        for lsa_name,lsa_fn in get_all_LSA_fns().items():
                        
            linear_layer= torch.nn.Linear(100,100)
            linear_layer2= torch.nn.Linear(100,100)#MyLinear(100,100)
            linear_layer3= torch.nn.Linear(100,100)
            linear_layer4= torch.nn.Linear(100,100)#MyLinear(100,100)
            linear_layer5= torch.nn.Linear(100,100)
            linear_layer6= torch.nn.Linear(100,100)#MyLinear(100,100)
            #list of activation functions
            activations=[
                torch.nn.ReLU(),
                torch.nn.Tanh(),
                torch.nn.Sigmoid(),
                torch.nn.Softmax(dim=1),
                torch.nn.Softmax(dim=0),
                torch.nn.Softmax(dim=-1),
                torch.nn.Softmax(dim=-2),
                #activation to make all positive
            ]

            model=torch.nn.Sequential(linear_layer,
                            torch.nn.ReLU(),
                            linear_layer2,
                            torch.nn.ReLU(),
                            linear_layer3,
                            torch.nn.ReLU(),

                            torch.nn.Softmax(dim=0),
                            linear_layer4,
                            torch.nn.Sigmoid(),
                            linear_layer5,
                            torch.nn.ReLU(),
                            linear_layer6,
                            # normalisation layer here
                            torch.nn.ReLU())
            if torch.cuda.is_available():
                model=model.cuda()
            optimizer=torch.optim.AdamW(model.parameters(),lr=0.0001)
            print("Testing {} with {}".format(name,lsa_name))
            prog_bar=tqdm(range(100000))
            training_Losses=[]
            #begin timer
            start=time.time()
            
            for i in prog_bar:# do 1000 iterations

    
                input_tensor=torch.randn(10,10,requires_grad=True,device="cuda" if torch.cuda.is_available() else "cpu")
                #in theory.... this could be batched to run faster. However, in this case, it's not really necessary and this is meant to simulate a single batch of data anyway. 
                output=model(input_tensor.flatten()).reshape(input_tensor.shape)
                
                LSA=lsa_fn(output)
                loss,logits=loss_fn(LSA,output)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                training_Losses.append(loss.item())
                #update tqdm bar
                log={"Loss":loss.item()}
                prog_bar.set_postfix(log)
            #plot final input and logits next to each other, and the loss in the title with time.
            end=time.time()

            t=end-start
            #round to 2dp
            t=round(t,2)
            input_tensor=torch.randn(10,10,requires_grad=True,device="cuda" if torch.cuda.is_available() else "cpu")
             
            output=model(input_tensor.flatten()).reshape(input_tensor.shape)
            
            LSA=lsa_fn(output)
            loss,logits=loss_fn(LSA,output)


            plt.subplot(2,2,1)
            plt.title("Input Tensor")
            plt.imshow(input_tensor.detach().cpu().numpy())
            plt.subplot(2,2,2)
            plt.title("Output Tensor")
            plt.imshow(output.detach().cpu().numpy())
            plt.subplot(2,2,3)
            plt.title("Logits Loss: {}".format(loss.item()))
            plt.imshow(logits.detach().cpu().numpy())
            plt.subplot(2,2,4)
            plt.title("Loss over time: {}".format(t))
            plt.plot(training_Losses)


            plt.suptitle("{} and {}, t={}".format(name,lsa_name,t))
            plt.savefig(os.path.join("results","adamWfinal output for {} and {}.png".format(name,lsa_name)))
            plt.clf()
            #end timer
            results.update({"{} and {} took {}".format(name,lsa_name,t):training_Losses})

    #plot results
    for loss_name in get_loss_fns().keys():
        for name,losses in results.items():
            if loss_name in name:

                plt.plot(losses,label=name)
                #shows the names, but they are too long and too big to and cover the graph, so we will move them to the side, but we need to make sure they are all visible first

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        #chang elayout so that the legend is not cut off
        plt.tight_layout()

        #the key is too big 
        plt.savefig(os.path.join("results","adamWresults for loss: {}.png".format(loss_name)))
        #clear the plot
        plt.clf()

    for loss_name in get_all_LSA_fns().keys():
        for name,losses in results.items():
            if loss_name in name:

                plt.plot(losses,label=name)
                #shows the names, but they are too long and too big to and cover the graph, so we will move them to the side, but we need to make sure they are all visible first
            #offset the legend to the right, but make sure it is still visible, make the plot bigger if necessary
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        #expand the plot if necessary
        plt.tight_layout()
        #the key is too big 
        plt.savefig(os.path.join("adamWresults for LSA {}.png".format(loss_name)))
        plt.clf()
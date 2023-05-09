

import torch
from functools import partial,reduce
'''
This is research code... it is not clean and it is not commented

If you wish to use it for TPUs, I strongly recommend you refactor your code to use this style of function factory.
 Otherwise your runs will be very slow.

This code is a copy of the LossCalculation.py file, but with the loss functions as functions that take any number of arguments.
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
    ##logfunction=lambda x:x
   
        # if log:
    #     logfunction=logargs
    if logitsversion==0:
        def baseLogits(*args):
            return calculate_loss(*args)
    elif logitsversion==1: 
        def baseLogits(*args):
            return oneminus(calculate_loss2(*args))
    elif logitsversion==2: 
        def baseLogits(*args):
            return oneminus(calculate_loss3(*args))      #one minus here
    elif logitsversion==3:
        #this does not work in any arrangement? 
        def baseLogits(*args):
            return oneminus(calculate_loss4(*args))
    elif logitsversion==4:
        #this does not work in any arrangement either?
        def baseLogits(*args):
            return oneminus(calculate_loss5(*args))
    elif logitsversion==5:
        def baseLogits(*args):
            return oneminus(calculate_loss6(*args))
    elif logitsversion==6:
        def baseLogits(*args):
            return oneminus(calculate_loss1(*args))
    elif logitsversion==13:
        def baseLogits(*args):
            return oneminus(calculate_loss7(*args))
    elif logitsversion==14:
        def baseLogits(*args):
            return torch.sum(oneminus(calculate_loss8(*args)),dim=-1)
    elif logitsversion==7:
        norm=True
        def baseLogits(*args):
            return calculate_lossNorms(*args)
            
    elif logitsversion==8:
        norm=True
        def baseLogits(*args):
            return calculate_lossNormsv2(*args)
        
                
    elif logitsversion==9:
        norm=True
        def baseLogits(*args):
            return calculate_lossNormsv3(*args)
                
    elif logitsversion==10:
        norm=True
        def baseLogits(*args):
            return calculate_lossNormsv4(*args)
        
             
    elif logitsversion==11:
        norm=True
        def baseLogits(*args):
            return calculate_lossNormsv5(*args)
             
    elif logitsversion==12:
        norm=False
        def baseLogits(*args):
            return calculate_lossNormsv5(*args)
    elif logitsversion==15:
        norm=True
        def baseLogits(*args):
            return calculate_lossNormsv6(*args)
    elif logitsversion==16:
        norm=True
        def baseLogits(*args):
            return calculate_lossNormsv7(*args)

    normfunction=lambda x:x
    if norm:
        normfunction=normargs
        def lossfn(*args):
            return baseLogits(*normfunction(*args)) 
    else:
        def lossfn(*args):
            return baseLogits(*args)
    return lossfn



def calculate_lossStock(args):
    I,C1=args[0],args[1]
    #normalize image and text features
    I = I / I.norm(dim=-1, keepdim=True)
    C1 = C1 / C1.norm(dim=-1, keepdim=True)
    #calculate logits
    logits_per_image =  I @ C1.T
    logits_per_text =  C1 @ I.T
    #calculate loss
    return logits_per_image, logits_per_text

def calculate_lossbase(args,norm=True,log=False):
    I,C1=args[0],args[1]
    #normalize image and text features
    I = I / I.norm(dim=-1, keepdim=True)
    C1 = C1 / C1.norm(dim=-1, keepdim=True)
    #calculate logits
    logits_per_image =  I @ C1.T
    #calculate loss
    return logits_per_image

def calculate_loss(  *Args):
    #find number of arguments
    n=len(Args)
    #find first n letters of alphabet
    alphabet=list(map(chr, range(97, 97+n)))
    #create tuples of (letter,argument)
    #we're going to do the next stage in 3s for speed.
    #we want to do einsum with each argument, so we need to create a string of the form "az,bz,cz->abcz"
    #we do this by joining the letters with commas, and then adding the arrow and the letters again
    #print(n)
    finalparts=[]
    components=[]
    parts=[]
    for i in range (0,n,3):
        #check the next 3 arguments exist
        if (i+3)>=n:
            #if not, just add the remaining arguments
            einsumparts=",".join(["{}z"]*(n-i)) + "->" + "{}" + "z"
            einsumparts=einsumparts.format(*alphabet[i:],"".join(alphabet[i:]))
            finalparts.append("".join(alphabet[i:])+"z")
            components.append(einsumparts)
            parts.append([*Args[i:]])
        else:
            einsumparts=",".join(["{}z"]*3) + "->" + "{}" + "z"
            einsumparts=einsumparts.format(*alphabet[i:i+3],"".join(alphabet[i:i+3]))
            finalparts.append("".join(alphabet[i:i+3])+"z")
            components.append(einsumparts)
            parts.append([*Args[i:i+3]])
    #we then do the einsum
    
    # #now we do the einsum with all components and all arguments
    # for component,part in zip(components,parts):
    #     print(component)
    #     print([p.shape for p in part])
    
    #einsums=[torch.einsum(component,*part) for component,part in zip(components,parts)]
    return torch.einsum(",".join(finalparts)+"->"+"".join(alphabet),*[torch.einsum(component,*part) for component,part in zip(components,parts)]) 
    #torch.einsum("abcz,defz->abcdef",torch.einsum("az,bz,cz->abcz",I,C1,C2),torch.einsum("az,bz,cz->abcz",C3,C4,C5))

    
    
def calculate_loss1(  *Args):
    #find number of arguments
    n=len(Args)
    #assume all args are shape BxF
    #each argmuents is going to become BxFx1x... (n dimensions)


    return torch.mean(torch.sqrt(torch.abs(reduce(torch.add,[torch.pow(arg,2).view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]).sub_(
                        torch.pow(reduce(torch.add,[arg.view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]),2),alpha=1/n))),dim=-1)


def calculate_loss2(  *Args):
    n=len(Args)
    
    # for i,arg in enumerate(Args):
    #     print([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])
    return torch.mean(torch.sqrt(reduce(torch.add,[torch.pow(arg,2).view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]).sub_(
                        torch.pow(reduce(torch.add,[arg.view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]),2),alpha=1/n)),dim=-1)
    #frequently ends with Nans, not sure why
    
def calculate_loss3( *Args):
    n=len(Args)

    return torch.sqrt(torch.sum(torch.pow(reduce(torch.add,[torch.pow(arg,2).view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]).sub_(
                        torch.pow(reduce(torch.add,[arg.view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]),2),alpha=1/n),2),dim=-1))
    #results in nan labels 
    


def calculate_loss4(*Args):
    #tested and not working
    n=len(Args)
    return torch.sum(torch.sqrt(reduce(torch.add,[torch.pow(arg,2).view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)])),dim=-1)
def calculate_loss5(*Args):
    n=len(Args)
    return torch.sum(reduce(torch.add,[ torch.pow(arg,2).view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]).sub_(
                    torch.pow(reduce(torch.add,[    arg.view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]),2),alpha=1/n),dim=-1)

    
def calculate_loss6(*Args):
    n=len(Args)
    return torch.sqrt(torch.sum(reduce(torch.add,[torch.pow(arg,2).view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]).sub_(
                            torch.pow(reduce(torch.add,[arg.view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]),2),alpha=1/n),dim=-1))
def calculate_loss7(  *Args):
    #find number of arguments
    n=len(Args)
    return torch.sum(torch.abs(reduce(torch.add,[torch.pow(arg,2).view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]).sub_(
                        torch.pow(reduce(torch.add,[arg.view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]),2),alpha=1/n)),dim=-1)

def calculate_loss8(  *Args):
    #find number of arguments
    n=len(Args)
    return reduce(torch.add,[torch.pow(arg,2).view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]).sub_(
                        torch.pow(reduce(torch.add,[arg.view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]),2),alpha=1/n)
################################################ NORMS ################################################

  
def calculate_lossNorms(  *Args):
    #find number of arguments
    n=len(Args)
    mean=reduce(torch.add,[arg.view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)])
    scalednorm=mean/mean.norm(dim=-1,keepdim=True)
    
    #return dot product of scalednorm and mean x 6
    out=torch.sum(torch.mul(scalednorm,mean),dim=-1) #this....doesnt work see v4
    return out

def calculate_lossNormsv2(*Args):
    #find number of arguments
    n=len(Args)
    sum=reduce(torch.add,[arg.view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)])
    out=sum.pow(2)/sum.norm(dim=-1,keepdim=True)
    out=torch.sum(out,dim=-1)
    return torch.squeeze(out)
    #return dot product of scalednorm and mean x 6
    #return torch.sum(torch.mul(scalednorm,mean),dim=-1)

  
def calculate_lossNormsv3(*Args):
    #find number of arguments
    n=len(Args)
    mean=reduce(torch.add,[torch.div(arg,n).view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)])
    return mean.norm(dim=-1,keepdim=True)

def calculate_lossNormsv4(*Args):
    #assert I.shape[0]==C1.shape[0]==C2.shape[0]==C3.shape[0]==C4.shape[0]==C5.shape[0]
    #check norms are 1
    # assert torch.allclose(I.norm(dim=-1, keepdim=True),torch.ones_like(I.norm(dim=-1, keepdim=True)))
    # assert torch.allclose(C1.norm(dim=-1, keepdim=True),torch.ones_like(C1.norm(dim=-1, keepdim=True)))
    # assert torch.allclose(C2.norm(dim=-1, keepdim=True),torch.ones_like(C2.norm(dim=-1, keepdim=True)))
    # assert torch.allclose(C3.norm(dim=-1, keepdim=True),torch.ones_like(C3.norm(dim=-1, keepdim=True)))
    # assert torch.allclose(C4.norm(dim=-1, keepdim=True),torch.ones_like(C4.norm(dim=-1, keepdim=True)))
    # assert torch.allclose(C5.norm(dim=-1, keepdim=True),torch.ones_like(C5.norm(dim=-1, keepdim=True)))


    # print("norms are 1")
    #find number of arguments
    n=len(Args)
    mean=reduce(torch.add,[arg.view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)])
    mean=torch.div(mean,mean.norm(dim=-1,keepdim=True))
    # print("max value in mean is ",torch.max(mean.flatten()).item())
    # print("min value in mean is ",torch.min(mean.flatten()).item())

    # print("max similarity to mean is ",torch.max(torch.sum(torch.mul(mean,I.view(I.shape[0],1,1,1,1,1,-1)),dim=-1).flatten()).item())
    
    return reduce(torch.add,[torch.sum(torch.mul(mean,arg.view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) ),dim=-1)for i,arg in enumerate(Args)])
    #return dot product of scalednorm and mean x 6

    
def calculate_lossNormsv5(*Args):
 
    #find number of arguments   
    n=len(Args)
    mean=reduce(torch.add,[arg.view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)])
    #perform dot similarity between input and normalised mean of other inputs
    return reduce(torch.add,[torch.sum(
                                        torch.mul(
                                            torch.div(
                                                    torch.sub(mean,
                                                              arg.view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1]))),
                                                    torch.sub(mean,
                                                              arg.view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1]))).norm(dim=-1,keepdim=True)),
                  arg.view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1]))),dim=-1) for i,arg in enumerate(Args)])
        # torch.sum(torch.mul(torch.div(mean.sub(I.view(I.shape[0],1,1,1,1,1,-1)),mean.sub(I.view(I.shape[0],1,1,1,1,1,-1)).norm(dim=-1,keepdim=True)),I.view(I.shape[0],1,1,1,1,1,-1)),dim=-1),#replicate down wards
        #                      torch.sum(torch.mul(torch.div(mean.sub(C1.view(1,C1.shape[0],1,1,1,1,-1)),mean.sub(C1.view(1,C1.shape[0],1,1,1,1,-1)).norm(dim=-1,keepdim=True)),C1.view(1,C1.shape[0],1,1,1,1,-1)),dim=-1),
        #                      torch.sum(torch.mul(torch.div(mean.sub(C2.view(1,1,C2.shape[0],1,1,1,-1)),mean.sub(C2.view(1,1,C2.shape[0],1,1,1,-1)).norm(dim=-1,keepdim=True)),C2.view(1,1,C2.shape[0],1,1,1,-1)),dim=-1),
        #                      torch.sum(torch.mul(torch.div(mean.sub(C3.view(1,1,1,C3.shape[0],1,1,-1)),mean.sub(C3.view(1,1,1,C3.shape[0],1,1,-1)).norm(dim=-1,keepdim=True)),C3.view(1,1,1,C3.shape[0],1,1,-1)),dim=-1),
        #                      torch.sum(torch.mul(torch.div(mean.sub(C4.view(1,1,1,1,C4.shape[0],1,-1)),mean.sub(C4.view(1,1,1,1,C4.shape[0],1,-1)).norm(dim=-1,keepdim=True)),C4.view(1,1,1,1,C4.shape[0],1,-1)),dim=-1),
        #                      torch.sum(torch.mul(torch.div(mean.sub(C5.view(1,1,1,1,1,C5.shape[0],-1)),mean.sub(C5.view(1,1,1,1,1,C5.shape[0],-1)).norm(dim=-1,keepdim=True)),C5.view(1,1,1,1,1,C5.shape[0],-1)),dim=-1)])
    #return dot product of scalednorm and mean x 6

       
def calculate_lossNormsv6(*Args):
    '''
    Trying to replicate the following numpy code in batch form

        mean=(x[i]+ y[j])/2
        deltaH= np.linalg.norm(np.array([x[i],y[j]]))- np.sqrt(2*np.power(mean,2))
        z[i,j]= mean-deltaH#np.linalg.norm(np.array([x[i],y[j]]))

        so z = root(6*mean^2)+mean - norm(x,y)
    '''
    #find number of arguments
    n=len(Args)
    mean=reduce(torch.add,[ torch.div(torch.abs(arg),n).view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)])
    mean=torch.sub(mean,torch.sub(torch.abs(torch.sqrt(n*torch.pow(mean,2))),
            torch.sqrt(reduce(torch.add,[torch.pow(arg,2).view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]))))                    
                         
    
    return torch.sum(mean,dim=-1)

def calculate_lossNormsv7(*Args):
    n=len(Args)
    mean2 = reduce(torch.add,[torch.div(torch.abs(arg),n).view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]) + torch.sqrt(torch.mul(reduce(torch.add,[ torch.div(torch.abs(arg),n).view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]).pow(2),n))
    #norm =sum(abs(x)**ord)**(1./ord)
    norm=torch.sqrt(reduce(torch.add,[ torch.pow(arg,2).view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]))
    return torch.sum(torch.sub(mean2,norm),dim=-1)


############
#loss functions

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


def get_loss_calc(reduction='sum',ver=0,mask=None):
    #returns a function that calculates the loss from target labels and logits and masks the output with the mask before reduction
#    is used, the loss is actually the negative of the loss
    #print(mask)
    if len(mask.shape)>1:
       masks=torch.unique(torch.flatten(mask,0,-1),dim=-1,sorted=False)
       print("mask shape is:", mask.shape)
       print("bool shape is:", (mask==6).shape)
       st=torch.stack([mask==masks[i] for i in range(len(masks))],dim=-1)
       print("stack shape is:", st.shape)
    else:
        #print("masks:", mask.shape)
        masks=None
        ver=0
    if ver==0:

        def loss(x,y,alpha):
            return torch.nn.functional.cross_entropy(x,y,reduction=reduction)

    elif ver==1:
        #onehot encode mask and multiply by alpha
        
        #masks=torch.stack([self.Lossmask==masks for masks in self.masks],dim=-1)
        
        #self.Lossmasks=torch.sum(masks*torch.nn.functional.softmax(self.alpha/torch.norm(self.alpha,keepdim=True)),dim=-1)
        def loss(x,y,alpha):
            #print(torch.nn.functional.one_hot(mask,num_classes=len(masks)).shape)
            #print(torch.nn.functional.softmax(alpha/torch.norm(alpha,keepdim=True)).shape)
            #print("loss masks shape before:",torch.nn.functional.one_hot(mask,num_classes=alpha.shape[0]).shape)
            #print("alpha shape:",alpha.shape)#11
            Lossmasks=torch.sum(torch.nn.functional.softmax(alpha/torch.norm(alpha,keepdim=True))*st.to(alpha.device),dim=-1)
            #print("losmasks:",Lossmasks.shape)
            #Lossmasks=Lossmasks.view(*mask.shape)
            return torch.nn.functional.cross_entropy(x*Lossmasks,y*Lossmasks,reduction=reduction,)
    elif ver==2:
        Lossmasks=reduce(torch.logical_or,[mask==masks[i] for i in range(-2,2)])

        def loss(x,y,alpha):
            
            return torch.nn.functional.cross_entropy(x*Lossmasks.to(y.device),y*Lossmasks.to(y.device),reduction=reduction)
            
    else:
        def loss(x,y,alpha):
            #l=torch.nn.functional.cross_entropy(x.where(mask,torch.tensor(-100)),y.where(mask,torch.tensor(-100)),ignore_index=-100,reduction="mean")
            #l1=torch.nn.functional.cross_entropy(x.where(mask,torch.tensor(-100)),y.where(mask,torch.tensor(-100)),ignore_index=-100,reduction="sum")
            #l2=torch.nn.functional.cross_entropy(x.where(mask,torch.tensor(-100)),y.where(mask,torch.tensor(-100)),ignore_index=-100,reduction="none")
            #print("function loss: {} \n vs \n {} \n vs \n {}".format(l,l1,l2))
            #mask=mask.to(x.device,non_blocking=True)

            return torch.nn.functional.cross_entropy(x*mask.to(x.device),y*mask.to(y.device))#*mask.shape[0]/mask.sum()
         #negative because when mask is used, the loss is actually the negative of the loss
    
    return loss
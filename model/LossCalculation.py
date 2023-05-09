

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
    elif logitsversion==17:
        norm=True
        def baseLogits(*args):
            return calculate_lossNormsv8(*args)
    elif logitsversion==18:
        norm=True
        def baseLogits(*args):
            return calculate_lossNormsv9(*args)
    normfunction=lambda x:x
    if norm:
        normfunction=normargs
        def lossfn(*args):
            return baseLogits(*normfunction(*args)) 
    else:
        def lossfn(*args):
            return baseLogits(*args)
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

    
    
def calculate_loss1(  I, C1, C2, C3, C4, C5):
    return torch.mean(torch.sqrt(torch.abs(reduce(torch.add,[torch.pow(I,2).view( I.shape[0],1,1,1,1,1,-1),
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
                                                    C5.view(1,1,1,1,1,C5.shape[0],-1)]),2),alpha=1/6))),dim=-1)
def calculate_loss2(  I, C1, C2, C3, C4, C5):
    return torch.mean(torch.sqrt(reduce(torch.add,[torch.pow(I,2).view( I.shape[0],1,1,1,1,1,-1),
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
def calculate_loss7(  I, C1, C2, C3, C4, C5):
    return torch.sum(torch.abs(reduce(torch.add,[torch.pow(I,2).view( I.shape[0],1,1,1,1,1,-1),
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

def calculate_loss8(  I, C1, C2, C3, C4, C5):
    return reduce(torch.add,[torch.pow(I,2).view( I.shape[0],1,1,1,1,1,-1),
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
                                                    C5.view(1,1,1,1,1,C5.shape[0],-1)]),2),alpha=1/6)
################################################ NORMS ################################################

  
def calculate_lossNorms(I, C1, C2, C3, C4, C5):
    mean=reduce(torch.add,[I.view( I.shape[0],1,1,1,1,1,-1),
                                                        C1.view(1,C1.shape[0],1,1,1,1,-1),
                                                        C2.view(1,1,C2.shape[0],1,1,1,-1),
                                                        C3.view(1,1,1,C3.shape[0],1,1,-1),
                                                        C4.view(1,1,1,1,C4.shape[0],1,-1),
                                                        C5.view(1,1,1,1,1,C5.shape[0],-1)])
    scalednorm=mean/mean.norm(dim=-1,keepdim=True)
    
    #return dot product of scalednorm and mean x 6
    out=torch.sum(torch.mul(scalednorm,mean),dim=-1) #this....doesnt work see v4
    return out

def calculate_lossNormsv2(I, C1, C2, C3, C4, C5):
    sum=reduce(torch.add,[I.view( I.shape[0],1,1,1,1,1,-1),
                        C1.view(1,C1.shape[0],1,1,1,1,-1),
                        C2.view(1,1,C2.shape[0],1,1,1,-1),
                        C3.view(1,1,1,C3.shape[0],1,1,-1),
                        C4.view(1,1,1,1,C4.shape[0],1,-1),
                        C5.view(1,1,1,1,1,C5.shape[0],-1)])
    out=sum.pow(2)/sum.norm(dim=-1,keepdim=True)
    out=torch.sum(out,dim=-1)
    return torch.squeeze(out)
    #return dot product of scalednorm and mean x 6
    #return torch.sum(torch.mul(scalednorm,mean),dim=-1)

  
def calculate_lossNormsv3(I, C1, C2, C3, C4, C5):
    mean=reduce(torch.add,[I.view( I.shape[0],1,1,1,1,1,-1).div(6),
                            C1.view(1,C1.shape[0],1,1,1,1,-1).div(6),
                            C2.view(1,1,C2.shape[0],1,1,1,-1).div(6),
                            C3.view(1,1,1,C3.shape[0],1,1,-1).div(6),
                            C4.view(1,1,1,1,C4.shape[0],1,-1).div(6),
                            C5.view(1,1,1,1,1,C5.shape[0],-1).div(6)])
    return mean.norm(dim=-1,keepdim=True)

def calculate_lossNormsv4(I, C1, C2, C3, C4, C5):
    #assert I.shape[0]==C1.shape[0]==C2.shape[0]==C3.shape[0]==C4.shape[0]==C5.shape[0]
    #check norms are 1
    # assert torch.allclose(I.norm(dim=-1, keepdim=True),torch.ones_like(I.norm(dim=-1, keepdim=True)))
    # assert torch.allclose(C1.norm(dim=-1, keepdim=True),torch.ones_like(C1.norm(dim=-1, keepdim=True)))
    # assert torch.allclose(C2.norm(dim=-1, keepdim=True),torch.ones_like(C2.norm(dim=-1, keepdim=True)))
    # assert torch.allclose(C3.norm(dim=-1, keepdim=True),torch.ones_like(C3.norm(dim=-1, keepdim=True)))
    # assert torch.allclose(C4.norm(dim=-1, keepdim=True),torch.ones_like(C4.norm(dim=-1, keepdim=True)))
    # assert torch.allclose(C5.norm(dim=-1, keepdim=True),torch.ones_like(C5.norm(dim=-1, keepdim=True)))


    # print("norms are 1")

    mean=reduce(torch.add,[I.view( I.shape[0],1,1,1,1,1,-1),
                            C1.view(1,C1.shape[0],1,1,1,1,-1),
                            C2.view(1,1,C2.shape[0],1,1,1,-1),
                            C3.view(1,1,1,C3.shape[0],1,1,-1),#.div(6),
                            C4.view(1,1,1,1,C4.shape[0],1,-1),#.div(6),
                            C5.view(1,1,1,1,1,C5.shape[0],-1)])
    mean=torch.div(mean,mean.norm(dim=-1,keepdim=True))
    # print("max value in mean is ",torch.max(mean.flatten()).item())
    # print("min value in mean is ",torch.min(mean.flatten()).item())

    # print("max similarity to mean is ",torch.max(torch.sum(torch.mul(mean,I.view(I.shape[0],1,1,1,1,1,-1)),dim=-1).flatten()).item())
    return reduce(torch.add,[torch.sum(torch.mul(mean,I.view(I.shape[0],1,1,1,1,1,-1)),dim=-1),#replicate down wards
                             torch.sum(torch.mul(mean,C1.view(1,C1.shape[0],1,1,1,1,-1)),dim=-1),
                             torch.sum(torch.mul(mean,C2.view(1,1,C2.shape[0],1,1,1,-1)),dim=-1),
                             torch.sum(torch.mul(mean,C3.view(1,1,1,C3.shape[0],1,1,-1)),dim=-1),
                             torch.sum(torch.mul(mean,C4.view(1,1,1,1,C4.shape[0],1,-1)),dim=-1),
                             torch.sum(torch.mul(mean,C5.view(1,1,1,1,1,C5.shape[0],-1)),dim=-1)])
    #return dot product of scalednorm and mean x 6

    
def calculate_lossNormsv5(I, C1, C2, C3, C4, C5):
    #assert I.shape[0]==C1.shape[0]==C2.shape[0]==C3.shape[0]==C4.shape[0]==C5.shape[0]
    #check norms are 1
    # assert torch.allclose(I.norm(dim=-1, keepdim=True),torch.ones_like(I.norm(dim=-1, keepdim=True)))
    # assert torch.allclose(C1.norm(dim=-1, keepdim=True),torch.ones_like(C1.norm(dim=-1, keepdim=True)))
    # assert torch.allclose(C2.norm(dim=-1, keepdim=True),torch.ones_like(C2.norm(dim=-1, keepdim=True)))
    # assert torch.allclose(C3.norm(dim=-1, keepdim=True),torch.ones_like(C3.norm(dim=-1, keepdim=True)))
    # assert torch.allclose(C4.norm(dim=-1, keepdim=True),torch.ones_like(C4.norm(dim=-1, keepdim=True)))
    # assert torch.allclose(C5.norm(dim=-1, keepdim=True),torch.ones_like(C5.norm(dim=-1, keepdim=True)))


    # print("norms are 1")

    mean=reduce(torch.add,[I.view( I.shape[0],1,1,1,1,1,-1),
                            C1.view(1,C1.shape[0],1,1,1,1,-1),
                            C2.view(1,1,C2.shape[0],1,1,1,-1),
                            C3.view(1,1,1,C3.shape[0],1,1,-1),#.div(6),
                            C4.view(1,1,1,1,C4.shape[0],1,-1),#.div(6),
                            C5.view(1,1,1,1,1,C5.shape[0],-1)])
    #perform dot similarity between input and normalised mean of other inputs
    return reduce(torch.add,[torch.sum(torch.mul(torch.div(mean.sub(I.view(I.shape[0],1,1,1,1,1,-1)),mean.sub(I.view(I.shape[0],1,1,1,1,1,-1)).norm(dim=-1,keepdim=True)),I.view(I.shape[0],1,1,1,1,1,-1)),dim=-1),#replicate down wards
                             torch.sum(torch.mul(torch.div(mean.sub(C1.view(1,C1.shape[0],1,1,1,1,-1)),mean.sub(C1.view(1,C1.shape[0],1,1,1,1,-1)).norm(dim=-1,keepdim=True)),C1.view(1,C1.shape[0],1,1,1,1,-1)),dim=-1),
                             torch.sum(torch.mul(torch.div(mean.sub(C2.view(1,1,C2.shape[0],1,1,1,-1)),mean.sub(C2.view(1,1,C2.shape[0],1,1,1,-1)).norm(dim=-1,keepdim=True)),C2.view(1,1,C2.shape[0],1,1,1,-1)),dim=-1),
                             torch.sum(torch.mul(torch.div(mean.sub(C3.view(1,1,1,C3.shape[0],1,1,-1)),mean.sub(C3.view(1,1,1,C3.shape[0],1,1,-1)).norm(dim=-1,keepdim=True)),C3.view(1,1,1,C3.shape[0],1,1,-1)),dim=-1),
                             torch.sum(torch.mul(torch.div(mean.sub(C4.view(1,1,1,1,C4.shape[0],1,-1)),mean.sub(C4.view(1,1,1,1,C4.shape[0],1,-1)).norm(dim=-1,keepdim=True)),C4.view(1,1,1,1,C4.shape[0],1,-1)),dim=-1),
                             torch.sum(torch.mul(torch.div(mean.sub(C5.view(1,1,1,1,1,C5.shape[0],-1)),mean.sub(C5.view(1,1,1,1,1,C5.shape[0],-1)).norm(dim=-1,keepdim=True)),C5.view(1,1,1,1,1,C5.shape[0],-1)),dim=-1)])
    #return dot product of scalednorm and mean x 6

       
def calculate_lossNormsv6(I, C1, C2, C3, C4, C5):
    '''
    Trying to replicate the following numpy code in batch form

        mean=(x[i]+ y[j])/2
        deltaH= np.linalg.norm(np.array([x[i],y[j]]))- np.sqrt(2*np.power(mean,2))
        z[i,j]= mean-deltaH#np.linalg.norm(np.array([x[i],y[j]]))

        so z = root(6*mean^2)+mean - norm(x,y)
    '''

    mean=reduce(torch.add,[ torch.abs(I.view( I.shape[0],1,1,1,1,1,-1).div(6)),
                            torch.abs(C1.view(1,C1.shape[0],1,1,1,1,-1).div(6)),
                            torch.abs(C2.view(1,1,C2.shape[0],1,1,1,-1).div(6)),
                            torch.abs(C3.view(1,1,1,C3.shape[0],1,1,-1).div(6)),
                            torch.abs(C4.view(1,1,1,1,C4.shape[0],1,-1).div(6)),
                            torch.abs(C5.view(1,1,1,1,1,C5.shape[0],-1).div(6))])
    mean=torch.sub(mean,torch.sub(torch.abs(torch.sqrt(6*torch.pow(mean,2))),
            torch.sqrt(reduce(torch.add,[I.view( I.shape[0],1,1,1,1,1,-1).pow(2),
                                        C1.view(1,C1.shape[0],1,1,1,1,-1).pow(2),
                                        C2.view(1,1,C2.shape[0],1,1,1,-1).pow(2),
                                        C3.view(1,1,1,C3.shape[0],1,1,-1).pow(2),
                                        C4.view(1,1,1,1,C4.shape[0],1,-1).pow(2),
                                        C5.view(1,1,1,1,1,C5.shape[0],-1).pow(2)]))))                    
                         
    
    return torch.sum(mean,dim=-1)

def calculate_lossNormsv7(I, C1, C2, C3, C4, C5):
    '''
    The above only works for values in the range 0-1 range. 

    However, we can improve this with the following formula:
    return mean2-deltaH 
    Where
    mean2=(np.abs(x[i])+ np.abs(y[j])+np.abs(w[k]))/N
    deltaH= np.linalg.norm(np.array([x[i],y[j],w[k]]))- np.sqrt(N*np.power(mean,2))
    
    we'll rewrite this as mean+sqrt(N*mean^2)-norm(x,y)
    
    '''
    mean2 = reduce(torch.add,[ torch.abs(I.view( I.shape[0],1,1,1,1,1,-1))/6,
                            torch.abs(C1.view(1,C1.shape[0],1,1,1,1,-1))/6,
                            torch.abs(C2.view(1,1,C2.shape[0],1,1,1,-1))/6,
                            torch.abs(C3.view(1,1,1,C3.shape[0],1,1,-1))/6,
                            torch.abs(C4.view(1,1,1,1,C4.shape[0],1,-1))/6,
                            torch.abs(C5.view(1,1,1,1,1,C5.shape[0],-1))/6]) + torch.sqrt(torch.mul(reduce(torch.add,[ torch.abs(I.view( I.shape[0],1,1,1,1,1,-1).div(6)),
                            torch.abs(C1.view(1,C1.shape[0],1,1,1,1,-1).div(6)),
                            torch.abs(C2.view(1,1,C2.shape[0],1,1,1,-1).div(6)),
                            torch.abs(C3.view(1,1,1,C3.shape[0],1,1,-1).div(6)),
                            torch.abs(C4.view(1,1,1,1,C4.shape[0],1,-1).div(6)),
                            torch.abs(C5.view(1,1,1,1,1,C5.shape[0],-1).div(6))]).pow(2),6))
    #norm =sum(abs(x)**ord)**(1./ord)
    norm=torch.sqrt(reduce(torch.add,[ torch.abs(I.view( I.shape[0],1,1,1,1,1,-1)).pow(2),
                            torch.abs(C1.view(1,C1.shape[0],1,1,1,1,-1)).pow(2),
                            torch.abs(C2.view(1,1,C2.shape[0],1,1,1,-1)).pow(2),
                            torch.abs(C3.view(1,1,1,C3.shape[0],1,1,-1)).pow(2),
                            torch.abs(C4.view(1,1,1,1,C4.shape[0],1,-1)).pow(2),
                            torch.abs(C5.view(1,1,1,1,1,C5.shape[0],-1)).pow(2)]))
    return torch.mean(torch.sub(mean2,norm),dim=-1)

def calculate_lossNormsv8(I, C1, C2, C3, C4, C5):

    #calculate the mean ^6 
    mean2 = reduce(torch.add,[ I.view( I.shape[0],1,1,1,1,1,-1)/6,
                            C1.view(1,C1.shape[0],1,1,1,1,-1)/6,
                            C2.view(1,1,C2.shape[0],1,1,1,-1)/6,
                            C3.view(1,1,1,C3.shape[0],1,1,-1)/6,
                            C4.view(1,1,1,1,C4.shape[0],1,-1)/6,
                            C5.view(1,1,1,1,1,C5.shape[0],-1)/6])
    prod=torch.add(mean2,reduce(torch.mul,
                    [ torch.abs(I.view( I.shape[0],1,1,1,1,1,-1)),
                      torch.abs(C1.view(1,C1.shape[0],1,1,1,1,-1)),
                      torch.abs(C2.view(1,1,C2.shape[0],1,1,1,-1)),
                      torch.abs(C3.view(1,1,1,C3.shape[0],1,1,-1)),
                      torch.abs(C4.view(1,1,1,1,C4.shape[0],1,-1)),
                      torch.abs(C5.view(1,1,1,1,1,C5.shape[0],-1))]) )    
    mean =torch.div(torch.pow(
                        reduce(torch.add,
                            [ torch.abs(I.view( I.shape[0],1,1,1,1,1,-1).div(6)),
                              torch.abs(C1.view(1,C1.shape[0],1,1,1,1,-1).div(6)),
                              torch.abs(C2.view(1,1,C2.shape[0],1,1,1,-1).div(6)),
                              torch.abs(C3.view(1,1,1,C3.shape[0],1,1,-1).div(6)),
                              torch.abs(C4.view(1,1,1,1,C4.shape[0],1,-1).div(6)),
                              torch.abs(C5.view(1,1,1,1,1,C5.shape[0],-1).div(6))])
                        ,6)
                    ,2)
    return torch.sum(torch.sub(mean2,mean),dim=-1)
    #calculate the product of the absolute values of the terms,
    #compare the product / mean  + mean  

def calculate_lossNormsv9(I,C1,C2,C3,C4,C5):
    #  # np.sqrt(np.power(mean,2)/N) == np.sqrt(np.power(SUM(X)/N,2)/N) == np.sqrt(np.power(SUM(X),2)/N^3)

    mean = torch.sqrt(torch.pow(reduce(torch.add,[ I.view( I.shape[0],1,1,1,1,1,-1),
                            C1.view(1,C1.shape[0],1,1,1,1,-1),
                            C2.view(1,1,C2.shape[0],1,1,1,-1),
                            C3.view(1,1,1,C3.shape[0],1,1,-1),
                            C4.view(1,1,1,1,C4.shape[0],1,-1),
                            C5.view(1,1,1,1,1,C5.shape[0],-1)]),2)/6^3)
    
    mean2=reduce(torch.add,[ torch.abs(I).view( I.shape[0],1,1,1,1,1,-1),
                            torch.abs(C1).view(1,C1.shape[0],1,1,1,1,-1),
                            torch.abs(C2).view(1,1,C2.shape[0],1,1,1,-1),
                            torch.abs(C3).view(1,1,1,C3.shape[0],1,1,-1),
                            torch.abs(C4).view(1,1,1,1,C4.shape[0],1,-1),
                            torch.abs(C5).view(1,1,1,1,1,C5.shape[0],-1)])/6
    norm=torch.sqrt(reduce(torch.add,[torch.pow(I,2).view( I.shape[0],1,1,1,1,1,-1),
                            torch.pow(C1,2).view(1,C1.shape[0],1,1,1,1,-1),
                            torch.pow(C2,2).view(1,1,C2.shape[0],1,1,1,-1),
                            torch.pow(C3,3).view(1,1,1,C3.shape[0],1,1,-1),
                            torch.pow(C4,2).view(1,1,1,1,C4.shape[0],1,-1),
                            torch.pow(C5,2).view(1,1,1,1,1,C5.shape[0],-1)]))

    print(mean.shape,mean2.shape,norm.shape)
    return torch.sum(torch.sub(torch.add(mean,mean2),norm),dim=-1)
def calculate_lossNormsvc(I, C1, C2, C3, C4, C5):
    #right! 
    #none of the above are working correctly, so lets just take the mean cosine distance between the vectors
    '''
    for 1D case of shape ( 1,F),
                it's      I x [I,C1,C2,C3,C4,C5]
                then its C1 x [I,C1,C2,C3,C4,C5]
                then its C2 x [I,C1,C2,C3,C4,C5]
                then its C3 x [I,C1,C2,C3,C4,C5]
                then its C4 x [I,C1,C2,C3,C4,C5]
                then its C5 x [I,C1,C2,C3,C4,C5]
                then we subtract I.I, C1.C1, C2.C2, C3.C3, C4.C4, C5.C5
                then we take the mean of the above 
    '''
    
    '''Lets try this in Batches where shape is (B,F)'''
    '''
    for 2D case of shape (B,F),
        I Similarity is sum(I x [I,C1,C2,C3,C4,C5], dim=-1) which returns a shape (6,B,B)
        C1 Similarity is sum(C1 x [I,C1,C2,C3,C4,C5], dim=-1) which returns a shape (6,B,B)
        C2 Similarity is sum(C2 x [I,C1,C2,C3,C4,C5], dim=-1) which returns a shape (6,B,B)
        C3 Similarity is sum(C3 x [I,C1,C2,C3,C4,C5], dim=-1) which returns a shape (6,B,B)
        C4 Similarity is sum(C4 x [I,C1,C2,C3,C4,C5], dim=-1) which returns a shape (6,B,B)
        C5 Similarity is sum(C5 x [I,C1,C2,C3,C4,C5], dim=-1) which returns a shape (6,B,B)
        we can sum these on dim 0 to get a shape (B,B). We can then subtract (B,B) matrices of  I.I C1.C1 C2.C2 C3.C3 C4.C4 C5.C5            
        This should give us a (B,B) matrix of the cosine distance between the vectors, this can then be halved because every vector is counted twice
    '''
        #stack=torch.stack([I,C1,C2,C3,C4,C5],dim=0)
    I=I/torch.norm(I,dim=-1,keepdim=True)
    C1=C1/torch.norm(C1,dim=-1,keepdim=True)
    C2=C2/torch.norm(C2,dim=-1,keepdim=True)
    C3=C3/torch.norm(C3,dim=-1,keepdim=True)
    C4=C4/torch.norm(C4,dim=-1,keepdim=True)
    C5=C5/torch.norm(C5,dim=-1,keepdim=True)
    total=reduce(torch.add, [
                            reduce(torch.add,[  C1@I.T,
                                                #torch.sum(torch.mul(C1,C1),dim=-1),
                                                C1@C2.T,
                                                C1@C3.T,
                                                C1@C4.T,
                                                C1@C5.T]),
                            reduce(torch.add,[  C2@I.T,
                                                C2@C1.T,
                                                #torch.sum(torch.mul(C2,C2),dim=-1),
                                                C2@C3.T,
                                                C2@C4.T,
                                                C2@C5.T]),
                            reduce(torch.add,[  C3@I.T,
                                                C3@C1.T,
                                                C3@C2.T,
                                                #torch.sumtorch.mul(C3,C3),dim=-1),
                                                C3@C4.T,
                                                C3@C5.T]),
                            reduce(torch.add,[  C4@I.T,
                                                C4@C1.T,
                                                C4@C2.T,
                                                C4@C3.T,
                                                #torch.sum(torch.mul(C4,C4),dim=-1),
                                                C4@C5.T]),
                            reduce(torch.add,[  C5@I.T,
                                                C5@C1.T,
                                                C5@C2.T,
                                                C5@C3.T,
                                                C5@C4.T,
                                                #torch.sum(torch.mul(C5,C5),dim=-1)
                                                ])])
    #print(total.shape)
    return reduce(torch.add,[ #torch.sum(torch.mul(I,I),dim=-1),
                                                I@C1.T,
                                                I@C2.T,
                                                I@C3.T,
                                                I@C4.T,
                                                I@C5.T])/10,total/50 #this is averaged across the 6 vectors, each with 5 elements, then halved because every vector is counted twice
                    
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
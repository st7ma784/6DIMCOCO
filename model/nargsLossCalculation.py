

from telnetlib import BM
import torch
from functools import reduce
'''
This is research code... it is not clean and it is not commented

If you wish to use it for TPUs, I strongly recommend you refactor your code to use this style of function factory.
 Otherwise your runs will be very slow.

This code is a copy of the LossCalculation.py file, but with the loss functions as functions that take any number of arguments.
'''


def calc_mean(*vecs):
    return reduce(torch.add,vecs)
def JSE_mean(*vecs):
   
   
    sum_of_squares=reduce(torch.add,[torch.pow(vec,2) for vec in vecs])
    JSEFactor=1-(4/sum_of_squares)
    return torch.mul(calc_mean(*vecs),JSEFactor)



def oneminus(args):
    return 1-args

def null(*args):
    return args

def normargs(*args):
    return map(lambda arg: arg/arg.norm(dim=-1, keepdim=True), args)
def logargs(args):
    return torch.log(args)


def get_loss_fn(logitsversion=0,norm=False,log=False,JSE=0):
    baseLogits=calculate_loss
    ##logfunction=lambda x:x
    mean_fn=calc_mean
    if JSE==1:
        mean_fn=JSE_mean
        # if log:
    #     logfunction=logargs
    if logitsversion==0:
        def baseLogits(*args):
            return calculate_loss(*args,mean_fn=mean_fn)
    elif logitsversion==1: 
        def baseLogits(*args):
            return oneminus(calculate_loss2(*args,mean_fn=mean_fn))
    elif logitsversion==2: 
        def baseLogits(*args):
            return oneminus(calculate_loss3(*args,mean_fn=mean_fn))      #one minus here
    elif logitsversion==3:
        #this does not work in any arrangement? 
        def baseLogits(*args):
            return oneminus(calculate_loss4(*args,mean_fn=mean_fn))
    elif logitsversion==4:
        #this does not work in any arrangement either?
        def baseLogits(*args):
            return oneminus(calculate_loss5(*args,mean_fn=mean_fn))
    elif logitsversion==5:
        def baseLogits(*args):
            return oneminus(calculate_loss6(*args,mean_fn=mean_fn))
    elif logitsversion==6:
        def baseLogits(*args):
            return oneminus(calculate_loss1(*args))
    elif logitsversion==13:
        def baseLogits(*args):
            return oneminus(calculate_loss7(*args,mean_fn=mean_fn))
    elif logitsversion==14:
        def baseLogits(*args):
            return torch.sum(oneminus(calculate_loss8(*args,mean_fn=mean_fn)),dim=-1)
    elif logitsversion==7:
        norm=True
        def baseLogits(*args):
            return calculate_lossNorms(*args,mean_fn=mean_fn)
            
    elif logitsversion==8:
        norm=True
        def baseLogits(*args):
            return calculate_lossNormsv2(*args,mean_fn=mean_fn)
        
                
    elif logitsversion==9:
        norm=True
        def baseLogits(*args):
            return calculate_lossNormsv3(*args,mean_fn=mean_fn)
                
    elif logitsversion==10:
        norm=True
        def baseLogits(*args):
            return calculate_lossNormsv4(*args,mean_fn=mean_fn)
        
             
    elif logitsversion==11:
        norm=True
        def baseLogits(*args):
            return calculate_lossNormsv5(*args,mean_fn=mean_fn)
             
    elif logitsversion==12:
        norm=False
        def baseLogits(*args):
            return calculate_lossNormsv5(*args,mean_fn=mean_fn)
    elif logitsversion==15:
        norm=True
        def baseLogits(*args):
            return calculate_lossNormsv6(*args,mean_fn=mean_fn)
    elif logitsversion==16:
        norm=True
        def baseLogits(*args):
            return calculate_lossNormsv7(*args,mean_fn=mean_fn)
    elif logitsversion==-1:
        def baseLogits(*args):
            return Fast_loss_Hdim(*args)
    normfunction=lambda x:x
    if norm:
        normfunction=normargs
        def lossfn(*args):
            return baseLogits(*normfunction(*args)) 
    else:
        def lossfn(*args):
            return baseLogits(*args)
    return lossfn



def calculate_lossStock(args,mean_fn=calc_mean):
    I,C1=args[0],args[1]
    #normalize image and text features
    I = I / I.norm(dim=-1, keepdim=True)
    C1 = C1 / C1.norm(dim=-1, keepdim=True)
    #calculate logits
    logits_per_image =  I @ C1.T
    logits_per_text =  C1 @ I.T
    #calculate loss
    return logits_per_image, logits_per_text

def calculate_lossbase(args,norm=True,log=False,mean_fn=calc_mean):
    I,C1=args[0],args[1]
    #normalize image and text features
    I = I / I.norm(dim=-1, keepdim=True)
    C1 = C1 / C1.norm(dim=-1, keepdim=True)
    #calculate logits
    logits_per_image =  I @ C1.T
    #calculate loss
    return logits_per_image

def calculate_loss(  *Args,mean_fn=calc_mean):
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

    
    
def calculate_loss1(  *Args,mean_fn=calc_mean):
    #find number of arguments
    n=len(Args)
    #assume all args are shape BxF
    #each argmuents is going to become BxFx1x... (n dimensions)


    return torch.mean(torch.sqrt(torch.abs( mean_fn(*[torch.pow(arg,2).view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]).sub_(
                        torch.pow( mean_fn(*[arg.view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]),2),alpha=1/n))),dim=-1)


def calculate_loss2(  *Args,mean_fn=calc_mean):
    n=len(Args)
    
    # for i,arg in enumerate(Args):
    #     print([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])
    return torch.mean(torch.sqrt( mean_fn(*[torch.pow(arg,2).view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]).sub_(
                        torch.pow( mean_fn(*[arg.view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]),2),alpha=1/n)),dim=-1)
    #frequently ends with Nans, not sure why
    
def calculate_loss3( *Args,mean_fn=calc_mean):
    n=len(Args)

    return torch.sqrt(torch.sum(torch.pow( mean_fn(*[torch.pow(arg,2).view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]).sub_(
                        torch.pow( mean_fn(*[arg.view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]),2),alpha=1/n),2),dim=-1))
    #results in nan labels 
    

def calcloss( *Args,mean_fn=calc_mean):
    n=len(Args)

    return torch.sum( mean_fn(*[torch.sqrt(torch.pow(arg,2).view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])).sub_(
                            torch.pow( mean_fn(*[arg.view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]),2),alpha=1/n)) for i,arg in enumerate(Args)]),dim=-1)


def calculate_loss4(*Args,mean_fn=calc_mean):
    #tested and not working
    return torch.sum(torch.sqrt( mean_fn(*[torch.pow(arg,2).view(*list([1]*i+[arg.shape[0]]+[1]*(len(Args)-1-i)+[-1])) for i,arg in enumerate(Args)])),dim=-1)

def calculate_loss5(*Args, mean_fn=calc_mean):
   
    return torch.sum( mean_fn(*[ torch.pow(arg,2).view(*list([1]*i+[arg.shape[0]]+[1]*(len(Args)-1-i)+[-1])) for i,arg in enumerate(Args)]).sub_(
                    torch.pow( mean_fn(*[    arg.view(*list([1]*i+[arg.shape[0]]+[1]*(len(Args)-1-i)+[-1])) for i,arg in enumerate(Args)]),2),alpha=1/len(Args)),dim=-1)

    
def calculate_loss6(*Args, mean_fn=calc_mean):
    
    return torch.sqrt(torch.abs(torch.sum( mean_fn(*[torch.pow(arg,2).view(*list([1]*i+[arg.shape[0]]+[1]*(len(Args)-1-i)+[-1])) for i,arg in enumerate(Args)]).sub_(
                            torch.pow( mean_fn(*[arg.view(*list([1]*i+[arg.shape[0]]+[1]*(len(Args)-1-i)+[-1])) for i,arg in enumerate(Args)]),2),alpha=1/len(Args)),dim=-1)))
def calculate_loss7(  *Args, mean_fn=calc_mean):
    #find number of arguments
    #n=len(Args)
    return torch.sum(torch.abs( mean_fn(*[torch.pow(arg,2).view(*list([1]*i+[arg.shape[0]]+[1]*(len(Args)-1-i)+[-1])) for i,arg in enumerate(Args)]).sub_(
                        torch.pow( mean_fn(*[arg.view(*list([1]*i+[arg.shape[0]]+[1]*(len(Args)-1-i)+[-1])) for i,arg in enumerate(Args)]),2),alpha=1/len(Args))),dim=-1)

def calculate_loss8(  *Args, mean_fn=calc_mean):
    #find number of arguments
    #n=len(Args)
    return  mean_fn(*[torch.pow(arg,2).view(*list([1]*i+[arg.shape[0]]+[1]*(len(Args)-1-i)+[-1])) for i,arg in enumerate(Args)]).sub_(
                        torch.pow( mean_fn(*[arg.view(*list([1]*i+[arg.shape[0]]+[1]*(len(Args)-1-i)+[-1])) for i,arg in enumerate(Args)]),2),alpha=1/len(Args))
################################################ NORMS ################################################

def calculate_lossNorms(  *Args,mean_fn=calc_mean):
    #find number of arguments
    #n=len(Args)
    mean= mean_fn(*[arg.view(*list([1]*i+[arg.shape[0]]+[1]*(len(Args)-1-i)+[-1])) for i,arg in enumerate(Args)])    
    #return dot product of scalednorm and mean x 6
    return torch.sum(torch.mul(mean/mean.norm(dim=-1,keepdim=True),mean),dim=-1) #this....doesnt work see v4
    #return out

def calculate_lossNormsv2(*Args, mean_fn=calc_mean):
    #find number of arguments
    #n=len(Args)
    sum= mean_fn(*[arg.view(*list([1]*i+[arg.shape[0]]+[1]*(len(Args)-1-i)+[-1])) for i,arg in enumerate(Args)])
    return torch.squeeze(torch.sum(sum.pow(2)/sum.norm(dim=-1,keepdim=True),dim=-1))
    #return dot product of scalednorm and mean x 6
    #return torch.sum(torch.mul(scalednorm,mean),dim=-1)

  
def calculate_lossNormsv3(*Args, mean_fn=calc_mean):
    #find number of arguments

    return  mean_fn(*[torch.div(arg,len(Args)).view(*list([1]*i+[arg.shape[0]]+[1]*(len(Args)-1-i)+[-1])) for i,arg in enumerate(Args)]).norm(dim=-1,keepdim=True).squeeze(-1)

def calculate_lossNormsv4(*Args, mean_fn=calc_mean):
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
    
    mean= mean_fn(*[arg.view(*list([1]*i+[arg.shape[0]]+[1]*(len(Args)-1-i)+[-1])) for i,arg in enumerate(Args)])
    mean=torch.div(mean,mean.norm(dim=-1,keepdim=True))

    return  mean_fn(*[torch.sum(torch.mul(mean,arg.view(*list([1]*i+[arg.shape[0]]+[1]*(len(Args)-1-i)+[-1])) ),dim=-1)for i,arg in enumerate(Args)])
    #return dot product of scalednorm and mean x 6

    
def calculate_lossNormsv5(*Args, mean_fn=calc_mean):
 
    #find number of arguments   
    #n=len(Args)
    mean= mean_fn(*[arg.view(*list([1]*i+[arg.shape[0]]+[1]*(len(Args)-1-i)+[-1])) for i,arg in enumerate(Args)])
    #perform dot similarity between input and normalised mean of other inputs
    return  mean_fn(*[torch.sum(
                                        torch.mul(
                                            torch.div(
                                                    torch.sub(mean,
                                                              arg.view(*list([1]*i+[arg.shape[0]]+[1]*(len(Args)-1-i)+[-1]))),
                                                    torch.sub(mean,
                                                              arg.view(*list([1]*i+[arg.shape[0]]+[1]*(len(Args)-1-i)+[-1]))).norm(dim=-1,keepdim=True)),
                  arg.view(*list([1]*i+[arg.shape[0]]+[1]*(len(Args)-1-i)+[-1]))),dim=-1) for i,arg in enumerate(Args)])
        # torch.sum(torch.mul(torch.div(mean.sub(I.view(I.shape[0],1,1,1,1,1,-1)),mean.sub(I.view(I.shape[0],1,1,1,1,1,-1)).norm(dim=-1,keepdim=True)),I.view(I.shape[0],1,1,1,1,1,-1)),dim=-1),#replicate down wards
        #                      torch.sum(torch.mul(torch.div(mean.sub(C1.view(1,C1.shape[0],1,1,1,1,-1)),mean.sub(C1.view(1,C1.shape[0],1,1,1,1,-1)).norm(dim=-1,keepdim=True)),C1.view(1,C1.shape[0],1,1,1,1,-1)),dim=-1),
        #                      torch.sum(torch.mul(torch.div(mean.sub(C2.view(1,1,C2.shape[0],1,1,1,-1)),mean.sub(C2.view(1,1,C2.shape[0],1,1,1,-1)).norm(dim=-1,keepdim=True)),C2.view(1,1,C2.shape[0],1,1,1,-1)),dim=-1),
        #                      torch.sum(torch.mul(torch.div(mean.sub(C3.view(1,1,1,C3.shape[0],1,1,-1)),mean.sub(C3.view(1,1,1,C3.shape[0],1,1,-1)).norm(dim=-1,keepdim=True)),C3.view(1,1,1,C3.shape[0],1,1,-1)),dim=-1),
        #                      torch.sum(torch.mul(torch.div(mean.sub(C4.view(1,1,1,1,C4.shape[0],1,-1)),mean.sub(C4.view(1,1,1,1,C4.shape[0],1,-1)).norm(dim=-1,keepdim=True)),C4.view(1,1,1,1,C4.shape[0],1,-1)),dim=-1),
        #                      torch.sum(torch.mul(torch.div(mean.sub(C5.view(1,1,1,1,1,C5.shape[0],-1)),mean.sub(C5.view(1,1,1,1,1,C5.shape[0],-1)).norm(dim=-1,keepdim=True)),C5.view(1,1,1,1,1,C5.shape[0],-1)),dim=-1)])
    #return dot product of scalednorm and mean x 6

       
def calculate_lossNormsv6(*Args, mean_fn=calc_mean):
    '''
    Trying to replicate the following numpy code in batch form

        mean=(x[i]+ y[j])/2
        deltaH= np.linalg.norm(np.array([x[i],y[j]]))- np.sqrt(2*np.power(mean,2))
        z[i,j]= mean-deltaH#np.linalg.norm(np.array([x[i],y[j]]))

        so z = root(6*mean^2)+mean - norm(x,y)
    '''
    #find number of arguments
    mean= mean_fn(*[ torch.div(torch.abs(arg),len(Args)).view(*list([1]*i+[arg.shape[0]]+[1]*(len(Args)-1-i)+[-1])) for i,arg in enumerate(Args)])
    mean=torch.sub(mean,torch.sub(torch.abs(torch.sqrt(len(Args)*torch.pow(mean,2))),
            torch.sqrt( mean_fn(*[torch.pow(arg,2).view(*list([1]*i+[arg.shape[0]]+[1]*(len(Args)-1-i)+[-1])) for i,arg in enumerate(Args)]))))                    
                         
    
    return torch.sum(mean,dim=-1)

def calculate_lossNormsv7(*Args, mean_fn=calc_mean):
    n=len(Args)
    mean2 =  mean_fn(*[torch.div(torch.abs(arg),n).view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]) + torch.sqrt(torch.mul( mean_fn(*[ torch.div(torch.abs(arg),n).view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]).pow(2),n))
    #norm =sum(abs(x)**ord)**(1./ord)
    norm=torch.sqrt( mean_fn(*[ torch.pow(arg,2).view(*list([1]*i+[arg.shape[0]]+[1]*(n-1-i)+[-1])) for i,arg in enumerate(Args)]))
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
            return torch.nn.functional.cross_entropy(x*Lossmasks,y*Lossmasks,reduction=reduction)
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
    
    return loss#

#fpor improved JSE implementation have a look at notes in the the other losscalculation file. 


def calc_mean(*vecs):
    return reduce(torch.add,vecs)

def JSE_mean(*vecs):
   
   
    sum_of_squares=reduce(torch.add,[torch.pow(vec,2) for vec in vecs])
    JSEFactor=1-(4/sum_of_squares)
    return torch.mul(calc_mean(*vecs),JSEFactor)


def Fast_loss_Hdim(*vecs):
    #step 1, stack all vectors into a single tensor, N,B,F
    #matrix multiply by transpose of itself to get shape N,N,B,B
    #rearrange to B,B,N,N
    #do CE loss along diagonal of B,B and sum
    #return loss

    #step 1
    stacked=torch.stack(vecs,dim=0)
    #step 2
    #print(stacked.shape)
    logits=torch.matmul(stacked.unsqueeze(0),stacked.transpose(-1,-2).unsqueeze(1))
    #print(logits.shape) # N,N,B,B
    #step 3
    logits=logits.permute(2,3,0,1)
    
    return logits.flatten(2,3)



#run each method on the same data and compare the results
if __name__ == "__main__":
    for i in range(17):
        print("method:",i)
        results=[]
        method=get_loss_fn(i)
        for n in range(2,10):
            results.append(torch.any(torch.isnan(method(*[torch.rand([2,10])]*n))) )
        if torch.any(torch.tensor(results)):
            print("method {} failed".format(i))
            print(results)
    loss=torch.nn.CrossEntropyLoss(reduction="mean")

    #next we're going to test Fast_loss_Hdim
    #we need to create a bunch of vectors of shape BxF
    vectors=torch.rand([10,512],dtype=torch.float32) *2 -1
    vectors2=torch.rand([10,512],dtype=torch.float32)   *2 -1
    vectors3=torch.rand([10,512],dtype=torch.float32)  *2 -1
    vectors4=torch.rand([10,512],dtype=torch.float32) *2 -1
    vectors5=torch.rand([10,512],dtype=torch.float32) *2 -1
    vectors6=torch.rand([10,512],dtype=torch.float32) *2 -1
    #try them all the same 
    vectors2=vectors
    vectors3=vectors
    vectors4=vectors
    vectors5=vectors
    vectors6=vectors

    #norm them
    vectors=vectors/vectors.norm(dim=-1,keepdim=True)
    vectors2=vectors2/vectors2.norm(dim=-1,keepdim=True)
    vectors3=vectors3/vectors3.norm(dim=-1,keepdim=True)
    vectors4=vectors4/vectors4.norm(dim=-1,keepdim=True)
    vectors5=vectors5/vectors5.norm(dim=-1,keepdim=True)
    vectors6=vectors6/vectors6.norm(dim=-1,keepdim=True)

    Logits=Fast_loss_Hdim(vectors,vectors2,vectors3,vectors4,vectors5,vectors6)
    print(Logits[0,0])
    print(Logits[1,1])
    labels=torch.arange(Logits.shape[0],dtype=torch.long).unsqueeze(1).repeat(1,Logits.shape[-1])
    print(Logits[0,1])
    CELoss=loss(Logits,labels)
    print(CELoss)
    loss2=torch.nn.functional.cross_entropy(Logits,labels)
    print(loss2)

    for N in range(3,7):
        B=4
        for j in range(0,20):
            testFn=get_loss_fn(j,True)
            testBatchA=torch.rand(B,512,device="cpu")
            testBatchB=torch.normal(0,0.3,(B,512),device="cpu")
            testBatchA=testBatchA/torch.norm(testBatchA,dim=-1,keepdim=True)
            testBatchB=testBatchB/torch.norm(testBatchB,dim=-1,keepdim=True)
            
            logtis=testFn(*[testBatchB]*N)#convert this to probabilities in range [0,1]
            logtis=torch.nan_to_num(logtis)

            label=torch.nn.functional.softmax(logtis)
            Views=torch.diag_embed(torch.ones(N,dtype=torch.long)*B-1)+1
            
            Lossmask=torch.sum(reduce(torch.add,list(map(lambda Arr: torch.nn.functional.one_hot(torch.arange(B).view(*Arr),num_classes=B),Views.tolist()))).pow(4),dim=-1)
            masks=torch.unique(torch.flatten(Lossmask,0,-1),dim=0,sorted=False)

            
            st=torch.stack([Lossmask==masks[i] for i in range(len(masks))],dim=0)

            from matplotlib import pyplot as plt
            fig = plt.plot(figsize=(1000, 1000))

            for i in range(st.shape[0]):
                plt.hist(logtis[st[i]].numpy(), label=masks[i])
            plt.legend(loc = "upper right")
            plt.savefig("ExactLabelsnormalPlotMethod{}B{}N{}.jpg".format(j,B,N))
            plt.clf()

        

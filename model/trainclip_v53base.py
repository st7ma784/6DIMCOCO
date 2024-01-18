
from cProfile import label
import pytorch_lightning
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional
from clip.model import Transformer,LayerNorm,VisionTransformer
from functools import partial
import clip
from warnings import warn
import matplotlib.pyplot as plt
from zmq import has
from CKA_test import add_colorbar 
from sklearn.linear_model import LogisticRegression
from model.trainclip_cka_base import LightningCLIPModule as base
class LightningCLIPModule(base):
    def __init__(self,
                
                learning_rate,
                logitsversion=0,
                normlogits=True,
                projection='inv',
                prune=True,
                exactlabels=False,
                adam_epsilon: float = 1e-8,
                warmup_steps: int = 0,
                weight_decay: float = 0.0,
                total_steps: int = 200000,
                train_batch_size: int = 64,
                eval_batch_size: int = 32,
                eval_splits: Optional[list] = None,
                embed_dim= 512,
                context_length= 77,
                vocab_size= 50257,
                transformer_width= 512,
                transformer_heads= 32,
                transformer_layers= 4,
                **kwargs,
                ):

        super().__init__()
        if hasattr(self,"clip"):
            del self.clip
        self.save_hyperparameters()
        print("learning_rate",learning_rate)

        self.context_length = context_length
        self.encoder = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
            )
        self.encode_image= VisionTransformer(
                input_resolution=224,
                patch_size=16,
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                output_dim=embed_dim
            )
        
        #self.linear.weight=torch.nn.Parameter(self.clip.token_embedding.weight.T)
        self.loss=torch.nn.CrossEntropyLoss(reduction='sum')

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.transformer_width=transformer_width
        self.handles=[]
        self.labels=[]
        self.model1_info={'Name':"SelfCLIP",}
        self.model2_info={'Name': "Stock CLIP",}
        self.naninfcount=0
        print("done")
        from model.LossCalculation import calculate_lossStock as sl
        from model.LossCalculation import calculate_lossNormsvc
        self.calculate_lossStock=sl

        self.calculate_lossStock2=calculate_lossNormsvc
        if logitsversion==0:
            from model.LossCalculation import calculate_loss as cl
        elif logitsversion==1: 
            from model.LossCalculation import calculate_loss2 as cl
        elif logitsversion==2: 
            from model.LossCalculation import calculate_loss3 as cl
        elif logitsversion==3:
            from model.LossCalculation import calculate_loss4 as cl
        elif logitsversion==4:
            from model.LossCalculation import calculate_loss5 as cl
        elif logitsversion==5:
            from model.LossCalculation import calculate_loss6 as cl
        else:
            from model.LossCalculation import calculate_loss as cl
        self.calculate_loss=cl
        self.normlogits=normlogits
        self.projection=projection
        self.prune=prune
        if self.prune:
            from model.PruneCalculation import PruneHook
            self.pruneHooks=[PruneHook(self.encode_image,[-1,0,1], 0.1, method="Hard", prune_eta=4, amount=4,fc_pru_bound=-1),
                             PruneHook(self.encoder,[-1,0,1], 0.1, method="Hard", prune_eta=4, amount=4,fc_pru_bound=-1)]
        else:
            self.pruneHooks=[]
        self.initialize_parameters()
  

    # @torch.jit.script
    def forward(self, im, captions1, captions2, captions3, captions4, captions5):
        image_features=self.encode_image(im)
        caption_features1=self.encode_text(captions1)
        caption_features2=self.encode_text(captions2)#
        caption_features3=self.encode_text(captions3)#
        caption_features4=self.encode_text(captions4)#
        caption_features5=self.encode_text(captions5)#

        if self.projection=="inv":
            image_features=image_features@ self.text_projection
        elif self.projection=="iinv":
            image_features=image_features@torch.inverse(self.text_projection)
        elif self.projection=="None":
            caption_features1=caption_features1@self.text_projection
            caption_features2=caption_features2@self.text_projection#
            caption_features3=caption_features3@self.text_projection# 
            caption_features4=caption_features4@self.text_projection#
            caption_features5=caption_features5@self.text_projection#       
        
        return self.calculate_lossStock2(image_features, caption_features1,caption_features2,caption_features3,caption_features4,caption_features5)
        #return self.calculate_lossStock(image_features, caption_features1)[0]*self.logit_scale.exp()
        
    def training_step(self, batch, batch_idx,optimizer_idx=0):

        labels=torch.arange(batch[0].shape[0],device=self.device)
        im,captions= batch[0],batch[1]
        
        logitsI,logits=self(im,captions[:,0],captions[:,1],captions[:,2],captions[:,3],captions[:,4])
        self.log("first logit",logits[0,0],enable_graph=False)
        self.log("BAD logit",logits[0,1],enable_graph=False)
        self.log("logit scale",self.logit_scale.exp())
        logitsI=logitsI*self.logit_scale.exp()
        logits=logits*self.logit_scale.exp()
        # The idea is that good logits are 1s,   bad should be -1s... so if logits are coming back as ~6000....
        #  Option 1: divide down.
        #  Option 2: 1- output...
        # option 3: logarithmic functions? 

        lossim = self.loss(logitsI, labels)
        loss1 = self.loss(logits, labels)
       
        loss = lossim+loss1
        loss=loss/2
        loss = loss.mean()
        self.log('train_loss', loss, prog_bar=True,enable_graph=False, rank_zero_only=True)
        
        return {"loss": loss}


    def validation_step(self,batch,*args):
        #do stock loss here
        labels=torch.arange(batch[0].shape[0],dtype=torch.long,device=self.device)
        self.model1_features = {}  #reset list of forward hooks
        self.model2_features = {}  #reset list of forward hooks
        image_features=self.encode_image(batch[0])
        self.model2.encode_image(batch[0])# to compare supervision model
        a=torch.nan_to_num(torch.stack(list(self.model1_features.values())))
        self.IMhsic_matrix0=torch.add(self.IMhsic_matrix0,torch.nan_to_num(batch_HSIC2(a),nan=0.0,posinf=100,neginf=-200)) 
        a=torch.nan_to_num(torch.stack(list(self.model2_features.values())))
      
        self.IMhsic_matrix2=torch.add(self.IMhsic_matrix2,torch.nan_to_num(batch_HSIC2(a),nan=0.0,posinf=100,neginf=-200))
        joint_HSIC=torch.nan_to_num(batch_HSIC3(a,torch.nan_to_num(torch.stack(list(self.model1_features.values())))), nan=0.0,posinf=1,neginf=-2)
        self.IMhsic_matrix1=torch.add(self.IMhsic_matrix1,joint_HSIC) 
        ##Now Do Text
        self.model1_features = {}  #reset list of forward hooks
        self.model2_features = {}  #reset list of forward hooks
        choice=torch.randint(0,5,(1,)).item()
        #print("choice", choice)
        c=batch[1][:,choice]
        c=c.squeeze()

        captions=self.encode_text(c) #run through main mode

        # c=captions.detach().clone().cpu()
        #run through main mode
        
        self.model2.encode_text(c)# to compare supervision model
        a=torch.nan_to_num(torch.stack(list(self.model1_features.values())))
        self.CAPhsic_matrix0=torch.add(self.CAPhsic_matrix0,batch_HSIC2(a)) 
        a=torch.nan_to_num(torch.stack(list(self.model2_features.values())))
        self.CAPhsic_matrix2=torch.add(self.CAPhsic_matrix2,batch_HSIC2(a))
        joint_HSIC=torch.nan_to_num(batch_HSIC3(a,torch.nan_to_num(torch.stack(list(self.model1_features.values())))))
        self.CAPhsic_matrix1=torch.add(self.CAPhsic_matrix1,joint_HSIC) 
        if self.projection=="inv":
            image_features=image_features@ self.text_projection
        elif self.projection=="iinv":
            image_features=image_features@torch.inverse(self.text_projection)
        elif self.projection=="None":
            captions=captions@self.text_projection


        # print("self.logit scale is 14 right? ",self.logit_scale.exp())
        logitsI,logitsT=self.calculate_lossStock(image_features, captions) 
        self.log("mean validation stock logits ", logitsI.mean())
        
        lossim = self.loss(logitsI*(self.logit_scale.exp()), labels)
        loss1 = self.loss(logitsT*(self.logit_scale.exp()), labels)
        loss = lossim+loss1
        loss=loss/2
        loss = loss.mean()
        self.results.append({"loss": loss, "imfeatures":image_features, "tfeatures":captions,"classes":batch[2]})
        return {"loss": loss, "imfeatures":image_features, "tfeatures":captions,"classes":batch[2]}
    def on_validation_epoch_end(self,acc_val):
        super().on_validation_epoch_end(acc_val)
        #step 3, repeat for each previous epoch (as a cum sum?))

    def _insert_hooks(self):
        self.handles=[]
        # if layer weight is has self.hparams.train_batch_size in shape or layer.weight is None])
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model1", name)) for name, layer in self.encode_image.named_modules()]) 
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model1", name)) for name, layer in self.encoder.named_modules() ]) 
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model2", name)) for name, layer in self.model2.visual.named_modules()]) 
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model2", name)) for name, layer in self.model2.transformer.named_modules()])
        


##########THIS IS WHERE I GOT TO BEFORE FOOD >>


        
def batch_HSIC2(K):
    #K is Layers x B x B
    a=torch.sum(K,dim=-1)
    #print(" K SHAPE ",K.shape)# 0,2,3, are all problem values..
    b=torch.sum(K,dim=-2)
    c=torch.sub(torch.pow(torch.sum(a,dim=-1),2)/(K.shape[-2] - 1),torch.sum(a*b,dim=1),alpha=2)
    #print(torch.sum(torch.sum(K*K.permute(0,2,1),dim=-1),dim=-1))
    output=torch.add(torch.sum(torch.sum(K*K.permute(0,2,1),dim=-1),dim=-1),torch.div(c,(K.shape[-2] - 2)))
    return torch.div(output,(K.shape[-2]*(K.shape[-2] - 3)))
    #check for why pos infs... 
def batch_HSIC3(K,L):
    K=K.unsqueeze(1) # 46,1,B,B
    L=L.unsqueeze(0) # 1,46, B,B
    a=torch.sum(L,dim=-1) #1,46,10
    b=torch.sum(K,dim=-2) #46,1,10
    #print(a.shape,b.shape)
    c=torch.sub(torch.mul(torch.sum(b,dim=-1),torch.sum(a,dim=-1)).div((K.shape[-2] - 1)),torch.sum(torch.mul(b,a),dim=-1),alpha=2) #[46,46]- [46,46] =[46,46]
    #print(c.shape) # expect LayerK, LayerL, 
    return torch.div(torch.add(torch.sum(torch.sum(K*L,dim=-1),dim=-1),torch.div(c,(K.shape[-2] - 2))),(K.shape[-2]*(K.shape[-2] - 3)))
    #returns many pos infs 
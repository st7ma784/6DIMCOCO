
from model.LossCalculation import calculate_lossStock as sl
from model.LossCalculation import calculate_lossNormsvc

from model.Projection import get_proj_fn
from model.nargsLossCalculation import get_loss_fn,get_loss_calc

import torch.nn as nn
import torch
import numpy as np
from typing import Optional
from clip.model import Transformer,LayerNorm
import clip

from functools import reduce
from model.trainclip_cka_base import LightningCLIPModule as base
class LightningCLIPModule(base):
    def __init__(self,
                
                learning_rate,
                logitsversion=0,
                normlogits=True,
                projection='inv',
                prune=True,
                exactlabels=False,
                logvariance=False,
                maskLosses=0,
                meanloss=False,
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
        self.save_hyperparameters()
        print("learning_rate",learning_rate)
        transformer_width=512
        embed_dim=512
        #this is needed for the clip model to work
        self.context_length = context_length
        self.encoder = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
            )
        self.clip,_=clip.load("ViT-B/32", device=self.device)
        #self.linear.weight=torch.nn.Parameter(self.clip.token_embedding.weight.T)

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.transformer_width=transformer_width
        self.handles=[]
        self.tfeatures=None

        self.labels=[]
        self.model1_info={'Name':"SelfCLIP",}
        self.model2_info={'Name': "Stock CLIP",}
        self.naninfcount=0
        
        self.calculate_lossStock=sl

        self.calculate_lossStock2=calculate_lossNormsvc
        self.calculate_loss=get_loss_fn(logitsversion=logitsversion,norm=normlogits,log=logvariance,JSE=kwargs.get("JSE",0))

        self.projection=get_proj_fn(projection)
        self.prune=prune
        if self.prune:
            from model.PruneCalculation import PruneHook
            self.pruneHooks=[PruneHook(self.encoder,[-1,0,1], 0.1, method="Hard", prune_eta=4, amount=4,fc_pru_bound=-1)]
        else:
            self.pruneHooks=[]
        self.initialize_parameters()
        # self.loss=get_loss_calc(reduction='sum',ver=0,mask=torch.ones([1]))

        if exactlabels==1:
            with torch.no_grad():
                testBatch=torch.rand(self.hparams.batch_size,self.transformer_width,device=self.device)
                if not normlogits:
                    testBatch=testBatch/torch.norm(testBatch,dim=-1,keepdim=True)
                self.label=self.calculate_loss(testBatch,testBatch,testBatch).to(self.device,non_blocking=True)
                #convert this to probabilities in range [0,1]
                self.label=torch.nn.functional.softmax(self.label)
                self.label=torch.nan_to_num(self.label, nan=1.0)
            print("using labels: ", self.label[:2,:2,:2])
        #elif add in the case where using -inf or -1 instead of zeros as below....
        else:
            self.label=torch.diag_embed(torch.diag_embed(torch.ones(self.hparams.batch_size,dtype=torch.float,device=self.device)))            #self.label=(self.label*2)-1 This makes loss negative! 
            print("using labelsv2: ", self.label[:2,:2,:2])
        self.label=torch.nan_to_num(self.label)
        self.maskLoss=maskLosses
        self.maskloss=torch.nn.MSELoss(reduction='none')

        B,N=self.hparams.batch_size,3
        Views=torch.diag_embed(torch.ones(N,dtype=torch.long)*B-1)+1
        self.Lossmask=torch.sum(reduce(torch.add,list(map(lambda Arr: torch.nn.functional.one_hot(torch.arange(B).view(*Arr),num_classes=B),Views.tolist()))).pow(4),dim=-1)
        assert self.label.shape == self.Lossmask.shape

        self.masks=torch.unique(torch.flatten(self.Lossmask,0,-1),dim=0,sorted=False)

        self.alpha=nn.Parameter(torch.ones_like(self.masks,dtype=torch.float))
        self.Lossmasks=torch.ones([1],device=self.device)

        from model.nargsLossCalculation import get_loss_sum
        self.meanloss=get_loss_sum(meanloss)
        self.loss=get_loss_calc(reduction='sum',ver=self.maskLoss,mask=self.Lossmask)
            
            #alpha for weighting regions. 
        #this is one set of masks, theres another set however, of



    # @torch.jit.script
    def forward(self, im, captions1,*captions):

        image_features=self.clip.encode_image(im)
        caption_features1=self.clip.encode_text(captions1)
        caption_features2=[self.encode_text(c) for c in captions]#
      

        if self.projection=="inv":
            image_features=image_features@ self.text_projection
        elif self.projection=="iinv":
            image_features=image_features@torch.inverse(self.text_projection)
        elif self.projection=="None":
            caption_features1=caption_features1@self.text_projection
            caption_features2=[c@self.text_projection for c in caption_features2]
          
        return self.calculate_loss(image_features, caption_features1,*caption_features2)
        #return self.calculate_lossStock(image_features, caption_features1)[0]*self.logit_scale.exp()

        
    def training_step(self, batch, batch_idx):

        im,captions= batch[0],batch[1]
        
        logits=self(im,*[captions[:,i] for i in range(captions.shape[1])])*self.logit_scale.exp()
        
        try:
            labels=self.label[:(im.shape[0]),:(im.shape[0]),:(im.shape[0])].to(self.device,non_blocking=True) 
        except:
            #labels wrong size!!?!
            labels=self.generate_labels((len(logits.shape),self.hparams.batch_size,self.transformer_width)).to(self.device,non_blocking=True)
        firstlogit=logits.flatten()[0]
       

        n_dims=len(logits.shape)
        dims=np.arange(n_dims).repeat(n_dims).reshape(n_dims,n_dims)
        dims_=np.arange(n_dims)
        dims_=np.expand_dims(dims_,axis=0)
        permutes=dims+dims_
        permutes=permutes%n_dims
        bad_logit=logits[permutes].mean()

        # assert bad_logit.shape[0]==firstlogit.shape[0]
        self.log("first logit",firstlogit,enable_graph=False)
        self.log("BAD logit",bad_logit,enable_graph=False)
        self.log("logit scale",self.logit_scale.exp())

        # The idea is that good logits are 1s,   bad should be -1s... so if logits are coming back as ~6000....
        #  Option 1: divide down.
        #  Option 2: 1- output...
        # option 3: logarithmic functions? 
        

        
        losses = [self.loss(logits.permute(*i), labels,alpha=self.alpha) for i in permutes]
        
        loss=self.meanloss(I=[losses[0]],T=losses[1:]).mean()
      
        self.log('train_loss', loss, prog_bar=True,enable_graph=False, rank_zero_only=True)

        return {"loss": loss}


    def on_validation_epoch_start(self):
        self.log("Mean Projection Value",self.text_projection.mean(),enable_graph=False)
        super().on_validation_epoch_start()


    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()

 
    

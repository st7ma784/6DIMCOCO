
from regex import B
import torch.nn as nn
import torch
import numpy as np
from typing import Optional
from clip.model import LayerNorm
from functools import reduce
from transformers import MarianMTModel,MarianConfig, CLIPTokenizer
from model.trainclip_cka_base import LightningCLIPModule as base
from model.trainclip_cka_base import batch_HSIC2,batch_HSIC3
from functools import partial
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
        # if hasattr(self,"clip"):
        #     del self.clip
        self.save_hyperparameters()
        print("learning_rate",learning_rate)
        self.transformer_width=transformer_width
        self.embed_dim=transformer_width
        #this is needed for the clip model to work
        self.context_length = context_length
        
        
        #on top of this transformer, we're going to have a full translation model to 
        #translate the language to
        config=MarianConfig(
            vocab_size=vocab_size,
            decoder_bos_token_id=self.clip.vocab_size-1,
            decoder_eos_token_id=self.clip.vocab_size,
            pad_token_id=0,
            activation_dropout=0,
            activation_function="swish", #"swish" 
            attention_dropout=0.0,
            classifier_dropout=0.0,
            d_model=512,
            decoder_attention_heads=16,
            decoder_ffn_dim=2048,
            decoder_layerdrop=0.0,
            decoder_layers=3, #would be higher if I had more VRAM
            decoder_start_token_id=self.clip.vocab_size-1,
            decoder_vocab_size=self.clip.vocab_size,
            dropout=0.0,
            encoder_attention_heads=16,
            encoder_ffn_dim=2048,
            encoder_layerdrop=0.0,
            encoder_layers=3, #would be higher if I had more VRAM
            eos_token_id=self.clip.vocab_size,
            forced_eos_token_id=0,
            init_std=0.02,
            is_encoder_decoder=True,
            max_position_embeddings=512,
            model_type="marian",
            num_hidden_layers=4,
            scale_embedding=False,
            share_encoder_decoder_embeddings=True,
            transformers_version="4.25.1",
            use_cache=True,
        )
        self.bos_token_id=self.clip.vocab_size-1
        self.transformerModel=MarianMTModel(config)
        self.transformerModel=self.transformerModel.to(self.device)
        self.transformerModel.train()
        #we need to first make some modifications to make this compatible with the CLIP tokenizers 
        #self.linear.weight=torch.nn.Parameter(self.clip.token_embedding.weight.T)

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.handles=[]

        self.labels=[]
        self.model1_info={'Name':"SelfCLIP",}
        self.model2_info={'Name': "Stock CLIP",}
        self.naninfcount=0
        from model.LossCalculation import calculate_lossStock as sl
        from model.LossCalculation import calculate_lossNormsvc
        self.calculate_lossStock=sl

        self.calculate_lossStock2=calculate_lossNormsvc
        from model.nargsLossCalculation import get_loss_fn,get_loss_calc
        self.calculate_loss=get_loss_fn(logitsversion=logitsversion,norm=normlogits,log=logvariance,JSE=kwargs.get("JSE",0))

        from model.Projection import get_proj_fn
        self.projection=get_proj_fn(projection)
        self.prune=prune
        if self.prune:
            from model.PruneCalculation import PruneHook
            self.pruneHooks=[PruneHook(self.transformerModel,[-1,0,1], 0.1, method="Hard", prune_eta=4, amount=4,fc_pru_bound=-1)]
        else:
            self.pruneHooks=[]
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

        torch.autograd.set_detect_anomaly(True)
        #with torch.no_grad():
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
            
        if kwargs.get("gumbel",False):
            self.token_select=partial(torch.nn.functional.gumbel_softmax,hard=True,dim=-1)
        else:
            self.token_select=partial(torch.nn.functional.softmax,dim=-1)
            
            #alpha for weighting regions. 
        #this is one set of masks, theres another set however, of
  
    def encode_text(self, text):
        decoder_ids=torch.zeros_like(text)
        decoder_ids[:,0]=self.clip.vocab_size-1
        output = self.transformerModel(text, decoder_input_ids=decoder_ids,)
        #take the output probabilities as a vector, 
        output=self.token_select(output.logits)
        x=output@self.token_embedding.weight 
        # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] 
        return x


    # @torch.jit.script
    def forward(self, im, captions1, *captions):
        captions=captions[:2]
        image_features=self.clip.encode_image(im)
        caption_features1=self.clip.encode_text(captions1)
        caption_features2=[self.encode_text(c) for c in captions]
      

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

        logits=self(im,*captions)*self.logit_scale.exp()
        labels=self.label
        
        if labels.shape!=logits.shape:
            labels=self.generate_labels((len(logits.shape),self.hparams.batch_size,self.transformer_width)).to(self.device,non_blocking=True)
            self.label=labels
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



    def validation_step(self,batch,*args):
        #do stock loss here
        labels=torch.diag_embed(torch.ones(self.hparams.batch_size,dtype=torch.float,device=self.device))
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


        # print("self.logit scale is 14 right? ",self.logit_scale.exp())
        #check captions isnt tuple
        if isinstance(captions,tuple):
            captions=captions[0]
        logitsI,logitsT=self.calculate_lossStock(image_features, captions) 
        self.log("mean validation stock logits ", logitsI.mean())
        #doing stock loss here! so we should assume that labels is of type long 
        lossim = self.loss(logitsI*(self.logit_scale.exp()), labels,alpha=self.alpha)
        loss1 = self.loss(logitsT*(self.logit_scale.exp()), labels,alpha=self.alpha)
        loss = lossim+loss1
        loss=loss/2
        loss = loss.mean()
        self.results.append({"loss": loss, "imfeatures":image_features, "tfeatures":captions,"classes":batch[2]})
        return {"loss": loss, "imfeatures":image_features, "tfeatures":captions,"classes":batch[2]}

    


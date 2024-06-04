import torch.nn as nn
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional
from clip.model import Transformer,LayerNorm,VisionTransformer
from functools import reduce
from model.trainclip_cka_base import LightningCLIPModule as CKA_base

torch.autograd.set_detect_anomaly(True)

class LightningCLIPModule(CKA_base):
    def __init__(self,
                
                learning_rate,
                logitsversion=0,
                normlogits=True,
                maskLosses=2,
                projection='None',
                logvariance=False,
                prune=True,
                meanloss=False,
                exactlabels=0,
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
        from model.LossCalculation import get_loss_sum
        self.meanloss=get_loss_sum(meanloss)
        self.context_length = context_length
        self.encoder = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
            )
        self.im_enc= VisionTransformer(
                input_resolution=224,
                patch_size=16,
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                output_dim=transformer_width
            )
        
        #self.linear.weight=torch.nn.Parameter(self.clip.token_embedding.weight.T)
        from model.LossCalculation import get_loss_calc
        self.valloss=torch.nn.CrossEntropyLoss(reduction='mean')
        self.logvariance=logvariance
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, transformer_width))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.transformer_width=transformer_width
        self.handles=[]
        self.model1_info={'Name':"SelfCLIP",}
        self.model2_info={'Name': "Stock CLIP",}
        self.naninfcount=0
        from model.LossCalculation import calculate_lossStock as sl
        self.calculate_lossStock=sl
        from model.nargsLossCalculation import get_loss_fn 
        self.calculate_loss=get_loss_fn(logitsversion=logitsversion,norm=normlogits,log=logvariance,JSE=kwargs.get("JSE",0))
        # if logitsversion==0:
        #     from model.LossCalculation import calculate_loss as cl
        # elif logitsversion==1: 
        #     from model.LossCalculation import calculate_loss2 as cl
        # elif logitsversion==2: 
        #     from model.LossCalculation import calculate_loss3 as cl
        # elif logitsversion==3:
        #     from model.LossCalculation import calculate_loss4 as cl
        # elif logitsversion==4:
        #     from model.LossCalculation import calculate_loss5 as cl
        # elif logitsversion==5:
        #     from model.LossCalculation import calculate_loss6 as cl
        # else:
        #     from model.LossCalculation import calculate_loss as cl
        # self.calculate_loss=partial(cl,norm=normlogits,log=logvariance)
        from model.Projection import get_proj_fn
        self.projection=get_proj_fn(projection)
        self.prune=prune
        if self.prune:
            from model.PruneCalculation import PruneHook
            self.pruneHooks=[PruneHook(self.im_enc,[-1,0,1], 0.1, method="Hard", prune_eta=4, amount=4,fc_pru_bound=-1),
                             PruneHook(self.encoder,[-1,0,1], 0.1, method="Hard", prune_eta=4, amount=4,fc_pru_bound=-1)]
        else:
            self.pruneHooks=[]
        self.initialize_parameters()
        # self.loss=get_loss_calc(reduction='sum',ver=0,mask=torch.ones([1]))

        if exactlabels==1:
            with torch.no_grad():
                #testBatch=torch.rand(self.hparams.batch_size,self.transformer_width,device=self.device)
                testBatch=torch.normal(0,0.3,(self.hparams.batch_size,self.transformer_width),device=self.device)
                if not normlogits:
                    testBatch=testBatch/torch.norm(testBatch,dim=-1,keepdim=True)
                self.label=self.calculate_loss(testBatch,testBatch,testBatch,testBatch,testBatch,testBatch).to(self.device,non_blocking=True)
                #convert this to probabilities in range [0,1]
                self.label=torch.nn.functional.softmax(self.label)
                self.label=torch.nan_to_num(self.label, nan=1.0)
            print("using labels: ", self.label[:2,:2,:2,:2,:2,:2])
        #elif add in the case where using -inf or -1 instead of zeros as below....
        else:
            self.label=torch.diag_embed(torch.diag_embed(torch.diag_embed(torch.diag_embed(torch.diag_embed(torch.ones(self.hparams.batch_size,dtype=torch.float,device=self.device))))))
            #self.label=(self.label*2)-1 This makes loss negative! 
            print("using labelsv2: ", self.label[:2,:2,:2,:2,:2,:2])
        self.label=torch.nan_to_num(self.label)
        self.maskLoss=maskLosses
        self.maskloss=torch.nn.MSELoss(reduction='none')

        #with torch.no_grad():
        B,N=self.hparams.batch_size,6
        Views=torch.diag_embed(torch.ones(N,dtype=torch.long)*B-1)+1
        self.Lossmask=torch.sum(reduce(torch.add,list(map(lambda Arr: torch.nn.functional.one_hot(torch.arange(B).view(*Arr),num_classes=B),Views.tolist()))).pow(4),dim=-1)
        assert self.label.shape == self.Lossmask.shape

        self.masks=torch.unique(torch.flatten(self.Lossmask,0,-1),dim=0,sorted=False)

        self.alpha=nn.Parameter(torch.ones_like(self.masks,dtype=torch.float))
        self.Lossmasks=torch.ones([1],device=self.device)

       

        self.loss=get_loss_calc(reduction='sum',ver=self.maskLoss,mask=self.Lossmask)
            
            #alpha for weighting regions. 
        #this is one set of masks, theres another set however, of


    def encode_image(self,*Args,**kwargs):
        return self.im_enc(*Args,**kwargs)
    # @torch.jit.script
    def forward(self, im, *captions):
        image_features=self.encode_image(im)
        caption_features=[self.encode_text(c) for c in captions]
        [i],captions=self.projection(self.text_projection,im=[image_features],text=caption_features)
        
        return self.calculate_loss(i,*captions).mul(torch.exp(self.logit_scale))

    def training_step(self, batch, batch_idx):

        im,captions= batch[0],batch[1]
        labels=self.label[:(im.shape[0]),:(im.shape[0]),:(im.shape[0]),:(im.shape[0]),:(im.shape[0]),:(im.shape[0])].to(self.device,non_blocking=True) 

        logits=self(im,*[captions[:,i] for i in range(captions.shape[1])])
        self.log("first logit",logits[0,0,0,0,0,0],enable_graph=False)
        self.log("BAD logit",logits[1,2,3,4,5,0],enable_graph=False)
        # The idea is that good logits are 1s,   bad should be -1s... so if logits are coming back as ~6000....
        #  Option 1: divide down.
        #  Option 2: 1- output...
        # option 3: logarithmic functions? 
        #print("logits",logits.shape)
        #print("labels",labels.shape)
        mloss=self.maskloss(logits,labels)
        meanloss=torch.mean(mloss)
        self.log("meanloss",meanloss,enable_graph=False, rank_zero_only=True)
        for mask in self.masks:
            mea=torch.mean(mloss[self.Lossmask==mask])
            self.log("maskVal={}".format(mask),mea,enable_graph=False, rank_zero_only=True)
            self.log("proportionmaskVal={}".format(mask),torch.div(mea,meanloss),enable_graph=False, rank_zero_only=True)
            #self.log("absdeltamaskVal={}".format(mask),torch.sub(loss,loss[self.Lossmasks==mask]),enable_graph=False, rank_zero_only=True)
        
      
        
        n_dims=captions.shape[1]+1
        dims=np.arange(n_dims).repeat(n_dims).reshape(n_dims,n_dims)
        dims_=np.arange(n_dims)
        dims_=np.expand_dims(dims_,axis=0)
        permutes=dims+dims_
        permutes=permutes%n_dims
        #create a list of [0,1,2,3,4,5] and rotated versions of it.


        
        losses = [self.loss(logits.permute(*i), labels,alpha=self.alpha) for i in permutes]
        
        loss=self.meanloss(I=[losses[0]],T=losses[1:]).mean()
      
        self.log('train_loss', loss, prog_bar=True,enable_graph=False, rank_zero_only=True)
        
        return {"loss": loss}

            

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.log("Mean Projection Value",self.text_projection.mean(),enable_graph=False)

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        
    def test_step(self, batch, batch_idx):
        super().test_step(batch, batch_idx)

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.results=[]
    
    def on_test_epoch_end(self):
        super().on_test_epoch_end()
    def test_token_embeddings(self):
        #create int of all the tokens in vocab size
        #embed them all with self.token_embeddings
        #perform kmeans on them all, 
        #log the clusters and the tokens nearest to each centroid. 
        tokens=torch.arange(self.token_embedding.num_embeddings,device=self.device)
        embeddings=self.token_embedding(tokens)
        # kmeans = KMeans(n_clusters=40, random_state=0).fit(embeddings)
        # for i in range(10):
        #     print(kmeans.cluster_centers_[i])
        #     print(tokens[kmeans.labels_==i])
        # self.logger.log_text("token embeddings cluster centers ",str(kmeans.cluster_centers_))
        # self.logger.log_text("token embeddings tokens nearest centers",str(tokens[kmeans.labels_==i]))
        # #log the tokens closest to the mean of all embeddings.
        values,indxs=torch.sort(torch.norm(embeddings-embeddings.mean(dim=0),dim=1),)
        self.logger.log_text("token embeddings center-most tokens",columns=tokens[indxs[:10]].tolist(),data=[values[:10].tolist()])
        self.logger.log_text("token embeddings furthest tokens",columns=tokens[indxs[-10:]].tolist(),data=[values[-10:].tolist()])

  


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
    c=torch.sub(torch.mul(torch.sum(b,dim=-1),torch.sum(a,dim=-1)).div((K.shape[-2] - 1)),torch.sum(torch.mul(b,a),dim=-1),alpha=2) #[46,46]- [46,46] =[46,46]
    #print(c.shape) # expect LayerK, LayerL, 
    return torch.div(torch.add(torch.sum(torch.sum(K*L,dim=-1),dim=-1),torch.div(c,(K.shape[-2] - 2))),(K.shape[-2]*(K.shape[-2] - 3)))
    #returns many pos infs 

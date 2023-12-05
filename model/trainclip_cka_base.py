from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from functools import partial,reduce
import clip

import matplotlib.pyplot as plt
from CKA_test import add_colorbar 
from sklearn.linear_model import LogisticRegression
from model.LossCalculation import get_loss_fn 
from model.Projection import get_proj_fn

class LightningCLIPModule(LightningModule):
    def __init__(self,
                **kwargs,
                ):

        super().__init__()
        self.save_hyperparameters()
        self.clip,_=clip.load("ViT-B/32", device=self.device)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        #self.linear.weight=torch.nn.Parameter(self.clip.token_embedding.weight.T)
        from model.LossCalculation import get_loss_calc
        self.valloss=torch.nn.CrossEntropyLoss(reduction='mean')
        self.handles=[]
        self.model1_info={'Name':"SelfCLIP",}
        self.model2_info={'Name': "Stock CLIP",}
        self.naninfcount=0
        # self.loss=get_loss_calc(reduction='sum',ver=0,mask=torch.ones([1]))

        self.label=torch.diag_embed(torch.diag_embed(torch.diag_embed(torch.diag_embed(torch.diag_embed(torch.ones(self.hparams.batch_size,dtype=torch.float,device=self.device))))))
        #self.label=(self.label*2)-1 This makes loss negative! 
        print("using labelsv2: ", self.label[:2,:2,:2,:2,:2,:2])
        self.label=torch.nan_to_num(self.label)
        self.calculate_loss=get_loss_fn(logitsversion=2)

        B,N=self.hparams.batch_size,6
        self.projection=get_proj_fn("none")


    # @torch.jit.script
    def forward(self, im, captions1, captions2, captions3, captions4, captions5):
        image_features=self.clip.encode_image(im)
        caption_features1=self.clip.encode_text(captions1)

        i,t=self.projection(self.text_projection,im=[image_features],text=[caption_features1,caption_features2,caption_features3,caption_features4,caption_features5])
        
        return self.calculate_loss(*[*i,*t]).mul(torch.exp(self.logit_scale))

    def training_step(self, batch, batch_idx,optimizer_idx=0):

        pass

            
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters()]+[p for p in self.encode_image.parameters()]+[p for p in self.encoder.parameters()], lr=self.hparams.learning_rate, eps=10e-7,
            #weight_decay=0.1,
            #betas=(0.9, 0.95),
            )
        lr_schedulers = {"scheduler": ReduceLROnPlateau(optimizer), "monitor": "train_loss"}

        return [optimizer],[lr_schedulers]

    def on_validation_epoch_start(self):
        self.log("logit scale",self.logit_scale.exp())
        self.naninfcount=0
        self.model2,_ = clip.load("ViT-B/32", device=self.device)
        self.model2.eval()
        self._insert_hooks()
        self.IMhsic_matrix0=torch.zeros([],device=self.device)
        self.IMhsic_matrix1=torch.zeros([],device=self.device)
        self.IMhsic_matrix2=torch.zeros([],device=self.device)
        self.CAPhsic_matrix0=torch.zeros([],device=self.device)
        self.CAPhsic_matrix1=torch.zeros([],device=self.device)
        self.CAPhsic_matrix2=torch.zeros([],device=self.device)
        self.eval()
        self.results=[]

    def validation_step(self,batch,*args):
        #do stock loss here
       
        self.model1_features = {}  #reset list of forward hooks
        self.model2_features = {}  #reset list of forward hooks
        image_features=self.clip.encode_image(batch[0])
        #if rank 0
        self.model2.encode_image(batch[0])# to compare supervision model
        a=torch.nan_to_num(torch.stack(list(self.model1_features.values())))
        self.IMhsic_matrix0=torch.add(self.IMhsic_matrix0,torch.nan_to_num(self.batch_HSIC2(a),nan=0.0,posinf=100,neginf=-200)) 
        a=torch.nan_to_num(torch.stack(list(self.model2_features.values())))
      
        self.IMhsic_matrix2=torch.add(self.IMhsic_matrix2,torch.nan_to_num(self.batch_HSIC2(a),nan=0.0,posinf=100,neginf=-200))
        joint_HSIC=torch.nan_to_num(self.batch_HSIC3(a,torch.nan_to_num(torch.stack(list(self.model1_features.values())))), nan=0.0,posinf=1,neginf=-2)
        self.IMhsic_matrix1=torch.add(self.IMhsic_matrix1,joint_HSIC) 
        ##Now Do Text
        self.model1_features = {}  #reset list of forward hooks
        self.model2_features = {}  #reset list of forward hooks
        choice=torch.randint(0,5,(1,)).item()
        #rint("choice", choice)
        c=batch[1][:,choice]
        c=c.squeeze()

        captions=self.clip.encode_text(c) #run through main mode
        self.model2.encode_text(c)# to compare supervision model
        a=torch.nan_to_num(torch.stack(list(self.model1_features.values())))
        self.CAPhsic_matrix0=torch.add(self.CAPhsic_matrix0,self.batch_HSIC2(a)) 
        a=torch.nan_to_num(torch.stack(list(self.model2_features.values())))
        self.CAPhsic_matrix2=torch.add(self.CAPhsic_matrix2,self.batch_HSIC2(a))
        joint_HSIC=torch.nan_to_num(self.batch_HSIC3(a,torch.nan_to_num(torch.stack(list(self.model1_features.values())))))
        self.CAPhsic_matrix1=torch.add(self.CAPhsic_matrix1,joint_HSIC) 
        
        [image_features], [captions] = self.projection(self.text_projection,im=[image_features],text=[captions])
        # print("self.logit scale is 14 right? ",self.logit_scale.exp())
        logitsI,logitsT=self.calculate_loss(image_features, captions) 
        self.log("mean validation stock logits ", logitsI.mean())
        labels=torch.arange(batch[0].shape[0],dtype=torch.long,device=self.device)

        lossim = self.valloss(logitsI*self.logit_scale.exp(), labels)
        loss1 = self.valloss(logitsT*self.logit_scale.exp(), labels)
        loss = lossim+loss1
        loss=loss/2
        loss = loss.mean()
        self.log('val_loss-stock', loss, prog_bar=True,enable_graph=False, rank_zero_only=True)
        self.results.append({"imfeatures":image_features, "tfeatures":captions,"classes":batch[2],"loss": loss})

        return {"loss": loss}

    def on_validation_epoch_end(self):
      
        self.unfreeze()
        self.train()
        self.plot_results("IM","IMHSIC{}.jpg".format(self.current_epoch))
        self.plot_results("CAP","CAPHSIC{}.jpg".format(self.current_epoch))
        if self.logger is not None:
            self.logger.log_image(key="IMHSIC{}".format(self.current_epoch), images=["IMHSIC{}.jpg".format(self.current_epoch)])        
            self.logger.log_image(key="CAPHSIC{}".format(self.current_epoch), images=["CAPHSIC{}.jpg".format(self.current_epoch)])
        for handle in self.handles:
            handle.remove()
        #print(self.naninfcount)
        del self.model2
      
      
    def test_step(self,batch,*args):
        #do stock loss here
        image_features=self.clip.encode_image(batch[0])
        self.results.append({"imfeatures":image_features, "classes":batch[1]})

        return {"imfeatures":image_features, "classes":batch[1]}

    def on_test_epoch_start(self):
        self.results=[]
    def on_test_epoch_end(self):
        imfeatures=torch.nan_to_num(torch.cat([val["imfeatures"] for val in self.results],dim=0)).cpu().numpy()
        labels=torch.cat([val["classes"] for val in self.results],dim=0).cpu().numpy()
        if not hasattr(self,"Iclassifier"):
            self.Iclassifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1, n_jobs=-1)
   
        self.Iclassifier.fit(imfeatures, labels)
        self.log( "TopK Imagenet",self.Iclassifier.score(imfeatures, labels))
        del self.results
    
    def _log_layer(self, model: str, name: str, layer: nn.Module,inp: torch.Tensor, out: torch.Tensor):
        if isinstance(out, tuple):
            out=out[0]       
        if out.shape[0] == self.hparams.train_batch_size:
            self.__store(out,name,model,layer)
        elif out.shape[1] == self.hparams.train_batch_size:
            self.__store(out.permute(1,0,*torch.arange(len(out.shape)-2)+2),name,model,layer)

    def __store(self,out,name, model,layer):
        X = out.flatten(1)
        X= torch.nan_to_num((X @ X.t()).fill_diagonal_(0))
        if (torch.isnan(X).any() or torch.isinf(X).any()):
            self.naninfcount+=1
        if model == "model1":
            while name in self.model1_features:
                name=name+"1"
            self.model1_features[name] = X

        elif model == "model2":
            while name in self.model1_features:
                name=name+"1"
            self.model2_features[name] = X

        else:
            raise RuntimeError("Unknown model name for _log_layer.")

    def _insert_hooks(self):
        self.handles=[]
        # if layer weight is has self.hparams.train_batch_size in shape or layer.weight is None])
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model1", name)) for name, layer in self.clip.encode_image.named_modules()]) 
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model1", name)) for name, layer in self.clip.encoder.named_modules() ]) 
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model2", name)) for name, layer in self.model2.visual.named_modules()]) 
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model2", name)) for name, layer in self.model2.transformer.named_modules()])
        
  
    def export(self):
      
        return {
            "model1_name": "Trained",
            "model2_name": "PretrainedModel",
            "IMCKA":self.IMhsic_matrix1 / (torch.sqrt(self.IMhsic_matrix0.unsqueeze(1))*torch.sqrt(self.IMhsic_matrix2.unsqueeze(0))),
            "CAPCKA":self.CAPhsic_matrix1 / (torch.sqrt(self.CAPhsic_matrix0.unsqueeze(1))*torch.sqrt(self.CAPhsic_matrix2.unsqueeze(0))),
            "model1_layers": self.named_modules(),
            "model2_layers": self.model2.named_modules(),
        }

    def plot_results(self,
                     model_name: str,
                     save_path: str = None,
                     title: str = None):
        title =model_name+" HSIC" if title is None else model_name+title
        fig, ax = plt.subplots()
        if model_name=="IM":
            #print(self.IMhsic_matrix0) #46 #Comes out inf on val step
            #print(self.IMhsic_matrix2) # 110
            t=self.IMhsic_matrix0.unsqueeze(1)*self.IMhsic_matrix2.unsqueeze(0) #46 x 110
        #print(torch.sum(torch.abs(t)==t))
            r=torch.sqrt(torch.abs(t))
            r[torch.abs(t)==-t]=-r[torch.abs(t)==-t]
            #print("im1",self.IMhsic_matrix1)
            #print("r", r)
            hsic_matrix = torch.div(self.IMhsic_matrix1.squeeze().t(), r)
            #print("hsic",hsic_matrix)
        else:
            print(self.CAPhsic_matrix0.shape,self.CAPhsic_matrix2.shape)
            t=self.CAPhsic_matrix0.unsqueeze(1)*self.CAPhsic_matrix2.unsqueeze(0)
            r=torch.sqrt(torch.abs(t))
            r[torch.abs(t)==-t]=-r[torch.abs(t)==-t]
            #print("cap1", self.CAPhsic_matrix1.shape)
            #print("r",r.shape)
            hsic_matrix = torch.div(self.CAPhsic_matrix1.squeeze().t() , r)
        hsic_matrix=torch.nan_to_num(hsic_matrix,nan=0)
        im = ax.imshow(hsic_matrix.cpu(), origin='lower', cmap='magma')
        ax.set_xlabel(f"Layers {self.model2_info['Name']}", fontsize=15)
        ax.set_ylabel(f"Layers {self.model1_info['Name']}", fontsize=15)
        if title is not None:
            ax.set_title(f"{title}", fontsize=18)
        else:
            ax.set_title(f"{self.model1_info['Name']} vs {self.model2_info['Name']}", fontsize=18)
        add_colorbar(im)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300)



    def batch_HSIC2(self,K):
        #K is Layers x B x B
        a=torch.sum(K,dim=-1)
        #print(" K SHAPE ",K.shape)# 0,2,3, are all problem values..
        b=torch.sum(K,dim=-2)
        c=torch.sub(torch.pow(torch.sum(a,dim=-1),2)/(K.shape[-2] - 1),torch.sum(a*b,dim=1),alpha=2)
        #print(torch.sum(torch.sum(K*K.permute(0,2,1),dim=-1),dim=-1))
        output=torch.add(torch.sum(torch.sum(K*K.permute(0,2,1),dim=-1),dim=-1),torch.div(c,(K.shape[-2] - 2)))
        return torch.div(output,(K.shape[-2]*(K.shape[-2] - 3)))
        #check for why pos infs... 
    def batch_HSIC3(self,K,L):
        print("K SHAPE ",K.shape)
        print("L SHAPE ",L.shape)
        K=K.unsqueeze(1) # 46,1,B,B
        L=L.unsqueeze(0) # 1,46, B,B
        a=torch.sum(L,dim=-1) #1,46,10
        b=torch.sum(K,dim=-2) #46,1,10
        c=torch.sub(torch.mul(torch.sum(b,dim=-1),torch.sum(a,dim=-1)).div((K.shape[-2] - 1)),torch.sum(torch.mul(b,a),dim=-1),alpha=2) #[46,46]- [46,46] =[46,46]
        #print(c.shape) # expect LayerK, LayerL, 
        return torch.div(torch.add(torch.sum(torch.sum(K*L,dim=-1),dim=-1),torch.div(c,(K.shape[-2] - 2))),(K.shape[-2]*(K.shape[-2] - 3)))
        #returns many pos infs 

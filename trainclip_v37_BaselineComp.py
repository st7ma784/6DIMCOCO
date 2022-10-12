
import pytorch_lightning
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import os
from functools import partial
import numpy as np
from typing import Optional
from clip.model import Transformer,LayerNorm,VisionTransformer
from pytorch_lightning.callbacks import TQDMProgressBar,EarlyStopping
# from deepspeed.ops.adam import FusedAdam,DeepSpeedCPUAdam
import clip
from warnings import warn
from mpl_toolkits import axes_grid1
import matplotlib.pyplot as plt
from CKA_test import add_colorbar 


class LightningCLIPModule(LightningModule):
    def __init__(self,
                
                learning_rate,
                useclip_en=True,
                useclip_im=True,
                JSE=False,
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
        self.JSE=JSE
        self.gelu=nn.GELU()
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

        #self.linear.weight=torch.nn.Parameter(self.clip.token_embedding.weight.T)
        self.lossim=torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss1=torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss2=torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss3=torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss4=torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss5=torch.nn.CrossEntropyLoss(reduction='mean')

        self.vocab_size = vocab_size
        self.automatic_optimization=False
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()
        self.handles=[]
        self.model1_info={'Name':"SelfCLIP",}
        self.model2_info={'Name': "Stock CLIP",}
        print("ici")

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
  
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
 
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.encoder.width ** -0.5) * ((2 * self.encoder.layers) ** -0.5)
        attn_std = self.encoder.width ** -0.5
        fc_std = (2 * self.encoder.width) ** -0.5

        for block in self.encoder.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.text_projection, std=self.encoder.width ** -0.5)
    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.encoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x.contiguous()

    def orig_HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.
        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N=K.shape[0]
        return torch.add(torch.trace(K@L),torch.div(torch.sum(K)*torch.sum(L)/(N - 1) - (torch.sum(K@L) * 2 ), (N - 2)))
        
    def on_validation_epoch_start(self):
        self.eval()
        self.freeze()
    #     #import clip model here]
        self.model2,_ = clip.load("ViT-B/32", device=self.device)
        self._insert_hooks()
        self.eval()
        self.model2.eval()


    def validation_step(self,batch,*args):

        self.model1_features = {}  #reset list of forward hooks
        self.model2_features = {}  
        self.encode_image(batch[0]) #run through main mode
        ###If your model has supervised data, then perhaps do a loss with your date here!
        self.model2.encode_image(batch[0])# to compare supervision model
        N = len(self.model1_features.values())
        M = len(self.model2_features.values())
        print("N",N)
        print("M",M)
        out=torch.stack([self.orig_HSIC(K, K) for K in self.model1_features.values()])
        self.hsic_matrix0=torch.add(self.hsic_matrix0,out) if hasattr(self, 'hsic_matrix0') else out
        out=torch.stack([self.orig_HSIC(L, L) for L in self.model2_features.values()])
        self.hsic_matrix2=torch.add(self.hsic_matrix2,out) if hasattr(self, 'hsic_matrix2') else out
        out=torch.stack([self.orig_HSIC(K, L) for K in self.model1_features.values() for L in self.model2_features.values()])
        self.hsic_matrix1=torch.add(self.hsic_matrix1,out.reshape(N,M)) if hasattr(self, 'hsic_matrix1') else out.reshape(N,M)
        self.hsic_matrix = self.hsic_matrix1 / (torch.sqrt(self.hsic_matrix0.unsqueeze(1))*torch.sqrt(self.hsic_matrix2.unsqueeze(0)))
        if not torch.isnan(self.hsic_matrix).any():
            warn("HSIC computation resulted in NANs")
            
    def on_validation_epoch_end(self,):
        self.unfreeze()
        self.train()
        self.plot_results("HSICBaselineComp{}.jpg".format(self.current_epoch))
        if self.logger is not None:
            self.logger.log_image(key="HSICBaselineComp{}".format(self.current_epoch), images=["HSICBaselineComp{}.jpg".format(self.current_epoch)])
        for handle in self.handles:
            handle.remove()
        del self.model2

    def _log_layer(self, model: str, name: str, layer: nn.Module,inp: torch.Tensor, out: torch.Tensor):
        with torch.no_grad():
            if isinstance(out, tuple):
                out = out[0]
            #if activation shape is the same as dataloader batch size, then it is a linear layer
            if out.shape[0] == self.hparams.train_batch_size:
                print("LOGGING : ", model, name, out.shape)
                if model == "model1":
                    X = out.flatten(1)
                    self.model1_features[name] = (X @ X.t()).fill_diagonal_(0)
                elif model == "model2":
                    X = out.flatten(1)
                    self.model2_features[name] = (X @ X.t()).fill_diagonal_(0)
                else:
                    raise RuntimeError("Unknown model name for _log_layer.")

    def _insert_hooks(self):
       
        for name, layer in self.named_modules():
            self.handles.append(layer.register_forward_hook(partial(self._log_layer, "model1", name)))
      
        for name, layer in self.model2.named_modules():
            self.handles.append(layer.register_forward_hook(partial(self._log_layer, "model2", name)))
       
  
    def export(self):
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
        return {
            "model1_name": "Trained",
            "model2_name": "PretrainedModel",
            "CKA": self.hsic_matrix,
            "model1_layers": self.named_modules(),
            "model2_layers": self.model2.named_modules(),
        }

    def plot_results(self,
                     save_path: str = None,
                     title: str = None):
        fig, ax = plt.subplots()
        im = ax.imshow(self.hsic_matrix.cpu(), origin='lower', cmap='magma')
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

    
    def training_step(self, batch, batch_idx,optimizer_idx=0):
        # access your optimizers with use_pl_optimizer=False. Default is True,
        # setting use_pl_optimizer=True will maintain plugin/precision support

        labels=torch.diag_embed(torch.arange(batch[0].shape[0],dtype=torch.long,device=self.device)-self.lossim.ignore_index)
        logs=self.logit_scale.exp()
        for i in range(3):
            labels=torch.diag_embed(labels)
        labels=labels+self.lossim.ignore_index
        #self.labels=self.labels.to(self.device)
        with torch.no_grad():
            cache=self.encode_text(batch[1].flatten(start_dim=0,end_dim=1)).unflatten(0,(batch[1].shape[0],5),)
            cache=cache/cache.norm(dim=-1, keepdim=True)
            cache1=cache[:,0]
            cache2=cache[:,1]#.to(torch.device("cpu"),non_blocking=True)
            cache3=cache[:,2]#.to(torch.device("cpu"),non_blocking=True)
            cache4=cache[:,3]#.to(torch.device("cpu"),non_blocking=True)
            cache5=cache[:,4]#.to(torch.device("cpu"),non_blocking=True)
            del cache
        cacheim=self.encode_image(batch[0])
        if self.JSE:
            JSEFactor=1-(4/torch.sum(torch.stack([cacheim,cache1,cache2,cache3,cache4,cache5],dim=0).pow(2),dim=0))
            cacheim=torch.mul(cacheim,JSEFactor)
            cacheim=self.gelu(cacheim)
            del JSEFactor

        cacheim = cacheim / cacheim.norm(dim=1, keepdim=True)
        lossim = self.lossim(logs*torch.einsum("abcz,defz->abcdef",torch.einsum("az,bz,cz->abcz",cache3,cache4,cache5),torch.einsum("az,bz,cz->abcz",cacheim,cache1,cache2)),labels)
        cacheim=cacheim.detach()#.to(torch.device("cpu"),non_blocking=True)
        self.log('imloss', lossim, prog_bar=True,enable_graph=False,rank_zero_only=True)      
        captions=batch[1]
        cap1,cap2,cap3,cap4,cap5=captions[:,0],captions[:,1],captions[:,2],captions[:,3],captions[:,4]        
        del batch

        caption_features1=self.encode_text(cap1)
        if self.JSE:
            JSEFactor=1-(4/torch.sum(torch.pow(torch.stack([caption_features1,cache2,cache3,cache4,cache5,cacheim]),2),dim=0))
            caption_features1=torch.mul(caption_features1,JSEFactor)
            caption_features1=self.gelu(caption_features1)
            del JSEFactor
        caption_features1 = caption_features1 / caption_features1.norm(dim=1, keepdim=True)
        loss1 = self.loss1(logs*torch.einsum("abcz,defz->abcdef",torch.einsum("az,bz,cz->abcz",cache3,cache4,cache5),torch.einsum("az,bz,cz->abcz",cacheim,caption_features1,cache2)),labels)

        self.log('caption1', loss1, prog_bar=True,enable_graph=False,rank_zero_only=True)
        del caption_features1,cap1


        caption_features2=self.encode_text(cap2)
        #print(caption_features2.requires_grad)
        if self.JSE:
            JSEFactor=1-(4/torch.sum(torch.pow(torch.stack([cache1,caption_features2,cache3,cache4,cache5,cacheim]),2),dim=0))
            caption_features2=torch.mul(caption_features2,JSEFactor)
            caption_features2=self.gelu(caption_features2)
            del JSEFactor
        caption_features2 = caption_features2 / caption_features2.norm(dim=1, keepdim=True) 
        loss2 = self.loss2(logs*torch.einsum("abcz,defz->abcdef",torch.einsum("az,bz,cz->abcz",cache3,cache4,cache5),torch.einsum("az,bz,cz->abcz",cacheim,cache1,caption_features2)),labels)        
        #self.manual_backward(loss,retain_graph=True)
        self.log('caption2', loss2, prog_bar=True,enable_graph=False,rank_zero_only=True)
        del caption_features2,cap2


        caption_features3=self.encode_text(cap3)
        if self.JSE:
            JSEFactor=1-(4/torch.sum(torch.pow(torch.stack([cache1,cache2,caption_features3,cache4,cache5,cacheim]),2),dim=0))
            caption_features3=torch.mul(caption_features3,JSEFactor)
            caption_features3=self.gelu(caption_features3)
            del JSEFactor
        caption_features3 = caption_features3 / caption_features3.norm(dim=1, keepdim=True)
        loss3 = self.loss3(logs*torch.einsum("abcz,defz->abcdef",torch.einsum("az,bz,cz->abcz",caption_features3,cache4,cache5),torch.einsum("az,bz,cz->abcz",cacheim,cache1,cache2)),labels)        
        #self.manual_backward(loss,retain_graph=True)
        self.log('caption3', loss3,  prog_bar=True,enable_graph=False,rank_zero_only=True)
        del caption_features3,cap3 


        caption_features4=self.encode_text(cap4)
        if self.JSE:
            JSEFactor=1-(4/torch.sum(torch.pow(torch.stack([cache1,cache2,cache3,caption_features4,cache5,cacheim]),2),dim=0))
            caption_features4=torch.mul(caption_features4,JSEFactor)
            caption_features4=self.gelu(caption_features4)
            del JSEFactor

        caption_features4 = caption_features4 / caption_features4.norm(dim=1, keepdim=True)
        
        loss4 = self.loss4(logs*torch.einsum("abcz,defz->abcdef",torch.einsum("az,bz,cz->abcz",cache3,caption_features4,cache5),torch.einsum("az,bz,cz->abcz",cacheim,cache1,cache2)),labels)        
        #self.manual_backward(loss,retain_graph=True)
        self.log('caption4', loss4,  prog_bar=True,enable_graph=False,rank_zero_only=True)
        del caption_features4,cap4


        caption_features5=self.encode_text(cap5)
        if self.JSE:
            JSEFactor=-torch.div(4,torch.sum(torch.pow(torch.stack([cache1,cache2,cache3,cache4,caption_features5,cacheim]),2),dim=0))
            caption_features5=torch.mul(caption_features5,torch.add(JSEFactor,1))
            caption_features5=self.gelu(caption_features5)
            del JSEFactor
        caption_features5 = caption_features5 / caption_features5.norm(dim=1, keepdim=True)
        loss5 = self.loss5(logs*torch.einsum("abcz,defz->abcdef",torch.einsum("az,bz,cz->abcz",cache3,cache4,caption_features5),torch.einsum("az,bz,cz->abcz",cacheim,cache1,cache2)),labels)        
        #self.manual_backward(loss4,retain_graph=True)
        self.log('caption5', loss5, prog_bar=True,enable_graph=False,rank_zero_only=True)
        del caption_features5,cap5
        loss=lossim+loss5+loss1+loss2+loss3+loss4
        self.log('loss', loss, prog_bar=True,enable_graph=False,rank_zero_only=True)

        return loss

            
    def configure_optimizers(self):
        
        optimizerA = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
      

        return [optimizerA]
 
def wandbtrain(config=None,dir="/Data",devices="auto",accelerator="auto",Dataset=None):
    if config is not None:
        #config=config.__dict__
        config=config.__dict__
        dir=config.get("dir",dir)
        logtool= pytorch_lightning.loggers.WandbLogger( project="6DIMBaselineCompSweep",entity="st7ma784", save_dir=dir)
        # print(logtool.experiment)
        # logtool.experiment.config={}
        # logtool.experiment.config.update(config)
        # logtool.log_hyperparams(config)

    else: 
        #We've got no config, so we'll just use the default, and hopefully a trainAgent has been passed
        import wandb
        print("here")
        run=wandb.init(project="6DIMBaselineCompSweep",entity="st7ma784",name="6DIMBaselineCompSweep",config=config)
        logtool= pytorch_lightning.loggers.WandbLogger( project="6DIMBaselineCompSweep",entity="st7ma784",experiment=run, save_dir=dir)
        config=run.config.as_dict()
    print("config",config)
    
    train(config,dir,devices,accelerator,Dataset,logtool)

def train(config={
        "batch_size":16,
        "learning_rate":2e-3,
        "precision":16,
        "embed_dim": 512,
        "transformer_width": 512,
        "transformer_heads": 32,
        "transformer_layers": 4,
        "JSE":False,
    },dir=".",devices="auto",accelerator="auto",Dataset=None,logtool=None):
    model=LightningCLIPModule(  learning_rate = config["learning_rate"],
                                JSE=config["JSE"],
                                    train_batch_size=config["batch_size"],
                                    embed_dim= config[ "embed_dim"],
                                    transformer_width= config["transformer_width"],
                                    transformer_heads= config["transformer_heads"],
                                    transformer_layers= config["transformer_layers"])
    if Dataset is None:
        from BuildSpainDataSet import COCODataModule

        Dataset=COCODataModule(Cache_dir=dir,batch_size=config["batch_size"])
    # print("Training with config: {}".format(config))
    Dataset.batch_size=config["batch_size"]
    callbacks=[
        TQDMProgressBar(),
        EarlyStopping(monitor="imloss", mode="min",patience=10,check_finite=True,stopping_threshold=0.001),
    ]
    p=config['precision']
    if isinstance(p,str):
        p=16 if p=="bf16" else int(p)  ##needed for BEDE
    print("Launching with precision",p)
    trainer=pytorch_lightning.Trainer(
            devices="auto",
            accelerator=accelerator,
            max_epochs=40,
            #profiler="advanced",
            logger=logtool,
            strategy="dp",
            num_nodes=int(os.getenv("SLURM_NNODES",1)),
            callbacks=callbacks,
            #gradient_clip_val=0.25, Not supported for manual optimization
            fast_dev_run=False,
            precision=p
    )
    if config["batch_size"] !=1:
        
        trainer.fit(model,Dataset)
    else:
        return 0 #No need to train if batch size is 1
if __name__ == '__main__':

    from HOparser import parser
    myparser=parser()
    hyperparams = myparser.parse_args()
    config=hyperparams.__dict__
    # config={
    #     "batch_size":4, #[1,4,8,16,32,64] #V2: 13 for 8GB VRAM, 22 for 24GB VRAM (ETA 00:48:00)
    #     #                                          #v3: 19 for 10GB VRAM (ETA 1:46:00),   23 for 24GB VRAM  
    #     # in 2 dim, 19 : 23 Batchs is the difference of 168 Samples, in 6 dim its 144 Million. 
    #     "learning_rate":2e-5,   #[2e-4,1e-4,5e-5,2e-5,1e-5,4e-6]
    #     "precision":'bf16',         #[32,16,'bf16']
    #     "embed_dim": 512,
    #     "transformer_width": 512,
    #     "transformer_heads": 16,
    #     "transformer_layers": 5,
    #     "JSE":True,
    # }
    train(config)
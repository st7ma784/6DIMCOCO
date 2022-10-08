
import pytorch_lightning
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import numpy as np
from typing import Optional
from clip.model import Transformer,LayerNorm,VisionTransformer
from pytorch_lightning.callbacks import TQDMProgressBar
from deepspeed.ops.adam import FusedAdam,DeepSpeedCPUAdam
from pytorch_lightning.strategies import DeepSpeedStrategy
import deepspeed

class LightningCLIPModule(LightningModule):
    def __init__(self,
                
                learning_rate,
                useclip_en=True,
                useclip_im=True,
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
        self.lossim=torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss1=torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss2=torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss3=torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss4=torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss5=torch.nn.CrossEntropyLoss(reduction='mean')
        self.vocab_size = vocab_size
        #self.automatic_optimization=False
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()
       


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
        return x

    def forward(self, im, captions1, captions2, captions3, captions4, captions5):
        #if self.useclip_im:
        image_features=self.encode_image(im)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        caption_features1=self.encode_text(captions1)
        caption_features1 = caption_features1 / caption_features1.norm(dim=1, keepdim=True)

        caption_features2=self.encode_text(captions2)
        caption_features2 = caption_features2 / caption_features2.norm(dim=1, keepdim=True)

        caption_features3=self.encode_text(captions3)
        caption_features3 = caption_features3 / caption_features3.norm(dim=1, keepdim=True)

        caption_features4=self.encode_text(captions4)
        caption_features4 = caption_features4 / caption_features4.norm(dim=1, keepdim=True)

        caption_features5=self.encode_text(captions5)
        caption_features5 = caption_features5 / caption_features5.norm(dim=1, keepdim=True)


        # normalized features

        #each of these is B x D
        #in a square we get BxB Matrix of DxD matrices for logits
        #for 3 features we get BxBxB matrix of DxDxD matrices for logits
        logs=self.logit_scale.exp()
        imlogits=logs*torch.einsum('a...,b...,c...,d...,e...,f...->abcdef',image_features,caption_features1,caption_features2,caption_features3,caption_features4,caption_features5)
        
        # logits1= logs*torch.einsum('a...,b...,c...,d...,e...,f...->abcdef',caption_features1,caption_features2,caption_features3,caption_features4,caption_features5,image_features)
        # logits2= logs*torch.einsum('a...,b...,c...,d...,e...,f...->abcdef',caption_features2,caption_features3,caption_features4,caption_features5,image_features,caption_features1)
        # logits3= logs*torch.einsum('a...,b...,c...,d...,e...,f...->abcdef',caption_features3,caption_features4,caption_features5,image_features,caption_features1,caption_features2)
        # logits4= logs*torch.einsum('a...,b...,c...,d...,e...,f...->abcdef',caption_features4,caption_features5,image_features,caption_features1,caption_features2,caption_features3)
        # logits5= logs*torch.einsum('a...,b...,c...,d...,e...,f...->abcdef',caption_features5,image_features,caption_features1,caption_features2,caption_features3,caption_features4)
        logits1=imlogits.permute(1,2,3,4,5,0)
        logits2=imlogits.permute(2,3,4,5,0,1)
        logits3=imlogits.permute(3,4,5,0,1,2)
        logits4=imlogits.permute(4,5,0,1,2,3)
        logits5=imlogits.permute(5,0,1,2,3,4)

        return imlogits,logits1,logits2,logits3,logits4,logits5


    def training_step(self, batch, batch_idx,optimizer_idx=0):
        # access your optimizers with use_pl_optimizer=False. Default is True,
        # setting use_pl_optimizer=True will maintain plugin/precision support
        labels=torch.diag_embed(torch.arange(batch[0].shape[0],dtype=torch.long,device=self.device)-self.lossim.ignore_index)

        for i in range(3):
            labels=torch.diag_embed(labels)
        labels=labels+self.lossim.ignore_index
        #self.labels=self.labels.to(self.device)
        im,captions= batch[0],batch[1]
        #print(captions.shape)#Batchx 5 Capions x Length
        imlogits,logits1,logits2,logits3,logits4,logits5=self(im,captions[:,0],captions[:,1],captions[:,2],captions[:,3],captions[:,4])
        #print(logits1.shape ,labels.shape)
        del im,captions
        loss1 = self.loss1(logits1, labels)
        loss2 = self.loss2(logits2, labels)
        loss3 = self.loss3(logits3, labels)
        loss4 = self.loss4(logits4, labels)
        loss5 = self.loss5(logits5, labels)
        lossim = self.lossim(imlogits, labels)

        loss = lossim+loss1+loss2+loss3+loss4+loss5
        loss=loss/6
        loss = loss.mean()
        self.log('train_loss', loss, prog_bar=True,enable_graph=False,rank_zero_only=True)
        return {"loss": loss}
    def configure_optimizers(self):
        
        optimizerA = FusedAdam(
            self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        return [optimizerA]
import wandb
def wandbtrain(config=None,dir="/Data",devices="auto",accelerator="auto",Dataset=None):
    with wandb.init(project="6DIMCacheSweep",entity="st7ma784",config=config) as run:

        logtool= pytorch_lightning.loggers.WandbLogger( project="6DIMCacheSweep",entity="st7ma784",experiment=run, save_dir=dir)
        #print(logtool.__dir__())
        config=logtool.experiment.config
        print("WANDB CONFIG",config)
        train(config,dir,devices,accelerator,Dataset,logtool)
def train(config={
        "batch_size":16,
        "learning_rate":2e-4,
        "precision":16,
    },dir="/Data",devices="auto",accelerator="auto",Dataset=None,logtool=None):
    model=LightningCLIPModule(  learning_rate = config["learning_rate"],
                                    train_batch_size=config["batch_size"],
                                    adam_epsilon = 1e-8)
    if Dataset is None:
        from BuildSpainDataSet import COCODataModule

        Dataset=COCODataModule(Cache_dir=dir,batch_size=config["batch_size"])
    Dataset.batch_size=config["batch_size"]
    trainer=pytorch_lightning.Trainer(
            devices=devices,
            accelerator=accelerator,
            max_epochs=100,
            #profiler="advanced",
            logger=logtool,
            #strategy="ddp",
            strategy=DeepSpeedStrategy(
            stage=1,
            #offload_optimizer=True, 


            #cpu_checkpointing=False,  # (Optional) offload activations to CPU
            ),            #callbacks=callbacks,
            #gradient_clip_val=0.25,
            precision=config["precision"]
    )
    if config["batch_size"] !=1:
        
        trainer.fit(model,Dataset)
    else:
        return 0 #No need to train if batch size is 1
if __name__ == '__main__':
    config={
        "batch_size":19,         #[1,4,8,16,32,64] #V2: 13 for 8GB VRAM, 22 for 24GB VRAM (ETA 01:13:00 -ddp )
        #                                          #v3: 19 for 10GB VRAM (ETA 1:46:00),   23 for 24GB VRAM  (eta 3hr:25)
        #                                          #v4: ? for 10GB VRAM (ETA 1:46:00),   19 for 24GB VRAM  (eta  1:20  min)
        # in 2 dim, 19 : 23 Batchs is the difference of 168 Samples, in 6 dim its 144 Million. 
        "learning_rate":2e-5,   #[2e-4,1e-4,5e-5,2e-5,1e-5,4e-6]
        "precision":'bf16',         #[32,16,'bf16']
    }
    train(config)
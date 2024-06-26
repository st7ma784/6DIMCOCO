
import pytorch_lightning
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional
from clip.model import Transformer,LayerNorm,VisionTransformer
from pytorch_lightning.callbacks import TQDMProgressBar

class LightningCLIPModule(LightningModule):
    def __init__(self,
                
                learning_rate,
                useclip_en='v40',
                useclip_im=False,
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

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.transformer_width=transformer_width
        self.initialize_parameters()
        print("done")


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

    def calculate_loss(self, I, C1, C2, C3, C4, C5):
      
        shapes=(I.shape[0],C1.shape[0],C2.shape[0],C3.shape[0],C4.shape[0],C5.shape[0],-1)
        arr=torch.stack([I.view(I.shape[0],1,1,1,1,1,-1).expand(shapes), 
                        C1.view(1,C1.shape[0],1,1,1,1,-1).expand(shapes),
                        C2.view(1,1,C2.shape[0],1,1,1,-1).expand(shapes),
                        C3.view(1,1,1,C3.shape[0],1,1,-1).expand(shapes),
                        C4.view(1,1,1,1,C4.shape[0],1,-1).expand(shapes),
                        C5.view(1,1,1,1,1,C5.shape[0],-1).expand(shapes)],dim=0)
        arr=arr-arr.mean(dim=0,keepdim=True)
        return 1-torch.sum(torch.norm(arr,p=2,dim=-1),dim=0)
    def forward(self, im, captions1, captions2, captions3, captions4, captions5):
        #if self.useclip_im:
        image_features=self.encode_image(im)
        image_features=image_features/ torch.norm(image_features, dim=1, keepdim=True)
        caption_features1=self.encode_text(captions1)
        caption_features1=caption_features1/ torch.norm(caption_features1, dim=1, keepdim=True)
        caption_features2=self.encode_text(captions2)
        caption_features2=caption_features2/ torch.norm(caption_features2, dim=1, keepdim=True)
        caption_features3=self.encode_text(captions3)
        caption_features3=caption_features3/ torch.norm(caption_features3, dim=1, keepdim=True)
        caption_features4=self.encode_text(captions4)
        caption_features4=caption_features4/ torch.norm(caption_features4, dim=1, keepdim=True)
        caption_features5=self.encode_text(captions5)
        caption_features5=caption_features5/ torch.norm(caption_features5, dim=1, keepdim=True)

        # normalized features

        #each of these is B x D
        #in a square we get BxB Matrix of DxD matrices for logits
        #for 3 features we get BxBxB matrix of DxDxD matrices for logits
        logs=self.logit_scale.exp()
        # Combine features into a grid of BxBxBxBxBxB X Featuresx6,  5 captions, 1 image
        #features = torch.stack([image_features, caption_features1, caption_features2, caption_features3, caption_features4, caption_features5], dim=2)
        # This gets us shape (B,F,6)
        #We want to get a BxBxBxBxBxB X 6 X F matrix
        #features = features.permute(0,2,1)
        Loss=self.calculate_loss(image_features, caption_features1, caption_features2, caption_features3, caption_features4, caption_features5)*logs
        logits1=Loss.permute(1,2,3,4,5,0)
        logits2=Loss.permute(2,3,4,5,0,1)
        logits3=Loss.permute(3,4,5,0,1,2)
        logits4=Loss.permute(4,5,0,1,2,3)
        logits5=Loss.permute(5,0,1,2,3,4)

        return Loss,logits1,logits2,logits3,logits4,logits5


    def training_step(self, batch, batch_idx,optimizer_idx=0):
        labels=torch.diag_embed(torch.diag_embed(torch.diag_embed(torch.diag_embed(torch.arange(batch[0].shape[0],dtype=torch.long,device=self.device)-self.lossim.ignore_index))))

        
        labels=labels+self.lossim.ignore_index
        #self.labels=self.labels.to(self.device)
        im,captions= batch[0],batch[1]
        #print(captions.shape)#Batchx 5 Capions x Length
        imlogits,logits1,logits2,logits3,logits4,logits5=self(im,captions[:,0],captions[:,1],captions[:,2],captions[:,3],captions[:,4])
        #print(logits1.shape ,labels.shape)
        loss1 = self.loss1(logits1, labels)
        loss2 = self.loss2(logits2, labels)
        loss3 = self.loss3(logits3, labels)
        loss4 = self.loss4(logits4, labels)
        loss5 = self.loss5(logits5, labels)
        lossim = self.lossim(imlogits, labels)

        loss = lossim+loss1+loss2+loss3+loss4+loss5
        loss=loss/6
        loss = loss.mean()
        self.log('train_loss', loss, prog_bar=True,enable_graph=False, rank_zero_only=True)
        return {"loss": loss}

            
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        lr_schedulers = {"scheduler": ReduceLROnPlateau(optimizer), "monitor": "train_loss"}

        return [optimizer],[lr_schedulers]

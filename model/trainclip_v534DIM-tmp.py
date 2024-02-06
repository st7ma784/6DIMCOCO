
from functools import reduce
from operator import add
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.append("/data/6DIMCOCO")


from model.trainclip_v5335DIM import LightningCLIPModule as base 
import torch
import torch.nn as nn
from transformers import CLIPTokenizer
import numpy as np
class LightningCLIPModule(base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        if kwargs["exactlabels"]==1:
            with torch.no_grad():
                testBatch=torch.rand(self.hparams.batch_size,self.transformer_width,device=self.device)*2 -1
                #norm the batch
                # testBatch=testBatch/torch.norm(testBatch,dim=-1,keepdim=True)
                if not kwargs["normlogits"]:
                    testBatch=testBatch/torch.norm(testBatch,dim=-1,keepdim=True)
                self.label=self.calculate_loss(testBatch,testBatch,testBatch,testBatch).to(self.device,non_blocking=True)
                #convert this to probabilities in range [0,1]
                self.label=torch.nn.functional.softmax(self.label)
                self.label=torch.nan_to_num(self.label, nan=1.0)
                print("using labels: ", self.label[:2,:2,:2,:2])
        #elif add in the case where using -inf or -1 instead of zeros as below....
        else:
            self.label=torch.diag_embed(torch.diag_embed(torch.diag_embed(torch.ones(self.hparams.batch_size,dtype=torch.float,device=self.device))) )           #self.label=(self.label*2)-1 This makes loss negative! 
            print("using labelsv2: ", self.label[:2,:2,:2,:2])
        self.pruneLabels=  len(self.label.shape)>=4
        self.token_emb=nn.Parameter(self.token_embedding.weight) #V,D
        self.token_scale=nn.Parameter(torch.ones(self.token_emb.shape[1],device=self.device))
        self.label=torch.nan_to_num(self.label)
        self.EOT_embedding=self.clip.token_embedding.weight[-1]
        #check shape is [512]
        self.clip.train()
        self.transformerModel.train()
        EOT_finder=kwargs.get("EOTGrad",0)
        if EOT_finder==0:
            self.EOT_summarization=self.EOT_finder
        elif EOT_finder==1:
            self.EOT_summarization=self.EOT_finder2
        else:
            raise ValueError("EOTGrad must be 0 or 1")
        

    def encode_text(self, text):

 
        EOT_indexes=torch.argmax(text,dim=-1)# already tokenized ready to go¬
        #or decoder inputs= sot tokens?
        #or decoder inputs= pad tokens? 
        decoder_input_ids=torch.full((text.shape[0],77),self.bos_token_id,dtype=torch.long,device=self.device)


        output = self.transformerModel.model(input_ids=text,
                                             decoder_input_ids=decoder_input_ids,
                                             output_hidden_states=True)
        #output = self.transformerModel(input_ids=text,decoder_input_ids=,return_dict=True,output_hidden_states=True)
        # print(output.keys())
        encoder_output=output.encoder_last_hidden_state[torch.arange(output.encoder_last_hidden_state.shape[0]),EOT_indexes]
        #print(encoder_output.has_g)
        #from the logits, we're going to find indexes (shape [B,S]) of the maximum cosine similarity between  token embedding for EOT [1,512] for each position of [B,S,512]
        eot=self.EOT_embedding.detach().to(self.device)
        x=output.last_hidden_state
        #scale x to be in range [-1,1]
        #EOT should be size B,S shape as a one hot vector
        #print(EOT_indexes.shape)
        #print(EOT_indexes)
        x=x/torch.norm(x,dim=-1,keepdim=True)
        #x=x*self.token_scale

        x = x + self.clip.positional_embedding.type(self.dtype)
        
        x = x.permute(1, 0, 2) # NLD -> LND
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = self.EOT_summarization(x)
        #x is B,S,D . EOT is B,S
        #EOT_indexes=torch.nn.functional.gumbel_softmax(x@eot,dim=-1,hard=True)# already tokenized ready to go¬

        # x=x * EOT_indexes.unsqueeze(-1)
        # x=x.sum(dim=1)
        return x,encoder_output
    def EOT_finder(self,x):
        eot=self.EOT_embedding.detach().to(self.device)
        return x[torch.arange(x.shape[0]), torch.argmax(x@eot,dim=-1)]
    def EOT_finder2(self,x):
        eot=self.EOT_embedding.detach().to(self.device)
        x=x * torch.nn.functional.gumbel_softmax(x@eot,dim=-1,hard=True).unsqueeze(-1)
        return x.sum(dim=1)
    # @torch.jit.script
    def forward(self, im, captions1,*captions):
        image_features=self.clip.encode_image(im)
        caption_features1=self.clip.encode_text(captions1)
        features=[self.encode_text(c) for c in captions[:2]]
        caption_features=[f[0] for f in features]+[f[1] for f in features]
        if self.projection=="inv":
            image_features=image_features@ self.text_projection
        elif self.projection=="iinv":
            image_features=image_features@torch.inverse(self.text_projection)
        elif self.projection=="None":
            caption_features=[c@self.text_projection for c in caption_features]
          
        return self.calculate_loss(image_features, caption_features1,*caption_features)
        #return self.calculate_lossStock(image_features, caption_features1)[0]*self.logit_scale.exp()


    def training_step(self, batch, batch_idx):

        im,captions= batch[0],batch[1]
        #assert len(self.label.shape)>=4

        logits=self.forward(im,*[captions[:,i] for i in range(captions.shape[1])])*self.logit_scale.exp()
        # print(logits.shape)

        labels=self.label
  
        if labels.shape != logits.shape:
            # print((len(logits.shape)))
            labels=self.generate_labels((len(logits.shape),self.hparams.batch_size,self.transformer_width)).to(self.device,non_blocking=True)
            self.label=labels
            # print(labels.shape)
            # print(self.label.shape)

        #     print(labels.shape)
        # print(labels.shape)
        # self.log("first logit",logits[0,0,0,0],enable_graph=False)
        # self.log("BAD logit",logits[0,1,2,3],enable_graph=False)
        # self.log("logit scale",self.logit_scale.exp())

        # The idea is that good logits are 1s,   bad should be -1s... so if logits are coming back as ~6000....
        #  Option 1: divide down.
        #  Option 2: 1- output...
        # option 3: logarithmic functions? 
        # option 4: 1/(1+e^-x)  (sigmoid) ? this is auto gened???? Look at this when you feel like it. 

        n_dims=len(logits.shape)
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

    def on_test_epoch_start(self):
        
        super().on_test_epoch_start()
        self.bertscores = []
    
    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        
        """ Post-training evaluation.
        Args:
            batch (dict): the data batch
            batch_idx (int): the batch index
        Returns:
            dict: the outputs.
        """
        super().test_step(batch, batch_idx)
        output=self.translate(batch["CN"])
            
        outs = {}
        logits = output.logits
        predictions = self.tokenizer.batch_decode(torch.argmax(logits, dim=-1),
                                                skip_special_tokens=True)
        predictions = [pred.strip() for pred in predictions]

        references = self.tokenizer.batch_decode(batch["EN"],
                                        skip_special_tokens=True)
        references = [label.strip() for label in references]
        refs = [[label] for label in references]

        metric = self.getMetric("bertscore")
        f1 = metric.compute(predictions=predictions,
                        references=references,
                        model_type="microsoft/deberta-xlarge-mnli",
                        lang="en",
                        device="cuda",
                        batch_size=48)["f1"]
        
        self.bertscores.extend(f1)

    def test_epoch_end(self, outputs: list) -> None:
        """ Post-training evaluation.
        Args:
            outputs (list): the outputs from all batches.
        """
        super().test_epoch_end(outputs)
        BertScore = reduce(add,self.bertscores) / len(self.bertscores)
        self.log("BertScore", BertScore, prog_bar=True,enable_graph=False, rank_zero_only=True)
    def on_validation_epoch_start(self):
        pass
    def validation_step(self, batch, batch_idx):
        pass
    def on_validation_epoch_end(self):
        pass



    def configure_optimizers(self):
        parameters = list(self.transformerModel.model.parameters()) + [self.text_projection]
        optimizer = torch.optim.AdamW(
                parameters, lr=self.hparams.learning_rate, eps=10e-8,
                #weight_decay=0.1,
             #betas=(0.9, 0.95),
             )
        lr_schedulers = {"scheduler": ReduceLROnPlateau(optimizer), "monitor": "train_loss"}

        return [optimizer],[lr_schedulers]

import evaluate
import time

def getMetric(self, metricName: str) -> evaluate.EvaluationModule:
    """ Gets a metric from HuggingFace's Evaluate API.
        Tries three times because their network can get flaky when busy.
        Call me once at the start.
    Args:
        metricName (str): the name of the metric to use.
    Returns:
        EvaluationModule: the metric.
    """
    try:
        return evaluate.load(metricName)
    except Exception:
        time.sleep(60)
        try:
            return evaluate.load(metricName)
        except Exception:
            time.sleep(60)
            try:
                return evaluate.load(metricName)
            except Exception as e:
                print(f"could not access HuggingFace {metricName}")
                raise e
            
def getBSf1(metric: evaluate.EvaluationModule,
            predictions: list,
            references: list) -> float:
    # Predictions and references are lists of plaintext tokens
    f1 = metric.compute(predictions=predictions,
                    references=references,
                    model_type="microsoft/deberta-xlarge-mnli", #  Pick a BERT.
                    lang="en",
                    device="cuda",
                    batch_size=48)["f1"]
    return sum(f1) / len(f1)



if __name__ == "__main__":
    print("Testing LightningCLIPModule")


    from BuildSpainDataSet import COCODataModule
    # from BuildImagenet import ImagenetDataModule

    # TestLoader=ImagenetDataModule(
    #     data_dir="/data", 
    #     meta_dir="/data",
    #     num_imgs_per_val_class=50,
    #     image_size=224,
    #     num_workers=4, 
    #     batch_size=8, 
    #     shuffle=True,
    #     pin_memory=True,
    #     drop_last=True)
    Dataset=COCODataModule(Cache_dir="/data",annotations="/data/annotations",batch_size=10)
    # from pl_bolts.datamodules import ImagenetDataModule
    model=LightningCLIPModule( 
        batch_size=10,
        learning_rate=3e-4,
        dir="/data",
        annotations="/data/annotations",
        debug=False,
        exactlabels=1,
        logitsversion=8,
        normlogits=1,
        alpha=0.5,
        precision=16,
        projection="None"
    )
  
        
    from pytorch_lightning.strategies import DDPStrategy as DDP
    import os
    import pytorch_lightning
    from pytorch_lightning.callbacks import EarlyStopping,TQDMProgressBar
    callbacks=[
        TQDMProgressBar(),
        EarlyStopping(monitor="train_loss", mode="min",patience=10,check_finite=True,stopping_threshold=0.001),
    ]
  
    #for windows .... 
    if sys.platform == "win32":
       os.environ["PL_TORCH_DISTRIBUTED_BACKEND"]='gloo'
    print("Launching with precision",16)
    trainer=pytorch_lightning.Trainer(
            devices="auto",
            #auto_select_gpus=True,
            accelerator="auto",
            max_epochs=20,
            #profiler="advanced",
            strategy=DDP(find_unused_parameters=True),
            num_nodes=int(os.getenv("SLURM_NNODES",1)),
            callbacks=callbacks,
            gradient_clip_val=0.25,# Not supported for manual optimization
            accumulate_grad_batches=16,
            fast_dev_run=False,
            precision=16,
    )
    trainer.fit(model,Dataset)
    # trainer.test(model,TestLoader)
   
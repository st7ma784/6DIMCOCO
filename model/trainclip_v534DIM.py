
from functools import reduce
from operator import add
from model.trainclip_v5335DIM import LightningCLIPModule as base 
import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM,CLIPTokenizer
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
    def encode_text(self, text):
 
        EOT_indexes=torch.argmax(text,dim=-1)# already tokenized ready to go¬
        #or decoder inputs= sot tokens?
        #or decoder inputs= pad tokens? 
        decoder_input_ids=torch.zeros_like(text)
        decoder_input_ids[:,0]=self.bos_token_id

        output = self.transformerModel(input_ids=text,decoder_input_ids=decoder_input_ids,return_dict=True,output_hidden_states=True)
        #output = self.transformerModel(input_ids=text,decoder_input_ids=,return_dict=True,output_hidden_states=True)

        encoder_output=output["encoder_last_hidden_state"][torch.arange(output["encoder_last_hidden_state"].shape[0]),EOT_indexes]
        #shape should be [batch_size, 1, d_model]
        EOT_locations=torch.argmax(torch.argmax(output.logits,dim=-1),dim=-1) #should be [batch_size,1]
        #print("EOT locations: ",EOT_locations.shape)
        output=self.token_select(output.logits)
        #print("output shape: ",output) #B,77,V #should be 1hot encoded?
        x=output@self.token_emb
        #scale x to be in range [-1,1]
        x=x/torch.norm(x,dim=-1,keepdim=True)
        x=x*self.token_scale
        # x=x+1 
        # x=x/2
        x = x + self.clip.positional_embedding.type(self.dtype)
        
        x = x.permute(1, 0, 2) # NLD -> LND
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), EOT_locations] 
        return x,encoder_output


    # @torch.jit.script
    def forward(self, im, captions1,*captions):
        image_features=self.clip.encode_image(im)
        caption_features1=self.clip.encode_text(captions1)
        features=[self.encode_text(c) for c in captions]
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
        assert len(self.label.shape)>=4

        logits=self(im,*[captions[:,i] for i in range(captions.shape[1])])*self.logit_scale.exp()

        try:
            labels=self.label[:(im.shape[0]),:(im.shape[0]),:(im.shape[0]),:(im.shape[0])].to(self.device,non_blocking=True) 
        except:
            #labels wrong size!!?!
            labels=self.generate_labels((len(logits.shape),self.hparams.batch_size,self.transformer_width)).to(self.device,non_blocking=True)

        self.log("first logit",logits[0,0,0,0],enable_graph=False)
        self.log("BAD logit",logits[0,1,2,3],enable_graph=False)
        self.log("logit scale",self.logit_scale.exp())

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


"""
    Don't publish this bit.

"""
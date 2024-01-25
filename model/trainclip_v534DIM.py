
from functools import reduce
from operator import add
from model.trainclip_v5335DIM import LightningCLIPModule as base 
import torch
from transformers import AutoModelForMaskedLM,AutoTokenizer
import numpy as np
class LightningCLIPModule(base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformerModel=AutoModelForMaskedLM.from_pretrained("bert-base-uncased",return_dict=True)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
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
        
        self.label=torch.nan_to_num(self.label)
    def encode_text(self, text):
        
        #keep note of the output of the translation model, 
        # but also the intermediate encoder outputs, and
        #  we're going to use the same trick as CLIP does
        #  for summarizing based on the EOT token.
        #assume text is [batch_size, n_ctx]
        print("text shape",text.shape)

        EOT_indexes=torch.argmax(text,dim=-1) 
        print("EOT_indexes",EOT_indexes.shape)
        output = self.translationModel(text,return_dict=True,output_hidden_states=True)
        #take the output probabilities as a vector, 
        #print(output.keys())
        hiddenstates=output.hidden_states
        #print(hiddenstates[0].shape) #torch.Size([10, 77, 512])
        print("hiddenstates",hiddenstates[-1].shape)
        #check shape is [batch_size, n_ctx, d_model]
        #we want to select the index in n_ctx that corresponds to the EOT tokens... 
        #so we need to find the index of the EOT token in the text, and then select that index from the hidden states
        encoder_output=hiddenstates[-1][torch.arange(hiddenstates[-1].shape[0]),EOT_indexes,:]
        #shape should be [batch_size, 1, d_model]

        output=torch.nn.functional.gumbel_softmax(output.logits,hard=True,dim=-1)
        x=output@self.token_embedding.weight 
        # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] 
        return x,encoder_output


    # @torch.jit.script
    def forward(self, im, captions1, captions2, captions3, captions4, captions5):
        image_features=self.clip.encode_image(im)
        caption_features1=self.clip.encode_text(captions1)
        caption_features2,hidden_states=self.encode_text(captions2)#
      

        if self.projection=="inv":
            image_features=image_features@ self.text_projection
        elif self.projection=="iinv":
            image_features=image_features@torch.inverse(self.text_projection)
        elif self.projection=="None":
            caption_features1=caption_features1@self.text_projection
            caption_features2=caption_features2@self.text_projection#
            hidden_states=hidden_states@self.text_projection     
          
        return self.calculate_loss(image_features, caption_features1,caption_features2,hidden_states)
        #return self.calculate_lossStock(image_features, caption_features1)[0]*self.logit_scale.exp()

  
    def training_step(self, batch, batch_idx):

        im,captions= batch[0],batch[1]
        assert len(self.label.shape)>=4
        if self.pruneLabels:
            labels=self.label[:(im.shape[0]),:(captions.shape[0]),:(captions.shape[0]),:(captions.shape[0])].to(self.device,non_blocking=True)
        #labels=self.label[:(im.shape[0]),:(im.shape[0]),:(im.shape[0]),:(im.shape[0])].to(self.device,non_blocking=True) 
        else:
            labels=self.label.to(self.device,non_blocking=True)
        logits=self(im,*[captions[:,i] for i in range(captions.shape[1])])*self.logit_scale.exp()
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
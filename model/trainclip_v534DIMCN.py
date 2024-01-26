
from functools import reduce, partial
from operator import add

from model.trainclip_v5335DIM import LightningCLIPModule as base 
import torch
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer, AutoConfig, CLIPTokenizer
import numpy as np

import evaluate
import time

class LightningCLIPModule(base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config=AutoConfig.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
        config.update({"decoder_vocab_size":self.clip.vocab_size})
        
        
        self.transformerModel=AutoModelForSeq2SeqLM.from_config(config)

        self.exact_labels=kwargs["exactlabels"]
        self.label=self.generate_labels((4,self.hparams.batch_size,self.transformer_width))
        self.pruneLabels=  len(self.label.shape)>=4


        if kwargs.get("gumbel",False):
            self.token_select=partial(torch.nn.functional.gumbel_softmax,hard=True,dim=-1)
        else:
            def token_select(i):
                return torch.nn.functional.one_hot(torch.argmax(i),num_classes=self.tokenizer.vocab_size).type(self.dtype)
            self.token_select=token_select
    def generate_labels(self, inputshape):
        if self.exact_labels==1:
            with torch.no_grad():
                testBatch=torch.rand((inputshape[1],self.transformer_width),device=self.device)*2 -1
                #norm the batch

                # testBatch=testBatch/torch.norm(testBatch,dim=-1,keepdim=True)
                testBatch=testBatch/torch.norm(testBatch,dim=-1,keepdim=True)

                label=self.calculate_loss(*[testBatch for _ in range(inputshape[0])]).to(self.device,non_blocking=True)
                #convert this to probabilities in range [0,1]
                label=torch.nn.functional.softmax(self.label)
                label=torch.nan_to_num(self.label, nan=1.0)
                
        #elif add in the case where using -inf or -1 instead of zeros as below....
        else:
            label=torch.ones(self.hparams.batch_size,dtype=torch.float,device=self.device)
            for i in range(inputshape[0]):
                label=torch.diag_embed(self.label)
        return torch.nan_to_num(label, nan=0.0)
    def on_validation_epoch_start(self):
        pass
    def on_validation_epoch_end(self):
        pass
    def validation_step(self, batch, batch_idx):
        pass
    def encode_text(self, text):
        
 
        EOT_indexes=torch.argmax(text,dim=-1)# already tokenized ready to go¬
        inputs=text  
        #or decoder inputs= sot tokens?
        #or decoder inputs= pad tokens? 
        output = self.transformerModel(input_ids=text,decoder_input_ids=text,return_dict=True,output_hidden_states=True)
        #output = self.transformerModel(input_ids=text,decoder_input_ids=,return_dict=True,output_hidden_states=True)

        #output is now in ENGLISH
        #print("hiddenstates",hiddenstates[-1].shape)
        #check shape is [batch_size, n_ctx, d_model]
        #we want to select the index in n_ctx that corresponds to the EOT tokens... 
        #so we need to find the index of the EOT token in the text, and then select that index from the hidden states
        encoder_output=output["encoder_last_hidden_state"][torch.arange(output["encoder_last_hidden_state"].shape[0]),EOT_indexes,:]
        #shape should be [batch_size, 1, d_model]

        output=torch.nn.functional.gumbel_softmax(output.logits,hard=True,dim=-1)
        #output is the shape sequence_length, batch_size, vocab_sizeZH
        #we need to convert this back to batch_size, sequence_length, vocab_sizeEN
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
    def forward(self, im, *captions):
        image_features=self.clip.encode_image(im)
        #  This line is the problem, might need to grab the encoder from the translation model. 

        features=[self.encode_text(c) for c in captions]
        caption_features=[f[0] for f in features]+[f[1] for f in features]
        if self.projection=="inv":
            image_features=image_features@ self.text_projection
        elif self.projection=="iinv":
            image_features=image_features@torch.inverse(self.text_projection)
        elif self.projection=="None":
            captions=[c@self.text_projection for c in captions]
          
        return self.calculate_loss(image_features, *caption_features)
        #return self.calculate_lossStock(image_features, caption_features1)[0]*self.logit_scale.exp()

  
    def training_step(self, batch, batch_idx):

        im,captions= batch[0],batch[1]
        try:
            labels=self.label[:(im.shape[0]),:(im.shape[0]),:(im.shape[0]),:(im.shape[0])].to(self.device,non_blocking=True) 
        except:
            #labels wrong size!!?!
            labels=self.generate_labels((len(captions)+1,self.hparams.batch_size,self.transformer_width)).to(self.device,non_blocking=True)

        zeros=np.zeros(len(labels.shape))
        rang=np.arange(len(labels.shape))
        logits=self(im,*[captions[:,i] for i in range(captions.shape[1])])*self.logit_scale.exp()
        self.log("first logit",logits[zeros],enable_graph=False)
        self.log("BAD logit",logits[rang],enable_graph=False)
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
    def translate(self, text):
        return self.translationModel(text,return_dict=True,output_hidden_states=True,output_encoder_states=True)
        
    def on_test_epoch_start(self):
        # super().on_test_epoch_start()

        self.tokenizer =CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",cache_dir=self.data_dir)
        self.metric=self.getMetric("bertscore")
        self.translations = []
        self.references=[]
    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """ Post-training evaluation.
        Args:
            batch (dict): the data batch
            batch_idx (int): the batch index
        Returns:
            dict: the outputs.
        """
        # super().test_step(batch, batch_idx)
        output=self.translate(batch["zh"])
            

        logits = output.logits # [batch_size, sequence_length, vocab_size]
        self.translations.extend(torch.argmax(logits, dim=-1))
        self.references.extend(batch["en"])


    def on_test_epoch_end(self):
        translated_tokens=torch.nan_to_num(torch.cat(self.results,dim=0)).cpu().numpy()
        labels=torch.cat(self.references,dim=0).cpu().numpy()
        
        predictions = self.tokenizer.batch_decode(translated_tokens, dim=-1,
                                                skip_special_tokens=True)
        predictions = [pred.strip() for pred in predictions]

        references = self.tokenizer.batch_decode(labels,
                                        skip_special_tokens=True)
        references = [label.strip() for label in references]
        references = [[label] for label in references]
        f1 = self.metric.compute(predictions=predictions,
                    references=references,
                    model_type="microsoft/deberta-xlarge-mnli", #  Pick a BERT.
                    lang="en",
                    device=self.device,
                    batch_size=48)["f1"]
        BertScore= sum(f1) / len(f1)

        self.log("BertScore", BertScore, prog_bar=True,enable_graph=False, rank_zero_only=True)

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
                


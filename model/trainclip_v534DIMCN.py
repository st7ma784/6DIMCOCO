
from sklearn.linear_model import LogisticRegression

from model.trainclip_v5335DIM import LightningCLIPModule as base 
import torch
from transformers import MarianMTModel,MarianConfig, CLIPTokenizer
import numpy as np

# import evaluate
import time
torch.set_anomaly_enabled(True)

class LightningCLIPModule(base):
    def __init__(self, vocab_size=21129,*args, **kwargs):
        super().__init__(*args, **kwargs)
        config=MarianConfig(
            vocab_size=self.clip.vocab_size,
            pad_token_id=0,
            activation_dropout=0,
            activation_function="gelu", #"swish" 
            attention_dropout=0.0,
            classifier_dropout=0.0,
            d_model=1024,
            decoder_attention_heads=16,
            decoder_ffn_dim=4096,
            decoder_layerdrop=0.0,
            decoder_layers=3, #would be higher if I had more VRAM
            decoder_start_token_id=self.clip.vocab_size-1,
            decoder_vocab_size=self.clip.vocab_size,
            decoder_bos_token_id=self.clip.vocab_size-1,
            decoder_eos_token_id=self.clip.vocab_size,
            dropout=0.0,
            encoder_attention_heads=16,
            encoder_ffn_dim=4096,
            encoder_layerdrop=0.0,
            encoder_layers=3, #would be higher if I had more VRAM
            eos_token_id=self.clip.vocab_size,
            forced_eos_token_id=0,
            init_std=0.02,
            is_encoder_decoder=True,
            max_position_embeddings=1024,
            model_type="marian",
            num_hidden_layers=4,
            scale_embedding=False,
            share_encoder_decoder_embeddings=False,
            transformers_version="4.25.1",
            use_cache=True,
        )
        self.bos_token_id=self.clip.vocab_size-1
        self.transformerModel=MarianMTModel(config)




        self.exact_labels=kwargs["exactlabels"]
        self.label=self.generate_labels((4,self.hparams.batch_size,self.transformer_width))
        self.pruneLabels=  len(self.label.shape)>=4
        self.EOT_embedding=self.clip.token_embedding.weight[-1]

        self.model_projection=torch.nn.Parameter(torch.empty(config.d_model, self.transformer_width))
        # if kwargs.get("gumbel",False):
        #     self.token_select=partial(torch.nn.functional.gumbel_softmax,hard=True,dim=-1)
        # else:
        #     self.token_select=partial(torch.nn.functional.softmax,dim=-1)
        EOT_finder=kwargs.get("EOTGrad",0)
        if EOT_finder==0:
            self.EOT_summarization=self.EOT_finder
        elif EOT_finder==1:
            self.EOT_summarization=self.EOT_finder2
        else:
            raise ValueError("EOTGrad must be 0 or 1")
    def on_validation_epoch_start(self):
        pass
    def on_validation_epoch_end(self):
        pass
    def validation_step(self, batch, batch_idx):
        pass
    def encode_text(self, text):
 
        EOT_indexes=torch.argmax(text,dim=-1)# already tokenized ready to go¬
        #or decoder inputs= sot tokens?
        #or decoder inputs= pad tokens? 
        decoder_input_ids=torch.full((text.shape[0],77),self.bos_token_id,dtype=torch.long).to(self.transformerModel.model.device)

        output = self.transformerModel.model(input_ids=text,
                                             decoder_input_ids=decoder_input_ids,
                                             return_dict=True,
                                             output_hidden_states=True)
        #output = self.transformerModel(input_ids=text,decoder_input_ids=,return_dict=True,output_hidden_states=True)

        encoder_output=output["encoder_last_hidden_state"][torch.arange(output["encoder_last_hidden_state"].shape[0]),EOT_indexes]
        #shape should be [batch_size, 1, d_model]
        
        #from the logits, we're going to find indexes (shape [B,S]) of the maximum cosine similarity between  token embedding for EOT [1,512] for each position of [B,S,512]
        
        x=output["last_hidden_state"]
        #scale x to be in range [-1,1]
        #EOT should be size B,S shape as a one hot vector
        #print(EOT_indexes.shape)
        #print(EOT_indexes)
        x=x/torch.norm(x,dim=-1,keepdim=True)
        x=x*self.token_scale

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
    def forward(self, im, *captions):
        image_features=self.clip.encode_image(im)
        #  This line is the problem, might need to grab the encoder from the translation model. 

        features=[self.encode_text(c) for c in captions[:2]]
        caption_features=[f[0] for f in features]+[f[1] for f in features]
        if self.projection=="inv":
            image_features=image_features@ self.text_projection
        elif self.projection=="iinv":
            image_features=image_features@torch.inverse(self.text_projection)
        elif self.projection=="None":
            caption_features=[c@self.text_projection for c in caption_features]
          
        return self.calculate_loss(image_features, *caption_features)
        #return self.calculate_lossStock(image_features, caption_features1)[0]*self.logit_scale.exp()

    def on_train_epoch_start(self):
        super().on_train_epoch_start()  
        self.transformerModel.to(self.device)
    def training_step(self, batch, batch_idx):

        im,captions= batch[0],batch[1][:2]
        
        logits=self(im,*[captions[:,i] for i in range(captions.shape[1])])*self.logit_scale.exp()
        labels=self.label
  
        if labels.shape != logits.shape:
            # print((len(logits.shape)))
            labels=self.generate_labels((len(logits.shape),self.hparams.batch_size,self.transformer_width)).to(self.device,non_blocking=True)
            self.label=labels
            # print(labels.shape)
            # print(self.label.shape)

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
    def translate(self, text):
        decoder_input_ids=torch.full((text.shape[0],77),self.bos_token_id,dtype=torch.long,device=self.device)


        return self.transformerModel.model(input_ids=text,decoder_input_ids=decoder_input_ids,return_dict=True,output_hidden_states=True)
         
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
        zh=batch["zh"]

        #train the lm_head linear layer
        output=self.translate(zh)

        logits = output.last_hidden_state # [batch_size, sequence_length, vocab_size]
        self.translations.extend(logits)
        self.references.extend(batch["en"])


    def on_test_epoch_end(self):
        translated_tokens=torch.nan_to_num(torch.cat(self.translations,dim=0)).cpu().numpy()
        labels=torch.cat(self.references,dim=0).cpu().numpy()
        
        #train linear regression on the translated tokens of shape [data_size,sequence_length,D]
        #with targets being the labels of 
        LinReg=LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1, n_jobs=-1)
        LinReg.fit(translated_tokens,labels)
        self.log( "Tranlation Vocab to embedding fit",LinReg.score(translated_tokens, labels))

        #do linear regression on the translated tokens of shape [data_size,sequence_length,D]
        token_outputs=LinReg.predict(translated_tokens)
        predictions = self.tokenizer.batch_decode(token_outputs, dim=-1,
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

    def getMetric(self, metricName: str):
        """ Gets a metric from HuggingFace's Evaluate API.
            Tries three times because their network can get flaky when busy.
            Call me once at the start.
        Args:
            metricName (str): the name of the metric to use.
        Returns:
            EvaluationModule: the metric.
        """
        try:
            import evaluate
            return evaluate.load(metricName)
        except Exception:
            time.sleep(60)
            try:
                import evaluate
                return evaluate.load(metricName)
            except Exception:
                time.sleep(60)
                try:
                    import evaluate
                    return evaluate.load(metricName)
                except Exception as e:
                    print(f"could not access HuggingFace {metricName}")
                    raise e
                


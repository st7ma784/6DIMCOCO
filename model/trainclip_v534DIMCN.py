
from itertools import chain
from sklearn.linear_model import SGDClassifier
from torch.autograd import Variable
from model.trainclip_v5335DIM import LightningCLIPModule as base 
import torch
from transformers import MarianMTModel,MarianConfig, CLIPTokenizer
import numpy as np

# import evaluate
import time
torch.set_anomaly_enabled(True)

class LightningCLIPModule(base):
    def __init__(self, vocab_size=50257,*args, **kwargs):
        super().__init__(*args, **kwargs)
        config=MarianConfig(
            vocab_size=self.clip.vocab_size,
            pad_token_id=0,
            activation_dropout=0,
            activation_function="swish", #"swish" 
            attention_dropout=0.0,
            classifier_dropout=0.0,
            d_model=512,
            decoder_attention_heads=16,
            decoder_ffn_dim=2048,
            decoder_layerdrop=0.0,
            decoder_layers=self.hparams.transformer_layers, #would be higher if I had more VRAM
            decoder_start_token_id=self.clip.vocab_size-1,
            decoder_vocab_size=self.clip.vocab_size,
            decoder_bos_token_id=self.clip.vocab_size-1,
            decoder_eos_token_id=self.clip.vocab_size,
            dropout=0.0,
            encoder_attention_heads=16,
            encoder_ffn_dim=2048,
            encoder_layerdrop=0.0,
            encoder_layers=self.hparams.transformer_layers, #would be higher if I had more VRAM
            eos_token_id=self.clip.vocab_size-1,
            forced_eos_token_id=0,
            init_std=0.02,
            is_encoder_decoder=True,
            max_position_embeddings=512,
            model_type="marian",
            num_hidden_layers=self.hparams.transformer_layers,
            scale_embedding=False,
            share_encoder_decoder_embeddings=False,
            transformers_version="4.25.1",
            use_cache=True,
        )
        self.bos_token_id=self.clip.vocab_size-1
        self.transformerModel=MarianMTModel(config)

    
        self.data_dir=kwargs.get("data_dir",self.hparams.get("dir","."))

        self.exact_labels=kwargs["exactlabels"]
        self.label=self.generate_labels((4,self.hparams.batch_size,self.transformer_width))
        self.pruneLabels=  len(self.label.shape)>=4
        self.EOT_embedding=self.clip.token_embedding.weight[-1]
        self.label=torch.diag_embed(
                                torch.ones(self.hparams.batch_size,
                                            dtype=torch.float,
                                            device=self.device)
                                ).unsqueeze(2).repeat(1,1,25)
        if self.hparams.logitsversion==-1:
            self.loss=torch.nn.CrossEntropyLoss(reduction="mean")
        self.model_projection=torch.nn.Parameter(torch.empty(config.d_model, self.transformer_width))
        # if kwargs.get("gumbel",False):
        #     self.token_select=partial(torch.nn.functional.gumbel_softmax,hard=True,dim=-1)
        # else:
        #     self.token_select=partial(torch.nn.functional.softmax,dim=-1)
        # self.label=torch.ones((self.hparams.batch_size,),dtype=torch.float,device=self.device)
        # self.label=self.label.diag_embed()
        # self.label=self.label.unsqueeze(-1).repeat(1,1,25).to(self.device)
        EOT_finder=kwargs.get("gumbel",False)
        if EOT_finder:
            self.EOT_summarization=self.EOT_finder
        else:
            self.EOT_summarization=self.EOT_finder2
        #else:
        #    raise ValueError("EOTGrad must be 0 or 1")
    def on_validation_epoch_start(self):
        # self.on_test_epoch_start()
        pass
    def on_validation_epoch_end(self):
        #self.on_test_epoch_end()
        #del self.linear_layer
        #del self.test_optimizer
        #del self.token_loss
        pass
    def validation_step(self, batch, batch_idx):
        #return self.test_step(batch, batch_idx)
        pass
    def encode_text(self, text):
 
        EOT_indexes=torch.argmax(text,dim=-1)# already tokenized ready to go¬
        #or decoder inputs= sot tokens?
        #or decoder inputs= pad tokens? 
        decoder_input_ids=torch.full((text.shape[0],77),self.bos_token_id,dtype=torch.long).to(self.transformerModel.model.device)

        output = self.transformerModel.model(input_ids=text,
                                             decoder_input_ids=decoder_input_ids,
                                             output_hidden_states=True)
        #output = self.transformerModel(input_ids=text,decoder_input_ids=,return_dict=True,output_hidden_states=True)

        encoder_output=output.encoder_last_hidden_state[torch.arange(output.encoder_last_hidden_state.shape[0]),EOT_indexes]
        #print(encoder_output.has_g)
        #from the logits, we're going to find indexes (shape [B,S]) of the maximum cosine similarity between  token embedding for EOT [1,512] for each position of [B,S,512]
        # eot=self.EOT_embedding.detach().to(self.device)
        x=output.last_hidden_state
        #scale x to be in range [-1,1]
        #EOT should be size B,S shape as a one hot vector
        #print(EOT_indexes.shape)
        #print(EOT_indexes)
        x=x/torch.norm(x,dim=-1,keepdim=True)
        # x=x*self.token_scale
        #print(x.shape)
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
        #print("inp",x.shape)

        eot=self.EOT_embedding.detach().to(self.device)
        x= x[torch.arange(x.shape[0]), torch.argmax(x@eot,dim=-1)]
        #print(x.shape) ##WHY 2,512???
        return x
    def EOT_finder2(self,x):
        eot=self.EOT_embedding.detach().to(self.device)
        #print("inp",x.shape)

        x=x * torch.nn.functional.gumbel_softmax(x@eot,dim=-1,hard=True).unsqueeze(-1)
        x=x.sum(dim=1)
        #
        # print(x.shape)
        return x


    # @torch.jit.script
    def forward(self, im, *captions):
        image_features=self.clip.encode_image(im)
        #  This line is the problem, might need to grab the encoder from the translation model. 

        features=[self.encode_text(c) for c in captions[:2]]
        caption_features=chain(*features)
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

        im,captions= batch[0],batch[1]
        
        logits=self(im,*[captions[:,i] for i in range(captions.shape[1])])*self.logit_scale.exp()
        labels=self.label.to(self.device,non_blocking=True)
        # print(logits.shape)
        # print(labels.shape)
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
        bad_logit=logits[permutes[0]]
        # bad_logit=bad_logits.sum(dim=0)
        # bad_logit=bad_logit/n_dims
        # assert bad_logit.shape[0]==firstlogit.shape[0]
        self.log("first logit",firstlogit,enable_graph=False)
        self.log("BAD logit",bad_logit.mean(),enable_graph=False)
        self.log("logit scale",self.logit_scale.exp())

        # The idea is that good logits are 1s,   bad should be -1s... so if logits are coming back as ~6000....
        #  Option 1: divide down.
        #  Option 2: 1- output...
        # option 3: logarithmic functions? 
        
        if self.hparams.logitsversion==-1:
            loss=self.loss(logits,labels)
        else:
            losses = [self.loss(logits.permute(*i), labels,alpha=self.alpha) for i in permutes]
        
            loss=self.meanloss(I=[losses[0]],T=losses[1:]).mean()
      
        self.log('train_loss', loss, prog_bar=True,enable_graph=False, rank_zero_only=True)

        return {"loss": loss}
    def translate(self, text):
        decoder_input_ids=torch.full((text.shape[0],77),self.bos_token_id,dtype=torch.long,device=self.device)


        return self.transformerModel.model(input_ids=text,decoder_input_ids=decoder_input_ids,return_dict=True,output_hidden_states=True)
         
    def on_test_epoch_start(self):
        # super().on_test_epoch_start()
        torch.set_grad_enabled(True)

        self.tokenizer =CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",cache_dir=self.data_dir)
        self.metric=self.getMetric("bertscore")
        self.translations=[]
        self.references=[]
        if not hasattr(self,"linear_layer") and not hasattr(self,"test_optimizer"):
            self.linear_layer=torch.nn.Linear(512,self.vocab_size).to(self.device)
            self.linear_layer.train()
            self.test_optimizer=torch.optim.AdamW(self.linear_layer.parameters(),lr=1e-3)
            self.token_loss=torch.nn.CrossEntropyLoss(reduction="mean")
        self.transformerModel.model.eval()
        
        self.linear_layer.register_forward_hook(self.Store_logits_hook)
        self.test_losses=[]
        self.test_idx=0
    def Store_logits_hook(self, module, input,output):
        if self.test_idx %10 ==0:
            self.translations.append(torch.argmax(output,dim=-1))

    def test_step(self, batch: dict, batch_idx: int):
        """ Post-training evaluation.
        Args:
            batch (dict): the data batch
            batch_idx (int): the batch index
        Returns:
            dict: the outputs.
        """
        with torch.inference_mode(False):
            zh=batch["zh"]
            zh=torch.stack(zh,dim=1)

            labels=torch.stack(batch["en"],dim=1)
            
            assert labels.shape[1]==77
            out=self.translate(zh)
            logits=Variable(out["last_hidden_state"],requires_grad=True)
            tokens=self.linear_layer(logits) #shape B,S,V
            
            loss=self.token_loss(tokens.permute(0,2,1),labels)
            self.test_optimizer.zero_grad()
            loss.backward()

            self.test_optimizer.step()
            self.log("test_loss",loss)
            self.test_losses.append(loss.item())
            if self.test_idx %10 ==0:
                self.references.append(labels)
            
            self.test_idx+=1


    def on_test_epoch_end(self):


        #log self.losses - min, mean max range 
        losses=torch.tensor(self.test_losses)
        self.log("Min Test Loss:",torch.min(losses))
        self.log("Max Test Loss", torch.max(losses))
        self.log("Mean Test Loss", torch.mean(losses))


        #train linear regression on the translated tokens of shape [data_size,sequence_length,D]
        #with targets being the labels of 
        translated_tokens=torch.cat(self.translations,dim=0).detach()
        labels=torch.cat(self.references,dim=0).detach()
        #replace 49407 with 0
        labels[labels>=49407]=0


        translated_tokens[translated_tokens>=49407]=0
 
        # print(translated_tokens.shape) B,S
        # print(labels.shape) B,S
        predictions = self.tokenizer.batch_decode(translated_tokens.tolist())
        predictions = [pred.strip() for pred in predictions]
        #print("predictions",predictions)
        references = self.tokenizer.batch_decode(labels.tolist())
        # except Exception as e:
        #     references=[]
        #     for ref in labels:
        #         #print(ref)
        #         # #decode
        #         # for r in ref:
        #         #     print(r)
        #         #     print(self.tokenizer.decode(r))
        #         #where ref is 49407, replace with 0
        #         ref = [r if r!=49407 else 0 for r in ref]
        #         ref = [r if r!=49408 else 0 for r in ref]

        #         reference=self.tokenizer.batch_decode(ref)
        #         #remove !
        #         reference = [r.replace("!", "") for r in reference]
        #         #print(reference)
        #         references.append(reference)
                
        # #references = [label.strip() for label in references]
        # #print("references",references)
        references = [[label.replace("!", " ")] for label in references]
        predictions = [pred.replace("!", " ") for pred in predictions]
        f1 = self.metric.compute(predictions=predictions,
                    references=references,
                    model_type="microsoft/deberta-xlarge-mnli", #  Pick a BERT.
                    lang="en",
                    device=self.device,
                    batch_size=48)["f1"]
        BertScore= sum(f1) / len(f1)

        self.log("BertScore across test_batch", BertScore, prog_bar=True,enable_graph=False, rank_zero_only=True)
        self.transformerModel.train()

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
                


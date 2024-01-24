from re import T

from regex import E
from model.trainclip_v5335DIM import LightningCLIPModule as base 
import torch
from transformers import AutoModelForMaskedLM

class LightningCLIPModule(base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformerModel=AutoModelForMaskedLM.from_pretrained("bert-base-uncased",return_dict=True)
        if kwargs["exactlabels"]==1:
            with torch.no_grad():
                testBatch=torch.rand(self.hparams.batch_size,self.transformer_width,device=self.device)
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
        labels=self.label[:(im.shape[0]),:(im.shape[0]),:(im.shape[0]),:(im.shape[0])].to(self.device,non_blocking=True) 

        logits=self(im,captions[:,0],captions[:,1],captions[:,2],captions[:,3],captions[:,4])*self.logit_scale.exp()
        self.log("first logit",logits[0,0,0,0],enable_graph=False)
        self.log("BAD logit",logits[0,1,2,3],enable_graph=False)
        self.log("logit scale",self.logit_scale.exp())

        # The idea is that good logits are 1s,   bad should be -1s... so if logits are coming back as ~6000....
        #  Option 1: divide down.
        #  Option 2: 1- output...
        # option 3: logarithmic functions? 
        # option 4: 1/(1+e^-x)  (sigmoid) ? this is auto gened???? Look at this when you feel like it. 



        lossim = self.loss(logits, labels,alpha=self.alpha)
            
        
        loss1 = self.loss(logits.permute(1,2,3,0), labels,alpha=self.alpha)
        loss2 = self.loss(logits.permute(2,3,0,1), labels,alpha=self.alpha)
        loss3 = self.loss(logits.permute(3,0,1,2), labels,alpha=self.alpha)
        loss=self.meanloss(I=[lossim],T=[loss1,loss2,loss3]).mean()
      
        self.log('train_loss', loss, prog_bar=True,enable_graph=False, rank_zero_only=True)
        
        return {"loss": loss}

            


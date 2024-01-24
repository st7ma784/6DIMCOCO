from model.nargsLossCalculation import Fast_loss_Hdim
import torch
from model.trainclip_v53 import LightningCLIPModule


torch.autograd.set_detect_anomaly(True)

class BaselineLightningCLIPModule(LightningCLIPModule):
   

    def forward(self, im, captions1, captions2, captions3, captions4, captions5):
        image_features=self.encode_image(im)
        caption_features1=self.encode_text(captions1)
        caption_features2=self.encode_text(captions2)
        caption_features3=self.encode_text(captions3)
        caption_features4=self.encode_text(captions4)
        caption_features5=self.encode_text(captions5)

        [i],[c1,c2,c3,c4,c5]=self.projection(self.text_projection,im=[image_features],text=[caption_features1,caption_features2,caption_features3,caption_features4,caption_features5])
        logits=Fast_loss_Hdim(i,c1,c2,c3,c4,c5)
        return logits

    def on_train_epoch_start(self) -> None:
        if self.prune:
            for hook in self.pruneHooks:
                hook.set_up()
        #self.Lossmasks=self.Lossmasks#.to(self.device)
        # if hasattr(self,"alpha"):
        #     self.logger.log_text("mask weights",columns=[str(i) for i in self.masks.tolist()],data=[self.alpha.tolist()])
        #     self.logger.log_text("effective weights", columns=[str(i) for i in self.masks.tolist()],data=[torch.nn.functional.softmax(self.alpha/torch.norm(self.alpha,keepdim=True)).tolist()])
        
    def on_train_epoch_end(self) -> None:
        if self.prune:
            for hook in self.pruneHooks:
                hook.remove()
        if hasattr(self,"alpha"):
            self.logger.log_text("mask weights",columns=self.masks.tolist(),data=[self.alpha.tolist()])
            self.logger.log_text("effective weights", columns=self.masks.tolist(),data=[torch.nn.functional.softmax(self.alpha/torch.norm(self.alpha,keepdim=True)).tolist()])
        
    def training_step(self, batch, batch_idx):

        im,captions= batch[0],batch[1]

        logits=self(im,captions[:,0],captions[:,1],captions[:,2],captions[:,3],captions[:,4])
        labels=torch.arange(logits.shape[0],dtype=torch.long,device=self.device).unsqueeze(1).repeat(1,logits.shape[-1])

        self.log("first logit",logits[0,0].mean(),enable_graph=False)
        self.log("BAD logit",logits[1,2].mean(),enable_graph=False)
        # The idea is that good logits are 1s,   bad should be -1s... so if logits are coming back as ~6000....
        #  Option 1: divide down.
        #  Option 2: 1- output...
        # option 3: logarithmic functions? 
        #print("logits",logits.shape)
        #print("labels",labels.shape)
        loss=torch.nn.functional.cross_entropy(logits,labels)
     
    
      
        self.log('train_loss', loss, prog_bar=True,enable_graph=False, rank_zero_only=True)
        
        return {"loss": loss}

# %%
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from functools import reduce
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
class PTLModule(pl.LightningModule):
    def __init__(self,
                batch_size=16,
                learning_rate=0.001,
                logitsversion=17,
                n=6,
                normlogits=False,
                logvariance=False,
                maskLosses=0,):
        super().__init__(
           
        )
        self.save_hyperparameters()
        self.emb=nn.Embedding(1000, 32)
        self.layer1 = nn.Linear(32, 128)
        self.layer2 = nn.Linear(128, 64)
        from model.nargsLossCalculation import get_loss_fn 
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.calculate_loss=get_loss_fn(logitsversion=logitsversion,norm=normlogits,log=logvariance)
        self.n=n
        self.maskLoss=maskLosses
        self.maskloss=torch.nn.MSELoss(reduction='none')

        torch.autograd.set_detect_anomaly(True)
        #with torch.no_grad():
       
    def forward(self, x):
        
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    def setup(self, stage):
        self.train_dataset = torch.utils.data.TensorDataset(torch.randint(0, 1000, (10000,)))
        # self.labels=torch.diag_embed(torch.diag_embed(torch.diag_embed(torch.diag_embed(torch.diag_embed(torch.ones(self.batch_size,dtype=torch.float,device=self.device))))))
        # self.labels=torch.nan_to_num(self.labels)

        #B,N=self.batch_size,6
        #Views=torch.diag_embed(torch.ones(N,dtype=torch.long)*B-1)+1
        #Lossmask=torch.sum(reduce(torch.add,list(map(lambda Arr: torch.nn.functional.one_hot(torch.arange(B).view(*Arr),num_classes=B),Views.tolist()))).pow(4),dim=-1)

        #self.masks=torch.unique(torch.flatten(Lossmask,0,-1),dim=0,sorted=False)

        self.alpha=None
        from model.LossCalculation import get_loss_sum
        self.meanloss=get_loss_sum(0)
       
        from model.LossCalculation import get_loss_calc

        self.loss=get_loss_calc(reduction='sum',ver=self.maskLoss,mask=torch.ones([1],device=self.device))
    def train_dataloader(self,batch_size=32):
      
        import torch.utils.data.dataloader as dataloader
        self.labels=torch.diag_embed(torch.diag_embed(torch.diag_embed(torch.diag_embed(torch.diag_embed(torch.ones(self.batch_size,dtype=torch.float,device=self.device))))))
        self.labels=torch.nan_to_num(self.labels)


        return dataloader.DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True,num_workers=8,drop_last=True)
    def training_step(self, batch, batch_idx):
        #batch shape is n
        #print("batch",batch)
        n=6
        x=self.emb(batch[0]) # should be Bxf 
        nx=[self(x+torch.randn_like(x))]*n # should be Bxf


        
        
        # labels=self.label[:(im.shape[0]),:(im.shape[0]),:(im.shape[0]),:(im.shape[0]),:(im.shape[0]),:(im.shape[0])].to(self.device,non_blocking=True) 

        logits=self.calculate_loss(*nx).mul(torch.exp(self.logit_scale))
        #print("logits",logits.shape)
        #self.log("first logit",logits[0,0,0,0,0,0],enable_graph=False)
        #self.log("BAD logit",logits[1,2,3,4,5,0],enable_graph=False)
        # The idea is that good logits are 1s,   bad should be -1s... so if logits are coming back as ~6000....
        #  Option 1: divide down.
        #  Option 2: 1- output...
        # option 3: logarithmic functions? 
        #print("logits",logits.shape)
        #print("labels",labels.shape)
        labels=torch.diag_embed(torch.diag_embed(torch.diag_embed(torch.diag_embed(torch.diag_embed(torch.ones_like(batch[0],dtype=torch.float))))))
        labels=torch.nan_to_num(labels)
        #print("labels",labels.shape)
        lossim = self.loss(logits, labels,alpha=self.alpha)
            
        
        loss1 = self.loss(logits.permute(1,2,3,4,5,0), labels,alpha=self.alpha)
        loss2 = self.loss(logits.permute(2,3,4,5,0,1), labels,alpha=self.alpha)
        loss3 = self.loss(logits.permute(3,4,5,0,1,2), labels,alpha=self.alpha)
        loss4 = self.loss(logits.permute(4,5,0,1,2,3), labels,alpha=self.alpha)
        loss5 = self.loss(logits.permute(5,0,1,2,3,4), labels,alpha=self.alpha)
        loss=self.meanloss(I=[lossim],T=[loss1,loss2,loss3,loss4,loss5]).mean()
      
        return  {"loss":loss, "labels":batch[0], "embs":nx[0]}  

    def training_epoch_end(self, outputs):
        #concatenate outputs,
        # 
        # create linear regression model
        # 
        # fit model
        # 
        # get score
        alllabels=torch.cat([x['labels'] for x in outputs],dim=0)
        allembs=torch.cat([x['embs'] for x in outputs],dim=0)
        # print("all labels",alllabels.shape)
        # print("all embs",allembs.shape)
        from sklearn.linear_model import LogisticRegression

        reg =LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=0, n_jobs=-1)
        reg.fit(allembs.detach().cpu().numpy(), alllabels.detach().cpu().numpy())
        
        self.log("score",reg.score(allembs.detach().cpu().numpy(), alllabels.detach().cpu().numpy()),on_step=False,on_epoch=True,prog_bar=True,logger=True)
        #store score to be used in numpy plot

            
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters()], lr=self.hparams.learning_rate, eps=10e-8,
            #weight_decay=0.1,
            #betas=(0.9, 0.95),
            )
        return [optimizer]
    
#we're going to create some cool graphs, each with epochs : score for each of the 6 models and for each method. 
results={n:[ {i:{}} for i in range(17)] for n in range(2,8)}


for n in range(2,8):
    for i in range(17):
        model=PTLModule(logitsversion=i)

        trainer = Trainer(
            gpus=1,
            max_epochs=20,
            logger=TensorBoardLogger("tb_logs", name="my_model{}".format(i),version=f"{i}"),
            #callbacks=[ModelCheckpoint(monitor='meanloss',mode='min',save_top_k=3,save_last=True)],
            #fast_dev_run=True,
            #limit_train_batches=0.01,
            #limit_val_batches=0.01,
            #limit_test_batches=0.01,
            #limit_predict_batches=0.01,
            #precision=16,
            #amp_level='O2',
            #amp_backend='apex',
            #auto_scale_batch_size='binsearch',
            auto_scale_batch_size="binsearch",
            auto_lr_find=True,
            #auto_select_gpus=True,
            #check_val_every_n_epoch=1,
        )
        
        trainer.tune(model)

        trainer.fit(model)
        #set results n i to be the list of scores
        results[n][i]=model.trainer.logged_metrics

#save results
import pickle
with open("results.pkl","wb") as f:
    pickle.dump(results,f)
    



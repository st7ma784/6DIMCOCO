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
                learning_rate=0.00001,
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

       
    def forward(self, x):
        
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    def setup(self, stage):
        self.train_dataset = torch.utils.data.TensorDataset(torch.randint(0, 1000, (10000,)))
   
        self.alpha=None
        from model.LossCalculation import get_loss_sum
        self.meanloss=get_loss_sum(0)
       
        from model.LossCalculation import get_loss_calc

        self.loss=get_loss_calc(reduction='sum',ver=self.maskLoss,mask=torch.ones([1],device=self.device))
    def train_dataloader(self,batch_size=32):
      
        import torch.utils.data.dataloader as dataloader


        return dataloader.DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True,num_workers=8,drop_last=True)
    def training_step(self, batch, batch_idx):
        #batch shape is n
        #print("batch",batch)

        x=self.emb(batch[0]) # should be Bxf 
        nx=[self(x+torch.randn_like(x))]*self.n # should be Bxf


        # labels=self.label[:(im.shape[0]),:(im.shape[0]),:(im.shape[0]),:(im.shape[0]),:(im.shape[0]),:(im.shape[0])].to(self.device,non_blocking=True) 

        logits=torch.mul(torch.nan_to_num(self.calculate_loss(*nx)),self.logit_scale.exp())

        labels=torch.ones_like(batch[0],dtype=torch.float)
        while len(labels.shape)<len(logits.shape):
            labels=torch.diag_embed(labels)
        labels=torch.nan_to_num(labels)

        loss = self.loss(logits, labels,alpha=self.alpha)
        self.log("loss",loss,enable_graph=False)
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


from functools import reduce
class PTLModuleStock(PTLModule):
    def training_step(self, batch, batch_idx):
           
        x=self.emb(batch[0]) # should be Bxf 
        nx=[self(x+torch.randn_like(x))]*self.n # should be Bxf

        loss=reduce(torch.add,[self.loss(item@ x.T *self.logit_scale.exp(),torch.arange(batch[0].shape[0],device=self.device),alpha=self.alpha) for x in nx  for item in nx])
        
        self.log("loss",loss,enable_graph=False)
        return  {"loss":loss, "labels":batch[0], "embs":nx[0]}  
    
#we're going to create some cool graphs, each with epochs : score for each of the 6 models and for each method. 
results={n:{ i:{} for i in range(17)} for n in range(2,14)}

for n in range(2,14):
   
    model=PTLModule(n=n
                    ,logitsversion=5)

    trainer = Trainer(
        gpus=1,
        max_epochs=20,
        logger=TensorBoardLogger("tb_logs"),
        auto_scale_batch_size="binsearch",
        auto_lr_find=False,
        #auto_select_gpus=True,
    )
    try:

        trainer.tune(model)

        trainer.fit(model)
        #set results n i to be the list of scores
        results[n][5]=model.trainer.logged_metrics
    except Exception as e:
        print(e)
        results[n][5]=None


for n in range(2,14):
    #do benchmark first
    model=PTLModuleStock(n=n)
    trainer = Trainer(
        gpus=1,
        max_epochs=20,
        logger=TensorBoardLogger("tb_logs"),
        auto_scale_batch_size="binsearch",
        auto_lr_find=False)
    try:
        trainer.tune(model)

        trainer.fit(model)
        #set results n i to be the list of scores
        results[n]["stock"]=model.trainer.logged_metrics
    except Exception as e:
        print(e)
        results[n]["stock"]=None

#save results
import pickle
with open("base-v-.pkl","wb") as f:
    pickle.dump(results,f)
    



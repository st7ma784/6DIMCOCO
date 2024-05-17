from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from functools import partial
import clip
import matplotlib.pyplot as plt
from CKA_test import add_colorbar 
from sklearn.linear_model import LogisticRegression
from model.nargsLossCalculation import get_loss_fn 
from model.Projection import get_proj_fn
import seaborn as sns

class LightningCLIPModule(LightningModule):
    def __init__(self,
                **kwargs,
                ):

        super().__init__()
        self.save_hyperparameters()
        #problem for using this as a inheritance...
        self.clip,_=clip.load("ViT-B/32", device=self.device)
        self.clip.train()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.context_length = self.clip.context_length
        #self.linear.weight=torch.nn.Parameter(self.clip.token_embedding.weight.T)
        self.valloss=torch.nn.CrossEntropyLoss(reduction='mean')
        self.handles=[]
        self.model1_info={'Name':"SelfCLIP",}
        self.model2_info={'Name': "Stock CLIP",}
        self.naninfcount=0
        # self.loss=get_loss_calc(reduction='sum',ver=0,mask=torch.ones([1]))

        self.label=torch.diag_embed(torch.diag_embed(torch.diag_embed
                (torch.diag_embed(torch.diag_embed(torch.ones(
                    self.hparams.batch_size,dtype=torch.float,device=self.device
                    ))))))
        #self.label=(self.label*2)-1 This makes loss negative! 
        print("using labelsv2: ", self.label[:2,:2,:2,:2,:2,:2])
        self.label=torch.nan_to_num(self.label)
        self.calculate_loss=get_loss_fn(logitsversion=2,JSE=kwargs.get("JSE",0))#        from model.LossCalculation import calculate_lossStock as sl???
        from model.nargsLossCalculation import calculate_lossStock
        self.stock_loss=calculate_lossStock
        self.tfeatures=None
        self.projection=get_proj_fn("none")
        self.im_enc=self.clip.visual
        self.encoder=self.clip.transformer
        self.token_embedding=self.clip.token_embedding
        self.positional_embedding=self.clip.positional_embedding
        self.ln_final=self.clip.ln_final
        self.exact_labels=kwargs.get("exact_labels",0)
    def encode_image(self,*args,**kwargs):
        return self.im_enc(*args,**kwargs)
    def generate_labels(self, inputshape):
        if self.exact_labels==1:
            with torch.no_grad():
                testBatch=torch.rand((inputshape[1],self.transformer_width),device=self.device)*2 -1
                #norm the batch

                # testBatch=testBatch/torch.norm(testBatch,dim=-1,keepdim=True)
                testBatch=testBatch/torch.norm(testBatch,dim=-1,keepdim=True)

                label=self.calculate_loss(*[testBatch for _ in range(inputshape[0])]).to(self.device,non_blocking=True)
                #convert this to probabilities in range [0,1]
                # label=torch.nn.functional.softmax(self.label)
                label=torch.nan_to_num(label, nan=1.0)
                
        #elif add in the case where using -inf or -1 instead of zeros as below....
        else:
            label=torch.ones(self.hparams.batch_size,dtype=torch.float,device=self.device)
            for i in range(1,inputshape[0]):
                label=torch.diag_embed(label)
        return torch.nan_to_num(label, nan=0.0)
    def initialize_parameters(self):
        if self.token_embedding:
            nn.init.normal_(self.token_embedding.weight, std=0.02)
        if hasattr(self, "positional_embedding"):    

            nn.init.normal_(self.positional_embedding, std=0.01)
        if hasattr(self, "encoder"):
            proj_std = (self.encoder.width ** -0.5) * ((2 * self.encoder.layers) ** -0.5)
            attn_std = self.encoder.width ** -0.5
            fc_std = (2 * self.encoder.width) ** -0.5
            for block in self.encoder.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if hasattr(self, "encode_image"):
            if hasattr(self.encode_image, "visual"):
                for block in self.encode_image.visual.blocks:
                    nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                    nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                    nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                    nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
            #check if ender has named modules
            if hasattr(self.encode_image, "named_modules"):
                for _,layer in self.encode_image.named_modules():
                    if isinstance(layer, nn.ModuleList):
                        for block in layer:

                            nn.init.normal_(block.weight, std=1)
                            nn.init.zeros_(block.bias)
                    elif isinstance(layer, nn.Linear):
                        nn.init.normal_(layer.weight, std=1)
                        nn.init.zeros_(layer.bias)
            for _,layer in self.encoder.named_modules():
                if isinstance(layer, nn.ModuleList):
                    for block in layer:
                        nn.init.normal_(block.weight, std=1)
                        nn.init.zeros_(block.bias)
                elif isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=fc_std)
                    nn.init.zeros_(layer.bias)
        if hasattr(self, "text_projection"):
            nn.init.normal_(self.text_projection, std=self.encoder.width ** -0.5)



    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.encoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] 
        return x
    # @torch.jit.script
    def forward(self, im, *captions):
        image_features=self.encode_image(im)
        caption_features=[self.clip.encode_text(c) for c in captions]

        i,t=self.projection(self.text_projection,im=[image_features],text=caption_features)
        
        return self.calculate_loss(*[*i,*t]).mul(torch.exp(self.logit_scale))

    def training_step(self, batch, batch_idx):

        pass

    
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
            if hasattr(self.logger,"log_text"):
                self.logger.log_text("mask weights",columns=self.masks.tolist(),data=[self.alpha.tolist()])
                self.logger.log_text("effective weights", columns=self.masks.tolist(),data=[torch.nn.functional.softmax(self.alpha/torch.norm(self.alpha,keepdim=True)).tolist()])
        
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
             
    def configure_optimizers(self):
        if self.hparams.precision==8:
            from model.LionOptimizer import Lion as lion
            #use lion optimizer
            optimizer=lion(self.parameters(), lr=self.hparams.learning_rate)
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.learning_rate, eps=10e-8,
                #weight_decay=0.1,
             #betas=(0.9, 0.95),
             )
        lr_schedulers = {"scheduler": ReduceLROnPlateau(optimizer), "monitor": "train_loss"}

        return [optimizer],[lr_schedulers]

    def on_validation_epoch_start(self):
        self.log("logit scale",self.logit_scale.exp())
        self.naninfcount=0
        self.model2,_ = clip.load("ViT-B/32", device=self.device)
        self.model2.eval()
        self._insert_hooks()
        self.IMhsic_matrix0=torch.zeros([],device=self.device)
        self.IMhsic_matrix1=torch.zeros([],device=self.device)
        self.IMhsic_matrix2=torch.zeros([],device=self.device)
        self.CAPhsic_matrix0=torch.zeros([],device=self.device)
        self.CAPhsic_matrix1=torch.zeros([],device=self.device)
        self.CAPhsic_matrix2=torch.zeros([],device=self.device)
        self.eval()
        self.results=[]
        if isinstance(self.projection,str):
            self.projection_fn=get_proj_fn(self.projection)
        else:
            self.projection_fn=self.projection

        if hasattr(self,"alpha") and hasattr(self,"masks"):
            if hasattr(self.logger,"log_text"):
                self.logger.log_text("mask weights",columns=self.masks.tolist(),data=[self.alpha.tolist()])
                self.logger.log_text("effective weights", columns=self.masks.tolist(),data=[torch.nn.functional.softmax(self.alpha/torch.norm(self.alpha,keepdim=True)).tolist()])
    def validation_step(self,batch,*args):
        #do stock loss here
       
        self.model1_features = {}  #reset list of forward hooks
        self.model2_features = {}  #reset list of forward hooks
        image_features=self.encode_image(batch[0])
        #if rank 0
        self.model2.encode_image(batch[0])# to compare supervision model
        a=torch.nan_to_num(torch.stack(list(self.model1_features.values())))
        self.IMhsic_matrix0=torch.add(self.IMhsic_matrix0,torch.nan_to_num(self.batch_HSIC2(a),nan=0.0,posinf=100,neginf=-200)) 
        a=torch.nan_to_num(torch.stack(list(self.model2_features.values())))
      
        self.IMhsic_matrix2=torch.add(self.IMhsic_matrix2,torch.nan_to_num(self.batch_HSIC2(a),nan=0.0,posinf=100,neginf=-200))
        joint_HSIC=torch.nan_to_num(self.batch_HSIC3(a,torch.nan_to_num(torch.stack(list(self.model1_features.values())))), nan=0.0,posinf=1,neginf=-2)
        self.IMhsic_matrix1=torch.add(self.IMhsic_matrix1,joint_HSIC) 
        ##Now Do Text
        self.model1_features = {}  #reset list of forward hooks
        self.model2_features = {}  #reset list of forward hooks
        choice=torch.randint(0,5,(1,)).item()
        #rint("choice", choice)
        c=batch[1][:,choice]
        c=c.squeeze()

        captions=self.encode_text(c) #run through main mode
        if isinstance(captions,tuple):
            print("captions is tuple")
            print("fixing")
            captions=captions[0]
        self.model2.encode_text(c)# to compare supervision model
        a=torch.nan_to_num(torch.stack(list(self.model1_features.values())))
        self.CAPhsic_matrix0=torch.add(self.CAPhsic_matrix0,self.batch_HSIC2(a)) 
        a=torch.nan_to_num(torch.stack(list(self.model2_features.values())))
        self.CAPhsic_matrix2=torch.add(self.CAPhsic_matrix2,self.batch_HSIC2(a))
        joint_HSIC=torch.nan_to_num(self.batch_HSIC3(a,torch.nan_to_num(torch.stack(list(self.model1_features.values())))))
        self.CAPhsic_matrix1=torch.add(self.CAPhsic_matrix1,joint_HSIC) 

        image_features, captions = self.projection_fn(self.text_projection,im=[image_features],text=[captions])
        # print("self.logit scale is 14 right? ",self.logit_scale.exp())
        logitsI,logitsT=self.stock_loss([*image_features, *captions]) 
        self.log("mean validation stock logits ", logitsI.mean())
        labels=torch.arange(batch[0].shape[0],dtype=torch.long,device=self.device)
        # print("logitsI",logitsI.shape)
        # print("logitsT",logitsT.shape)
        # print("labels",labels.shape)
        lossim = self.valloss(logitsI*self.logit_scale.exp(), labels)
        loss1 = self.valloss(logitsT*self.logit_scale.exp(), labels)
        loss = lossim+loss1
        loss=loss/2
        loss = loss.mean()
        self.log('val_loss-stock', loss, prog_bar=True,enable_graph=False, rank_zero_only=True)
        self.results.append({"imfeatures":image_features[0], "tfeatures":torch.cat(captions),"classes":batch[2],"loss": loss})

        return {"loss": loss}

    def on_validation_epoch_end(self):
        imfeatures=torch.nan_to_num(torch.cat([val["imfeatures"] for val in self.results],dim=0)).cpu().numpy()
        tfeatures=torch.nan_to_num(torch.cat([val["tfeatures"] for val in self.results],dim=0)).cpu().numpy()
        #check that B is not 2 and self.epoch is >0
        print("imfeatures",imfeatures.shape)
        print("tfeatures",tfeatures.shape)#20,512
        if self.tfeatures is None and self.current_epoch>0:
            self.tfeatures=np.expand_dims(tfeatures,0) #1 ,5,B,512
        elif self.current_epoch>0 and self.tfeatures is not None:
            self.tfeatures=np.concatenate([self.tfeatures,np.expand_dims(tfeatures,0)],axis=0)
        
        #step 3, repeat for each previous epoch (as a cum sum?))
        #step 4, take the first 5 tfeatures. compare their cartesian distance and mean cosine similarity. 
        #step 5, take every 5th tfeature. compare their cartesian distance and mean cosine similarity.


        labels=torch.cat([val["classes"] for val in self.results],dim=0).cpu().numpy()
        if not hasattr(self,"Iclassifier"):
            self.Iclassifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1, n_jobs=-1)
        if not hasattr(self,"Tclassifier"):
            self.Tclassifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1, n_jobs=-1)
        
        self.Iclassifier.fit(imfeatures, labels)
        self.log( "ImProbe",self.Iclassifier.score(imfeatures, labels))
        self.Tclassifier.fit(tfeatures, labels)
        self.log( "TProbe",self.Tclassifier.score(tfeatures, labels))

        self.log('val_loss-stock', torch.stack([val["loss"] for val in self.results],dim=0).mean(), prog_bar=True,enable_graph=False, rank_zero_only=True)
        self.unfreeze()
        self.train()
        self.plot_results("IM","IMHSIC{}.jpg".format(self.current_epoch))
        self.plot_results("CAP","CAPHSIC{}.jpg".format(self.current_epoch))
        if self.logger is not None:
            if hasattr(self.logger,"log_image"):
                self.logger.log_image(key="IMHSIC{}".format(self.current_epoch), images=["IMHSIC{}.jpg".format(self.current_epoch)])        
                self.logger.log_image(key="CAPHSIC{}".format(self.current_epoch), images=["CAPHSIC{}.jpg".format(self.current_epoch)])
        for handle in self.handles:
            handle.remove()
        #print(self.naninfcount)
        del self.model2
        if self.prune:
            for hook in self.pruneHooks:
                global_entropy = hook.retrieve()
                hook.remove()        
        self.test_token_embeddings()
                # im_scores =map(lambda name, block: prune_Residual_Attention_block(block, global_entropy[name], self.args["prune_eta"]), filter(lambda name,block: isinstance(block, ResidualAttentionBlock) and name in global_entropy.keys(), self.encode_image.named_modules()[:-1]))
                # for imscoredict in im_scores:
                #     for (param_to_prune, im_score) in imscoredict.items():
                #         prune_module(param_to_prune, im_score, self.args)
                #then purun accordingly 
        #log the tokenembeddings for the text encoder, 
        del self.results
    
    def test_token_embeddings(self):
        #create int of all the tokens in vocab size
        #embed them all with self.token_embeddings
        #perform kmeans on them all, 
        #log the clusters and the tokens nearest to each centroid. 
        tokens=torch.arange(self.token_embedding.num_embeddings,device=self.device)
        embeddings=self.token_embedding(tokens)
        # kmeans = KMeans(n_clusters=40, random_state=0).fit(embeddings)
        # for i in range(10):
        #     print(kmeans.cluster_centers_[i])
        #     print(tokens[kmeans.labels_==i])
        # self.logger.log_text("token embeddings cluster centers ",str(kmeans.cluster_centers_))
        # self.logger.log_text("token embeddings tokens nearest centers",str(tokens[kmeans.labels_==i]))
        # #log the tokens closest to the mean of all embeddings.
        values,indxs=torch.sort(torch.norm(embeddings-embeddings.mean(dim=0),dim=1),)
        if hasattr(self.logger,"log_text"):
            self.logger.log_text("token embeddings center-most tokens",columns=tokens[indxs[:10]].tolist(),data=[values[:10].tolist()])
            self.logger.log_text("token embeddings furthest tokens",columns=tokens[indxs[-10:]].tolist(),data=[values[-10:].tolist()])
    def test_step(self,batch,*args):
        #do stock loss here
        image_features=self.encode_image(batch[0])
        self.results.append({"imfeatures":image_features, "classes":batch[1]})

        return {"imfeatures":image_features, "classes":batch[1]}

    def on_test_epoch_start(self):
        self.results=[]
        #were going to generate the plots here for movement in out validation features.
        #first plot will be the distribution of each entry...
        if not self.tfeatures is None:
            plot=plt.figure()
            for i in range(self.tfeatures.shape[0]):
                sns.distplot(self.tfeatures[i][1],label="Epoch {}".format(i))
            plt.legend()
            plt.title("Distribution of Validation Features for sample1")
            plt.savefig("distribution_of_validation_features.jpg")
            plt.close()
            plot=plt.figure()
            for i in range(self.tfeatures.shape[0]):
                sns.distplot(self.tfeatures[i],label="Epoch {}".format(i))
            plt.legend()
            plt.title("mean Distribution of Validation Features")
            plt.savefig("mean_distribution_of_validation_features.jpg")
            plt.close()
            deltas=np.diff(self.tfeatures,axis=0)
            plot=plt.figure()
            for i in range(deltas.shape[0]):
                sns.distplot(deltas[i][1],label="Epoch {}".format(i))
            plt.legend()
            plt.title("Distribution of Validation Feature Deltas for sample1")
            plt.savefig("distribution_of_validation_feature_deltas.jpg")
            plt.close()
            plot=plt.figure()
            for i in range(deltas.shape[0]):
                sns.distplot(deltas[i],label="Epoch {}".format(i))
            plt.legend()
            plt.title("mean Distribution of Validation Feature Deltas")
            plt.savefig("mean_distribution_of_validation_feature_deltas.jpg")

            #we're going to repeat the same process, taking the first 5 features, and every 5th feature.
            #step 1, plot the 5 vectors as 
            first_five=self.tfeatures[:,0:5] # is shape epochs x 5 x 512
            similarity_matrix=np.zeros((first_five.shape[0],first_five.shape[1],first_five.shape[1])) 
            #norm of each vector
            norms=np.linalg.norm(first_five,axis=2) # epochs x 5
            normed_first_five=np.divide(first_five,norms[:,:,None])
            similarity_matrix=np.matmul(normed_first_five,normed_first_five.transpose(0,2,1))# shape epochs x 5 x 5
            # block out the diagonal
            similarity_matrix[:,:,np.arange(similarity_matrix.shape[1]),np.arange(similarity_matrix.shape[1])]=0
            #sum last 2 dimensions and divide by 20
            similarity_matrix=np.sum(similarity_matrix,axis=(1,2))/20
            plot=plt.figure()
            #do plot of similarity matrix with epoch on x axis and similarity on y axis
            plt.plot(np.arange(similarity_matrix.shape[0]),similarity_matrix)
            plt.title("Mean Similarity of first 5 features across validation epochs")
            plt.savefig("mean_similarity_of_first_5_features_across_validation_epochs.jpg")
            plt.close()
            #now do the same for every 5th feature
            every_five=self.tfeatures[:,np.arange(0,self.tfeatures.shape[1],5)]
            similarity_matrix=np.zeros((every_five.shape[0],every_five.shape[1],every_five.shape[1]))
            norms=np.linalg.norm(every_five,axis=2)
            normed_every_five=np.divide(every_five,norms[:,:,None])
            similarity_matrix=np.matmul(normed_every_five,normed_every_five.transpose(0,2,1))
            similarity_matrix[:,:,np.arange(similarity_matrix.shape[1]),np.arange(similarity_matrix.shape[1])]=0
            similarity_matrix=np.sum(similarity_matrix,axis=(1,2))/ ((every_five.shape[1]*every_five.shape[1]) - every_five.shape[1])
            plot=plt.figure()
            plt.plot(np.arange(similarity_matrix.shape[0]),similarity_matrix)
            plt.title("Mean Similarity between each sample across validation epochs")
            plt.savefig("mean_similarity_between_each_sample_across_validation_epochs.jpg")
            plt.close()

            #log all these plots
            if self.logger is not None:
                if hasattr(self.logger,"log_image"):
                    # self.logger.log_image(key=[
                    #     "distribution_of_validation_features.jpg",
                    #     "mean_distribution_of_validation_features.jpg",
                    #     "distribution_of_validation_feature_deltas.jpg",
                    #     "mean_distribution_of_validation_feature_deltas.jpg",
                    #     "mean_similarity_of_first_5_features_across_validation_epochs.jpg",
                    #     "mean_similarity_between_each_sample_across_validation_epochs.jpg",
                    #     ], images=[
                    #     "distribution_of_validation_features.jpg",
                    #     "mean_distribution_of_validation_features.jpg",
                    #     "distribution_of_validation_feature_deltas.jpg",
                    #     "mean_distribution_of_validation_feature_deltas.jpg",
                    #     "mean_similarity_of_first_5_features_across_validation_epochs.jpg",
                    #     "mean_similarity_between_each_sample_across_validation_epochs.jpg",
                    #     ])
                    self.logger.log_image(key="distribution_of_validation_features", images=["distribution_of_validation_features.jpg"])
                    self.logger.log_image(key="mean_distribution_of_validation_features", images=["mean_distribution_of_validation_features.jpg"])
                    self.logger.log_image(key="distribution_of_validation_feature_deltas", images=["distribution_of_validation_feature_deltas.jpg"])
                    self.logger.log_image(key="mean_distribution_of_validation_feature_deltas", images=["mean_distribution_of_validation_feature_deltas.jpg"])
                    self.logger.log_image(key="mean_similarity_of_first_5_features_across_validation_epochs", images=["mean_similarity_of_first_5_features_across_validation_epochs.jpg"])
                    self.logger.log_image(key="mean_similarity_between_each_sample_across_validation_epochs", images=["mean_similarity_between_each_sample_across_validation_epochs.jpg"])
                #remove the tfeatures
                
            self.tfeatures=None

    def on_test_epoch_end(self):
        imfeatures=torch.nan_to_num(torch.cat([val["imfeatures"] for val in self.results],dim=0)).cpu().numpy()
        labels=torch.cat([val["classes"] for val in self.results],dim=0).cpu().numpy()
        if not hasattr(self,"Iclassifier"):
            self.Iclassifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1, n_jobs=-1)
   
        self.Iclassifier.fit(imfeatures, labels)
        self.log( "TopK Imagenet",self.Iclassifier.score(imfeatures, labels))
        del self.results
    
    def _log_layer(self, model: str, name: str, layer: nn.Module,inp: torch.Tensor, out: torch.Tensor):
        if isinstance(out, tuple):
            out=out[0]       
        if out.shape[0] == self.hparams.train_batch_size:
            self.__store(out,name,model,layer)
        elif out.shape[1] == self.hparams.train_batch_size:
            self.__store(out.permute(1,0,*torch.arange(len(out.shape)-2)+2),name,model,layer)

    def __store(self,out,name, model,layer):
        X = out.flatten(1)
        X= torch.nan_to_num((X @ X.t()).fill_diagonal_(0))
        if (torch.isnan(X).any() or torch.isinf(X).any()):
            self.naninfcount+=1
        if model == "model1":
            while name in self.model1_features:
                name=name+"1"
            self.model1_features[name] = X

        elif model == "model2":
            while name in self.model1_features:
                name=name+"1"
            self.model2_features[name] = X

        else:
            raise RuntimeError("Unknown model name for _log_layer.")

    def _insert_hooks(self):
        self.handles=[]
        # if layer weight is has self.hparams.train_batch_size in shape or layer.weight is None])
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model1", name)) for name, layer in self.encode_image.named_modules()]) 
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model1", name)) for name, layer in self.encoder.named_modules() ]) 
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model2", name)) for name, layer in self.model2.visual.named_modules()]) 
        self.handles.extend([layer.register_forward_hook(partial(self._log_layer, "model2", name)) for name, layer in self.model2.transformer.named_modules()])
        
  
  
    def export(self):
      
        return {
            "model1_name": "Trained",
            "model2_name": "PretrainedModel",
            "IMCKA":self.IMhsic_matrix1 / (torch.sqrt(self.IMhsic_matrix0.unsqueeze(1))*torch.sqrt(self.IMhsic_matrix2.unsqueeze(0))),
            "CAPCKA":self.CAPhsic_matrix1 / (torch.sqrt(self.CAPhsic_matrix0.unsqueeze(1))*torch.sqrt(self.CAPhsic_matrix2.unsqueeze(0))),
            "model1_layers": self.named_modules(),
            "model2_layers": self.model2.named_modules(),
        }

    def plot_results(self,
                     model_name: str,
                     save_path: str = None,
                     title: str = None):
        title =model_name+" HSIC" if title is None else model_name+title
        fig, ax = plt.subplots()
        if model_name=="IM":
            #print(self.IMhsic_matrix0) #46 #Comes out inf on val step
            #print(self.IMhsic_matrix2) # 110
            t=self.IMhsic_matrix0.unsqueeze(1)*self.IMhsic_matrix2.unsqueeze(0) #46 x 110
        #print(torch.sum(torch.abs(t)==t))
            r=torch.sqrt(torch.abs(t))
            r[torch.abs(t)==-t]=-r[torch.abs(t)==-t]
            #print("im1",self.IMhsic_matrix1)
            #print("r", r)
            hsic_matrix = torch.div(self.IMhsic_matrix1.squeeze().t(), r)
            #print("hsic",hsic_matrix)
        else:
            print(self.CAPhsic_matrix0.shape,self.CAPhsic_matrix2.shape)
            t=self.CAPhsic_matrix0.unsqueeze(1)*self.CAPhsic_matrix2.unsqueeze(0)
            r=torch.sqrt(torch.abs(t))
            r[torch.abs(t)==-t]=-r[torch.abs(t)==-t]
            #print("cap1", self.CAPhsic_matrix1.shape)
            #print("r",r.shape)
            hsic_matrix = torch.div(self.CAPhsic_matrix1.squeeze().t() , r)
        hsic_matrix=torch.nan_to_num(hsic_matrix,nan=0)
        im = ax.imshow(hsic_matrix.cpu(), origin='lower', cmap='magma')
        ax.set_xlabel(f"Layers {self.model2_info['Name']}", fontsize=15)
        ax.set_ylabel(f"Layers {self.model1_info['Name']}", fontsize=15)
        if title is not None:
            ax.set_title(f"{title}", fontsize=18)
        else:
            ax.set_title(f"{self.model1_info['Name']} vs {self.model2_info['Name']}", fontsize=18)
        add_colorbar(im)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300)



    def batch_HSIC2(self,K):
        #K is Layers x B x B
        K=torch.nan_to_num(K,nan=0.0,posinf=1.0,neginf=-1.0)
        # print("K",K)
        a=torch.sum(K,dim=-1)
        #print(" K SHAPE ",K.shape)# 0,2,3, are all problem values..
        b=torch.sum(K,dim=-2)
        a=torch.nan_to_num(a,nan=0.0,posinf=1.0,neginf=-1.0)
        b=torch.nan_to_num(b,nan=0.0,posinf=1.0,neginf=-1.0)
        c=torch.sub(torch.div(torch.pow(torch.sum(a,dim=-1),2),(K.shape[-2] - 1)),torch.sum(torch.mul(a,b),dim=1),alpha=2)
        #print(torch.sum(torch.sum(K*K.permute(0,2,1),dim=-1),dim=-1))
        c=torch.nan_to_num(c,nan=0.0,posinf=1.0,neginf=-1.0)
        # print("c",c)
        output=torch.add(torch.sum(torch.sum(torch.mul(K,K.permute(0,2,1)),dim=-1),dim=-1),torch.div(c,(K.shape[-2] - 2)))
        #print(output)# all nans 
        output=torch.nan_to_num(output,nan=0.0,posinf=1.0,neginf=-1.0)
        return output
        #check for why pos infs... 
    def batch_HSIC3(self,K,L):

        
        K=K.unsqueeze(1) # 46,1,B,B
        #convert nan to 0 and inf to 1 
        K=torch.nan_to_num(K,nan=0.0,posinf=1.0,neginf=-1.0)
        
        L=L.unsqueeze(0) # 1,46, B,B
        L=torch.nan_to_num(L,nan=0.0,posinf=1.0,neginf=-1.0)
        a=torch.sum(L/L.shape[-1],dim=-1) #1,46,10
        b=torch.sum(K/K.shape[-2],dim=-2) #46,1,10
        a=torch.nan_to_num(a,nan=0.0,posinf=1.0,neginf=-1.0)
        b=torch.nan_to_num(b,nan=0.0,posinf=1.0,neginf=-1.0)
       
        c=torch.sub(torch.mul(torch.sum(b,dim=-1),torch.sum(a,dim=-1)).div((K.shape[-2] - 1)),torch.sum(torch.mul(b,a),dim=-1),alpha=2) #[46,46]- [46,46] =[46,46]
        #print(c.shape) # expect LayerK, LayerL, 
        # print("cHSIC3",c)
        c=torch.nan_to_num(c,nan=0.0,posinf=1.0,neginf=-1.0)
        output= torch.div(torch.add(torch.sum(torch.sum(K*L,dim=-1),dim=-1),torch.div(c,(K.shape[-2] - 2))),(K.shape[-2]*(K.shape[-2] - 3)))
        #returns many pos infs 
        output=torch.nan_to_num(output,nan=0.0,posinf=1.0,neginf=-1.0)
        return output
    
    
def batch_HSIC2(K):
    #K is Layers x B x B
    a=torch.sum(K,dim=-1)
    #print(" K SHAPE ",K.shape)# 0,2,3, are all problem values..
    b=torch.sum(K,dim=-2)
    c=torch.sub(torch.pow(torch.sum(a,dim=-1),2)/(K.shape[-2] - 1),torch.sum(a*b,dim=1),alpha=2)
    #print(torch.sum(torch.sum(K*K.permute(0,2,1),dim=-1),dim=-1))
    output=torch.add(torch.sum(torch.sum(K*K.permute(0,2,1),dim=-1),dim=-1),torch.div(c,(K.shape[-2] - 2)))
    return torch.div(output,(K.shape[-2]*(K.shape[-2] - 3)))
    #check for why pos infs... 
def batch_HSIC3(K,L):
    K=K.unsqueeze(1) # 46,1,B,B
    L=L.unsqueeze(0) # 1,46, B,B
    a=torch.sum(L,dim=-1) #1,46,10
    b=torch.sum(K,dim=-2) #46,1,10
    #print(a.shape,b.shape)
    c=torch.sub(torch.mul(torch.sum(b,dim=-1),torch.sum(a,dim=-1)).div((K.shape[-2] - 1)),torch.sum(torch.mul(b,a),dim=-1),alpha=2) #[46,46]- [46,46] =[46,46]
    #print(c.shape) # expect LayerK, LayerL, 
    return torch.div(torch.add(torch.sum(torch.sum(K*L,dim=-1),dim=-1),torch.div(c,(K.shape[-2] - 2))),(K.shape[-2]*(K.shape[-2] - 3)))
    #returns many pos infs 

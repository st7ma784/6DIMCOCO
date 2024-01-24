import torch.nn as nn
import torch
import pytorch_lightning as pl
from functools import partial
import sys
sys.path.append("./APCT")
from torch.nn.utils.prune import l1_unstructured, random_structured, ln_structured, identity
from functools import partial
from random import random
from collections import defaultdict
import logging
from functools import partial
from typing import Sequence, Tuple

import torch.nn.utils.prune as pytorch_prune

import pytorch_lightning as pl



class BaseHook:
    """
    Base method for adding hooks of a network.
    """
    def __init__(self, model):
        """
        Initialization method
        :param model: The model to be added
        """
        self.model = model
        self.handles = []       # handles for registered hooks
        self.features = {}      # recorder for computed attributes

    def set_up(self):
        """
        Remove all previous hooks and register hooks for each of t
        :return:
        """
        self.remove()
        for block_name, block in self.model.named_modules():
            if type(block) in [LinearBlock, ConvBlock]:
                self.features[block_name] = {}
                self.add_block_hook(block_name, block)

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.features = {}

    def add_block_hook(self, block_name, block):
        for module_name, module in block.named_modules():
            if check_activation(module):
                self.features[block_name][module_name] = None
                handle = module.register_forward_hook(self.hook(block_name, module_name))
                self.handles.append(handle)

    def hook(self, block_name, module_name):
        def fn(layer, input_var, output_var):
            pass
        return fn


class EntropyHook(BaseHook):
    """
    Entropy hook is a forward hood that computes the neuron entropy of the network.
    """
    def __init__(self, model, Gamma, ratio=1):
        """
        Initialization method.
        :param model: Pytorch model, which should be a sequential blocks
        :param Gamma: The breakpoint for a given activation function, i.e.
                        {0} separates ReLU and PReLU into two linear regions.
                        {-0.5, 0.5} separates Sigmoid and tanH into 2 semi-constant region and 1 semi-linear region.
        """
        super().__init__(model)
        self.Gamma = Gamma
        self.num_pattern = len(Gamma) + 1
        self.ratio = ratio

    def hook(self, block_name, layer_name):
        """

        :param block_name:
        :param layer_name:
        :return:
        """
        def fn(layer, input_var, output_var):
            """
            Count the frequency of each pattern
            """
            if random() < self.ratio:
                input_var = input_var[0]
                pattern = get_pattern(input_var, self.Gamma)
                if self.features[block_name][layer_name] is None:
                    self.features[block_name][layer_name] = np.zeros((self.num_pattern,) + pattern.shape[1:])
                for i in range(1 + len(self.Gamma)):
                    self.features[block_name][layer_name][i] += (pattern == i).sum(axis=0)
        return fn

    def retrieve(self, reshape=True):
        entropy = []
        for block in self.features.values():
            for layer in block.values():
                layer = layer.reshape(self.num_pattern, -1)
                layer /= layer.sum(axis=0)
                s = np.zeros(layer.shape[1:])
                for j in range(self.num_pattern):
                    s += -layer[j] * np.log(1e-8 + layer[j])
                entropy.append(s)
        return entropy


log = logging.getLogger(__name__)

_PYTORCH_PRUNING_FUNCTIONS = {
    "ln_structured": pytorch_prune.ln_structured,
    "l1_unstructured": pytorch_prune.l1_unstructured,
    "random_structured": pytorch_prune.random_structured,
    "random_unstructured": pytorch_prune.random_unstructured,
}

_PYTORCH_PRUNING_METHOD = {
    "ln_structured": pytorch_prune.LnStructured,
    "l1_unstructured": pytorch_prune.L1Unstructured,
    "random_structured": pytorch_prune.RandomStructured,
    "random_unstructured": pytorch_prune.RandomUnstructured,
}

_PARAM_TUPLE = Tuple[nn.Module, str]
_PARAM_LIST = Sequence[_PARAM_TUPLE]
_MODULE_CONTAINERS = (pl.LightningModule, nn.Sequential, nn.ModuleList, nn.ModuleDict)

class PruneHook(EntropyHook):
    def __init__(self, model, Gamma, ratio=1, **kwargs):
        super().__init__(model, Gamma, ratio)
        self.device="cuda"
        self.args=kwargs
        self.activations =set([nn.Linear])#set([nn.LeakyReLU, nn.ReLU, nn.ELU, nn.Sigmoid, nn.GELU,QuickGELU, nn.Tanh, nn.PReLU])
        self.Gamma=torch.tensor(Gamma, dtype=torch.float32,device=self.device)
        
    def set_up(self):
        self.remove()
        self.features=defaultdict(lambda: torch.zeros((1,self.Gamma.shape[0]+1), dtype=torch.float32, device=self.device))
        self.handles.extend( [module.register_forward_hook(partial(self.hook, layer_name=module_name)) for module_name, module in self.model.named_modules() if type(module) in self.activations])
# line 62, function has no attribute named modules fix: self.model.named... -> self.named...
    def hook(self, layer, input_var, output_var, layer_name):
        #if random()>self.ratio:# here because adds random noise to the data
        input=output_var.view(output_var.shape[-1],-1)
        hist=torch.bucketize(input, self.Gamma)# returns index of gamma to each value.
        counts=torch.nn.functional.one_hot(hist, self.Gamma.shape[0]+1).sum(dim=1)

        #counts=torch.bincount(hist,minlength=self.Gamma.shape[0]+1)
        self.features[layer_name]= counts.add(self.features[layer_name])
   
    def process_layer(self,layer):
        layer = layer.reshape(self.Gamma.shape[0]+1, -1)
        layer /= layer.sum(axis=0)
        return torch.sum(-layer*torch.log(1e-8+layer),dim=0) # changed from 0 

    def process_block_entropy(self, blockdict):
        if len(blockdict)==0:
            return torch.zeros(1)
        return {k:self.process_layer(v) for k,v in blockdict.items()}

    def retrieve(self,eta=-1):
        if len(self.features.keys())==0:
            return {}
        entropy=self.process_block_entropy(self.features) 
        print("entropy",entropy.keys())

        #for block_name, block in self.model.named_modules():
        for (module_name, module) in filter(lambda item : type(item[1]) in self.activations and random()<self.ratio, self.model.named_modules()):
            im_score = compute_importance(module.weight.detach(), entropy[module_name], eta)
            prune_module(module,"weight", im_score, self.args)
            #now consider pruning the near by batch norm layers 
    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.features = {}
def prune_module(layer,name, im_score, args):
    
    cur_param = getattr(layer, name)
    num_dims = cur_param.dim()
    if args["method"] == 'LnStructured':
        #compute importance according to the filters, 
        if num_dims > 1:
            ln_structured(layer, name, args["amount"], 2, dim=0, importance_scores=im_score.cuda())
        else:
            l1_unstructured(layer, name, args["amount"], importance_scores=im_score.cuda())
    elif args["method"] == 'RandomStructured':
        #use the absolute weight value and remove it (set weight to 0) tocreate sparsity.
        random_structured(layer, name, args["amount"], dim=0)
    elif args["method"] == 'Hard':
        #use entropy as importance score not the weight value
        slc = [slice(None)] * num_dims
        tensor_to_pru = im_score[slc]

        hard_ind = tensor_to_pru[(slice(None, ),) + (0,) * (num_dims - 1)]
        num_filters = torch.sum(hard_ind < args["fc_pru_bound"]).to(torch.int)
        if num_filters == 0:
            identity(layer, name)
        elif 0 < num_filters < len(tensor_to_pru):
            if num_dims > 1:
                ln_structured(layer, name, int(num_filters), 2, dim=0, importance_scores=im_score)
            else:
                l1_unstructured(layer, name, int(num_filters), importance_scores=im_score)
       

def compute_importance(weight, channel_entropy, eta):

    if not weight.shape[0] == channel_entropy.shape[0] and channel_entropy.shape[0] == weight.t().shape[0]:
        weight = weight.t()
        print("Transposing weight")
    assert weight.shape[0] == channel_entropy.shape[0] and channel_entropy.ndim == 1   
    weight = abs(weight)

    if eta == -1:
        importance_scores = weight
    elif eta == 0:
        importance_scores = weight
    elif eta == 2:
        importance_scores = channel_entropy * weight
    elif eta == 3:
        importance_scores =1 / (torch.div(1,channel_entropy) +torch.div(1,weight))
    elif eta == 4:
        normed_entropy = (channel_entropy - channel_entropy.mean()) / channel_entropy.std()
        normed_weight = (weight - weight.mean()) / weight.std()
        importance_scores = normed_entropy * normed_weight
    elif eta == 5:
        normed_entropy = (channel_entropy - channel_entropy.mean()) / channel_entropy.std()
        normed_weight = (weight - weight.mean()) / weight.std()
        importance_scores = normed_entropy + normed_weight
    else:
        raise ValueError()

    return importance_scores

#ENTRY POINT
def prune_Residual_Attention_block(block, block_entropy, eta):
    """
    :param block: RA block to be pruned
    :param block_entropy: entropy of the block output (out_channels * H * W)
    :param eta: hyper parameter.
    :return:
    """
    if block_entropy is None:
        return {}
    print("Pruning Residual Attention Block", block_entropy)

    LTWeightsDict={name:layer.weight.detach() for name,layer in block.named_modules() if isinstance(layer,nn.Linear)}
    channel_entropy = block_entropy#.mean(tuple(range(1, num_dim)))   # averaged entropy (out_channels, )
    print("channel_entropy",channel_entropy.shape)
    #lt_im_score = compute_importance(weights, channel_entropy, eta)
    lt_importance_dict={K: compute_importance(V, channel_entropy, eta) for K,V in LTWeightsDict.items()}
    block_type = 'RABlock'

    linear_im_dict = {
        (K,"weight",block_type):V for K,V in lt_importance_dict.items()}
    
    return linear_im_dict


class myPruneCallback(pl.callbacks.ModelPruning):

    def __init__(self,Gamma,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.activations =set([nn.Linear])#set([nn.LeakyReLU, nn.ReLU, nn.ELU, nn.Sigmoid, nn.GELU,QuickGELU, nn.Tanh, nn.PReLU])
        self.Gamma=torch.tensor(Gamma, dtype=torch.float32)

    def _hook(self, layer, input_var, output_var, layer_name):
        input=output_var.view(output_var.shape[-1],-1)
        hist=torch.bucketize(input, self.Gamma)# returns index of gamma to each value.
        counts=torch.nn.functional.one_hot(hist, self.Gamma.shape[0]+1).sum(dim=1)
        self.features[layer_name]= counts.add(self.features[layer_name])

    def _process_layer(self,layer):
        layer = layer.reshape(self.Gamma.shape[0]+1, -1)
        layer /= layer.sum(axis=0)
        return torch.sum(-layer*torch.log(1e-8+layer),dim=0) # changed from 0 

    def _process_block_entropy(self, blockdict):
        if len(blockdict)==0:
            return torch.zeros(1)
        return {k:self.process_layer(v) for k,v in blockdict.items()}

    def _retrieve(self,plmodule,eta=-1):
        if len(self.features.keys())==0:
            return {}
        entropy=self._process_block_entropy(self.features) 

        #for block_name, block in self.model.named_modules():
        for (module_name, module) in filter(lambda item : type(item[1]) in self.activations and random()<self.ratio, plmodule.named_modules()):
            im_score = compute_importance(module.weight.detach(), entropy[module_name], eta)
            prune_module(module,"weight", im_score, self.args)
            #now consider pruning the near by batch norm layers 

    

    def on_train_epoch_start(self,trainer,plmodule) -> None:
        self.PruneHandles=[]
        self.features=defaultdict(lambda: torch.zeros((1,self.Gamma.shape[0]+1), dtype=torch.float32, device=self.device))
        self.PruneHandles.extend( [module.register_forward_hook(partial(self._hook, layer_name=module_name)) for module_name, module in plmodule.named_modules() if type(module) in self.activations])

    def on_train_epoch_end(self,trainer,plmodule) -> None:
        self._retrieve(plmodule)
        map(lambda hook: hook.remove(),self.PruneHandles)
    def on_validation_epoch_end(self,trainer,plmodule):
        self._retrieve(plmodule)
        map(lambda hook: hook.remove(),self.PruneHandles)



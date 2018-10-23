import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


## Basic idea is to create a model which can have variable number of hidden layers as per users need.

class qnetwork(nn.Module):
    
    def __init__(self,input_size,output_size,seed=0,hidden_node= [64,32,8]):
        super().__init__()
        self.seed = torch.manual_seed(seed)

        self.hidden_layer = nn.ModuleList([nn.Linear(input_size,hidden_node[0])])
        layer_size = zip(hidden_node[:-1],hidden_node[1:])
        self.hidden_layer.extend([nn.Linear(h1,h2) for h1,h2 in layer_size])
        
        self.output = nn.Linear(hidden_node[-1],output_size)
        
    def forward(self,x):
        for linear in self.hidden_layer:
            x = linear(x)
        return self.output(x)
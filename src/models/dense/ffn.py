import torch
from typing import Optional
from torch import nn
from torch.nn import functional as F


class SwiGLUFFN(nn.Module):
    def __init__(self,input_dim:int,hidden_dim:int):
        """
        Initializes the SwiGLUFFN module.
        
        Args:
            input_dim (int): The dimensionality of the input features.
            hidden_dim (int): The dimensionality of the hidden layer.
            
        Initializes three linear layers:
        - `w_1`: Projects input features to the hidden dimension.
        - `w_2`: Projects input features to the hidden dimension using a separate path.
        - `out`: Projects the transformed hidden representation back to the input dimension.
        """
        super().__init__()
        self.w_1=nn.Linear(input_dim,hidden_dim)
        self.w_2=nn.Linear(input_dim,hidden_dim)
        self.out=nn.Linear(hidden_dim,input_dim)
    def forward(self,x:torch.Tensor):
        """
        Computes the output of the SwiGLUFFN module.
        """
        return self.out(self.w_1(x) * F.silu(self.w_2(x)))
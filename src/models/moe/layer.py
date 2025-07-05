import torch
from torch import nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from attention import SimpleMultiHeadAttention
from .moe import SparseMOE
from utils import RMSNorm
from utils import clones,SubLayerConnection
class layer(nn.Module):
    def __init__(self,d_model:int,n_heads:int,num_experts:int,top_k:int,device,attn_eps:float,dropout:float,ffn_eps:float=1e-6):
        """
        Initialize the layer.

        Args:
            d_model (int): The dimensionality of the input and output features.
            n_heads (int): The number of attention heads.
            num_experts (int): The number of expert networks.
            top_k (int): The number of experts to be selected for each input.
            device (str): The device to use (cpu or cuda).
            attn_eps (float): The small value added to the denominator in the attention normalization for numerical stability.
            dropout (float): The dropout rate to be applied after the sublayer.
            ffn_eps (float, optional): The small value added to the denominator in the feed-forward normalization for numerical stability. Defaults to 1e-6.
        """
        super().__init__()
        self.d_model=d_model
        self.n_heads=n_heads
        self.device=device
        
        self.attention=SimpleMultiHeadAttention(dim=self.d_model,num_heads=self.n_heads,device=self.device,
                                                dropout=dropout,bias=False)
        
        self.ffn=SparseMOE(d_model=self.d_model,d_hidden=self.d_model * 2, num_experts=num_experts,top_k=top_k) # for matching total params just do d_hidden = d_model // 2 & d_hidden = d_model * 2 for matching active params
        
        self.attn_norm=RMSNorm(dim=d_model,eps=attn_eps)
        self.ffn_norm=RMSNorm(dim=d_model,eps=ffn_eps)
        
        
    
    def forward(self,x:torch.Tensor,freqs_complex:torch.Tensor,start_pos:int):
        
        """
        Computes the output of the layer.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
            freqs_complex (torch.Tensor): The complex position frequencies tensor of shape (seq_len, d_head/2).
            start_pos (int): The starting position of the sequence.

        Returns:
            tuple: (output tensor of shape (batch_size, seq_len, d_model), load_balancing_loss)
        """
        # print(x.shape)
        # print(freqs_complex.shape)

        h=x + self.attention(self.attn_norm(x),freqs_complex=freqs_complex,start_pos=start_pos)
        ffn_output,router_loss=self.ffn(self.ffn_norm(h))
        out=h+ffn_output
        
        
        return out, router_loss
        
        
        
        
        
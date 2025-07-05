import torch
import torch.nn as nn
from .config import ModelArgs
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import clones,RMSNorm,precompute_theta_pos_frequencies
from .layer import layer


class tiny_gpt(nn.Module):
    def __init__(self,args:ModelArgs):
        super(tiny_gpt, self).__init__()
        self.args=args
        self.vocab_size=args.vocab_size
        self.n_layers=args.n_layers
        self.tok_embedding=nn.Embedding(self.vocab_size,args.d_model)
        self.layers=clones(layer(d_model=args.d_model,
                                 n_heads=args.n_heads,
                                 device=args.device,
                                 attn_eps=args.attn_eps,
                                 dropout=args.attn_dropout,
                                 ffn_eps=args.ffn_eps),self.n_layers)
        self.norm=RMSNorm(args.d_model,eps=args.norm_eps)
        
        self.output=nn.Linear(in_features=args.d_model,out_features=self.vocab_size)
        
        self.freqs_complex=precompute_theta_pos_frequencies(d_head=args.d_model//args.n_heads,seq_len=args.max_seq_len,device=args.device)
    
    
    def forward(self,x:torch.Tensor,start_pos:int):
        batch_size,seq_len=x.shape
        h=self.tok_embedding(x)
        freqs_complex=self.freqs_complex[start_pos:start_pos+seq_len]
        
        for layer in self.layers:
            h = layer(h,freqs_complex=freqs_complex,start_pos=start_pos)
        
        h=self.norm(h)
        out=self.output(h).float()
        
        return out
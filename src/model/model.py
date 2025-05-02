import torch
import torch.nn as nn
from model.config import ModelArgs
from model.utils import clones,RMSNorm,precompute_theta_pos_frequencies
from model.layer import layer
class tiny_mixtral(nn.Module):
    def __init__(self,args:ModelArgs):
        super(tiny_mixtral, self).__init__()
        self.args=args
        self.vocab_size=args.vocab_size
        self.n_layers=args.n_layers
        self.tok_embedding=nn.Embedding(self.vocab_size,args.d_model)
        self.layers=clones(layer(d_model=args.d_model,d_head=args.d_head,n_heads=args.n_heads,
                                 n_kv_heads=args.n_kv_heads,window_size=args.window_size,num_experts=args.n_experts,
                                 top_k=args.top_k,device=args.device,max_seq_len=args.max_seq_len,attn_eps=args.attn_eps,
                                 dropout=args.attn_dropout,ffn_eps=args.ffn_eps),self.n_layers)
        self.norm=RMSNorm(args.d_model,eps=args.norm_eps)
        
        self.output=nn.Linear(in_features=args.d_model,out_features=self.vocab_size)
        
        self.freqs_complex=precompute_theta_pos_frequencies(d_head=args.d_head,seq_len=args.max_seq_len,device=args.device)
    
    
    def forward(self,x:torch.Tensor,start_pos:int):
        batch_size,seq_len=x.shape
        h=self.tok_embedding(x)
        freqs_complex=self.freqs_complex[start_pos:start_pos+seq_len]
        for layer in self.layers:
            h=layer(h,freqs_complex=freqs_complex,start_pos=start_pos)
        
        h=self.norm(h)
        out=self.output(h).float()
        
        return out
        

    
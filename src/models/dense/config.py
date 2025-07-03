from dataclasses import dataclass
import torch
@dataclass
class ModelArgs:
    vocab_size:int=10000
    d_model: int = 512 #embedding size
    d_head: int = 64 #head size
    n_heads:int=8 #number of heads
    n_kv_heads:int=2 #number of key-value heads
    n_layers:int=8 #number of layers
    train_epochs:int=4 #number of epochs
    batch_size:int=64 #batch size
    val_epochs:int=2 #number of validation epochs
    window_size:int=5 #window size
    seq_len:int=512 #sequence length
    max_seq_len:int=1024 #maximum sequence length
    clip:int=1 #gradient clipping
    attn_dropout:float=0.1 #attention dropout
    dropout:float=0.1 #dropout
    max_lr:float=1e-3 #maximum learning rate
    beta1:float=0.9 #beta1
    beta2:float=0.999 #beta2
    device:str='cuda' if torch.cuda.is_available() else 'cpu' 
    wandb_project:str='gpt'
    norm_eps:float=1e-6
    attn_eps:float=1e-6
    ffn_eps:float=1e-6
    

from dataclasses import dataclass
import torch
@dataclass
class ModelArgs:
    vocab_size:int = 50_256 # 50_256
    d_model: int = 768 #embedding size # 768
    d_head: int = 64 #head size
    n_heads:int = 12 #number of heads # 12
    n_kv_heads:int = 4 #number of key-value heads # 4
    n_layers:int = 5 #number of layers # 5
    train_epochs:int = 2 #number of epochs # 1-2
    batch_size:int = 64 #batch size # 512
    val_epochs:int = 1 #number of validation epochs # 1
    window_size:int = 128 #window size # 128
    seq_len:int = 512 #sequence length # 512
    max_seq_len:int = 1024 #maximum sequence length
    max_lr:float = 5e-4 #maximum learning rate
    n_experts:int = 8 #number of experts # 8
    top_k:int = 2 #top k # 2
    val_steps:int = 1000 #validation steps # TODO: val_steps ????
    # do not change
    clip:int = 1 #gradient clipping
    attn_dropout:float = 0.1 #attention dropout
    dropout:float = 0.1 #dropout
    beta1:float = 0.9 #beta1
    beta2:float = 0.999 #beta2
    device:str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    wandb_project:str = 'moe-active'
    norm_eps:float = 1e-6
    attn_eps:float = 1e-6
    ffn_eps:float = 1e-6
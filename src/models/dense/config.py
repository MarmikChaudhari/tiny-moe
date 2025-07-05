from dataclasses import dataclass
import torch
@dataclass
class ModelArgs:
    vocab_size:int=50_256 # 50_256
    d_model: int = 768 #embedding size # 768
    d_head: int = 64 #head size
    n_heads:int = 12 #number of heads # 12
    n_kv_heads:int = 4 #number of key-value heads # 4
    n_layers:int = 5 #number of layers # 5
    train_epochs:int = 1 #number of epochs # 1
    batch_size:int = 256 #batch size # 256
    val_epochs:int = 1 #number of validation epochs # 1
    window_size:int = 128 #window size # 128
    seq_len:int = 512 #sequence length # 512
    max_seq_len:int = 1024 #maximum sequence length # 1024
    max_lr:float = 1e-3 #maximum learning rate
    val_steps:int = 300 #validation steps # 250-500 depending on total_train_steps
    save_steps:int = 1000 #save steps # 1000, total_train_steps = total_toks // (batch_size * seq_len)
    # do not change
    clip:int = 1 #gradient clipping
    attn_dropout:float = 0.1 #attention dropout
    dropout:float = 0.1 #dropout
    beta1:float = 0.9 #beta1
    beta2:float = 0.999 #beta2
    device:str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    wandb_project:str = 'dense'
    norm_eps:float = 1e-6
    attn_eps:float = 1e-6
    ffn_eps:float = 1e-6    
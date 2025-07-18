from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class ModelConfig:
    """config for tiny gpt inference"""
    vocab_size:int=50_257 # 50_256
    d_model: int = 768 #embedding size # 768
    d_head: int = 64 #head size
    n_heads:int = 12 #number of heads # 12
    n_layers:int = 5 #number of layers # 5
    max_seq_len:int = 1024 #maximum sequence length # 1024
    attn_eps:float = 1e-6
    ffn_eps:float = 1e-6
    norm_eps:float = 1e-6
    attn_dropout:float = 0.0 #attention dropout (disabled for inference)
    device:str = 'cuda' if torch.cuda.is_available() else 'cpu' 


@dataclass
class ModelArgs:
    vocab_size:int=50_256 # 50_256
    d_model: int = 768 #embedding size # 768
    d_head: int = 64 #head size
    n_heads:int = 12 #number of heads # 12
    n_kv_heads:int = 8 #number of key-value heads # 8
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


class TinyGPTConfig(PretrainedConfig):
    model_type = "tiny_gpt"
    def __init__(self,
                 vocab_size = ModelConfig.vocab_size,
                 d_model = ModelConfig.d_model,
                 d_head = ModelConfig.d_head,
                 n_heads = ModelConfig.n_heads,
                 n_layers = ModelConfig.n_layers,
                 max_seq_len = ModelConfig.max_seq_len,
                 norm_eps = ModelConfig.norm_eps,
                 attn_eps = ModelConfig.attn_eps,
                 ffn_eps = ModelConfig.ffn_eps,
                 attn_dropout = ModelConfig.attn_dropout,
                 device = ModelConfig.device,
                 **kwargs
                 ):
        super().__init__(top_k=None,**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.norm_eps = norm_eps
        self.attn_eps = attn_eps
        self.ffn_eps = ffn_eps
        self.attn_dropout = attn_dropout
        self.device = device






class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-6):
        """
        Initializes the RMSNorm module.

        Args:
            dim (int): The dimensionality of the input feature space.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        """
        super().__init__()
        self.eps=eps
        self.w=nn.Parameter(torch.ones(dim))
    
    def norm(self,x:torch.Tensor):
        """
        Computes the root mean square normalization of the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        return x * torch.rsqrt(torch.mean(x**2,-1, keepdim=True) + self.eps)
    def forward(self,x:torch.Tensor):
        """
        Forward pass of the RMSNorm module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        return self.w * self.norm(x.float()).type_as(x)




#----Rotary Embeddings---

def precompute_theta_pos_frequencies(d_head:int,seq_len:int,device:str,theta:float=10000.0):
    """
    Precomputes the position frequencies for Rotary Position Embeddings.

    Args:
        d_head (int): The number of dimensions in the attention head.
        seq_len (int): The sequence length of the input sequence.
        device (str): The device on which to create the tensor.
        theta (float, optional): The base for the exponential decay. Defaults to 10000.0.

    Returns:
        torch.Tensor: A tensor of shape (seq_len, d_head/2) containing the complex position frequencies.
    """
    assert d_head%2==0,"d_head must be even"
    #theta_i=1000^-2(i-1)/d_head for i [1,2...d_head/2]
    theta_nr=torch.arange(0,d_head,2,device=device)
    theta=1.0/(theta**(theta_nr/d_head)).to(device)
    
    m=torch.arange(seq_len,device=device)
    m_theta=torch.outer(m,theta).float()
    freq_complex=torch.polar(torch.ones_like(m_theta),m_theta)
    
    return freq_complex #(seq_len,d_head/2)


def apply_rotary_embeddings(x:torch.Tensor,freq_complex:torch.Tensor,device:str):
    """
    Applies Rotary Position Embeddings to the input tensor.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_head).
        freq_complex (torch.Tensor): The complex position frequencies tensor of shape (seq_len, d_head/2).

    Returns:
        torch.Tensor: The tensor after applying Rotary Position Embeddings.
    """
    x_complex=torch.view_as_complex(x.float().reshape(*x.shape[:-1],-1,2)) #N,seq_len,h,head_dim/2,2 
    
    freq_complex=freq_complex.unsqueeze(0).unsqueeze(2) # 1,seq_len,1,head_dim/2
    
    x_rotated=x_complex * freq_complex #(N,seq_len,h,head_dim/2)
    x_out=torch.view_as_real(x_rotated) #(N,seq_len,h,head_dim/2,2)
    x_out=x_out.reshape(*x.shape)
    
    return x_out.type_as(x).to(device)



class SubLayerConnection(nn.Module):
    def __init__(self,size,dropout):
        """
        Initializes the SubLayerConnection module.

        Args:
            size (int): The size of the input for the layer normalization.
            dropout (float): The dropout rate to be applied after the sublayer.
        """
        super(SubLayerConnection,self).__init__()
        self.norm=nn.LayerNorm(size)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,x,sublayer):
        """
        Computes the output of the SubLayerConnection module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            sublayer (nn.Module): The sublayer module to be applied to the input tensor.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, d_model).
        """

        return x + self.dropout(sublayer(self.norm(x)))
    

def clones(module,N):
    """
    Creates a list of N copies of the given nn.Module.

    Args:
        nn.Module: The nn.Module to be cloned.
        N (int): The number of copies to be made.

    Returns:
        nn.ModuleList: A list of N identical nn.Module objects.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])




class SimpleMultiHeadAttention(nn.Module):
    """Simple multi-head attention without GQA, sliding window, or KV cache"""
    
    def __init__(self, dim: int, num_heads: int, device, dropout: float = 0.0, bias: bool = False):
        """
        Initialize the SimpleMultiHeadAttention module.

        Args:
            dim (int): The dimensionality of the input and output features.
            num_heads (int): The number of attention heads.
            device: The device to use (cpu or cuda).
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            bias (bool, optional): Whether to use bias in linear layers. Defaults to False.
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.device = device
        self.dropout = dropout
        
        # Combined projection for queries, keys, and values
        self.c_attn = nn.Linear(dim, 3 * dim, bias=bias)
        # Output projection
        self.c_proj = nn.Linear(dim, dim, bias=bias)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Use flash attention if available
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor = None, start_pos: int = 0):
        """
        Compute multi-head attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_complex (torch.Tensor, optional): Complex position frequencies for RoPE. Defaults to None.
            start_pos (int, optional): Starting position (unused in simple attention). Defaults to 0.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        batch_size, seq_len, _ = x.shape
        
        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.dim, dim=2)
        
        # Reshape and transpose for multi-head attention
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings if provided
        if freqs_complex is not None:
            # Note: apply_rotary_embeddings expects (batch, seq_len, num_heads, head_dim)
            q_rotary = q.transpose(1, 2)  # (batch_size, seq_len, num_heads, head_dim)
            k_rotary = k.transpose(1, 2)  # (batch_size, seq_len, num_heads, head_dim)
            
            q_rotary = apply_rotary_embeddings(q_rotary, freqs_complex, device=self.device)
            k_rotary = apply_rotary_embeddings(k_rotary, freq_complex=freqs_complex, device=self.device)
            
            q = q_rotary.transpose(1, 2)  # Back to (batch_size, num_heads, seq_len, head_dim)
            k = k_rotary.transpose(1, 2)  # Back to (batch_size, num_heads, seq_len, head_dim)
        
        # Compute attention
        if self.flash:
            # Use flash attention for efficiency
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.dropout if self.training else 0, 
                is_causal=True
            )
        else:
            # Manual implementation of attention
            # Compute attention scores
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Apply causal mask
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
            causal_mask = causal_mask.view(1, 1, seq_len, seq_len)
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
            
            # Apply softmax
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            
            # Apply attention to values
            y = torch.matmul(attn_weights, v)
        
        # Reshape back to (batch_size, seq_len, dim)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y

    def reset_cache(self):
        """Reset cache (no-op for simple attention)"""
        pass




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



class layer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, device, attn_eps: float, dropout: float, ffn_eps: float = 1e-6):
        """
        Initialize the layer.

        Args:
            d_model (int): The dimensionality of the input and output features.
            n_heads (int): The number of attention heads.
            device (str): The device to use (cpu or cuda).
            attn_eps (float): The small value added to the denominator in the attention normalization for numerical stability.
            dropout (float): The dropout rate to be applied after the sublayer.
            ffn_eps (float, optional): The small value added to the denominator in the feed-forward normalization for numerical stability. Defaults to 1e-6.
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.device = device
        
        self.attention = SimpleMultiHeadAttention(
            dim=self.d_model, 
            num_heads=self.n_heads, 
            device=self.device,
            dropout=dropout,
            bias=False
        )
        
        # Use 4*d_model as hidden dimension, following standard transformer practice
        self.ffn = SwiGLUFFN(input_dim=self.d_model, hidden_dim = 4 * self.d_model)
        
        self.attn_norm = RMSNorm(dim=d_model, eps=attn_eps)
        self.ffn_norm = RMSNorm(dim=d_model, eps=ffn_eps)
        
    def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor, start_pos: int):
        """
        Computes the output of the layer.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
            freqs_complex (torch.Tensor): The complex position frequencies tensor of shape (seq_len, d_head/2).
            start_pos (int): The starting position of the sequence.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Attention block with residual connection
        h = x + self.attention(self.attn_norm(x), freqs_complex=freqs_complex, start_pos=start_pos)
        
        # FFN block with residual connection
        # SwiGLUFFN returns only the output tensor, no router loss for dense model
        ffn_output = self.ffn(self.ffn_norm(h))
        out = h + ffn_output

        
        return out




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




class TinyGPTForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = TinyGPTConfig
    base_model_prefix = "gpt_model"

    def __init__(self, config):
        super().__init__(config)
        args = ModelConfig(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            d_head=config.d_head,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            max_seq_len=config.max_seq_len,
            norm_eps=config.norm_eps,
            attn_eps=config.attn_eps,
            ffn_eps=config.ffn_eps,
            attn_dropout=config.attn_dropout,
            device=config.device,
        )
        self.model = tiny_gpt(args=args)
        self.config = config
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):

        outputs = self.model(input_ids, start_pos=0)

        return CausalLMOutputWithPast(
            loss=None,
            logits=outputs,
            attentions=None,
        )
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": input_ids,
        }
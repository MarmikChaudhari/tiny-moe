import torch
from torch import nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from attention import AttentionWithKVCache
from .ffn import SwiGLUFFN
from utils import RMSNorm
from utils import clones,SubLayerConnection

class layer(nn.Module):
    def __init__(self, d_model: int, d_head: int, n_heads: int, n_kv_heads: int, window_size: int, device, max_seq_len: int, attn_eps: float, dropout: float, ffn_eps: float = 1e-6):
        """
        Initialize the layer.

        Args:
            d_model (int): The dimensionality of the input and output features.
            d_head (int): The dimensionality of the hidden state in the attention mechanism.
            n_heads (int): The number of attention heads.
            n_kv_heads (int): The number of attention heads for the key-value projection.
            window_size (int): The size of the window for rolling buffer KV cache.
            device (str): The device to use (cpu or cuda).
            max_seq_len (int): The maximum sequence length for initialization of KV cache.
            attn_eps (float): The small value added to the denominator in the attention normalization for numerical stability.
            dropout (float): The dropout rate to be applied after the sublayer.
            ffn_eps (float, optional): The small value added to the denominator in the feed-forward normalization for numerical stability. Defaults to 1e-6.
        """
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.window_size = window_size
        self.device = device
        self.n_kv_heads = n_kv_heads
        
        self.attention = AttentionWithKVCache(
            dim=self.d_model, 
            num_heads=self.n_heads, 
            window_size=self.window_size,
            device=self.device, 
            max_seq_len=max_seq_len, 
            num_kv_heads=self.n_kv_heads
        )
        
        # Use 4*d_model as hidden dimension, following standard transformer practice
        self.ffn = SwiGLUFFN(input_dim=self.d_model, hidden_dim=4 * self.d_model)
        
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
            tuple: (output tensor of shape (batch_size, seq_len, d_model), load_balancing_loss)
        """
        # Attention block with residual connection
        h = x + self.attention(self.attn_norm(x), freqs_complex=freqs_complex, start_pos=start_pos)
        
        # FFN block with residual connection
        # SwiGLUFFN returns only the output tensor, no router loss for dense model
        ffn_output = self.ffn(self.ffn_norm(h))
        out = h + ffn_output

        
        return out
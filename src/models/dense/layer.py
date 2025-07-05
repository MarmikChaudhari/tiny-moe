import torch
from torch import nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from attention import SimpleMultiHeadAttention
from .ffn import SwiGLUFFN
from utils import RMSNorm
from utils import clones,SubLayerConnection

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
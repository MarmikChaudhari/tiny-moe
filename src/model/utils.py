#rmsnorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
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
    
    print('x_complex',x_complex.shape)
    freq_complex=freq_complex.unsqueeze(0).unsqueeze(2) # 1,seq_len,1,head_dim/2
    print('freq_',freq_complex.shape)
    
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
from transformers import PretrainedConfig
from src.models.moe.config import ModelConfig

class TinyMixtralConfig(PretrainedConfig):
    model_type = "tiny_mixtral_5l_active"
    def __init__(self,
                 vocab_size = ModelConfig.vocab_size,
                 d_model = ModelConfig.d_model,
                 d_head = ModelConfig.d_head,
                 n_heads = ModelConfig.n_heads,
                 n_layers = ModelConfig.n_layers,
                 max_seq_len = ModelConfig.max_seq_len,
                 n_experts = ModelConfig.n_experts,
                 top_k_experts = ModelConfig.top_k,
                 norm_eps = ModelConfig.norm_eps,
                 attn_eps = ModelConfig.attn_eps,
                 ffn_eps = ModelConfig.ffn_eps,
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
        self.n_experts = n_experts
        self.top_k_experts = top_k_experts
        self.norm_eps = norm_eps
        self.attn_eps = attn_eps
        self.ffn_eps = ffn_eps
        self.device = device
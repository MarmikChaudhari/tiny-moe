from transformers import PreTrainedModel, GenerationMixin
from configuration_tiny_gpt import TinyGPTConfig
from src.models.dense.config import ModelConfig
from src.models.dense.model import tiny_gpt
from transformers.modeling_outputs import CausalLMOutputWithPast


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
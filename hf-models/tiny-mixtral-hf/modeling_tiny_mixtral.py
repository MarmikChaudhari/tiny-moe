from transformers import PreTrainedModel, GenerationMixin
from configuration_tiny_mixtral import TinyMixtralConfig
from src.models.moe.config import ModelConfig
from src.models.moe.model import tiny_mixtral
from transformers.modeling_outputs import MoECausalLMOutputWithPast


class TinyMixtralForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = TinyMixtralConfig
    base_model_prefix = "moe_model"

    def __init__(self, config):
        super().__init__(config)
        args = ModelConfig(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            d_head=config.d_head,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            max_seq_len=config.max_seq_len,
            n_experts=config.n_experts,
            top_k=config.top_k_experts,
            norm_eps=config.norm_eps,
            attn_eps=config.attn_eps,
            ffn_eps=config.ffn_eps,
            device=config.device,
        )
        self.model = tiny_mixtral(args=args)
        self.config = config
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):

        outputs, load_balancing_loss = self.model(input_ids, start_pos=0)

        return MoECausalLMOutputWithPast(
            loss=None,
            logits=outputs,
            aux_loss=load_balancing_loss,
            attentions=None,
        )
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": input_ids,
        }
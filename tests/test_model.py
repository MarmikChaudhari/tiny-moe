import torch
from model.model import tiny_mixtral
def test_tiny_mixtral_training():
    from model.config import ModelArgs

    args = ModelArgs(
        vocab_size=100,
        d_model=32,
        d_head=8,
        n_heads=4,
        n_kv_heads=2,
        window_size=5,
        n_experts=4,
        top_k=2,
        device="cpu",
        max_seq_len=16,
        seq_len=5,
        n_layers=2,
        attn_eps=1e-5,
        attn_dropout=0.1,
        ffn_eps=1e-5,
        norm_eps=1e-5
    )

    model = tiny_mixtral(args)
    model.train()

    x = torch.randint(0, args.vocab_size, (3, 5))  # (batch_size=3, seq_len=5)
    output = model(x, start_pos=0)

    assert output.shape == (3, 5, args.vocab_size)
    assert output.dtype == torch.float32
    print("✅ Training forward pass passed.")
    
    

def test_tiny_mixtral_inference():
    from model.config import ModelArgs

    args = ModelArgs(
        vocab_size=100,
        d_model=32,
        d_head=8,
        n_heads=4,
        n_kv_heads=2,
        window_size=5,
        n_experts=4,
        top_k=2,
        device="cpu",
        max_seq_len=16,
        seq_len=1,
        n_layers=2,
        attn_eps=1e-5,
        attn_dropout=0.0,  # Turn off dropout for inference
        ffn_eps=1e-5,
        norm_eps=1e-5
    )

    model = tiny_mixtral(args)
    model.eval()

    input_ids = torch.randint(0, args.vocab_size, (1, 1))  # (batch_size=1, seq_len=1)
    for t in range(5):  # simulate autoregressive generation
        out = model(input_ids, start_pos=t)
        assert out.shape == (1, 1, args.vocab_size)
        assert out.dtype == torch.float32

    print("✅ Inference step-by-step pass passed.")
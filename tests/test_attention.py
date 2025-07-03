import torch
from src.attention import AttentionWithKVCache
from src.utils import precompute_theta_pos_frequencies

def test_attention_output_shape():
    model = AttentionWithKVCache(dim=32, num_heads=8, window_size=5,device="cpu")
    d_head=8
    # freq = precompute_theta_pos_frequencies(8, 10, "cpu")
    # x = torch.randn(2, 10, 64)  # (batch, seq_len, dim)

    x=torch.randn(size=(3,5,32))
    freq=precompute_theta_pos_frequencies(4, seq_len=5, device="cpu")  # shape: (5, 4)
    out = model(x,freqs_complex=freq)
    assert out.shape == (3, 5, 32), "Output shape mismatch"
    

def test_training_windowed_attention():
    model = AttentionWithKVCache(dim=64, num_heads=4, window_size=3,device="cpu")
    freq = precompute_theta_pos_frequencies(16, 6, "cpu")
    model.train()
    x = torch.randn(1, 6, 64)  # (batch, seq_len, dim)
    out = model(x,freqs_complex=freq)
    assert not torch.isnan(out).any(), "Output has NaNs"
    assert out.shape == (1, 6, 64)

def test_inference_kv_cache_rollover():
    model = AttentionWithKVCache(dim=64, num_heads=4, window_size=4, max_seq_len=8,device="cpu")
    model.eval()
    model.reset_cache()
    freq = precompute_theta_pos_frequencies(16, 1, "cpu")
    with torch.no_grad():
        outputs = []
        for i in range(10):  # Feed 10 tokens, triggering cache rolling
            x = torch.randn(1, 1, 64)
            y = model(x, start_pos=i,freqs_complex=freq)
            outputs.append(y)
        final_out = torch.cat(outputs, dim=1)

    assert final_out.shape == (1, 10, 64)
    assert not torch.isnan(final_out).any(), "NaNs in output after cache rollover"
    assert model.cache_k.shape == (8, 2, 16), "KV cache size mismatch"

def test_grouped_query_attention_sharing():
    freq = precompute_theta_pos_frequencies(8, 6, "cpu")
    model = AttentionWithKVCache(dim=64, num_heads=8, num_kv_heads=2, window_size=5,device="cpu")
    x = torch.randn(1, 6, 64)
    model.train()
    out = model(x)

    assert out.shape == (1, 6, 64)
    assert model.repeats == 4, "KV head sharing is incorrect"

def test_cache_reset():
    freq = precompute_theta_pos_frequencies(16, 2, "cpu")
    model = AttentionWithKVCache(dim=64, num_heads=4, window_size=4,device="cpu")
    model.eval()
    model.reset_cache()

    x = torch.randn(1, 2, 64)
    _ = model(x,freqs_complex=freq)
    assert model.cache_pos > 0

    model.reset_cache()
    assert model.cache_pos == 0
    assert torch.all(model.cache_k == 0)
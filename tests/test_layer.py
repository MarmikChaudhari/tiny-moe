import torch
import pytest
from utils import precompute_theta_pos_frequencies
from layer import layer  # Replace with actual import

def make_dummy_inputs(batch_size, seq_len, d_model, d_head, device):
    assert d_head % 2 == 0, "d_head must be even"
    
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    freqs = precompute_theta_pos_frequencies(d_head, seq_len, device)  # returns shape (seq_len, d_head // 2)
    
    return x, freqs


def test_layer_output_shape():
    d_model = 32
    d_head = 4 # full dimension (not d_head // 2)
    n_heads = 8
    n_kv_heads = 2
    window_size = 5
    num_experts = 4
    top_k = 2
    dropout = 0.1
    seq_len = 5
    batch_size = 3
    device = "cpu"

    l = layer(d_model, d_head, n_heads, n_kv_heads, window_size, num_experts,
              top_k, device, max_seq_len=seq_len, attn_eps=1e-5, dropout=dropout).to(device)

    x=torch.randn(batch_size, seq_len, d_model, device=device)
    freqs = precompute_theta_pos_frequencies(d_head=d_head, seq_len=seq_len, device=device)
    # d_model = d_head * n_heads → so shape of x matches what attention expects

    print("x:", x.shape)
    print("freqs:", freqs.shape)
    
    output = l(x, freqs, start_pos=0)

    assert output.shape == x.shape, "Output shape mismatch"
    assert output.dtype == x.dtype, "Output dtype mismatch"
    assert output.device == x.device, "Output device mismatch"


def test_layer_forward_device_consistency():
    if torch.cuda.is_available():
        device = "cuda"
        d_model = 32
        d_head = 16
        seq_len = 4
        l = layer(d_model, d_head, 4, 2, 4, 4, 2, device, seq_len, 1e-5, 0.1).to(device)

        x, freqs = make_dummy_inputs(2, seq_len, d_model, d_head, device=device)
        out = l(x, freqs, start_pos=0)

        assert out.device.type == "cuda"
        assert out.shape == x.shape


def test_layer_invalid_input_dimension():
    d_model = 32
    d_head = 16
    l = layer(d_model, d_head, 4, 2, 4, 4, 2, "cpu", max_seq_len=4, attn_eps=1e-5, dropout=0.1)

    wrong_x = torch.randn(2, 4, d_model + 1)  # Wrong feature size
    freqs = torch.polar(torch.ones(4, d_head // 2), torch.rand(4, d_head // 2))

    with pytest.raises(RuntimeError):
        l(wrong_x, freqs, start_pos=0)
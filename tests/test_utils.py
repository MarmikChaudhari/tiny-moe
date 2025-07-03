import pytest
from src.utils import RMSNorm, apply_rotary_embeddings,precompute_theta_pos_frequencies
import torch

def test_output_shape():
    x = torch.randn(8, 16)
    norm = RMSNorm(dim=16)
    out = norm(x)
    assert out.shape == x.shape, "Output shape should match input shape"

def test_known_input():
    x = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    norm = RMSNorm(dim=2, eps=0.0)
    norm.w.data = torch.tensor([1.0, 1.0])
    out = norm(x)
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
    expected = x / rms
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)

def test_gradient_flow():
    x = torch.randn(10, 32, requires_grad=True)
    norm = RMSNorm(dim=32)
    out = norm(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Gradients should be propagated through RMSNorm"
    assert not torch.isnan(x.grad).any(), "No NaNs should appear in the gradient"

# def test_eps_effect():
#     x = torch.ones(2, 4) * 1000  # large magnitude to test eps
#     norm_small_eps = RMSNorm(dim=4, eps=1e-12)
#     norm_large_eps = RMSNorm(dim=4, eps=1e-1)
#     out_small = norm_small_eps(x)
#     out_large = norm_large_eps(x)
#     assert not torch.allclose(out_small, out_large), "Outputs should differ due to different eps values"

def test_weight_initialization():
    norm = RMSNorm(dim=10)
    assert torch.allclose(norm.w.data, torch.ones(10)), "Weights should initialize to ones"
    
    
    
def test_precompute_theta_pos_frequencies_even_check():
    with pytest.raises(AssertionError, match="d_head must be even"):
        precompute_theta_pos_frequencies(7, 4, "cpu")


def test_apply_rotary_emb_output_shape():
    batch_size = 2
    seq_len = 4
    d_head = 8
    device = "cpu"

    x = torch.randn(batch_size, seq_len, 2,d_head, device=device)
    freq = precompute_theta_pos_frequencies(d_head, seq_len, device)
    print(x.shape)
    print(freq.shape)
    out = apply_rotary_embeddings(x, freq, device)

    assert out.shape == x.shape, "Output shape mismatch"
    assert out.dtype == x.dtype, "Output dtype mismatch"
    assert out.device.type == device, "Output device mismatch"


def test_apply_rotary_embeddings_precision_and_device():
    for dtype in [torch.float32, torch.float16]:
        x = torch.randn(1, 2, 2,8, dtype=dtype)
        freq = precompute_theta_pos_frequencies(8, 2, "cpu")
        out = apply_rotary_embeddings(x, freq, "cpu")

        assert torch.all(torch.isfinite(out)), "Output has non-finite values"
        assert out.dtype == dtype, f"Expected dtype {dtype}, got {out.dtype}"


def test_apply_rotary_embeddings_on_cuda_if_available():
    if torch.cuda.is_available():
        device = "cuda"
        x = torch.randn(1, 3, 8, device=device)
        freq = precompute_theta_pos_frequencies(8, 3, device)
        out = apply_rotary_embeddings(x, freq, device)
        assert out.device.type == "cuda"
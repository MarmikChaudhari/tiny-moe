import torch
import pytest
from torch import nn
from torch.nn import functional as F

from src.models.moe.moe import SparseMOE, SwiGLUFFN
# === SwiGLUFFN tests ===

def test_swigluffn_output_shape():
    x = torch.randn(4, 16)
    model = SwiGLUFFN(input_dim=16, hidden_dim=32)
    out = model(x)
    assert out.shape == x.shape, "Output shape must match input shape"

def test_swigluffn_forward_values():
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    model = SwiGLUFFN(input_dim=4, hidden_dim=8)
    out = model(x)
    assert torch.isfinite(out).all(), "SwiGLUFFN output should contain no NaNs or Infs"

def test_swigluffn_gradients():
    x = torch.randn(5, 10, requires_grad=True)
    model = SwiGLUFFN(input_dim=10, hidden_dim=20)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Gradients should flow through SwiGLUFFN"
    assert torch.isfinite(x.grad).all(), "Gradients should not contain NaNs or Infs"

# === SparseMOE tests ===

def test_sparsemoe_output_shape():
    x = torch.randn(2, 3, 16)
    model = SparseMOE(d_model=16, d_hidden=32, num_experts=4, top_k=2)
    out, loss = model(x)
    assert out.shape == x.shape, "SparseMOE output shape must match input"
    assert isinstance(loss, torch.Tensor), "Load balancing loss must be a tensor"

def test_sparsemoe_gradients():
    x = torch.randn(2, 5, 8, requires_grad=True)
    model = SparseMOE(d_model=8, d_hidden=16, num_experts=3, top_k=2)
    out, loss = model(x)
    total_loss = out.sum() + loss
    total_loss.backward()
    assert x.grad is not None, "Gradients should propagate through SparseMOE"
    assert torch.isfinite(x.grad).all(), "Gradients should not contain NaNs or Infs"

def test_sparsemoe_topk_behavior():
    x = torch.randn(1, 2, 10)
    model = SparseMOE(d_model=10, d_hidden=20, num_experts=5, top_k=2)
    _, loss = model(x)
    assert loss > 0, "Load balancing loss should be positive"
    assert model.top_k <= model.num_experts, "top_k should not exceed num_experts"

def test_sparsemoe_router_softmax_distribution():
    x = torch.randn(3, 4, 6)
    model = SparseMOE(d_model=6, d_hidden=12, num_experts=4, top_k=2)
    with torch.no_grad():
        x_flat = x.view(-1, 6)
        router_logits = model.router(x_flat)
        probs = F.softmax(router_logits, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs[:, 0]), atol=1e-5), "Softmax probs should sum to 1"

def test_sparsemoe_expert_usage():
    x = torch.randn(2, 2, 8)
    model = SparseMOE(d_model=8, d_hidden=16, num_experts=3, top_k=1)
    with torch.no_grad():
        _, loss = model(x)
        assert loss.item() > 0, "Load balancing loss should be non-zero to encourage expert diversity"

def test_sparsemoe_d_hidden_div_topk():
    d_model = 12
    top_k = 3
    num_experts = 4
    # d_hidden should be d_model // top_k
    expected_d_hidden = d_model // top_k
    model = SparseMOE(d_model=d_model, d_hidden=expected_d_hidden, num_experts=num_experts, top_k=top_k)
    # Check that each expert has the correct hidden dim
    for expert in model.experts:
        assert expert.w_1.in_features == d_model
        assert expert.w_1.out_features == expected_d_hidden
        assert expert.w_2.in_features == d_model
        assert expert.w_2.out_features == expected_d_hidden
        assert expert.out.in_features == expected_d_hidden
        assert expert.out.out_features == d_model
    # Forward pass sanity check
    x = torch.randn(2, 5, d_model)
    out, loss = model(x)
    assert out.shape == x.shape, "Output shape should match input shape"
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"


test_swigluffn_gradients()
test_sparsemoe_output_shape()
test_sparsemoe_gradients()
test_sparsemoe_topk_behavior()
test_sparsemoe_router_softmax_distribution()
test_sparsemoe_expert_usage()

print("All tests passed")

import torch
from data import tokenizer
from model.model import tiny_mixtral  # Update path if different
from model.config import ModelArgs
from data import vocab_size,tokenizer,train_dataset,val_dataset,train_loader,val_loader,batch_size
import argparse
import os
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = ModelArgs(vocab_size=vocab_size,d_model=512,d_head=64,n_heads=8,n_kv_heads=2,window_size=257,n_experts=8,
                 top_k=2,n_layers=8,batch_size=batch_size,train_epochs=1,val_epochs=1,seq_len=150,max_seq_len=256,
                 clip=1,attn_dropout=0.1,dropout=0.1,max_lr=1e-3,beta1=0.9,beta2=0.999,device=device,wandb_project="mixtral",norm_eps=1e-6,attn_eps=1e-6,ffn_eps=1e-6)
def load_model(checkpoint_path):
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"

    model = tiny_mixtral(args=config).to(device)  # Pass args if your model needs them
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def sample_next_token(logits, temperature=1.0, top_k=50, top_p=0.9):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))  # Safety
        values, _ = torch.topk(probs, top_k)
        probs[probs < values[..., -1, None]] = 0
        probs = probs / probs.sum(dim=-1, keepdim=True)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = sorted_probs.cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_probs[sorted_indices_to_remove] = 0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        probs = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)

    return torch.multinomial(probs, num_samples=1)


def generate_text(prompt, model, tokenizer, max_new_tokens=100, temperature=1.0, top_k=50, top_p=0.95):
    if not prompt.strip():
        raise ValueError("Prompt is empty. Please provide a non-empty prompt.")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if input_ids.shape[1] == 0:
        raise ValueError(f"Prompt '{prompt}' produced 0 tokens after tokenization.")

    output_ids = input_ids.clone()
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(output_ids, start_pos=output_ids.shape[1] - input_ids.shape[1])
            next_token_logits = outputs[:, -1, :]  # Shape: [1, vocab_size]

            next_token_id = sample_next_token(
                next_token_logits, temperature=temperature, top_k=top_k, top_p=top_p
            )

            if next_token_id.item() == tokenizer.eos_token_id:
                break

            output_ids = torch.cat([output_ids, next_token_id], dim=1)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# if __name__ == "__main__":
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="models/best_epoch.pt", help="Path to model checkpoint")
parser.add_argument("--prompt", type=str, required=True, help="Prompt to generate text from")
parser.add_argument("--max_new_tokens", type=int, default=100, help="Max tokens to generate")
args = parser.parse_args()

model = load_model(args.checkpoint)
result = generate_text(
    args.prompt,
    model,
    tokenizer=tokenizer,
    max_new_tokens=args.max_new_tokens,
    temperature=1.3,   
    top_k=20,
    top_p=0.95
)

print("\n📝 Generated Text:")
print(result)
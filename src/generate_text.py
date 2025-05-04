import torch
from data import tokenizer
from model.model import tiny_mixtral  # Update path if different
from model.config import ModelArgs
from data import vocab_size,tokenizer,train_dataset,val_dataset,train_loader,val_loader,batch_size
import argparse
import os

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

def generate_text(prompt, model, tokenizer, max_new_tokens=100):
    if not prompt.strip():
        raise ValueError("Prompt is empty. Please provide a non-empty prompt.")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(input_ids.shape)
    if input_ids.shape[1] == 0:
        raise ValueError(f"Prompt '{prompt}' produced 0 tokens after tokenization.")

    output_ids = input_ids.clone()
    print(f"Current output_ids shape: {output_ids.shape}")
    
    print(f"Initial input_ids shape: {input_ids.shape}")

    print(f"Initial output_ids shape: {output_ids.shape}")

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            print(f"Current output_ids shape: {output_ids.shape}")  # Debugging line
            outputs = model(output_ids, start_pos=output_ids.shape[1] - input_ids.shape[1])
            print(f"Current output_ids shape: {output_ids.shape}")
            next_token_logits = outputs[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

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
result = generate_text(args.prompt, model, max_new_tokens=args.max_new_tokens,tokenizer=tokenizer)

print("\n📝 Generated Text:")
print(result)
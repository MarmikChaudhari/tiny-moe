import os
import math
import torch
import wandb
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dense.model import tiny_gpt
from models.dense.config import ModelArgs
from data import vocab_size,tokenizer,train_loader,val_loader,batch_size
from tqdm import tqdm
import argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device="cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
config = ModelArgs(vocab_size=vocab_size, batch_size = BATCH_SIZE, device=device)


def save_checkpoint(model,optimizer,scheduler,step,path,best_val_loss=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)  
    torch.save({
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "scheduler_state_dict":scheduler.state_dict(),
        "step":step,
        "best_val_loss":best_val_loss,
    },path)


def load_checkpoint(model,optimizer,scheduler,path):

    
    checkpoint=torch.load(path,map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["step"]


def evaluate(model, dataloader, criterion):
    """
    Evaluate model on a given dataset using a given loss criterion.

    Args:
        model: (nn.Module) the model to evaluate
        dataloader: (DataLoader) the dataset to evaluate on
        criterion: (nn.Module) the loss criterion

    Returns:
        total_loss: (float) the average loss per sample over the dataset
    """
    model.eval()
    total_loss = 0
    total_samples = 0  # Track total samples for correct averaging
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating Batches"):
            inputs, targets = [x.to(device) for x in batch]
            
            # Ensure batch size is 1 for inference (if required by your model)
            # if inputs.size(0) != 1:
            #     raise ValueError(
            #         f"Inference requires batch_size=1, got {inputs.size(0)}. "
            #         "Modify your DataLoader or handle batching differently."
            #     )
            
            # Forward pass - dense model only returns outputs, no load balancing loss
            outputs = model(inputs, start_pos=0)
            
            # Calculate loss (flatten outputs and targets for cross-entropy)
            loss = criterion(
                outputs.view(-1, outputs.shape[-1]),  # shape: [seq_len * batch_size, vocab_size]
                targets.view(-1)                      # shape: [seq_len * batch_size]
            )
            
            # Accumulate loss and sample count
            total_loss += loss.item() * inputs.size(0)  # weight by batch size
            total_samples += inputs.size(0)
    
    # Return average loss per sample
    return total_loss / total_samples if total_samples > 0 else 0.0


def train(resume_path=None,use_wandb=False):
    model=tiny_gpt(args=config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.max_lr,weight_decay=0.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.train_epochs)
    
    criterion=torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    start_step = 0
   
        
    best_val_loss = float("inf")
    #optional resume
    if resume_path is not None and os.path.exists(resume_path):

        print(f"🔁 Resuming from {resume_path}")
        checkpoint=torch.load(resume_path,map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = checkpoint.get("step", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        # for layer in model.layers:
        #     if hasattr(layer, 'attention') and hasattr(layer.attention, 'reset_cache'):
        #         layer.attention.reset_cache()
        #         #print("cache reset")
        
    
    if use_wandb:
        wandb.init(project=config.wandb_project,config=config,resume="allow")
        wandb.watch(model)
    
    step = start_step
    
    for epoch in range(config.train_epochs):
        model.train()
        running_loss=0.0
            
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.train_epochs}', unit='batch'):
            inputs,targets=[x.to(device) for x in batch]
            
            # Dense model only returns outputs, no load balancing loss
            outputs = model(inputs,start_pos=0)
            loss = criterion(outputs.view(-1,outputs.shape[-1]),targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),config.clip)
            optimizer.step()
            scheduler.step()
            step += 1
            running_loss += loss.item()
            if use_wandb:
                wandb.log({
                    "train/loss":loss.item(),
                    "train/lr":scheduler.get_last_lr()[0],
                    "epoch":epoch,
                    "step":step,
                })
            
            if step % config.val_steps == 0:
                print(f'val loss at step {step}...')
                val_loss = evaluate(model,val_loader,criterion)
                model.train()
                if use_wandb:
                    wandb.log({
                        "val/loss":val_loss,
                        "epoch":epoch,
                        "step":step,
                    })
                # if best_val_loss is None:
                #     best_val_loss=val_loss
                if val_loss < best_val_loss:
                    best_val_loss=val_loss
                    save_checkpoint(model,optimizer,scheduler,step,f"trained_models/best_val_loss_dense_step_{step}.pt",best_val_loss=best_val_loss)
                    print("best val loss model saved at step",step)

            if step % config.save_steps == 0: # save model every save_steps
                save_checkpoint(model,optimizer,scheduler,step,f"trained_models/dense_step_{step}.pt")
                print(f"model saved at step {step}")
    

        save_checkpoint(model, optimizer, scheduler, step, f"trained_models/last_epoch_dense_{epoch}.pt")
        print(f"Epoch {epoch+1} complete. Avg Loss: {running_loss / len(train_loader):.4f}")

    print("✅ Training complete.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="trained_models/best_model_dense.pt", help="Path to model checkpoint")
    parser.add_argument("--usewandb", action="store_true", default=False, help="Use Weights & Biases logging")
    args = parser.parse_args()

    train(resume_path=args.checkpoint,use_wandb=args.usewandb)


# parser = argparse.ArgumentParser()
# parser.add_argument("--checkpoint", type=str, default="trained_models/best_model_dense.pt", help="Path to model checkpoint")
# parser.add_argument("--usewandb", action="store_true", default=False, help="Use Weights & Biases logging")
# args = parser.parse_args()

# train(resume_path=args.checkpoint,use_wandb=args.usewandb)
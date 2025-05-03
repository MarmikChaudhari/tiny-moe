import os
import math
import torch
import wandb

from model.model import tiny_mixtral
from model.config import ModelArgs
from data import vocab_size,tokenizer,train_dataset,val_dataset,train_loader,val_loader,batch_size,max_seq_len

device="cuda" if torch.cuda.is_available() else "cpu"
args = ModelArgs(vocab_size=vocab_size,d_model=512,d_head=64,n_heads=8,n_kv_heads=2,window_size=257,n_experts=8,
                 top_k=2,n_layers=8,batch_size=batch_size,train_epochs=1,val_epochs=1,seq_len=150,max_seq_len=256,
                 clip=1,attn_dropout=0.1,dropout=0.1,max_lr=1e-3,beta1=0.9,beta2=0.999,device=device,wandb_project="mixtral",norm_eps=1e-6,attn_eps=1e-6,ffn_eps=1e-6)
# model = tiny_mixtral(args)


def save_checkpoint(model,optimizer,scheduler,step,path,best_val_loss=None):    
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


def evaluate(model,dataloader,criterion):
    """
    Evaluate model on a given dataset using a given loss criterion.

    Args:
        model: (nn.Module) the model to evaluate
        dataloader: (DataLoader) the dataset to evaluate on
        criterion: (nn.Module) the loss criterion

    Returns:
        total_loss: (float) the total loss over the dataset
    """
    model.eval()
    total_loss=0
    with torch.no_grad():
        for batch in dataloader:
            inputs,targets=[x.to(device) for x in batch]
            outputs=model(inputs)
            loss=criterion(outputs.view(-1,outputs.shape[-1]),targets.view(-1))
            total_loss+=loss.item()
    
    return total_loss/len(dataloader)


def train(resume_path=None,wandb=False):
    model=tiny_mixtral(args=args).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args["max_lr"],weight_decay=0.01)
    scheduler=torch.optim.CosineAnnealingLR(optimizer,T_max=args["train_epochs"])
    
    criterion=torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    start_step=0
   
        
    best_val_loss = float("inf")
    #optional resume
    if resume_path is not None and os.path.exists(resume_path):
        print(f"🔁 Resuming from {resume_path}")
        checkpoint=torch.load(resume_path,map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = checkpoint.get("step", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        
    
    if wandb:
        wandb.init(project=args["wandb_project"],config=args,resume="allow")
        wandb.watch(model)
    
    step=start_step
    
    for epoch in range(args["train_epochs"]):
        model.train()
        running_loss=0.0
            
        
        for batch in train_loader:
            inputs,targets=[x.to(device) for x in batch]
            
            outputs=model(inputs)
            loss=criterion(outputs.view(-1,outputs.shape[-1]),targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),args["clip"])
            optimizer.step()
            scheduler.step()
            step+=1
            running_loss+=loss.item()
            if wandb:
                wandb.log({
                    "train/loss":loss.item(),
                    "train/lr":scheduler.get_last_lr()[0],
                    "epoch":epoch,
                    "step":step,
                })
            
            if step%100==0:
                val_loss=evaluate(model,val_loader,criterion)
                if wandb:
                    wandb.log({
                        "val/loss":val_loss,
                        "epoch":epoch,
                        "step":step,
                    })
                if val_loss<best_val_loss:
                    best_val_loss=val_loss
                    save_checkpoint(model,optimizer,scheduler,step,"models/best_model.pt")
                    print("best model saved at step",step)
    

        save_checkpoint(model, optimizer, scheduler, step, "models/last_epoch.pt")
        print(f"Epoch {epoch+1} complete. Avg Loss: {running_loss / len(train_loader):.4f}")

    print("✅ Training complete.")
            
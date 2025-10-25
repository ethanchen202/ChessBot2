import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
import os
import shutil
import weightwatcher as ww
import pandas as pd

from run_timer import TIMER
from dataloader import CCRL4040LMDBDataset, worker_init_fn
from model import ChessViT
from model2 import ChessViTv2

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000) # Adjust width for better column display


# ----------------------------
# Training Pipeline
# ----------------------------
def train_pipeline(
        model, 
        train_loader, 
        val_loader=None, 
        num_epochs=10, 
        lr=1e-3, 
        weight_decay=0.05, 
        device='cuda', 
        accumulation_steps=1,
        load_from_checkpoint_path=None,
        checkpoint_dir='./results/checkpoints/_last_run'
    ):
    
    TIMER.start("Initializing training process")

    if load_from_checkpoint_path is not None:
        checkpoint = torch.load(load_from_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)

    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    watcher = ww.WeightWatcher(model)

    # LR Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])
    
    # Losses
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    
    # Mixed precision
    scaler = GradScaler(device)

    TIMER.stop("Initializing training process")
    
    for epoch in range(1, num_epochs+1):
        TIMER.start(f"Training epoch {epoch}")
        model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        optimizer.zero_grad()
        
        for step, (x, policy_labels, value_labels, legal_mask) in enumerate(train_loader):

            x = x.to(device)
            policy_labels = policy_labels.to(device)
            value_labels = value_labels.to(device)
            legal_mask = legal_mask.to(device)
            
            with autocast(device):
                policy_logits, value_preds, outcome_logits, extras = model(x, legal_mask)
                loss_policy = policy_loss_fn(policy_logits, policy_labels)
                loss_value = value_loss_fn(value_preds, value_labels)
                # Weighted sum of losses (can adjust alpha)
                loss = loss_policy + loss_value
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_policy_loss += loss_policy.item()
            total_value_loss += loss_value.item()

            if step % 100 == 0:
                TIMER.lap(f"Training for {len(train_loader)} batches", step + 1, len(train_loader))
            TIMER.lap(f"Training for {len(train_loader)} batches", step + 1, len(train_loader))
        
        avg_policy_loss = total_policy_loss / len(train_loader)
        avg_value_loss = total_value_loss / len(train_loader)
        scheduler.step()
        print(f"Epoch {epoch}: Avg Policy Loss={avg_policy_loss:.4f}, Avg Value Loss={avg_value_loss:.4f}")
        
        TIMER.stop(f"Training epoch {epoch}")

        TIMER.start(f"Validating epoch {epoch}")

        # ----------------------------
        # Validation loop (optional)
        # ----------------------------
        if val_loader is not None:
            model.eval()
            val_policy_loss = 0.0
            val_value_loss = 0.0

            correct_top1 = 0
            correct_top5 = 0
            total = 0

            with torch.no_grad():
                for x, policy_labels, value_labels in val_loader:
                    x = x.to(device)
                    policy_labels = policy_labels.to(device)
                    value_labels = value_labels.to(device)
                    
                    with autocast(device):
                        # compute loss
                        policy_logits, value_preds, outcome_logits, extras = model(x)
                        val_policy_loss += policy_loss_fn(policy_logits, policy_labels).item()
                        val_value_loss += value_loss_fn(value_preds, value_labels).item()

                        # accuracy
                        _, pred_top1 = policy_logits.max(dim=1)  # Top-1
                        correct_top1 += pred_top1.eq(policy_labels.argmax(dim=1)).sum().item()
                        
                        # Top-5
                        top5_preds = policy_logits.topk(5, dim=1).indices  # (batch_size, 5)
                        correct_top5 += top5_preds.eq(policy_labels.argmax(dim=1).view(-1, 1)).sum().item()
                        
                        total += policy_labels.size(0)

            val_policy_loss /= len(val_loader)
            val_value_loss /= len(val_loader)
            top1_acc = correct_top1 / total
            top5_acc = correct_top5 / total

            print(f"Validation: "
                f"Policy Loss={val_policy_loss:.4f}, "
                f"Value Loss={val_value_loss:.4f}, "
                f"Top-1 Acc={top1_acc:.4f}, "
                f"Top-5 Acc={top5_acc:.4f}")
        
        TIMER.stop(f"Validating epoch {epoch}")

        # ----------------------------
        # Save checkpoint
        # ----------------------------
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pt')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        print(watcher.analyze(plot=True))


if __name__ == "__main__":

    TIMER.start("Initializing Dataloader")
    # Paths
    # lmdb_path_train = r'/teamspace/studios/this_studio/chess_bot/datasets/processed/CCRL-4040-train-2m-100k-0.2-0.8-1.lmdb'
    # lmdb_path_val = r'/teamspace/studios/this_studio/chess_bot/datasets/processed/CCRL-4040-val-2m-100k-0.2-0.8-1.lmdb'
    lmdb_path_train = r'/teamspace/studios/this_studio/chess_bot/datasets/processed/CCRL-4040-train-20000000-500000-0.2-0.8-1.lmdb'
    lmdb_path_val = r'/teamspace/studios/this_studio/chess_bot/datasets/processed/CCRL-4040-val-20000000-500000-0.2-0.8-1.lmdb'
    checkpoint_path = r'/teamspace/studios/this_studio/chess_bot/results/checkpoints/CCRL-4040-val-1000000-1000-0.2-0.8-1-lr_1e-4'

    # Hyperparameters
    batch_size = 512
    accumulation_steps = 1  # effective batch size = batch_size * accumulation_steps
    num_epochs = 200
    lr = 1e-4
    
    # Create datasets
    train_dataset = CCRL4040LMDBDataset(lmdb_path_train)
    val_dataset = CCRL4040LMDBDataset(lmdb_path_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn, persistent_workers=True, pin_memory=True)
    
    TIMER.stop("Initializing Dataloader")

    TIMER.start("Initializing Model")
    
    # Initialize model

    model = ChessViTv2()

    # Check if cuda is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    TIMER.stop("Initializing Model")
    
    # Train
    train_pipeline(
        model, 
        train_loader, 
        val_loader=val_loader, 
        num_epochs=num_epochs, 
        lr=lr, 
        accumulation_steps=accumulation_steps,
        device=device,
        checkpoint_dir=checkpoint_path,
    )
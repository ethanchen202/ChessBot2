import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import os

from run_timer import TIMER
from dataloader import CCRL4040LMDBDataset, worker_init_fn
from model import ChessViT


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
        checkpoint_dir='./checkpoints'
    ):
    
    TIMER.start("Initializing training process")

    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
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
        
        TIMER.start(f"Training for {len(train_loader)} batches")
        for step, (x, policy_labels, value_labels) in enumerate(train_loader):

            x = x.to(device)
            policy_labels = policy_labels.to(device)
            value_labels = value_labels.to(device)
            
            with autocast(device):
                policy_logits, value_preds = model(x)
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
        
        avg_policy_loss = total_policy_loss / len(train_loader)
        avg_value_loss = total_value_loss / len(train_loader)
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
                        policy_logits, value_preds = model(x)
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


if __name__ == "__main__":

    TIMER.start("Initializing Dataloader")
    # Paths
    lmdb_path_train = r'/teamspace/studios/this_studio/chess_bot/datasets/processed/CCRL-4040-train.lmdb'
    lmdb_path_val = r'/teamspace/studios/this_studio/chess_bot/datasets/processed/CCRL-4040-val.lmdb'
    checkpoint_path = r'/teamspace/studios/this_studio/chess_bot/results/checkpoints'

    # Hyperparameters
    batch_size = 320
    accumulation_steps = 2  # effective batch size = batch_size * accumulation_steps
    num_epochs = 200
    
    # Create datasets
    train_dataset = CCRL4040LMDBDataset(lmdb_path_train)
    val_dataset = CCRL4040LMDBDataset(lmdb_path_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn, persistent_workers=True)
    
    TIMER.stop("Initializing Dataloader")

    TIMER.start("Initializing Model")
    
    # Initialize model
    model = ChessViT()

    # Check if cuda is available
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    TIMER.stop("Initializing Model")
    
    # Train
    train_pipeline(
        model, 
        train_loader, 
        val_loader=val_loader, 
        num_epochs=num_epochs, 
        lr=1e-3, 
        accumulation_steps=accumulation_steps,
        device=device,
        checkpoint_dir=checkpoint_path,
    )
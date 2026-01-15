import torch
import torch.nn as nn
import os

def save_checkpoint(filepath: str, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer):   
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    checkpoint_name = f"checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, os.path.join(filepath, checkpoint_name))
    print(f"--- Checkpoint saved: {checkpoint_name} ---")


def load_checkpoint(filepath:str, model: nn.Module, optimizer: torch.optim.Optimizer):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Loaded the model from checkpoint from epoch nr {epoch}")
        return epoch
    else:
        print("Couldnt find the checkpoint.")
        return 0
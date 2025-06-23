from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime

from models.transformer import PatchedTransformerEncoderStack, ReconstructionHead, SelfSupervisedBackbone

class NoamOpt:
    """Optim wrapper that implements rate."""
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        if step == 0:
            return 0
        return self.factor *             (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def patchify(X, n_patches):
    """
    X: (batch_size, n_modalities, sequence_length)
    """
    patch_size = X.shape[-1] // n_patches
    return X.view(X.shape[0], X.shape[1], n_patches, patch_size)

def get_mask(shape, masking_ratio, device):
    """
    shape: (batch_size, n_modalities, n_patches)
    """
    len_keep = int(shape[2] * (1 - masking_ratio))
    noise = torch.rand(shape, device=device)
    ids_shuffle = torch.argsort(noise, dim=2)
    ids_restore = torch.argsort(ids_shuffle, dim=2)
    ids_keep = ids_shuffle[:, :, :len_keep]
    mask = torch.ones(shape, device=device)
    mask[:, :, :len_keep] = 0
    mask = torch.gather(mask, dim=2, index=ids_restore)
    return mask.bool()

def train_epoch(model, dataloader, optimizer, device, masking_ratio, n_patches):
    model.train()
    total_loss = 0
    for X_batch in tqdm(dataloader, desc="Training"):
        X_batch = X_batch[0].to(device)
        
        # Patchify
        X_patched = patchify(X_batch, n_patches)
        
        # Create mask
        mask = get_mask(X_patched.shape[:3], masking_ratio, device)
        
        X_masked = X_patched.clone()
        X_masked[mask.unsqueeze(-1).expand_as(X_masked)] = 0

        optimizer.zero_grad()
        
        # Forward pass
        X_hat = model(X_masked)
        
        # Calculate loss only on masked patches
        loss = torch.nn.functional.mse_loss(X_hat[mask], X_patched[mask])
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, device, masking_ratio, n_patches):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch in tqdm(dataloader, desc="Validation"):
            X_batch = X_batch[0].to(device)
            
            X_patched = patchify(X_batch, n_patches)
            
            mask = get_mask(X_patched.shape[:3], masking_ratio, device)
            
            X_masked = X_patched.clone()
            X_masked[mask.unsqueeze(-1).expand_as(X_masked)] = 0
            
            X_hat = model(X_masked)
            
            loss = torch.nn.functional.mse_loss(X_hat[mask], X_patched[mask])
            
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    X_train = np.load(args.data_path / 'X_train.npy')
    X_val = np.load(args.data_path / 'X_val.npy')

    # Replace NaNs and transpose
    X_train = np.nan_to_num(X_train, nan=0.0).transpose(0, 2, 1)
    X_val = np.nan_to_num(X_val, nan=0.0).transpose(0, 2, 1)

    train_dataset = TensorDataset(torch.from_numpy(X_train).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Model setup
    seq_len = X_train.shape[2]
    n_patches = seq_len // args.patch_size
    
    transformer = PatchedTransformerEncoderStack(
        n_patches=n_patches,
        patch_size=args.patch_size,
        n_modalities=X_train.shape[1],
        d_embedding=args.d_embedding,
        n_layers=args.n_layers,
        transformer_dropout=args.transformer_dropout
    )
    reconstruction_head = ReconstructionHead(d_embedding=args.d_embedding, patch_size=args.patch_size)
    model = SelfSupervisedBackbone(transformer=transformer, self_supervised_head=reconstruction_head).to(device)

    base_optimizer = optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    optimizer = NoamOpt(args.d_embedding, args.noam_factor, args.warmup_steps, base_optimizer)
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = args.output_dir / timestamp
    checkpoint_dir = run_dir / 'checkpoints'
    log_dir = run_dir / 'logs'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, args.masking_ratio, n_patches)
        val_loss = validate_epoch(model, val_loader, device, args.masking_ratio, n_patches)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch+1}/{args.epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save final model checkpoint
    final_model_path = checkpoint_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")

    writer.close()
    print("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer Pre-training Script")
    parser.add_argument('--data_path', type=Path, default='/data/IDLab/aar_foundation_models/training_snapshots/pretraining', help='Path to the training data')
    parser.add_argument('--output_dir', type=Path, default='runs', help='Directory for logs and checkpoints')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--d_embedding', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--transformer_dropout', type=float, default=0.1, help='Dropout rate for transformer')
    parser.add_argument('--masking_ratio', type=float, default=0.75, help='Ratio of patches to mask')
    parser.add_argument('--patch_size', type=int, default=25, help='Size of each patch')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='Warmup steps for Noam scheduler')
    parser.add_argument('--noam_factor', type=float, default=1.0, help='Factor for Noam scheduler')
    
    args = parser.parse_args()
    main(args)

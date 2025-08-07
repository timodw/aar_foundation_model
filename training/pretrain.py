from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.transformer import PatchedTransformerEncoderStack, ReconstructionHead, SelfSupervisedBackbone
from utils import (
    patchify, 
    unpatchify, 
    NoamOpt, 
    get_mask,
    setup_logging,
    save_config,
    log_reconstruction_to_tensorboard,
    load_pretraining_data
)


def train_pretrain_epoch(model, dataloader, optimizer, device, masking_ratio, n_patches, unmasked_loss_weight):
    """
    Performs one epoch of pretraining using masked reconstruction.

    Args:
        model: The self-supervised Transformer model to train.
        dataloader: DataLoader for the training data.
        optimizer: The optimizer (typically Adam wrapped in a NoamOpt object).
        device: The device to train on.
        masking_ratio (float): Fraction of patches to mask.
        n_patches (int): Number of patches to divide sequences into.
        unmasked_loss_weight (float): Weight for unmasked reconstruction loss.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    for X_batch in tqdm(dataloader, desc="Training"):
        X_batch = X_batch[0].to(device)
        
        X_patched = patchify(X_batch, n_patches)
        mask = get_mask(X_patched.shape[:3], masking_ratio, device)
        
        X_masked = X_patched.clone()
        X_masked[mask.unsqueeze(-1).expand_as(X_masked)] = 0

        optimizer.zero_grad()
        X_hat = model(X_masked)
        
        loss = 0
        # Extra checks for when masking ratio is either 0.0 or 1.0
        if torch.any(mask):
            loss += torch.nn.functional.mse_loss(X_hat[mask], X_patched[mask])
        if unmasked_loss_weight > 0 and torch.any(~mask):
            loss += unmasked_loss_weight * torch.nn.functional.mse_loss(X_hat[~mask], X_patched[~mask])
        
        if isinstance(loss, torch.Tensor):
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0


def validate_pretrain_epoch(model, dataloader, device, masking_ratio, n_patches, unmasked_loss_weight):
    """
    Performs one epoch of validation using masked reconstruction.

    Args:
        model: The self-supervised Transformer model to validate.
        dataloader: DataLoader for the validation data.
        device: The device to run validation on.
        masking_ratio (float): Fraction of patches to mask.
        n_patches (int): Number of patches to divide sequences into.
        unmasked_loss_weight (float): Weight for unmasked reconstruction loss.

    Returns:
        float: Average validation loss for the epoch.
    """
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
            
            loss = 0
            # Extra checks for when masking ratio is either 0.0 or 1.0
            if torch.any(mask):
                loss += torch.nn.functional.mse_loss(X_hat[mask], X_patched[mask])
            if unmasked_loss_weight > 0 and torch.any(~mask):
                loss += unmasked_loss_weight * torch.nn.functional.mse_loss(X_hat[~mask], X_patched[~mask])
            
            if isinstance(loss, torch.Tensor):
                total_loss += loss.item()
            
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0


def main(args):
    # Setup
    writer, log_dir, model_path = setup_logging(args.output_dir)
    save_config(args, log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loading
    X_train, X_val = load_pretraining_data(args.data_path, args.input_mode)
    
    train_dataset = TensorDataset(torch.from_numpy(X_train).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Model Initialization
    seq_len = X_train.shape[2]
    n_patches = seq_len // args.patch_size
    n_modalities = X_train.shape[1]

    transformer = PatchedTransformerEncoderStack(
        n_patches=n_patches,
        patch_size=args.patch_size,
        n_modalities=n_modalities,
        d_embedding=args.d_embedding,
        n_layers=args.n_layers,
        transformer_dropout=args.transformer_dropout,
        embedding_type=args.embedding_type
    ).to(device)
    
    reconstruction_head = ReconstructionHead(d_embedding=args.d_embedding, patch_size=args.patch_size).to(device)
    model = SelfSupervisedBackbone(transformer, reconstruction_head).to(device)

    # Optimizer
    base_optimizer = optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    optimizer = NoamOpt(args.d_embedding, args.noam_factor, args.warmup_steps, base_optimizer)
    
    # Visualization sample
    stds = torch.from_numpy(X_val).std(dim=(2))
    vis_sample_idx = torch.argmax(torch.sum(stds, dim=1))
    vis_sample = torch.from_numpy(X_val[vis_sample_idx:vis_sample_idx+1]).float().to(device)
    vis_mask = get_mask((vis_sample.shape[0], vis_sample.shape[1], n_patches), args.masking_ratio, device)

    # Training Loop
    for epoch in range(args.epochs):
        train_loss = train_pretrain_epoch(model, train_loader, optimizer, device, args.masking_ratio, n_patches, args.unmasked_loss_weight)
        val_loss = validate_pretrain_epoch(model, val_loader, device, args.masking_ratio, n_patches, args.unmasked_loss_weight)

        torch.save(model.state_dict(), model_path)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % 10 == 0:
            log_reconstruction_to_tensorboard(writer, model, vis_sample, vis_mask, epoch, n_patches)

        print(f"Epoch {epoch+1}/{args.epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer Pre-training Script")
    parser.add_argument('--data_path', type=Path, help='Path to the training data directory')
    parser.add_argument('--output_dir', type=Path, help='Directory for logs and checkpoints')
    parser.add_argument('--batch_size', type=int, default=512, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--d_embedding', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--transformer_dropout', type=float, default=0.1, help='Dropout rate for transformer')
    parser.add_argument('--masking_ratio', type=float, default=0.5, help='Ratio of patches to mask')
    parser.add_argument('--patch_size', type=int, default=25, help='Size of each patch')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='Warmup steps for Noam scheduler')
    parser.add_argument('--noam_factor', type=float, default=1.0, help='Factor for Noam scheduler')
    parser.add_argument('--input_mode', type=str, default='multi', choices=['multi', 'single'], help='Input mode: multi-modality or single-modality')
    parser.add_argument('--unmasked_loss_weight', type=float, default=0.0, help='Weight for the unmasked part of the loss (0.0 to 1.0)')
    parser.add_argument('--embedding_type', type=str, default='linear', choices=['linear', 'conv'], help='Type of embedding to use')
    args = parser.parse_args()
    main(args)
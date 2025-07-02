import argparse
import json
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.transformer import PatchedTransformerEncoderStack, ClassificationHead
from utils import (
    load_finetuning_data,
    apply_label_mapping,
    subsample_per_class,
    prepare_dataloaders,
    setup_logging,
    save_config,
    train_epoch,
    validate_epoch,
    patchify
)


def main(args):
    # Setup
    writer, log_dir, model_path = setup_logging(args.output_dir, args.data_path.name, args.fold)
    save_config(args, log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Loading and Preprocessing
    label_mapping_path = args.data_path / 'label_mapping.json'
    if not label_mapping_path.exists():
        raise FileNotFoundError(f"Mandatory 'label_mapping.json' not found in {args.data_path}")
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)

    X_train, y_train, X_val, y_val = load_finetuning_data(args.data_path, args.fold)
    X_train, y_train = apply_label_mapping(X_train, y_train, label_mapping)
    X_val, y_val = apply_label_mapping(X_val, y_val, label_mapping)

    if args.max_samples is not None:
        X_train, y_train = subsample_per_class(X_train, y_train, args.max_samples, args.data_path, args.fold)

    train_loader, val_loader, n_classes = prepare_dataloaders(X_train, y_train, X_val, y_val, args.batch_size)

    # Model Initialization
    with open(args.pretrained_model_path / 'config.json', 'r') as f:
        pretrained_config = json.load(f)

    seq_len = X_train.shape[2]
    n_patches = seq_len // pretrained_config['patch_size']
    n_modalities = X_train.shape[1]

    transformer = PatchedTransformerEncoderStack(
        n_patches=n_patches,
        patch_size=pretrained_config['patch_size'],
        n_modalities=n_modalities,
        d_embedding=pretrained_config['d_embedding'],
        n_layers=pretrained_config['n_layers'],
        transformer_dropout=pretrained_config['transformer_dropout']
    ).to(device)

    classification_head = ClassificationHead(
        n_modalities=n_modalities,
        d_embedding=pretrained_config['d_embedding'],
        n_patches=n_patches,
        n_classes=n_classes
    ).to(device)

    model = torch.nn.Sequential(transformer, classification_head)

    # Load Pretrained Weights
    print(f"Loading pretrained weights from: {args.pretrained_model_path}")
    pretrained_state_dict = torch.load(args.pretrained_model_path / 'model.pt', map_location=device)
    transformer_weights = {k.replace('transformer.', '', 1): v for k, v in pretrained_state_dict.items() if k.startswith('transformer.')}
    transformer.load_state_dict(transformer_weights)

    # Optimizer and Loss Function
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    print(f"Finetuning on fold {args.fold} of {args.data_path.name}")
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss, train_acc, train_bal_acc = train_epoch(model, train_loader, optimizer, criterion, device, patchify, n_patches)
        val_loss, val_acc, val_bal_acc = validate_epoch(model, val_loader, criterion, device, patchify, n_patches)

        writer.add_scalar(f'Loss/train_fold_{args.fold}', train_loss, epoch)
        writer.add_scalar(f'Accuracy/train_fold_{args.fold}', train_acc, epoch)
        writer.add_scalar(f'Balanced_Accuracy/train_fold_{args.fold}', train_bal_acc, epoch)
        writer.add_scalar(f'Loss/val_fold_{args.fold}', val_loss, epoch)
        writer.add_scalar(f'Accuracy/val_fold_{args.fold}', val_acc, epoch)
        writer.add_scalar(f'Balanced_Accuracy/val_fold_{args.fold}', val_bal_acc, epoch)

        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Bal Acc: {train_bal_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Bal Acc: {val_bal_acc:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)

    writer.close()
    print('Finetuning complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finetuning script for the AAR foundation model")
    parser.add_argument('--data_path', type=Path, default='/data/IDLab/aar_foundation_models/training_snapshots/finetuning/horsing_around', help='Path to the finetuning data directory')
    parser.add_argument('--output_dir', type=Path, default='logs/finetuning', help='Directory for logs and checkpoints')
    parser.add_argument('--fold', type=int, default=0, help='Fold to use for training and validation')
    parser.add_argument('--max_samples', type=int, default=400, help='Maximum number of samples to use for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=.0, help='Weight decay')
    parser.add_argument('--avg_pool', action='store_true', default=False, help='Use average pooling instead of CLS token')
    parser.add_argument('--pretrained_model_path', type=Path, required=True, help='Path to pretrained model weights for the transformer')
    args = parser.parse_args()
    main(args)

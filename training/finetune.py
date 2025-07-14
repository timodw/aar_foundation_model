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
    if args.pretrained_model_path:
        with open(args.pretrained_model_path / 'config.json', 'r') as f:
            model_config = json.load(f)
        embedding_type = model_config.get('embedding_type', 'linear')
        patch_size = model_config['patch_size']
        d_embedding = model_config['d_embedding']
        n_layers = model_config['n_layers']
        transformer_dropout = model_config['transformer_dropout']
    else:
        # Use command-line arguments if not loading a pretrained model
        embedding_type = args.embedding_type
        patch_size = args.patch_size
        d_embedding = args.d_embedding
        n_layers = args.n_layers
        transformer_dropout = args.transformer_dropout

    seq_len = X_train.shape[2]
    n_patches = seq_len // patch_size
    n_modalities = X_train.shape[1]

    transformer = PatchedTransformerEncoderStack(
        n_patches=n_patches,
        patch_size=patch_size,
        n_modalities=n_modalities,
        d_embedding=d_embedding,
        n_layers=n_layers,
        transformer_dropout=transformer_dropout,
        embedding_type=embedding_type
    ).to(device)

    classification_head = ClassificationHead(
        n_modalities=n_modalities,
        d_embedding=d_embedding,
        n_patches=n_patches,
        n_classes=n_classes,
        avg_pool=args.avg_pool
    ).to(device)

    model = torch.nn.Sequential(transformer, classification_head)

    # Load Pretrained Weights if path is provided
    if args.pretrained_model_path:
        print(f"Loading pretrained weights from: {args.pretrained_model_path}")
        pretrained_state_dict = torch.load(args.pretrained_model_path / 'model.pt', map_location=device)
        transformer_weights = {k.replace('transformer.', '', 1): v for k, v in pretrained_state_dict.items() if k.startswith('transformer.')}
        transformer.load_state_dict(transformer_weights, strict=False)

        # # Freeze layers only when using a pretrained model
        # for param in model[0].parameters():
        #     param.requires_grad = False
        # for param in model[0].transformer.layers[-1].parameters():
        #     param.requires_grad = True
        # for param in model[1].parameters():
        #     param.requires_grad = True

    # Optimizer and Loss Function
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
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
    parser.add_argument('--pretrained_model_path', type=Path, default=None, help='Path to pretrained model weights for the transformer')
    
    # Add arguments for model architecture, used only if --pretrained_model_path is not provided
    parser.add_argument('--patch_size', type=int, default=25, help='Size of the patches')
    parser.add_argument('--d_embedding', type=int, default=128, help='Dimension of the embedding')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--transformer_dropout', type=float, default=0.1, help='Dropout for the transformer')
    parser.add_argument('--embedding_type', type=str, default='linear', choices=['linear', 'conv'], help='Type of embedding to use')

    args = parser.parse_args()
    main(args)

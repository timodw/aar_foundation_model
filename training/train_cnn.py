import argparse
import json
from pathlib import Path
import sys
import torch
import torch.optim as optim
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.cnn import CNN
from training.utils import (
    apply_label_mapping,
    load_finetuning_data,
    prepare_dataloaders,
    save_config,
    setup_logging,
    subsample_per_class,
    train_epoch,
    validate_epoch,
)


def main(args):
    # Load mandatory label mapping
    label_mapping_path = args.data_path / 'label_mapping.json'
    if not label_mapping_path.exists():
        raise FileNotFoundError(f"Mandatory 'label_mapping.json' not found in {args.data_path}")
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)

    # Load data
    X_train, y_train, X_val, y_val = load_finetuning_data(args.data_path, args.fold)

    # Apply label mapping and filtering
    X_train, y_train = apply_label_mapping(X_train, y_train, label_mapping)
    X_val, y_val = apply_label_mapping(X_val, y_val, label_mapping)

    # Subsample the training data if max_samples is set
    if args.max_samples is not None:
        X_train, y_train = subsample_per_class(X_train, y_train, args.max_samples, args.data_path, args.fold)

    # Create datasets and dataloaders
    train_loader, val_loader, n_classes = prepare_dataloaders(X_train, y_train, X_val, y_val, args.batch_size)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_modalities = X_train.shape[1]

    model = CNN(
        n_classes=n_classes,
        n_modalities=n_modalities
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Logging and checkpointing setup
    writer, log_dir, model_path = setup_logging(args.output_dir, args.data_path.name, args.fold)
    save_config(args, log_dir)

    print(f"Training CNN on fold {args.fold} of {args.data_path.name}")
    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of validation samples: {len(X_val)}")
    print(f"Number of classes: {n_classes}")
    print(f"Logs and checkpoints will be saved to: {log_dir}")

    # --- Training loop ---
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss, train_acc, train_bal_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_bal_acc = validate_epoch(model, val_loader, criterion, device)

        writer.add_scalar(f'Loss/train_fold_{args.fold}', train_loss, epoch)
        writer.add_scalar(f'Accuracy/train_fold_{args.fold}', train_acc, epoch)
        writer.add_scalar(f'Balanced_Accuracy/train_fold_{args.fold}', train_bal_acc, epoch)
        writer.add_scalar(f'Loss/val_fold_{args.fold}', val_loss, epoch)
        writer.add_scalar(f'Accuracy/val_fold_{args.fold}', val_acc, epoch)
        writer.add_scalar(f'Balanced_Accuracy/val_fold_{args.fold}', val_bal_acc, epoch)

        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Bal Acc: {train_bal_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Bal Acc: {val_bal_acc:.4f}')

        # --- Save best model ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)

    writer.close()
    print('Training complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for the CNN model")
    parser.add_argument('--data_path', type=Path, default='/data/IDLab/aar_foundation_models/training_snapshots/finetuning/horsing_around', help='Path to the finetuning data directory')
    parser.add_argument('--output_dir', type=Path, default='logs/cnn_training', help='Directory for logs and checkpoints')
    parser.add_argument('--fold', type=int, default=1, help='Fold to use for training and validation')
    parser.add_argument('--max_samples', type=int, default=50, help='Maximum number of samples to use for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()
    main(args)

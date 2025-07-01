import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import sys
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, balanced_accuracy_score

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.cnn import CNN
from utils import subsample_per_class


def load_data(data_path: Path, fold: int):
    fold_dir = data_path / f'fold_{fold}'
    X_train = np.load(fold_dir / 'X_train.npy')
    y_train = np.load(fold_dir / 'y_train.npy', allow_pickle=True)
    X_val = np.load(fold_dir / 'X_val.npy')
    y_val = np.load(fold_dir / 'y_val.npy', allow_pickle=True)

    X_train = np.swapaxes(X_train, 1, 2)
    y_train = np.array([l.decode('utf-8') for l in y_train])
    X_val = np.swapaxes(X_val, 1, 2)
    y_val = np.array([l.decode('utf-8') for l in y_val])

    return X_train, y_train, X_val, y_val


def apply_label_mapping(X, y, label_mapping):
    # Create a mask for samples that have a label in the mapping
    mask = np.array([label in label_mapping for label in y], dtype=bool)
    
    # Filter X and y using the mask
    X_filtered = X[mask]
    y_filtered = np.array([label_mapping[label] for label in y[mask]])

    return X_filtered, y_filtered

def main(args):
    # Load mandatory label mapping
    label_mapping_path = args.data_path / 'label_mapping.json'
    if not label_mapping_path.exists():
        raise FileNotFoundError(f"Mandatory 'label_mapping.json' not found in {args.data_path}")
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)

    # Load data
    X_train, y_train, X_val, y_val = load_data(args.data_path, args.fold)

    # Apply label mapping and filtering
    X_train, y_train = apply_label_mapping(X_train, y_train, label_mapping)
    X_val, y_val = apply_label_mapping(X_val, y_val, label_mapping)

    # Subsample the training data if max_samples is set
    if args.max_samples is not None:
        X_train, y_train = subsample_per_class(X_train, y_train, args.max_samples, args.data_path, args.fold)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    n_classes = len(label_encoder.classes_)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train_encoded).long())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val_encoded).long())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

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
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir = args.output_dir / args.data_path.name / f"fold_{args.fold}" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    model_path = log_dir / "model.pt"

    # Save config
    config_path = log_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4, default=str)

    print(f"Training CNN on fold {args.fold} of {args.data_path.name}")
    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of validation samples: {len(X_val)}")
    print(f"Number of classes: {n_classes}")
    print(f"Logs and checkpoints will be saved to: {log_dir}")

    # --- Training loop ---
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        all_train_preds = []
        all_train_labels = []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(batch_y.cpu().numpy())

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        train_balanced_accuracy = balanced_accuracy_score(all_train_labels, all_train_preds)

        writer.add_scalar(f'Loss/train_fold_{args.fold}', avg_train_loss, epoch)
        writer.add_scalar(f'Accuracy/train_fold_{args.fold}', train_accuracy, epoch)
        writer.add_scalar(f'Balanced_Accuracy/train_fold_{args.fold}', train_balanced_accuracy, epoch)

        # --- Validation loop ---
        model.eval()
        total_val_loss = 0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(batch_y.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        val_balanced_accuracy = balanced_accuracy_score(all_val_labels, all_val_preds)

        writer.add_scalar(f'Loss/val_fold_{args.fold}', avg_val_loss, epoch)
        writer.add_scalar(f'Accuracy/val_fold_{args.fold}', val_accuracy, epoch)
        writer.add_scalar(f'Balanced_Accuracy/val_fold_{args.fold}', val_balanced_accuracy, epoch)

        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train Bal Acc: {train_balanced_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val Bal Acc: {val_balanced_accuracy:.4f}')

        # --- Save best model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
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

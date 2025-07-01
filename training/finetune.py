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

from models.transformer import PatchedTransformerEncoderStack, ClassificationHead, SelfSupervisedBackbone
from utils import patchify, unpatchify, subsample_per_class


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
    mask = np.zeros_like(y, dtype=bool)
    for i, l in enumerate(y):
        if l in label_mapping:
            mask[i] = True
    
    X_filtered = X[mask]
    y_filtered = np.array([label_mapping[l] for l in y[mask]])

    return X_filtered, y_filtered


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finetuning script for the AAR foundation model")
    parser.add_argument('--data_path', type=Path, default='/data/IDLab/aar_foundation_models/training_snapshots/finetuning/horsing_around', help='Path to the finetuning data directory')
    parser.add_argument('--output_dir', type=Path, default='logs/finetuning', help='Directory for logs and checkpoints')
    parser.add_argument('--fold', type=int, default=0, help='Fold to use for training and validation')
    parser.add_argument('--max_samples', type=int, default=400, help='Maximum number of samples to use for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--pretrained_model_path', type=Path, required=False, help='Path to pretrained model weights for the transformer')
    args = parser.parse_args()

    args.pretrained_model_path = Path('/home/timodw/IDLab/aar_foundation_model/logs/pretraining/20250625091554')
    
    # Load mandatory label mapping
    label_mapping_path = args.data_path / 'label_mapping.json'
    if not label_mapping_path.exists():
        raise FileNotFoundError(f"Mandatory 'label_mapping.json' not found in {args.data_path}")
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)

    # Load pretrained config to get model parameters
    config_path = args.pretrained_model_path / 'config.json'
    with open(config_path, 'r') as f:
        pretrained_config = json.load(f)
    
    d_embedding = pretrained_config['d_embedding']
    n_layers = pretrained_config['n_layers']
    patch_size = pretrained_config['patch_size']
    transformer_dropout = pretrained_config['transformer_dropout']

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
    seq_len = X_train.shape[2]
    n_patches = seq_len // patch_size
    n_modalities = X_train.shape[1]

    transformer = PatchedTransformerEncoderStack(
        n_patches=n_patches,
        patch_size=patch_size,
        n_modalities=n_modalities,
        d_embedding=d_embedding,
        n_layers=n_layers,
        transformer_dropout=transformer_dropout
    ).to(device)
    classification_head = ClassificationHead(
        n_modalities=n_modalities,
        d_embedding=d_embedding,
        n_classes=n_classes
    ).to(device)

    # Load pretrained weights, freeze transformer, and setup optimizer
    print(f"Loading pretrained weights from: {args.pretrained_model_path}")
    pretrained_state_dict = torch.load(args.pretrained_model_path / 'model.pt', map_location=device)
    
    transformer_weights = {
        k.replace('transformer.', '', 1): v 
        for k, v in pretrained_state_dict.items() 
        if k.startswith('transformer.')
    }
    transformer.load_state_dict(transformer_weights)

    print("Freezing transformer weights for fine-tuning.")
    for param in transformer.parameters():
        param.requires_grad = False
    # transformer.transformer.layers[-1].requires_grad = True
    
    optimizer = optim.Adam([
        {'params': classification_head.parameters()},
        {'params': transformer.parameters()}
    ], lr=args.learning_rate)
    
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

    print(f"Finetuning on fold {args.fold} of {args.data_path.name}")
    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of validation samples: {len(X_val)}")
    print(f"Number of classes: {n_classes}")
    print(f"Logs and checkpoints will be saved to: {log_dir}")

    # --- Training loop ---
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        classification_head.train()
        total_train_loss = 0
        all_train_preds = []
        all_train_labels = []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            X_patched = patchify(batch_X, n_patches)

            optimizer.zero_grad()
            z = transformer(X_patched)
            outputs = classification_head(z)
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
        classification_head.eval()
        total_val_loss = 0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                X_patched = patchify(batch_X,n_patches)
                z = transformer(X_patched)
                outputs = classification_head(z)
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
            torch.save(classification_head.state_dict(), model_path)

    writer.close()
    print('Finetuning complete.')

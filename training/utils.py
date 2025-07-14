import torch
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class NoamOpt:
    
    def __init__(self, d_embedding, factor, warmup, optimizer):
        self.d_embedding = d_embedding 
        self.factor = factor
        self.warmup = warmup
        self.optimizer = optimizer
        self._step = 0
        self._rate = 0

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        if step is None:
            step = self._step
        if step == 0:
            return 0
        return self.factor * (self.d_embedding ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()


def patchify(X, n_patches):
    # X: (batch_size, n_modalities, sequence_length)
    patch_size = X.shape[-1] // n_patches
    return X.view(X.shape[0], X.shape[1], n_patches, patch_size)


def unpatchify(X_patched):
    # X_patched: (n_modalities, n_patches, patch_size)
    n_modalities, n_patches, patch_size = X_patched.shape
    return X_patched.reshape(n_modalities, n_patches * patch_size)


def get_mask(shape, masking_ratio, device):
    # shape: (batch_size, n_modalities, n_patches)
    len_keep = int(shape[2] * (1 - masking_ratio))
    noise = torch.rand(shape, device=device)
    ids_shuffle = torch.argsort(noise, dim=2)
    ids_restore = torch.argsort(ids_shuffle, dim=2)
    ids_keep = ids_shuffle[:, :, :len_keep]
    mask = torch.ones(shape, device=device)
    mask[:, :, :len_keep] = 0
    mask = torch.gather(mask, dim=2, index=ids_restore)
    return mask.bool()


def subsample_per_class(X, y, max_samples, data_path, fold):
    indices_path = data_path / f'fold_{fold}' / f'subsample_indices_{max_samples}.npy'
    
    if indices_path.exists():
        print(f"Loading existing subsample indices from {indices_path}")
        indices = np.load(indices_path)
    else:
        print(f"Generating new subsample indices for {max_samples} samples per class.")
        unique_classes = np.unique(y)
        indices = []
        for cls in unique_classes:
            class_indices = np.where(y == cls)[0]
            if len(class_indices) > max_samples:
                selected_indices = np.random.choice(class_indices, max_samples, replace=False)
            else:
                selected_indices = class_indices
            indices.extend(selected_indices)
        
        indices = np.array(indices)
        np.save(indices_path, indices)
        print(f"Saved subsample indices to {indices_path}")

    return X[indices], y[indices]


def load_finetuning_data(data_path: Path, fold: int):
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


def prepare_dataloaders(X_train, y_train, X_val, y_val, batch_size):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    n_classes = len(label_encoder.classes_)

    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train_encoded).long())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val_encoded).long())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, n_classes


def setup_logging(output_dir, data_path_name, fold):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir = output_dir / data_path_name / f"fold_{fold}" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    model_path = log_dir / "model.pt"
    return writer, log_dir, model_path


def save_config(args, log_dir):
    config_path = log_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4, default=str)


def train_epoch(model, dataloader, optimizer, criterion, device, patchify_func=None, n_patches=None):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        if patchify_func:
            batch_X = patchify_func(batch_X, n_patches)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, balanced_accuracy


def validate_epoch(model, dataloader, criterion, device, patchify_func=None, n_patches=None):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            if patchify_func:
                batch_X = patchify_func(batch_X, n_patches)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, balanced_accuracy


def setup_pretrain_logging(output_dir):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir = output_dir / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    model_path = log_dir / "model.pt"
    return writer, log_dir, model_path


def load_pretraining_data(data_path: Path, input_mode: str):
    X_train_path = data_path / 'X_train.npy'
    X_val_path = data_path / 'X_val.npy'
    if not X_train_path.exists() or not X_val_path.exists():
        raise FileNotFoundError(f"Pretraining data not found in {data_path}. Please run data preparation.")

    X_train = np.load(X_train_path)
    X_val = np.load(X_val_path)

    # Replace NaNs and transpose
    X_train = np.nan_to_num(X_train, nan=0.0).transpose(0, 2, 1)
    X_val = np.nan_to_num(X_val, nan=0.0).transpose(0, 2, 1)

    if input_mode == 'single':
        print("Using single-modality input mode. Reshaping data.")
        n_train, n_modalities_train, seq_len_train = X_train.shape
        X_train = X_train.reshape(n_train * n_modalities_train, 1, seq_len_train)
        n_val, n_modalities_val, seq_len_val = X_val.shape
        X_val = X_val.reshape(n_val * n_modalities_val, 1, seq_len_val)
    
    return X_train, X_val
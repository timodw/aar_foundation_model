import torch
import numpy as np
import argparse
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from numpy.typing import NDArray
from torch import Tensor


class NoamOpt:
    """
    A wrapper for an optimizer that implements the Noam learning rate schedule,
    as described in "Attention is All You Need".
    """
    
    def __init__(self, d_embedding: int, factor: float, warmup: int, optimizer: torch.optim.Optimizer):
        """
        Initializes the NoamOpt scheduler.

        Args:
            d_embedding (int): The dimensionality of the model's embedding layer.
                               Used in the learning rate calculation.
            factor (float): A scaling factor for the learning rate.
            warmup (int): The number of warmup steps for the learning rate scheduler.
            optimizer (torch.optim.Optimizer): The optimizer object to wrap.
        """
        self.d_embedding = d_embedding 
        self.factor = factor
        self.warmup = warmup
        self.optimizer = optimizer
        self._step = 0
        self._rate = 0

    @property
    def param_groups(self):
        """
        Property to access the parameter groups of the wrapped optimizer.

        Returns:
            The parameter groups from the optimizer.
        """
        return self.optimizer.param_groups

    def step(self):
        """
        Performs a single optimization step, updating the learning rate according
        to the Noam schedule. Increments the internal step counter and updates
        the learning rate for all parameter groups before calling the
        optimizer's step function.
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step: Optional[int] = None) -> float:
        """
        Calculates the learning rate for a given step.

        Args:
            step (int, optional): The step for which to calculate the rate. If
                                  None, uses the internal step counter.

        Returns:
            float: The calculated learning rate.
        """
        if step is None:
            step = self._step
        if step == 0:
            return 0
        return self.factor * (self.d_embedding ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        """
        Clears the gradients of all optimized torch.Tensors.
        """
        self.optimizer.zero_grad()


def patchify(X: Tensor, n_patches: int) -> Tensor:
    """
    Reshapes a batch of sequences into patches.

    Args:
        X (Tensor): The input tensor of shape (batch_size, n_modalities,
                    sequence_length).
        n_patches (int): The number of patches to divide the sequence into.

    Returns:
        Tensor: The reshaped tensor of shape (batch_size, n_modalities,
                n_patches, patch_size).
    """
    patch_size = X.shape[-1] // n_patches
    return X.view(X.shape[0], X.shape[1], n_patches, patch_size)


def unpatchify(X_patched: Tensor) -> Tensor:
    """
    Reverses the patchify operation, reconstructing sequences from patches.

    Args:
        X_patched (Tensor): A tensor of patches with shape (n_modalities,
                            n_patches, patch_size).

    Returns:
        Tensor: The reconstructed sequence tensor of shape (n_modalities,
                sequence_length).
    """
    n_modalities, n_patches, patch_size = X_patched.shape
    return X_patched.reshape(n_modalities, n_patches * patch_size)


def get_mask(shape: Tuple[int, int, int], masking_ratio: float, device: torch.device) -> Tensor:
    """
    Generates a random boolean mask for a sequence of patches

    Args:
        shape (tuple): The shape of the tensor to be masked, typically
                       (batch_size, n_modalities, n_patches).
        masking_ratio (float): The fraction of patches to mask (e.g., 0.75 for
                               75% masking).
        device (torch.device): The device to create the mask tensor on.

    Returns:
        Tensor: A boolean tensor of the given shape where True indicates a
                masked position.
    """
    len_keep = int(shape[2] * (1 - masking_ratio))
    noise = torch.rand(shape, device=device)
    ids_shuffle = torch.argsort(noise, dim=2)
    ids_restore = torch.argsort(ids_shuffle, dim=2)
    mask = torch.ones(shape, device=device)
    mask[:, :, :len_keep] = 0
    mask = torch.gather(mask, dim=2, index=ids_restore)
    return mask.bool()


def subsample_per_class(X: NDArray, y: NDArray, max_samples: int, data_path: Path, fold: int, label_mapping_id: str) -> Tuple[NDArray, NDArray]:
    """
    Subsamples the training data to a maximum number of samples per class.

    To ensure reproducibility, it saves the indices of the subsampled data to a
    file. If the file already exists, it loads the indices from there instead
    of generating new ones.

    Args:
        X (NDArray): The input data features.
        y (NDArray): The corresponding labels.
        max_samples (int): The maximum number of samples per class.
        data_path (Path): The root directory of the dataset.
        fold (int): The current fold number, used for naming the indices file.
        label_mapping_id (str): An identifier for the label mapping, used for
                                naming the indices file.

    Returns:
        Tuple[NDArray, NDArray]: The subsampled data (X) and labels (y).
    """
    indices_path = data_path / f'fold_{fold}' / f'subsample_indices_{max_samples}_{label_mapping_id}.npy'
    
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


def load_finetuning_data(data_path: Path, fold: int) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Loads the training and validation data for a specific fold from .npy files.

    Args:
        data_path (Path): The directory containing the fold subdirectories.
        fold (int): The fold number to load data for.

    Returns:
        Tuple[NDArray, NDArray, NDArray, NDArray]: A tuple containing
        X_train, y_train, X_val, y_val.
    """
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


def apply_label_mapping(X: NDArray, y: NDArray, label_mapping: dict) -> Tuple[NDArray, NDArray]:
    """
    Filters the data based on a label mapping and applies the mapping.

    Samples whose labels are not in the mapping are removed. The remaining
    labels are replaced with the values from the mapping dictionary.

    Args:
        X (NDArray): The input data features.
        y (NDArray): The corresponding labels.
        label_mapping (dict): A dictionary mapping original labels to new labels.

    Returns:
        Tuple[NDArray, NDArray]: The filtered and mapped data (X) and labels (y).
    """
    mask = np.zeros_like(y, dtype=bool)
    for i, l in enumerate(y):
        if l in label_mapping:
            mask[i] = True
    
    X_filtered = X[mask]
    y_filtered = np.array([label_mapping[l] for l in y[mask]])

    return X_filtered, y_filtered


def prepare_dataloaders(X_train: NDArray, y_train: NDArray, X_val: NDArray, y_val: NDArray, batch_size: int) -> Tuple[DataLoader, DataLoader, int]:
    """
    Creates PyTorch DataLoaders for training and validation sets.

    It also encodes the string labels into integer format for training.

    Args:
        X_train (NDArray): Training data features.
        y_train (NDArray): Training data labels.
        X_val (NDArray): Validation data features.
        y_val (NDArray): Validation data labels.
        batch_size (int): The batch size for the DataLoaders.

    Returns:
        Tuple[DataLoader, DataLoader, int]: A tuple containing the training
        DataLoader, validation DataLoader, and the number of unique classes.
    """
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    n_classes = len(label_encoder.classes_)

    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train_encoded).long())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val_encoded).long())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, n_classes


def setup_logging(output_dir: Path, data_path_name: Optional[str] = None, fold: Optional[int] = None) -> Tuple[SummaryWriter, Path, Path]:
    """
    Sets up the logging directory for training experiments.

    Creates a timestamped directory. For fine-tuning, includes dataset name 
    and fold number in the path structure. For pre-training, creates a simple
    timestamped directory.

    Args:
        output_dir (Path): The root directory for logs.
        data_path_name (str, optional): The name of the dataset directory.
                                        Required for fine-tuning experiments.
        fold (int, optional): The current fold number. Required for 
                              fine-tuning experiments.

    Returns:
        Tuple[SummaryWriter, Path, Path]: A tuple containing the TensorBoard
        SummaryWriter, the log directory path, and the model checkpoint path.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    if data_path_name is not None and fold is not None:
        log_dir = output_dir / data_path_name / f"fold_{fold}" / timestamp
    else:
        log_dir = output_dir / timestamp
    
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    model_path = log_dir / "model.pt"
    return writer, log_dir, model_path


def save_config(args: argparse.Namespace, log_dir: Path):
    """
    Saves the experiment configuration to a JSON file.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
        log_dir (Path): The directory where the config.json file will be saved.
    """
    config_path = log_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4, default=str)


def train_classifier_epoch(model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, device: torch.device, patchify_func: Optional[callable] = None, n_patches: Optional[int] = None) -> Tuple[float, float, float]:
    """
    Performs one epoch of training for a classification model.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (DataLoader): The DataLoader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to train on.
        patchify_func (callable, optional): A function to patchify the input data.
                                            Required for transformer models.
        n_patches (int, optional): The number of patches. Required if
                                   patchify_func is provided.

    Returns:
        Tuple[float, float, float]: The average training loss, accuracy, and
        balanced accuracy for the epoch.
    """
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


def validate_classifier_epoch(model: torch.nn.Module, dataloader: DataLoader, criterion: torch.nn.Module, device: torch.device, patchify_func: Optional[callable] = None, n_patches: Optional[int] = None) -> Tuple[float, float, float]:
    """
    Performs one epoch of validation for a classification model.

    Args:
        model (torch.nn.Module): The model to validate.
        dataloader (DataLoader): The DataLoader for the validation data.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to run validation on.
        patchify_func (callable, optional): A function to patchify the input data.
                                            Required for transformer models.
        n_patches (int, optional): The number of patches. Required if
                                   patchify_func is provided.

    Returns:
        Tuple[float, float, float]: The average validation loss, accuracy, and
        balanced accuracy for the epoch.
    """
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


def print_label_distribution(y: NDArray, dataset_name: str = "training"):
    """
    Prints the label distribution of a dataset.

    Args:
        y (NDArray): An array of labels.
        dataset_name (str): The name of the dataset (e.g., "training",
                            "validation") to include in the printout.
    """
    unique, counts = np.unique(y, return_counts=True)
    print(f"Label distribution of the {dataset_name} set:")
    for label, count in zip(unique, counts):
        print(f"  - {label}: {count}")


def log_reconstruction_to_tensorboard(writer: SummaryWriter, model: torch.nn.Module, sample: Tensor, mask: Tensor, epoch: int, n_patches: int):
    """
    Creates plots comparing original, masked input, and reconstructed signals
    for each modality and logs them to TensorBoard.

    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter for logging.
        model (torch.nn.Module): The trained model for generating reconstructions.
        sample (Tensor): Input sample tensor to visualize.
        mask (Tensor): Boolean mask indicating which patches were masked.
        epoch (int): Current epoch number for logging.
        n_patches (int): Number of patches to divide sequences into.
    """
    model.eval()
    with torch.no_grad():
        sample_patched = patchify(sample, n_patches)
        sample_masked = sample_patched.clone()
        sample_masked[mask.unsqueeze(-1).expand_as(sample_masked)] = 0
        reconstruction_patched = model(sample_masked)

        original_signal = unpatchify(sample_patched.squeeze(0))
        masked_signal = unpatchify(sample_masked.squeeze(0))
        reconstructed_signal = unpatchify(reconstruction_patched.squeeze(0))

        fig, axes = plt.subplots(original_signal.shape[0], 1, figsize=(15, 5 * original_signal.shape[0]), sharex=True, squeeze=False)
        axes = axes.flatten()
        fig.suptitle(f'Epoch {epoch+1} Reconstruction')
        for i, ax in enumerate(axes):
            ax.plot(original_signal[i].cpu().numpy(), label='Original')
            ax.plot(masked_signal[i].cpu().numpy(), label='Masked Input', linestyle='--')
            ax.plot(reconstructed_signal[i].cpu().numpy(), label='Reconstruction')
            ax.set_ylabel(f'Modality {i+1}')
            ax.legend()
        
        axes[-1].set_xlabel('Time step')
        writer.add_figure('Reconstruction', fig, global_step=epoch)
        plt.close(fig)


def load_pretraining_data(data_path: Path, input_mode: str) -> Tuple[NDArray, NDArray]:
    """
    Loads the pre-training data from .npy files.

    Handles NaN values and transposes the data for model consumption. Also
    reshapes the data for single-modality input if specified.

    Args:
        data_path (Path): The directory containing X_train.npy and X_val.npy.
        input_mode (str): The input mode, either 'multi' or 'single'.

    Returns:
        Tuple[NDArray, NDArray]: A tuple containing the training and validation
        data arrays (X_train, X_val).
    """
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


def get_predictions(model: torch.nn.Module, dataloader: DataLoader, device: torch.device, patchify_func: Optional[callable] = None, n_patches: Optional[int] = None) -> Tuple[NDArray, NDArray]:
    """
    Gets predictions and true labels from a model on a given dataset.
    
    Args:
        model (torch.nn.Module): The model to get predictions from.
        dataloader (DataLoader): The DataLoader for the data.
        device (torch.device): The device to run inference on.
        patchify_func (callable, optional): A function to patchify the input data.
                                            Required for transformer models.
        n_patches (int, optional): The number of patches. Required if
                                   patchify_func is provided.
    
    Returns:
        Tuple[NDArray, NDArray]: Arrays of true labels and predictions.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            if patchify_func:
                batch_X = patchify_func(batch_X, n_patches)
            
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds)
import torch
import numpy as np
from pathlib import Path


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
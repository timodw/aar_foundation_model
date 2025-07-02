import torch
import numpy as np

from typing import Optional
from torch import Tensor


class LinearEmbedding(torch.nn.Module):

    def __init__(self, n_features: int, d_embedding: int):
        super().__init__()
        self.layer = torch.nn.Linear(n_features, d_embedding, bias=False)

    def forward(self, X: Tensor) -> Tensor:
        return self.layer(X)


class LearnablePositionalEncoding(torch.nn.Module):

    def __init__(self, n_input: int, d_embedding: int):
        super().__init__()
        initial_value = torch.nn.init.uniform_(torch.empty((1, n_input, d_embedding)), -0.02, 0.02)
        self.positional_encoding = torch.nn.Parameter(initial_value, requires_grad=True)

    def __radd__(self, other: Tensor) -> Tensor:
        return other + self.positional_encoding


class ReconstructionHead(torch.nn.Module):
    
    def __init__(self, d_embedding: int, patch_size: int):
        super().__init__()
        self.weight_matrix = torch.nn.Linear(d_embedding, patch_size, bias=False)

    def forward(self, X: Tensor) -> Tensor:
        return self.weight_matrix(X)


class ClassificationHead(torch.nn.Module):

    def __init__(self, n_modalities: int, d_embedding: int, n_patches: int, n_classes: int, dropout: float=0.0, avg_pool=False):
        super().__init__()
        self.avg_pool = avg_pool
        layers = [
            torch.nn.Flatten(),
            torch.nn.Dropout(),
            torch.nn.Linear(n_modalities * d_embedding, n_classes)
        ]
        if self.avg_pool:
            layers.insert(0, torch.nn.AvgPool1d(n_patches))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, X: Tensor) -> Tensor:
        # X: (B, n_modalities, d_embedding, n_patches)
        # out: (B, n_classes)
        if not self.avg_pool:
            X = X[:, :, -1, :] # Use last embedding as input for classification
        return self.layers(X)


class SelfSupervisedBackbone(torch.nn.Module):

    def __init__(self, transformer: torch.nn.Module, self_supervised_head: torch.nn.Module):
        super().__init__()
        self.transformer = transformer
        self.self_supervised_head = self_supervised_head

    def forward(self, X: Tensor) -> Tensor:
        z: Tensor = self.transformer(X)
        X_hat: Tensor = self.self_supervised_head(z)
        return X_hat


class PatchedTransformerEncoderStack(torch.nn.Module):

    def __init__(self, n_patches: int, patch_size: int, n_modalities: int, d_embedding=128, n_layers=4, transformer_dropout=0.1):
        super().__init__()

        self.n_heads = d_embedding // 64
        self.d_feedforward = d_embedding * 4
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.n_modalities = n_modalities
        self.d_embedding = d_embedding

        self.embedding = LinearEmbedding(self.patch_size, self.d_embedding)
        self.positional_encoding = LearnablePositionalEncoding(self.n_patches, self.d_embedding)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_embedding, nhead=self.n_heads, dim_feedforward=self.d_feedforward,
            batch_first=True, dropout=transformer_dropout, norm_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                                       norm=torch.nn.LayerNorm(d_embedding))

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Number of parameters: {n_params:,}")


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: [batch_size, n_modalities, n_patches, patch_size]
        batch_size, n_modalities, n_patches, patch_size = X.shape

        X_emb = self.embedding(X) # [batch_size, n_modalities, n_patches, d_embedding]
        X_emb = X_emb.view(batch_size * n_modalities, n_patches, self.d_embedding) # [batch_size * n_modalities, n_patches, d_embedding]
        X_w_pos_enc = X_emb + self.positional_encoding # [batch_size * n_modalities, n_patches, d_embedding]
        z = self.transformer(X_w_pos_enc) # [batch_size * n_modalities, n_patches, d_embedding]
        z = z.view(batch_size, n_modalities, n_patches, self.d_embedding) # [batch_size, n_modalities, n_patches, d_embedding]

        return z

if __name__ == '__main__':
    transformer = PatchedTransformerEncoderStack(n_patches=32, patch_size=32, n_modalities=3)
    reconstruction_head = ReconstructionHead(d_embedding=128, patch_size=32)
    self_supervised_model = SelfSupervisedBackbone(transformer=transformer, self_supervised_head=reconstruction_head)
    print(self_supervised_model)

    X = torch.empty(64, 3, 32, 32)
    print(self_supervised_model(X).shape)
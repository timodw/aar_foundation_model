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

class PatchedTransformerEncoderStack(torch.nn.Module):

    def __init__(self, n_patches: int, n_modalities: int, d_embedding=128, n_layers=4, transformer_dropout=0.1):
        super().__init__()

        self.n_heads = d_embedding // 64
        self.d_feedforward = d_embedding * 4
        self.n_patches = n_patches
        self.n_modalities = n_modalities
        self.d_embedding = d_embedding

        self.embedding = LinearEmbedding(self.n_modalities, self.d_embedding)
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

        # Apply linear embedding elementwise
        X_emb = self.embedding(X) # X_emb: [batch_size, n_modalities, n_patches, d_embedding]

        # Apply channel independence
        X_emb = X_emb.view(batch_size * n_modalities, n_patches, self.d_embedding) # X_emb: [batch_size * n_modalities, n_patches, d_embedding]

        # Apply positional_encoding
        X_w_pos_enc = X_emb + self.positional_encoding # X_w_pos_enc: [batch_size * n_modalities, n_patches, d_embedding]

        # Pass trough Transformer stack
        z = self.transformer(X_w_pos_enc) # z: [batch_size * n_modalities, n_patches, d_embedding]

        # Revert to original dimensions
        z = z.view(batch_size, n_modalities, n_patches, self.d_embedding) # z: [batch_size, n_modalities, n_patches, d_embedding]

        return z

if __name__ == '__main__':
    model = PatchedTransformerEncoderStack(32, 32)
    print(model)

    X = torch.empty(64, 3, 32, 32)
    print(model(X).shape)
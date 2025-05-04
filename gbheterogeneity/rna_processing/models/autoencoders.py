import torch

import rna_processing.models.common as common

from typing import List


class LinearBlock(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout_prob: float = None) -> None:
        super().__init__()

        if dropout_prob:
            self.linear_layer = torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim),
                                                    torch.nn.ReLU(),
                                                    torch.nn.BatchNorm1d(output_dim),
                                                    torch.nn.Dropout(dropout_prob))
        else:
            self.linear_layer = torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim),
                                                    torch.nn.ReLU(),
                                                    torch.nn.BatchNorm1d(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_layer(x)
        return x


class RNAEncoder(torch.nn.Module):
    def __init__(self, layer_dims: List, dropout_probs: List) -> None:
        super().__init__()

        encoding_layers = []
        num_layers = len(layer_dims[:-2])
        for i in range(num_layers):
            encoding_layers.append(LinearBlock(layer_dims[i], layer_dims[i + 1], dropout_probs[i]))

        self.encoding_layers = torch.nn.ModuleList(encoding_layers)
        self.last_layer = torch.nn.Linear(layer_dims[-2], layer_dims[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoding_layers:
            x = layer(x)

        x = self.last_layer(x)
        return x


class RNADecoder(torch.nn.Module):
    def __init__(self, layer_dims: List) -> None:
        super().__init__()

        decoding_layers = []
        num_layers = len(layer_dims[:-2])
        for i in range(num_layers):
            decoding_layers.append(LinearBlock(layer_dims[i], layer_dims[i + 1]))

        self.decoding_layers = torch.nn.ModuleList(decoding_layers)
        self.last_layer = torch.nn.Linear(layer_dims[-2], layer_dims[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.decoding_layers:
            x = layer(x)

        x = self.last_layer(x)
        return x


class AutoencoderRNA(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List, latent_dim: int, dropout_prob: float,**kwargs) -> None:
        super().__init__()

        # Hyperparams
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoding_layer_dims = [input_dim] + hidden_dims + [latent_dim]
        self.decoding_layer_dims = list(reversed(self.encoding_layer_dims))
        self.dropout_probs = [dropout_prob] * len(hidden_dims)

        # Layers
        self.encoder = RNAEncoder(self.encoding_layer_dims, self.dropout_probs)
        self.decoder = RNADecoder(self.decoding_layer_dims)

        # Info
        print("Number of autoencoder parameters: {}".format(common.count_parameters(self)))

    def forward(self, rna_seq: torch.Tensor) -> torch.Tensor:
        rna_feat = self.encoder(rna_seq)
        rna_rec = self.decoder(rna_feat)

        return rna_rec


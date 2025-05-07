import torch
import functools

import torch.nn.functional as F
import timm.layers as layers

from typing import Callable, Dict
from gbheterogeneity.utils.pytorch.parameter_getter import count_parameters


class Mlp(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: Callable = torch.nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()

        # Hyperparameters
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Layers
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = torch.nn.Linear(hidden_features, out_features)
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()

        # Hyperparameters
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # Layers
        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

        # Attention maps/gradients
        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients: torch.Tensor) -> None:
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self) -> torch.Tensor:
        return self.attn_gradients

    def save_attention_map(self, attention_map: torch.Tensor) -> None:
        self.attention_map = attention_map

    def get_attention_map(self) -> torch.Tensor:
        return self.attention_map

    def forward(self, x: torch.Tensor, register_hook: bool = False) -> torch.Tensor:
        batch_size, num_tokens, embedding_dim = x.shape
        qkv_with_heads_shape = (
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            embedding_dim // self.num_heads,
        )
        qkv = self.qkv(x).reshape(*qkv_with_heads_shape).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # q, k and v tensors of shape (batch_size, num_heads, num_tokens, dim_head)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, embedding_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        num_words: int = 24,
        drop_rate: float = 0.0,
        act_layer: Callable = torch.nn.GELU,
        norm_layer: Callable = torch.nn.LayerNorm,
    ) -> None:
        super().__init__()

        # Hyperparameters
        mlp_hidden_dim = int(dim * mlp_ratio)

        # Global patches
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, num_words + 1, dim))
        self.pos_drop = torch.nn.Dropout(p=drop_rate)

        # Layers
        self.norm = norm_layer(dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # Initialize weights
        layers.trunc_normal_(self.pos_embed, std=0.02)
        layers.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: torch.nn.Module) -> None:
        if isinstance(m, torch.nn.Linear):
            layers.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, register_hook: bool = False) -> torch.Tensor:
        # Compute attention
        batch_size = x.shape[0]

        # Add embedding CLS
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add position information
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)

        # Attention block
        x = x + self.attn(self.norm1(x), register_hook=register_hook)
        x = x + self.mlp(self.norm2(x))

        # Normalize
        x = self.norm(x)
        return x


class RNAAttentionEncoder(torch.nn.Module):
    def __init__(
        self,
        genes_per_cluster: Dict,
        embedding_dim: int,
        dropout_prob: float,
        num_heads: int,
        **kwargs,
    ) -> None:
        super().__init__()

        # Hyperparams
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout_prob
        self.num_heads = num_heads
        self.genes_per_cluster = genes_per_cluster

        # Layers
        self.embedder = self.build_embedder()
        self.attention_net = self.build_attention_net()

        # Info
        print(
            "Number of attention-based model parameters: {}".format(
                count_parameters(self)
            )
        )

    def build_embedder(self) -> torch.nn.Module:
        embedder_dict = {}
        for key, value in self.genes_per_cluster.items():
            if isinstance(value, list):
                input_dim = len(value)
            else:
                input_dim = int(value)

            embedding_layer = torch.nn.Sequential(
                torch.nn.Linear(input_dim, self.embedding_dim),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(self.embedding_dim),
                torch.nn.Dropout(self.dropout_prob),
            )
            embedder_dict[key] = embedding_layer

        embedder = torch.nn.ModuleDict(embedder_dict)
        return embedder

    def build_attention_net(self) -> torch.nn.Module:
        norm_layer = functools.partial(torch.nn.LayerNorm, eps=1e-6)
        attention_net = Block(
            dim=self.embedding_dim,
            num_heads=self.num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=norm_layer,
        )

        return attention_net

    def encode(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        sequence_list = []
        for key, net in self.embedder.items():
            input = x[key]
            output = net(input)
            sequence_list.append(output)
        x = torch.stack(sequence_list, dim=1)
        x = self.attention_net(x)

        return x

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        representations = self.encode(x)

        return representations


class AttentionRNA(torch.nn.Module):
    def __init__(
        self,
        genes_per_cluster: Dict,
        embedding_dim: int,
        dropout_prob: float,
        num_heads: int,
        projection_dim: int,
        **kwargs,
    ) -> None:
        super().__init__()

        # Hyperparams
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout_prob
        self.num_heads = num_heads
        self.genes_per_cluster = genes_per_cluster

        # Layers
        self.embedder = self.build_embedder()
        self.attention_net = self.build_attention_net()
        self.decoder = self.build_decoder()
        self.proj_head = torch.nn.Linear(embedding_dim, projection_dim)

        # Info
        print(
            "Number of attention-based model parameters: {}".format(
                count_parameters(self)
            )
        )

    def build_embedder(self) -> torch.nn.Module:
        embedder_dict = {}
        for key, value in self.genes_per_cluster.items():
            if isinstance(value, list):
                input_dim = len(value)
            else:
                input_dim = int(value)

            embedding_layer = torch.nn.Sequential(
                torch.nn.Linear(input_dim, self.embedding_dim),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(self.embedding_dim),
                torch.nn.Dropout(self.dropout_prob),
            )
            embedder_dict[key] = embedding_layer

        embedder = torch.nn.ModuleDict(embedder_dict)
        return embedder

    def build_attention_net(self) -> torch.nn.Module:
        norm_layer = functools.partial(torch.nn.LayerNorm, eps=1e-6)
        attention_net = Block(
            dim=self.embedding_dim,
            num_heads=self.num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=norm_layer,
        )

        return attention_net

    def build_decoder(self) -> torch.nn.Module:
        decoder_dict = {}
        for key, value in self.genes_per_cluster.items():
            if isinstance(value, list):
                input_dim = len(value)
            else:
                input_dim = int(value)
            decoder_dict[key] = torch.nn.Linear(self.embedding_dim, input_dim)

        decoder = torch.nn.ModuleDict(decoder_dict)
        return decoder

    def encode(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        sequence_list = []
        for key, net in self.embedder.items():
            input = x[key]
            output = net(input)
            sequence_list.append(output)
        x = torch.stack(sequence_list, dim=1)
        x = self.attention_net(x)

        return x

    def decode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        reconstruction = {}
        for i, (key, net) in enumerate(self.decoder.items()):
            output = net(x[:, i + 1, :])
            reconstruction[key] = output

        return reconstruction

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj_head(x[:, 0, :]), dim=-1)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        representations = self.encode(x)
        reconstruction = self.decode(representations)
        projections = self.project(representations)

        if self.training:
            return reconstruction, projections
        else:
            return reconstruction, representations, projections


class AttentionRNASurvivalTime(RNAAttentionEncoder):
    def __init__(
        self,
        genes_per_cluster: Dict,
        embedding_dim: int,
        dropout_prob: float,
        num_heads: int,
        projection_dim: int,
        freeze_encoder: bool,
        **kwargs,
    ) -> None:
        super().__init__(
            genes_per_cluster=genes_per_cluster,
            embedding_dim=embedding_dim,
            dropout_prob=dropout_prob,
            num_heads=num_heads,
        )

        self.projection_dim = projection_dim
        self.survival_time_head = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, projection_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(projection_dim, 1),
        )

        if freeze_encoder:
            for param in self.embedder.parameters():
                param.requires_grad = False
            for param in self.attention_net.parameters():
                param.requires_grad = False

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        representations = self.encode(x)
        global_representation = representations[:, 0, :]
        predictions = self.survival_time_head(global_representation)

        return predictions

import torch
from typing import Callable


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


class Coattention(torch.nn.Module):
    def __init__(
        self,
        dim: int = 256,
        img_dim: int = 768,
        rna_dim: int = 128,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()

        # Hyperparameters
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        # Layers
        self.img_qkv = torch.nn.Linear(img_dim, dim * 3, bias=qkv_bias)
        self.rna_qkv = torch.nn.Linear(rna_dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.img_proj = torch.nn.Linear(dim, img_dim)
        self.rna_proj = torch.nn.Linear(dim, rna_dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.norm_img = torch.nn.LayerNorm(img_dim, eps=1e-6)
        self.norm_rna = torch.nn.LayerNorm(rna_dim, eps=1e-6)

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

    def forward(
        self, image: torch.Tensor, rna: torch.Tensor, register_hook: bool = False
    ) -> torch.Tensor:
        # Get (Q,K,V) on image
        batch_size, num_patches, _ = image.shape
        qkv_with_heads_shape = batch_size, num_patches, 3, self.num_heads, self.head_dim
        img_qkv = (
            self.img_qkv(image).reshape(*qkv_with_heads_shape).permute(2, 0, 3, 1, 4)
        )
        img_q, img_k, img_v = img_qkv[0], img_qkv[1], img_qkv[2]
        # img_q, img_k and img_v tensors of shape (batch_size, num_heads, num_patches + 1, dim_head)

        # Get (Q,K,V) on rna
        batch_size, num_clusters, _ = rna.shape
        qkv_with_heads_shape = (
            batch_size,
            num_clusters,
            3,
            self.num_heads,
            self.head_dim,
        )
        rna_qkv = (
            self.rna_qkv(rna).reshape(*qkv_with_heads_shape).permute(2, 0, 3, 1, 4)
        )
        rna_q, rna_k, rna_v = rna_qkv[0], rna_qkv[1], rna_qkv[2]
        # img_q, img_k and img_v tensors of shape (batch_size, num_heads, num_rna_clusters + 1, dim_head)

        # Compute attention rna-image
        attn_rna_image = (rna_q @ img_k.transpose(-2, -1)) * self.scale
        attn_rna_image = attn_rna_image.softmax(dim=-1)
        attn_rna_image = self.attn_drop(attn_rna_image)

        co_rna_image = (
            (attn_rna_image @ img_v)
            .transpose(1, 2)
            .reshape(batch_size, num_clusters, self.head_dim * self.num_heads)
        )
        co_rna_image = self.rna_proj(co_rna_image)
        co_rna_image = self.proj_drop(co_rna_image)
        co_rna = self.norm_rna(co_rna_image + rna)

        # Compute attention image-rna
        attn_image_rna = (img_q @ rna_k.transpose(-2, -1)) * self.scale
        attn_image_rna = attn_image_rna.softmax(dim=-1)
        attn_image_rna = self.attn_drop(attn_image_rna)

        if register_hook:
            self.save_attention_map(attn_image_rna)
            attn_image_rna.register_hook(self.save_attn_gradients)

        co_image_rna = (
            (attn_image_rna @ rna_v)
            .transpose(1, 2)
            .reshape(batch_size, num_patches, self.head_dim * self.num_heads)
        )
        co_image_rna = self.img_proj(co_image_rna)
        co_image_rna = self.proj_drop(co_image_rna)
        co_image = self.norm_img(co_image_rna + image)

        return co_image, co_rna


if __name__ == "__main__":
    print("Co-attention encoder")
    batch_size = 5
    rna_seq_size = 3
    image_seq_size = 4
    rna_emb_size = 10
    image_emb_size = 6

    rna_input = torch.rand(batch_size, rna_seq_size, rna_emb_size)
    image_input = torch.rand(batch_size, image_seq_size, image_emb_size)
    coatt = Coattention(
        dim=12, img_dim=image_emb_size, rna_dim=rna_emb_size, num_heads=2
    )
    co_image, co_rna = coatt(image_input, rna_input)
    print(rna_input.shape)
    print(image_input.shape)
    print(co_rna.shape)
    print(co_image.shape)

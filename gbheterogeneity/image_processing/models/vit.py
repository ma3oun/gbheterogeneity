import torch
import functools

import timm.models.vision_transformer as vision_t
import timm.models.layers as layers

from typing import Callable, Dict


class Mlp(torch.nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self,
                 in_features: int,
                 hidden_features: int = None,
                 out_features: int = None,
                 act_layer: Callable = torch.nn.GELU,
                 drop: float = 0.0) -> None:
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
    """Attention layer as used in Vision Transformer"""

    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0) -> None:
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
        qkv_with_heads_shape = batch_size, num_tokens, 3, self.num_heads, embedding_dim // self.num_heads
        qkv = self.qkv(x).reshape(*qkv_with_heads_shape).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # q, k and v tensors of shape (batch_size, num_heads, num_tokens, dim_head)

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
    """Attention block as used in Vision Transformer"""

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: int = 4.0,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 drop_path: float = 0.0,
                 act_layer: Callable = torch.nn.GELU,
                 norm_layer: Callable = torch.nn.LayerNorm) -> None:
        super().__init__()

        # Hyperparameters
        mlp_hidden_dim = int(dim * mlp_ratio)

        # Layers
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim=dim, 
                              num_heads=num_heads, 
                              qkv_bias=qkv_bias, 
                              qk_scale=qk_scale, 
                              attn_drop=attn_drop, 
                              proj_drop=drop)
        self.drop_path = layers.DropPath(drop_path) if drop_path > 0.0 else torch.nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor, register_hook: bool = False) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(torch.nn.Module):
    """Vision Transformer. A PyTorch implementation of
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`: https://arxiv.org/abs/2010.11929
    """

    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: int = 4.0,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_layer: Callable = None) -> None:
        """
        Args:
            img_size (int, tuple): input image size.
            patch_size (int, tuple): patch size.
            in_chans (int): number of input channels.
            embed_dim (int): embedding dimension.
            depth (int): depth of transformer.
            num_heads (int): number of attention heads.
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): enable bias for qkv if True.
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set.
            drop_rate (float): dropout rate.
            attn_drop_rate (float): attention dropout rate.
            drop_path_rate (float): stochastic depth rate.
            norm_layer: (torch.nn.Module): normalization layer.
        """
        super().__init__()

        # Hyperparameters
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or functools.partial(torch.nn.LayerNorm, eps=1e-6)

        # Layers
        # Image to patches
        self.patch_embed = vision_t.PatchEmbed(img_size=img_size, 
                                               patch_size=patch_size, 
                                               in_chans=in_chans, 
                                               embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = torch.nn.Dropout(p=drop_rate)
        # Attention blocks
        # Define drop_path for each block using stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        self.blocks = torch.nn.ModuleList([Block(dim=embed_dim,
                                                 num_heads=num_heads,
                                                 mlp_ratio=mlp_ratio,
                                                 qkv_bias=qkv_bias,
                                                 qk_scale=qk_scale,
                                                 drop=drop_rate,
                                                 attn_drop=attn_drop_rate,
                                                 drop_path=dpr[i],
                                                 norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)

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

    @torch.jit.ignore
    def no_weight_decay(self) -> Dict:
        return {"pos_embed", "cls_token"}

    def forward(self, x: torch.Tensor, register_blk: int = -1) -> torch.Tensor:
        # Image to patches
        batch_size = x.shape[0]
        x = self.patch_embed(x)

        # Add embedding CLS
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add position information
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)

        # Go through the attention blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x, register_blk == i)

        x = self.norm(x)
        return x


def interpolate_pos_embed(pos_embed_checkpoint: torch.Tensor, visual_encoder: torch.nn.Module) -> torch.Tensor:
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches**0.5)

    if orig_size != new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(pos_tokens, 
                                                     size=(new_size, new_size), 
                                                     mode="bicubic", 
                                                     align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print("Reshape position embedding from %d to %d" % (orig_size**2, new_size**2))

        return new_pos_embed
    else:
        return pos_embed_checkpoint

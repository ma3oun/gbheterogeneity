import torch
import functools

import torch.nn.functional as F

from .vit import VisionTransformer, interpolate_pos_embed

from typing import Dict


DEIT_URL = "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth"


class ImageEncoder(torch.nn.Module):
    def __init__(self, config: Dict = None) -> None:
        super().__init__()

        #################################################
        ############### Create main model ###############
        #################################################

        # Build vision encoder
        self.img_size = config["image_res"]
        norm_layer = functools.partial(torch.nn.LayerNorm, eps=1e-6)
        self.visual_encoder = VisionTransformer(img_size=self.img_size,
                                                    patch_size=16,
                                                    embed_dim=768,
                                                    depth=12,
                                                    num_heads=12,
                                                    mlp_ratio=4,
                                                    qkv_bias=True,
                                                    norm_layer=norm_layer)

        # Initilize DEIT model
        if config["init_deit"]:
            checkpoint = torch.hub.load_state_dict_from_url(url=DEIT_URL, map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict["pos_embed"], self.visual_encoder)
            state_dict["pos_embed"] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
            print(msg)

        # Optionally freeze the visual encoder
        if config["freeze_vision_encoder"]:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False

        # Define projection heads
        embed_dim = config["embed_dim"]
        vision_width = config["vision_width"]
        self.vision_proj = torch.nn.Linear(vision_width, embed_dim)

        # Freeze projection heads
        if config["freeze_projection_heads"]:
            for param in self.vision_proj.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        image_embeddings = self.visual_encoder(images)
        image_pooled = image_embeddings[:, 0, :]
        image_features = F.normalize(self.vision_proj(image_pooled), dim=-1)

        return image_embeddings, image_features


if __name__ == "__main__":
    print("Attention encoder")
    test_image = torch.rand(4, 3, 256, 256)
    configuration = {"image_res": 256,
                     "init_deit": True,
                     "freeze_vision_encoder": False,
                     "freeze_projection_heads": False,
                     "vision_width": 768,
                     "embed_dim": 256}
    encoder_fn = ImageEncoder(configuration)
    image_embeddings, image_features = encoder_fn(test_image)
    print(test_image.shape)
    print(image_embeddings.shape)
    print(image_features.shape)
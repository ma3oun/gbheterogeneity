import torch

import torch.nn.functional as F

from .co_attention import Coattention


class CoattentionModel(torch.nn.Module):
    def __init__(
        self,
        co_attention_dim: int,
        image_emb_size: int,
        rna_emb_size: int,
        projection_size: int,
        num_heads: int,
        img_model: torch.nn.Module,
        rna_model: torch.nn.Module,
        freeze_img_model: bool,
        freeze_rna_model: bool,
    ) -> None:
        super().__init__()

        # Hyperparams
        self.img_model = img_model
        self.rna_model = rna_model
        self.co_attention = Coattention(
            dim=co_attention_dim,
            img_dim=image_emb_size,
            rna_dim=rna_emb_size,
            num_heads=num_heads,
        )

        self.image_projector = torch.nn.Linear(image_emb_size, projection_size)
        self.rna_projector = torch.nn.Linear(rna_emb_size, projection_size)
        self.classification_head = torch.nn.Linear(image_emb_size + rna_emb_size, 2)

        # Optionally freeze image or rna encoders
        if freeze_img_model:
            for param in self.img_model.parameters():
                param.requires_grad = False

        if freeze_rna_model:
            for param in self.rna_model.parameters():
                param.requires_grad = False

    def forward(
        self, image: torch.Tensor, rna: torch.Tensor, register_hook: bool = False
    ) -> torch.Tensor:
        batch_size = image.shape[0]

        # Get representations
        representations_image, _ = self.img_model(image)
        representations_rna = self.rna_model.encode(rna)

        # Get projections
        projections_image = F.normalize(
            self.image_projector(representations_image[:, 0, :]), dim=-1
        )
        projections_rna = F.normalize(
            self.rna_projector(representations_rna[:, 0, :]), dim=-1
        )

        # Shuffle representations to create negative rna-image pairs
        neg_representations_image = torch.cat(
            [representations_image[1:], representations_image[0].unsqueeze(dim=0)],
            dim=0,
        )
        neg_representations_rna = torch.cat(
            [representations_rna[1:], representations_rna[0].unsqueeze(dim=0)], dim=0
        )

        # Compute cross-attention representations
        pos_coattentions_image, pos_coattentions_rna = self.co_attention(
            representations_image, representations_rna, register_hook=register_hook
        )
        neg_coattentions_image_a, neg_coattentions_rna_a = self.co_attention(
            representations_image, neg_representations_rna
        )
        neg_coattentions_image_b, neg_coattentions_rna_b = self.co_attention(
            neg_representations_image, representations_rna
        )

        # Reconstruct RNA input from cross-attention representation
        reconstructions_rna = self.rna_model.decode(pos_coattentions_rna)

        # Classify rna-image into match ("1") or no match ("0")
        pos_vl_representations = torch.cat(
            [pos_coattentions_image[:, 0, :], pos_coattentions_rna[:, 0, :]], dim=1
        )
        neg_vl_representations_a = torch.cat(
            [neg_coattentions_image_a[:, 0, :], neg_coattentions_rna_a[:, 0, :]], dim=1
        )
        neg_vl_representations_b = torch.cat(
            [neg_coattentions_image_b[:, 0, :], neg_coattentions_rna_b[:, 0, :]], dim=1
        )
        vl_representations = torch.cat(
            [
                pos_vl_representations,
                neg_vl_representations_a,
                neg_vl_representations_b,
            ],
            dim=0,
        )

        vl_labels = [
            torch.ones(batch_size, dtype=torch.long),
            torch.zeros(2 * batch_size, dtype=torch.long),
        ]
        vl_labels = torch.cat(vl_labels, dim=0).to(image.device)

        vl_outputs = self.classification_head(vl_representations)

        return (
            projections_image,
            projections_rna,
            vl_labels,
            vl_outputs,
            reconstructions_rna,
        )


if __name__ == "__main__":
    import image_processing.models.encoder as img_encoder
    import rna_processing.models.transformer as rna_encoder

    print("Image encoder")
    test_image = torch.rand(4, 3, 256, 256)
    configuration = {
        "image_res": 256,
        "init_deit": True,
        "freeze_vision_encoder": False,
        "freeze_projection_heads": False,
        "vision_width": 768,
        "embed_dim": 256,
        "temp": 0.07,
    }
    img_model = img_encoder.ImageEncoder(configuration)
    # image_embeddings, image_features = img_model(test_image)
    # print(image_embeddings.shape)
    # print(image_features.shape)

    print("RNA encoder")
    genes_per_cluster = {
        "C0": [i for i in range(100)],
        "C1": [i for i in range(200)],
        "C2": [i for i in range(150)],
    }
    rna_model = rna_encoder.AttentionRNA(
        genes_per_cluster=genes_per_cluster,
        embedding_dim=128,
        dropout_prob=0.5,
        num_heads=8,
        projection_dim=64,
    )

    test_input = {}
    for key, list_of_genes in genes_per_cluster.items():
        chromosome_tensor = torch.rand(len(list_of_genes))
        chromosome_tensor = torch.stack(
            [
                chromosome_tensor,
                chromosome_tensor,
                chromosome_tensor,
                chromosome_tensor,
            ],
            dim=0,
        )
        test_input[key] = chromosome_tensor
    # rna_embeddings = rna_model.encode(test_input)
    # rna_features = rna_model.project(rna_embeddings)
    # print(rna_embeddings.shape)
    # print(rna_features.shape)

    co = CoattentionModel(
        co_attention_dim=768,
        image_emb_size=768,
        rna_emb_size=128,
        num_heads=8,
        img_model=img_model,
        rna_model=rna_model,
    )

    output = co(test_image, test_input)

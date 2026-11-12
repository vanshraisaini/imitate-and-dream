import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2Tokenizer(nn.Module):
    """
    Frozen DINOv2 that turns images into patch tokens.

    Input : (B, 3, H, W) float in [0, 1]
    Output: (B, N, D_vis) where N = (image_size / 14) ** 2

    image_size must be a multiple of 14. 84 gives a 6x6 grid (36 tokens),
    112 gives an 8x8 grid (64 tokens). Images are bilinearly resized if needed.
    """

    PATCH_SIZE = 14

    def __init__(self, model_name="dinov2_vits14", image_size=84):
        super().__init__()
        if image_size % self.PATCH_SIZE != 0:
            raise ValueError(f"image_size must be a multiple of {self.PATCH_SIZE}")

        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        self.image_size = image_size
        self.grid_size = image_size // self.PATCH_SIZE
        self.num_patches = self.grid_size ** 2
        self.embed_dim = self.backbone.embed_dim  # 384 for ViT-S/14

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )

    def train(self, mode=True):
        # Keep the frozen backbone in eval mode no matter what.
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, images):
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        images = (images - self.mean) / self.std
        out = self.backbone.forward_features(images)
        return out["x_norm_patchtokens"]  # (B, N, D_vis)


if __name__ == "__main__":
    input = torch.rand(2, 3, 84, 84)
    model = DINOv2Tokenizer(model_name="dinov2_vits14", image_size=84)
    output = model(input)
    print(output.shape)
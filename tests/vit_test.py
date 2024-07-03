from images_segmentation.models.vit import ViT, Mask_Vit
import torch
import torch.nn.functional as F

def try_vit():
    model = ViT(
        image_size=300,
        patch_size=15,
        in_channels=3,
        num_classes=1,
        embed_dim=512,
        depth=4,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        drop_rate=0.1,
        norm_type="postnorm"
    )

    x = torch.rand((4, 3, 300, 300))
    print(x.shape)

    answer = model(x)
    print(answer.shape)

def try_mask_vit():
    model = Mask_Vit(
        image_size=300,
        patch_size=15,
        in_channels=3,
        num_classes=1,
        embed_dim=512,
        depth=4,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        drop_rate=0.1,
        norm_type="postnorm"
    )

    x = torch.rand((4, 3, 300, 300))
    print(x.shape)
    answer = model(x)
    print(answer.shape)

if __name__ == "__main__":
    try_mask_vit()
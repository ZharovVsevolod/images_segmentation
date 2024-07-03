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
    # x = torch.rand((4, 3, 300, 300))
    # model = Mask_Vit(
    #     image_size_h = 300,
    #     image_size_w = 300,
    #     patch_size_h = 15,
    #     patch_size_w = 15,
    #     in_channels = 3,
    #     num_classes = 1,
    #     embed_dim = 512,
    #     depth = 4,
    #     num_heads = 8,
    #     mlp_ratio = 4,
    #     qkv_bias = False,
    #     drop_rate = 0.1,
    #     norm_type = "postnorm"
    # )


    x = torch.rand((4, 3, 1000, 1400))
    model = Mask_Vit(
        image_size_h = 1000,
        image_size_w = 1400,
        patch_size_h = 100,
        patch_size_w = 140,
        in_channels = 3,
        num_classes = 1,
        embed_dim = 512,
        depth = 4,
        num_heads = 8,
        mlp_ratio = 4,
        qkv_bias = False,
        drop_rate = 0.1,
        norm_type = "postnorm"
    )


    print(x.shape)
    answer = model(x)
    print(answer.shape)

if __name__ == "__main__":
    try_mask_vit()
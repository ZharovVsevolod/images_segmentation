from images_segmentation.data import ImagesDataModule, ImagesDataset2
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import einops
from images_segmentation.models.shell import Image_Save_CheckPoint

def show_tensor_image(image):
    image = einops.rearrange(image, "c h w -> h w c")
    plt.imshow(image)
    plt.show()

def get_images(dataset, i_n):
    images = []
    original_masks = []
    predicted_masks = []

    for i in i_n:
        image, mask = dataset[i]

        image = einops.rearrange(image, "c h w -> h w c")
        mask = einops.rearrange(mask, "c h w -> h w c")

        images.append(image)
        original_masks.append(mask)
        predicted_masks.append(mask)
    
    return images, original_masks, predicted_masks

if __name__ == "__main__":
    # dm2 = ImagesDataModule(
    #     data_dir = "dataset/dataset2",
    #     batch_size = 4,
    #     height = 300,
    #     wigth = 300
    # )

    dm = ImagesDataModule(
        data_dir = "dataset/dataset3",
        batch_size = 4,
        height = 1000,
        wigth = 1400
    )

    dm.setup(stage = "fit")
    
    images, original_masks, predicted_masks = get_images(dm.val_dataset, [3, 17, 23, 29])

    save_img = Image_Save_CheckPoint(n = 4, border = 0.5)
    fig = save_img.prepare_image_for_logging(
        images=images,
        original_masks=original_masks,
        predicted_masks=predicted_masks
    )
    plt.show()
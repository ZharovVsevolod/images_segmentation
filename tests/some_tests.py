from images_segmentation.data import ImagesDataModule, ImagesDataset2
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import einops
from images_segmentation.models.shell import Image_Save_CheckPoint

def show_tensor_image(image):
    image = einops.rearrange(image, "c h w -> h w c")
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    dm = ImagesDataModule(
        data_dir = "dataset/dataset2",
        batch_size = 4,
        height = 512,
        wigth = 512
    )

    pathes = dm.load_dataset()
    train, test = random_split(pathes, [0.8, 0.2])
    dataset = ImagesDataset2(train)
   
    # image, mask = dataset[17]
    # show_tensor_image(image)
    # show_tensor_image(mask)

    images = []
    original_masks = []
    predicted_masks = []

    for i in [17, 28, 56, 100, 105]:
        image, mask = dataset[i]

        image = einops.rearrange(image, "c h w -> h w c")
        mask = einops.rearrange(mask, "c h w -> h w c")

        images.append(image)
        original_masks.append(mask)
        predicted_masks.append(mask)

    
    save_img = Image_Save_CheckPoint(4)

    fig = save_img.prepare_image_for_logging(
        images=images,
        original_masks=original_masks,
        predicted_masks=predicted_masks
    )

    plt.show()
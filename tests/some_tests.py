from images_segmentation.data import ImagesDataModule, ImagesDataset1
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import einops
from images_segmentation.models.shell import Image_Save_CheckPoint
import torch
import pandas as pd

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

def dm23_test(type_dm:int = 2):
    if type_dm == 2:
        dm = ImagesDataModule(
            data_dir = "dataset/dataset2",
            batch_size = 4,
            height = 300,
            wigth = 300
        )
    if type_dm == 3:
        dm = ImagesDataModule(
            data_dir = "dataset/dataset3",
            batch_size = 4,
            height = 1000,
            wigth = 1400
        )

    dm.prepare_data()
    dm.setup(stage = "fit")
    
    images, original_masks, predicted_masks = get_images(dm.val_dataset, [3, 17, 23, 29])

    save_img = Image_Save_CheckPoint(n = 4, border = 0.5)
    fig = save_img.prepare_image_for_logging(
        images=images,
        original_masks=original_masks,
        predicted_masks=predicted_masks
    )
    plt.show()

def search_ds1_test():
    df = pd.read_csv("dataset/dataset1/cut_and_labels.csv")

    path_to_cut = "dataset/dataset1/cut"
    
    image_name, alpha = df.iloc[3]

    gl_path_to_image = path_to_cut + "/" + image_name

    print(gl_path_to_image)
    print(alpha)

    image = plt.imread(gl_path_to_image)
    plt.imshow(image)
    plt.show()

def ds1_test():
    df = pd.read_csv("dataset/dataset1/cut_and_labels.csv")
    ds1 = ImagesDataset1(
        paths = df,
        need_height = 150,
        need_width = 120,
        path_to_cut = "dataset/dataset1/cut"
    )

    print(f"Lenght of whole dataset1 = {len(ds1)}")

    image, alpha = ds1[15]

    print(f"alpha = {alpha}")

    image_for_show = torch.tensor(image, dtype = torch.int)
    image_for_show = einops.rearrange(image_for_show, "c h w -> h w c")
    plt.imshow(image_for_show)
    plt.show()

def dm1_test():
    dm = ImagesDataModule(
            data_dir = "dataset/dataset1",
            batch_size = 4,
            height = 150,
            wigth = 120
        )
    
    dm.prepare_data()
    dm.setup(stage = "fit")

    for i in [3, 17, 588, 623]:
        image, alpha = dm.val_dataset[i]

        print(f"alpha = {alpha}")

        image_for_show = torch.tensor(image, dtype = torch.int)
        image_for_show = einops.rearrange(image_for_show, "c h w -> h w c")
        plt.imshow(image_for_show)
        plt.show()


if __name__ == "__main__":
    dm23_test()
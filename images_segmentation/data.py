import torch
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as L

from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import os
import requests, zipfile
import numpy as np

import pathlib
from urllib.parse import urlencode

import albumentations as A
import cv2
import einops

import pandas as pd
from tqdm import tqdm

class ImagesDataset1(Dataset):
    def __init__(
            self, 
            paths,
            need_height:int = 150,
            need_width:int = 120,
            path_to_cut = "dataset/dataset1/cut"
        ) -> None:
        super().__init__()

        self.path_to_cut = path_to_cut
        self.paths = paths
        self.transform = A.Resize(height = need_height, width = need_width)

    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, index:int):
        image_path, alpha = self.paths.iloc[index]
        image_path = self.path_to_cut + "/" + image_path
        image = cv2.imread(image_path)

        if image.shape[0] < image.shape[1]:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed_image = self.transform(image = image)["image"]
        
        tensor_transformed_image = torch.tensor(transformed_image, dtype = torch.float32)
        tensor_transformed_image = einops.rearrange(tensor_transformed_image, "h w c -> c h w")

        alpha = torch.tensor(alpha, dtype = torch.float32)

        return tensor_transformed_image, alpha

class ImagesDataset2(Dataset):
    def __init__(
            self, 
            patches,
            crop_height:int = 256,
            crop_width:int = 256,
            flip_probability:float = 0.5,
            brightness_probability:float = 0.2,
            what_dataset:int = 2,
            need_resize:bool = False
        ) -> None:
        super().__init__()
        self.patches = patches

        if need_resize:
            self.transform = A.Compose([
                A.Resize(height = crop_height, width = crop_width),
                A.HorizontalFlip(p = flip_probability),
                A.VerticalFlip(p = flip_probability),
                A.RandomBrightnessContrast(p = brightness_probability)
            ])
        else:
            self.transform = A.Compose([
                A.PadIfNeeded(min_height = crop_height, min_width = crop_width),
                A.RandomCrop(height = crop_height, width = crop_width),
                A.HorizontalFlip(p = flip_probability),
                A.VerticalFlip(p = flip_probability),
                A.RandomBrightnessContrast(p = brightness_probability)
            ])

        self.what_dataset = what_dataset
    
    def __len__(self) -> int:
        return len(self.patches)
    
    def __getitem__(self, index:int):
        image_path, mask_path = self.patches[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path)
        if self.what_dataset == 3:
            mask = np.where(mask == 0, 1, 0)

        transformed = self.transform(image = image, mask = mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]

        tensor_transformed_image = torch.tensor(transformed_image, dtype=torch.float32)
        tensor_transformed_image = einops.rearrange(tensor_transformed_image, "h w c -> c h w")

        tensor_transformed_mask = torch.tensor(transformed_mask, dtype=torch.float32)
        tensor_transformed_mask = einops.rearrange(tensor_transformed_mask, "h w c -> c h w")
        tensor_transformed_mask = tensor_transformed_mask[2, :, :].unsqueeze(0)
        tensor_transformed_mask = torch.as_tensor(tensor_transformed_mask > 0, dtype=torch.float32)

        return tensor_transformed_image, tensor_transformed_mask

def download_datasets():
    print("Downloading dataset...")
    yd_url = "https://disk.yandex.ru/d/cXHdTIPM0eMuew"
    final_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?' + urlencode(dict(public_key=yd_url))
    response = requests.get(final_url)
    download_url = response.json()['href']

    dataset_file = pathlib.Path('dataset.zip')

    if not dataset_file.exists():
        download_response = requests.get(download_url)

        with open(dataset_file, 'wb') as f:
            f.write(download_response.content)

        with zipfile.ZipFile(dataset_file, 'r') as zip_ref:
            zip_ref.extractall("dataset")

def cut_image_and_return_df(
        path_to_dir, 
        path_to_image,
        path_to_labels = "dataset/dataset1/labels.csv",
        cut_and_labels = pd.DataFrame({"path" : [], "alpha" : []})
    ):
    img_path = pathlib.Path(path_to_image)
    image = cv2.imread(img_path)

    labels_path = pathlib.Path(path_to_labels)
    df = pd.read_csv(labels_path)

    image_labels = df[df["image"] == img_path.name.split("_")[-1]]

    if not os.path.isdir(f"{path_to_dir}/cut"):
        os.mkdir(f"{path_to_dir}/cut")

    i = 0
    for line in image_labels.iloc():
        idx, img, xtl, ytl, xbr, ybr, alpha = line.values
        cropped_image = image[ytl:ybr, xtl:xbr, :]

        img_cut = img.split(".")[0]
        new_name = f"{img_cut}_{i}.jpg"
        i += 1

        temp = pd.DataFrame({
                "path" : [new_name],
                "alpha" : [alpha]
        })

        cut_and_labels = pd.concat([cut_and_labels, temp], ignore_index = True)
        cv2.imwrite(path_to_dir + "/cut/" + new_name, cropped_image)
    
    return cut_and_labels

def cut_ds1(root_dir):
    image_names = os.listdir(root_dir)

    cut_and_labels = pd.DataFrame({"path" : [], "alpha" : []})

    print("Extracting object from images...")

    for img_name in tqdm(image_names):
        cut_and_labels = cut_image_and_return_df(
            path_to_dir = "dataset/dataset1",
            path_to_image = f"{root_dir}/{img_name}",
            cut_and_labels = cut_and_labels
        )

    cut_and_labels.to_csv("dataset/dataset1/cut_and_labels.csv", index = False)

    print("Extraction complete")


class ImagesDataModule(L.LightningDataModule):
    def __init__(
            self,
            data_dir:str,
            batch_size:int,
            height:int,
            wigth:int,
            flip_probability:float = 0.5,
            brightness_probability:float = 0.2,
            need_resize:bool = False
        ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = [height, wigth]

        self.flip_probability = flip_probability
        self.brightness_probability = brightness_probability
        self.need_resize = need_resize
    
    def prepare_data(self) -> None:
        if not os.path.isdir(self.data_dir):
            download_datasets()
            dm1_cut_path = self.data_dir.split("/")[0] + "/dataset1/images"
            cut_ds1(dm1_cut_path)
        else:
            print("Dataset in it`s place")
    
    def load_dataset(self):
        match self.data_dir:
            case "dataset/dataset1":
                print("Dataset #1")
                self.dataset_number = 1

                paths_and_alphas = pd.read_csv("dataset/dataset1/cut_and_labels.csv")
                return paths_and_alphas

            case "dataset/dataset2":
                print("Dataset #2")
                self.dataset_number = 2
                img_folder = pathlib.Path("dataset/dataset2/aug_data/aug_data/images")
                mask_folder = pathlib.Path("dataset/dataset2/aug_data/aug_data/masks")
                
                image_names = [str(i) for i in img_folder.iterdir()]
                image_names.sort()
                mask_names = [str(i) for i in mask_folder.iterdir()]
                mask_names.sort()

                pathes = list(zip(image_names, mask_names))
                return pathes

            case "dataset/dataset3":
                print("Dataset #3")
                self.dataset_number = 3
                img_folder = pathlib.Path("dataset/dataset3/images")
                mask_folder = pathlib.Path("dataset/dataset3/masks")

                image_names = [str(i) for i in img_folder.iterdir()]
                image_names.sort()
                mask_names = [str(i) for i in mask_folder.iterdir()]
                mask_names.sort()

                pathes = list(zip(image_names, mask_names))
                return pathes
        
        raise Exception("There is no matching dataset in ", self.data_dir)
    
    def setup(self, stage: str) -> None:
        print("Loading dataset...")
        full_images = self.load_dataset()
        print("Dataset has been loaded")
        print("Splitting the full dataset")
        if self.dataset_number == 1:
            images_train = full_images.sample(frac = 0.8)
            images_val = full_images.drop(images_train.index)
            dl = images_val["alpha"].value_counts().to_dict()
            print("How many examples to each class in validation part")
            print(dl)
            del dl
        else:
            images_train, images_val = random_split(full_images, [0.8, 0.2])
        
        del full_images
        print("Dataset has been splitted")
        print(f"Length of train part: {len(images_train)}")
        print(f"Length of validation part: {len(images_val)}")

        if self.dataset_number == 1:
            if stage == "fit" or stage is None:
                self.train_dataset = ImagesDataset1(
                    images_train,
                    need_height = self.image_size[0],
                    need_width = self.image_size[1],
                    path_to_cut = "dataset/dataset1/cut"
                )
                self.val_dataset = ImagesDataset1(
                    images_val,
                    need_height = self.image_size[0],
                    need_width = self.image_size[1],
                    path_to_cut = "dataset/dataset1/cut"
                )

        else:
            if stage == "fit" or stage is None:
                self.train_dataset = ImagesDataset2(
                    images_train,
                    crop_height = self.image_size[0],
                    crop_width = self.image_size[1],
                    flip_probability = self.flip_probability,
                    brightness_probability = self.brightness_probability,
                    what_dataset = self.dataset_number,
                    need_resize = self.need_resize
                )
                self.val_dataset = ImagesDataset2(
                    images_val,
                    crop_height = self.image_size[0],
                    crop_width = self.image_size[1],
                    flip_probability = self.flip_probability,
                    brightness_probability = self.brightness_probability,
                    what_dataset = self.dataset_number,
                    need_resize = self.need_resize
                )
            
            print("Stage `fit` is set")

        if stage == "test" or stage is None:
            pass
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass
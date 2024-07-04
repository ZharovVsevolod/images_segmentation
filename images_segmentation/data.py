import torch
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as L

from typing import Any, List, Tuple, Literal
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import os
import requests, zipfile, io
import numpy as np

import pathlib
from urllib.parse import urlencode

import albumentations as A
import cv2
import einops

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
        else:
            print("Dataset is on his place")
    
    def load_dataset(self):
        match self.data_dir:
            case "dataset/dataset1":
                print("Dataset #1")


                raise Exception("Dataset #1 will be support soon", self.data_dir)

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
        images_train, images_val = random_split(full_images, [0.8, 0.2])
        del full_images
        print("Dataset has been splitted")
        print(f"Length of train part: {len(images_train)}")
        print(f"Length of validation part: {len(images_val)}")

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
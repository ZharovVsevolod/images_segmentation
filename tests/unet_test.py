from images_segmentation.models.unet import UNet
from images_segmentation.data import ImagesDataModule, ImagesDataset2
from torch.utils.data import random_split

import wandb, os
from dotenv import load_dotenv

def net_test():
    HEIGHT = 256
    WIGTH = 256

    net = UNet(
        enc_channels = [3, 16, 32, 64],
        dec_channels = [64, 32, 16],
        n_classes = 1,
        retain_dim = True,
        out_size = [HEIGHT, WIGTH]
    )

    dm = ImagesDataModule(
            data_dir = "dataset/dataset2",
            batch_size = 4,
            height = HEIGHT,
            wigth = WIGTH
        )

    pathes = dm.load_dataset()
    train, test = random_split(pathes, [0.8, 0.2])
    dataset = ImagesDataset2(train)

    image, mask = dataset[17]

    print(image.shape)
    print(mask.shape)

    output = net(image.unsqueeze(0))
    print(output.shape)
    print(output)


def wandb_test():
    load_dotenv()
    wandb.login(key=os.environ["WANDB_KEY"])
    wandb.finish()


if __name__ == "__main__":
    net_test()
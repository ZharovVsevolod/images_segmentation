import hydra
from hydra.core.config_store import ConfigStore

from images_segmentation.config import Params, UnetModel
from images_segmentation.config import Scheduler_ReduceOnPlateau, Scheduler_OneCycleLR
from images_segmentation.models.shell import Model_Lightning_Shell, Image_Save_CheckPoint
from images_segmentation.data import ImagesDataModule

import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

import os
from dotenv import load_dotenv

load_dotenv()

cs = ConfigStore.instance()
cs.store(name="params", node=Params)
cs.store(group="model", name="base_unet", node=UnetModel)
cs.store(group="scheduler", name="base_rop", node=Scheduler_ReduceOnPlateau)
cs.store(group="scheduler", name="base_oclr", node=Scheduler_OneCycleLR)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    L.seed_everything(cfg.training.seed)
    wandb.login(key=os.environ["WANDB_KEY"])

    dm = ImagesDataModule(
        data_dir = cfg.data.data_directory,
        batch_size = cfg.training.batch,
        height = cfg.data.height,
        wigth = cfg.data.width,
        flip_probability = cfg.data.flip_probability,
        brightness_probability = cfg.data.brightness_probability
    )

    model = Model_Lightning_Shell(cfg)

    os.mkdir(cfg.training.wandb_path)
    wandb_log = WandbLogger(
        project = cfg.training.project_name, 
        name = cfg.training.train_name, 
        save_dir = cfg.training.wandb_path
    )

    checkpoint = ModelCheckpoint(
        dirpath = cfg.training.model_path,
        filename = "epoch_{epoch}-{val_loss:.3f}",
        save_top_k = cfg.training.save_best_of,
        monitor = cfg.training.checkpoint_monitor
    )
    lr_monitor = LearningRateMonitor(logging_interval = "epoch")
    early_stop = EarlyStopping(monitor = cfg.training.checkpoint_monitor, patience = cfg.training.early_stopping_patience)

    image_save = Image_Save_CheckPoint(
        n = cfg.training.num_image_to_save,
        border = cfg.training.mask_border
    )

    trainer = L.Trainer(
        max_epochs = cfg.training.epochs,
        accelerator = "auto",
        devices = 1,
        log_every_n_steps=10,
        logger = wandb_log,
        callbacks = [checkpoint, lr_monitor, early_stop, image_save],
        # fast_dev_run = 5
    )
    trainer.fit(model = model, datamodule = dm)

    wandb.finish()


if __name__ == "__main__":
    main()
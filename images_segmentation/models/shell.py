import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from torchmetrics.classification import BinaryAccuracy, BinaryJaccardIndex, BinaryF1Score
from torchmetrics import Dice
from torchmetrics.regression import MeanSquaredError

from images_segmentation.config import Params
from images_segmentation.models.unet import UNet
from images_segmentation.models.vit import Mask_Vit
from images_segmentation.models.simple_model import SimpleModel

import matplotlib.pyplot as plt
import einops
import numpy as np
import itertools

class Model_Lightning_Shell(L.LightningModule):
    def __init__(
            self,
            args: Params
        ) -> None:
        super().__init__()

        # Match model that we need
        match args.model.name:
            case "unet":
                self.inner_model = UNet(
                      enc_channels = args.model.enc_channels,
                      dec_channels = args.model.dec_channels,
                      n_classes = args.model.n_classes,
                      retain_dim = args.model.retain_dim,
                      out_size = [args.data.height, args.data.width]
                )
            case "vit":
                self.inner_model = Mask_Vit(
                    image_size_h = args.model.image_size_h, 
                    image_size_w = args.model.image_size_w, 
                    patch_size_h = args.model.patch_size_h, 
                    patch_size_w = args.model.patch_size_w, 
                    in_channels = args.model.in_channels,
                    num_classes = args.model.n_classes,
                    embed_dim = args.model.embedding_dim,
                    depth = args.model.layers,
                    num_heads = args.model.heads,
                    mlp_ratio = args.model.mlp_ratio,
                    qkv_bias = args.model.qkv_bias,
                    drop_rate = args.model.dropout,
                    norm_type = args.model.norm_type
                )


        self.metric_acc = BinaryAccuracy(threshold = args.training.mask_border)
        self.metric_dice = Dice(threshold = args.training.mask_border)
        self.metric_f1 = BinaryF1Score(threshold = args.training.mask_border)
        self.metric_jac = BinaryJaccardIndex(threshold = args.training.mask_border)
        self.lr = args.training.lr

        #-----
        self.args = args
        self.save_hyperparameters()
    
    def forward(self, x) -> torch.Any:
        return self.inner_model(x)
    
    def loss(self, y, y_hat):
        return F.binary_cross_entropy_with_logits(y, y_hat)
    
    def lr_scheduler(self, optimizer):
        if self.args.scheduler.name == "ReduceOnPlateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                patience = self.args.scheduler.patience, 
                factor = self.args.scheduler.factor
            )
            scheduler_out = {"scheduler": sched, "monitor": "val_loss"}
        
        if self.args.scheduler.name == "OneCycleLR":
            sched = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr = self.lr * self.args.scheduler.expand_lr, 
                total_steps = self.args.training.epochs
            )
            scheduler_out = {"scheduler": sched}
        
        return scheduler_out
    
    def training_step(self, batch) -> STEP_OUTPUT:
        x, y_hat = batch

        y = self(x)

        answer_loss = self.loss(y, y_hat)
        score_acc = self.metric_acc(preds = y, target = y_hat)
        score_jac = self.metric_jac(preds = y, target = y_hat)
        score_f1 = self.metric_f1(preds = y, target = y_hat)
        score_dice = self.metric_dice(
            preds = torch.as_tensor(torch.nn.functional.sigmoid(y > self.args.training.mask_border), dtype=torch.int), 
            target = torch.as_tensor(y_hat, dtype=torch.int)
        )

        self.log("train_loss", answer_loss)
        self.log("train_acc", score_acc)
        self.log("train_jac", score_jac)
        self.log("train_dice", score_dice)
        self.log("train_f1", score_f1)

        return answer_loss
    
    def validation_step(self, batch) -> STEP_OUTPUT:
        x, y_hat = batch

        y = self(x)

        answer_loss = self.loss(y, y_hat)
        score_acc = self.metric_acc(preds = y, target = y_hat)
        score_jac = self.metric_jac(preds = y, target = y_hat)
        score_f1 = self.metric_f1(preds = y, target = y_hat)
        score_dice = self.metric_dice(
            preds = torch.as_tensor(torch.nn.functional.sigmoid(y > self.args.training.mask_border), dtype=torch.int), 
            target = torch.as_tensor(y_hat, dtype=torch.int)
        )

        self.log("val_loss", answer_loss)
        self.log("val_acc", score_acc)
        self.log("val_jac", score_jac)
        self.log("val_dice", score_dice)
        self.log("val_f1", score_f1)
    
    def test_step(self, batch) -> STEP_OUTPUT:
        pass
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.1)
        scheduler_dict = self.lr_scheduler(optimizer)
        return (
            {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}
        )

class Image_Save_CheckPoint(L.Callback):
    def __init__(self, n:int = 3, border:float = 0.9):
        super().__init__()
        self.n = n
        self.border = border

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        val_ds = trainer.datamodule.val_dataset
        val_len = len(val_ds)

        images = []
        original_masks = []
        predicted_masks = []

        for _ in range(self.n):
            num = np.random.randint(low = 0, high = val_len-1)
            image, mask = val_ds[num]
            
            image = image.unsqueeze(0).to(pl_module.device)
            mask_answer = pl_module(image)
            image = image.cpu()
            mask_answer = mask_answer.cpu()

            image = einops.rearrange(image, "1 c h w -> h w c")
            mask = einops.rearrange(mask, "c h w -> h w c")
            mask_answer = einops.rearrange(mask_answer, "1 c h w -> h w c")

            images.append(image)
            original_masks.append(mask)
            predicted_masks.append(mask_answer)

        figure = self.prepare_image_for_logging(
            images = images,
            original_masks = original_masks,
            predicted_masks = predicted_masks
        )
        trainer.logger.log_image(
            key = "Mask prediction",
            images = [figure]
        )
        plt.close()

    def prepare_image_for_logging(self, images, original_masks, predicted_masks):
        figure, ax = plt.subplots(
            nrows = self.n,
            ncols = 4,
            figsize = (10, 10)
        )

        for i in range(self.n):
            ax[i][0].imshow(torch.as_tensor(images[i], dtype=torch.int))
            ax[i][1].imshow(original_masks[i])
            ax[i][2].imshow(predicted_masks[i])
            ax[i][3].imshow(torch.as_tensor(torch.nn.functional.sigmoid(predicted_masks[i]) > self.border, dtype=torch.int))
                
        ax[0][0].set_title("Image")
        ax[0][1].set_title("Original Mask")
        ax[0][2].set_title("Model Output")
        ax[0][3].set_title("Predicted Mask")

        figure.tight_layout()
        return figure


#--------------------------------------------
#-----Part of Simple Model for dataset 1-----
#--------------------------------------------

class SM_Shell(L.LightningModule):
    def __init__(
            self,
            args: Params
        ) -> None:
        super().__init__()        
        self.inner_model = SimpleModel(
            enc_channels = args.model.enc_channels,
            head_channels = args.model.head_channels,
            head_droprate = args.model.head_droprate
        )

        self.metric = MeanSquaredError()
        self.loss_func = torch.nn.MSELoss()
        self.lr = args.training.lr

        self.clear_matrix()

        #-----
        self.args = args
        self.save_hyperparameters()
    
    def forward(self, x) -> torch.Any:
        return self.inner_model(x)
    
    def loss(self, y, y_hat):
        return self.loss_func(y, y_hat)
    
    def lr_scheduler(self, optimizer):
        if self.args.scheduler.name == "ReduceOnPlateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                patience = self.args.scheduler.patience, 
                factor = self.args.scheduler.factor
            )
            scheduler_out = {"scheduler": sched, "monitor": "val_loss"}
        
        if self.args.scheduler.name == "OneCycleLR":
            sched = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr = self.lr * self.args.scheduler.expand_lr, 
                total_steps = self.args.training.epochs
            )
            scheduler_out = {"scheduler": sched}
        
        return scheduler_out
    
    def update_matrix(self, y:torch.Tensor, y_hat:torch.Tensor) -> None:
        n = y.shape[0]
        for i in range(n):
            yht = int(y_hat[i] * 10)
            yt = int(torch.round(y[i], decimals=1)  * 10)

            self.ans_matrix[yht][yt] += 1
    
    def clear_matrix(self) -> None:
        self.ans_matrix = np.zeros((11, 11), dtype = int)            
    
    def training_step(self, batch) -> STEP_OUTPUT:
        x, y_hat = batch

        y = self(x).squeeze()

        answer_loss = self.loss(y, y_hat)
        self.log("train_loss", answer_loss)

        return answer_loss
    
    def validation_step(self, batch) -> STEP_OUTPUT:
        x, y_hat = batch

        y = self(x).squeeze()

        self.update_matrix(y, y_hat)

        answer_loss = self.loss(y, y_hat)
        self.log("val_loss", answer_loss)
    
    def test_step(self, batch) -> STEP_OUTPUT:
        pass
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.1)
        scheduler_dict = self.lr_scheduler(optimizer)
        return (
            {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}
        )


class AnswerMatrix(L.Callback):
    def __init__(self) -> None:
        super().__init__()
    
    def make_img_matrix(self, matr):
        fig = plt.figure(figsize = (12, 8), dpi = 80)
        plt.imshow(matr,  interpolation = 'nearest', cmap = plt.cm.Blues)
        plt.colorbar()

        tick_marks = np.arange(11)
        labels = np.arange(start = 0.0, stop = 1.1, step = 0.1)
        labels = [round(mrk, 1) for mrk in labels]
        plt.xticks(tick_marks, labels)
        plt.yticks(tick_marks, labels)

        fmt = 'd'
        thresh = matr.max() / 2.
        for i, j in itertools.product(range(matr.shape[0]), range(matr.shape[1])):
            plt.text(j, i, format(matr[i, j], fmt), horizontalalignment="center", color="white" if matr[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # plt.show()
        return [fig]

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule | SM_Shell) -> None:
        answer_matrix = pl_module.ans_matrix
        pl_module.clear_matrix()
        answer_matrix_image = self.make_img_matrix(answer_matrix)

        trainer.logger.log_image(key = "Validation Confusion Matrix", images = answer_matrix_image)
        plt.close()
from typing import Literal, List
from dataclasses import dataclass

@dataclass
class Model:
    name: str

@dataclass
class SimpleMd(Model):
    enc_channels: List
    head_channels: List
    head_droprate: float

@dataclass
class UnetModel(Model):
    n_classes: int
    enc_channels: List
    dec_channels: List
    retain_dim: bool

@dataclass
class VitModel(Model):
    n_classes: int
    image_size_h: int
    image_size_w: int
    patch_size_h: int
    patch_size_w: int
    in_channels: int
    layers: int
    heads: int
    embedding_dim: int
    mlp_ratio: int
    norm_type: str
    dropout: float
    qkv_bias: bool

@dataclass
class Scheduler:
    name: str

@dataclass
class Scheduler_ReduceOnPlateau(Scheduler):
    patience: int
    factor: float

@dataclass
class Scheduler_OneCycleLR(Scheduler):
    expand_lr: int

@dataclass
class Data:
    data_directory: str
    height: int
    width: int
    flip_probability: float
    brightness_probability: float
    need_resize: bool

@dataclass
class Training:
    project_name: str
    train_name: str
    seed: int
    epochs: int
    batch: int
    lr: float
    wandb_path: str
    model_path: str
    save_best_of: int
    checkpoint_monitor: str
    early_stopping_patience: int
    num_image_to_save: int
    mask_border: float

@dataclass
class Params:
    model: Model
    data: Data
    training: Training
    scheduler: Scheduler
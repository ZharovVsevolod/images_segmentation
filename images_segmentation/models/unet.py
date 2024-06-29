from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch
from torch import nn
from typing import Literal, List
import einops

class Block(nn.Module):
	def __init__(self, in_channels, out_channels = None, mode:Literal["down", "up"] = "down"):
		super().__init__()
		# store the convolution and RELU layers
		# if not (mode in ["down", "up"]):
		# 	raise Exception("Не тот тип mode, он может быть только ['down', 'up'], а он = ", mode)

		if out_channels is None:
			match mode:
				case "down":
					out_channels = in_channels // 2
				case "up":
					out_channels = in_channels * 2
				case _:
					out_channels = in_channels
		
		self.convblock = nn.Sequential(
			nn.Conv2d(
				in_channels = in_channels,
				out_channels = out_channels,
				kernel_size = 3
			),
			nn.ReLU(),
			nn.Conv2d(
				in_channels = out_channels,
				out_channels = out_channels,
				kernel_size = 3
			),
			nn.ReLU()
		)

	def forward(self, x):
		# apply CONV => RELU => CONV block to the inputs and return it
		x = self.convblock(x)
		return x

class Encoder(nn.Module):
	def __init__(self, channels=[3, 16, 32, 64]):
		super().__init__()

		# store the encoder blocks and maxpooling layer
		self.enc_blocks = nn.ModuleList([
			Block(
				in_channels=channels[i],
				out_channels=channels[i+1]
			)
			for i in range(len(channels) - 1)
		])

		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

	def forward(self, x):
		# initialize an empty list to store the intermediate outputs
		block_outputs = []

		# loop through the encoder blocks
		for block in self.enc_blocks:
			# pass the inputs through the current encoder block, store
			# the outputs, and then apply maxpooling on the output
			x = block(x)
			block_outputs.append(x)
			x = self.pool(x)

		# return the list containing the intermediate outputs
		return block_outputs

class Decoder(nn.Module):
	def __init__(self, channels=[64, 32, 16]):
		super().__init__()
		# initialize the number of channels, upsampler blocks, and
		# decoder blocks
		self.channels = channels

		self.upconvs = nn.ModuleList([
			nn.ConvTranspose2d(
				in_channels=channels[i],
				out_channels=channels[i+1],
				kernel_size=2,
				stride=2
			)
			for i in range(len(channels) - 1)
		])

		self.dec_blocks = nn.ModuleList([
			Block(
				in_channels=channels[i],
				out_channels=channels[i+1]
			)
			for i in range(len(channels) - 1)
		])

	def forward(self, x, enc_features):
		# loop through the number of channels
		for i in range(len(self.channels) - 1):
			# pass the inputs through the upsampler blocks
			x = self.upconvs[i](x)
			# crop the current features from the encoder blocks,
			# concatenate them with the current upsampled features,
			# and pass the concatenated output through the current
			# decoder block
			croped_enc_features = self.crop(enc_features[len(self.channels) - i - 2], x)
			croped_enc_features = einops.rearrange(croped_enc_features, "b c h w -> b c h w", c=x.shape[1]) # Проверка размерности
			x = torch.cat((x, croped_enc_features), dim=1)
			x = einops.rearrange(x, "b c h w -> b c h w", c=self.channels[i]) # Проверка размерности
			x = self.dec_blocks[i](x)

		# return the final decoder output
		return x

	def crop(self, enc_features, x):
		# grab the dimensions of the inputs, and crop the encoder
		# features to match the dimensions
		(_, _, H, W) = x.shape
		enc_features = CenterCrop([H, W])(enc_features)

		# return the cropped features
		return enc_features

class UNet(nn.Module):
	def __init__(
			self, 
			enc_channels: List, 
			dec_channels: List,
			n_classes: int, 
			retain_dim: bool,
			out_size: List,
			need_softmax:bool = False
		):
		super().__init__()

		# initialize the encoder and decoder
		self.encoder = Encoder(enc_channels)
		self.decoder = Decoder(dec_channels)

		# initialize the regression head and store the class variables
		self.head = nn.Conv2d(
			in_channels=dec_channels[-1],
			out_channels=n_classes,
			kernel_size=1
		)
		self.retain_dim = retain_dim
		self.out_size = out_size

		self.need_softmax = need_softmax

	def forward(self, x):
		# grab the features from the encoder
		enc_features = self.encoder(x)

		# pass the encoder features through decoder making sure that
		# their dimensions are suited for concatenation
		dec_features = self.decoder(enc_features[-1], enc_features[:-1])

		# pass the decoder features through the regression head to
		# obtain the segmentation mask
		map = self.head(dec_features)

		# check to see if we are retaining the original output
		# dimensions and if so, then resize the output to match them
		if self.retain_dim:
			map = F.interpolate(map, self.out_size)

		#---------------------------
		if self.need_softmax:
			map = torch.nn.functional.sigmoid(map)
		#---------------------------

		# return the segmentation map
		return map

import torch
from torch import nn
from typing import Literal, List

class MLP(nn.Module):
    def __init__(
            self, 
            in_features:int, 
            hidden_features:int = None, 
            out_features:int = None, 
            drop:float = 0.0, 
            act_layer = nn.GELU()
        ):
        super().__init__()

        if out_features is None:
            out_features = in_features
        
        if hidden_features is None:
            hidden_features = in_features

        # Linear Layers
        self.lin1 = nn.Linear(
            in_features=in_features,
            out_features=hidden_features
        )
        self.lin2 = nn.Linear(
            in_features=hidden_features,
            out_features=out_features
        )

        # Activation(s)
        self.act = act_layer
        self.dropout = nn.Dropout(p=drop)

    def forward(self, x):
        x = self.act(self.dropout(self.lin1(x)))
        x = self.act(self.lin2(x))
        return x


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

class SimpleModel(nn.Module):
    def __init__(
            self,
            enc_channels:List = [3, 8, 3, 1],
            head_channels:List = [690, 256, 128, 16, 1],
            head_droprate:float = 0.05
    ) -> None:
        super().__init__()
        if len(head_channels) != 5:
              raise Exception(f"Lenght of head_channels must be 5, but it is {len(head_channels)}")

        self.encoder = Encoder(enc_channels)

        self.head = nn.Sequential(
            MLP(
                in_features = head_channels[0],
                hidden_features = head_channels[1],
                out_features = head_channels[2],
                drop = head_droprate
            ),
            MLP(
                in_features = head_channels[2],
                hidden_features = head_channels[3],
                out_features = head_channels[4],
                drop = head_droprate,
				act_layer = nn.SELU()
            )
        )
    
    def forward(self, x):
        x = self.encoder(x)[-1]
        x = torch.flatten(x, start_dim=1)
        x = self.head(x)
        x = torch.sigmoid(x)
        return x

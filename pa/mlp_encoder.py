from typing import Tuple, List, Optional, Dict
import torch
import numpy as np

from .config import Config as MainConfig
from .attention_layer import Dense


class Config:
    layers = [512, 256, 256]


class MLPItemEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = Dense(MainConfig.item_dims, Config.layers, MainConfig.items_emb_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.to(MainConfig.device)
        outputs = self.dense(inputs)
        return outputs

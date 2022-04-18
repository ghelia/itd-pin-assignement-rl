"""train recorder class"""
from typing import MutableMapping, List

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

class Recorder:
    """Wrapper of tensorboard api used during training
    Parameters
    ----------
    tb_path: str
        path where to save the tensorboard logfiles
    """
    def __init__(self, tb_path: str) -> None:
        self.writer = SummaryWriter(tb_path)
        self.scalars: MutableMapping[str, List[float]] = dict()
        self.hists: MutableMapping[str, List[float]] = dict()
        self.epoch = 0

    def scalar(self, value: float, label: str) -> None:
        """record a scalar
        Parameters
        ----------
        value: float
            scallar value
        label: str
            name of the scallar
        """
        assert isinstance(value, float)
        if label not in self.scalars:
            self.scalars[label] = []
        self.scalars[label].append(value)

    def hist(self, value: List[float], label: str) -> None:
        """record a scalar
        Parameters
        ----------
        value: List[float]
            list of values of the histogram
        label: str
            name of the histogram
        """
        assert isinstance(value, list)
        if label not in self.hists:
            self.hists[label] = []
        self.hists[label] = self.hists[label] + value

    def gradients_and_weights(self, model: torch.nn.Module) -> None:
        """record gradients and weights of a pytorch model
        Parameters
        ----------
        model: torch.nn.Module
            pytorch model
        """
        for name, param  in model.named_parameters():
            self.hist(param.data.view([-1]).tolist(), f"{name}")
            if param.grad is not None:
                self.hist(param.grad.view([-1]).tolist(), f"{name}/gradients")

    def end_epoch(self) -> None:
        """logs registered data for tensorboard"""
        self.epoch += 1
        for label, scalar in self.scalars.items():
            self.writer.add_scalar(label, np.mean(scalar), self.epoch)
        for label, hist in self.hists.items():
            self.writer.add_histogram(label, np.array(hist), self.epoch)
        self.scalars = dict()
        self.hists = dict()

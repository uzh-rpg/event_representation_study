from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class ModelContainer(ABC):
    """
    Abstract class defining model container.
    A typical model container will encompass all the utilities related to model preparation.
    Note that cfg contains all the neccessary data needed to create a model container.
    """

    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        # self.models is a python dictionary containing models
        self.models = {"model": None}

    @abstractmethod
    def gen_model(self, **kwargs):
        """
        Generate self.models
        """
        pass

    @abstractmethod
    def load_saved(self):
        """
        Load models from config file directories
        """
        pass

    def infer(
        self, target_model: str, input_tensor: torch.Tensor, no_grad: bool, **kwargs
    ):
        """
        Make inference with target_model.

        Args:
            target_model: Name of model to perform inference with
            input_tensor: torch.Tensor containing input values
            no_grad: If True, performs inference with no gradient computation
        """
        if no_grad:
            with torch.no_grad():
                return self.models[target_model](input_tensor)
        else:
            return self.models[target_model](input_tensor)

    def complex_infer(
        self, target_model: str, input_dict: dict, no_grad: bool, **kwargs
    ):
        """
        Make complex inference with target_model, where input tensor is more than one.

        Args:
            target_model: Name of model to perform inference with
            input_dict: Dictionary of arguments to pass to the model with which to perform inference
            no_grad: If True, performs inference with no gradient computation
        """
        if no_grad:
            with torch.no_grad():
                return self.models[target_model](**input_dict)
        else:
            return self.models[target_model](**input_dict)

    def parallelize(self, target: list, devices: list, **kwargs):
        """
        Parallelize target models for single/multi-gpu training

        Args:
            target: List of strings containing names of models to be parallelized
            device: List of torch.device(s) used for parallelizing models
        """
        multi_gpu = torch.cuda.device_count() >= 1 and self.cfg.parallel
        single_gpu = torch.cuda.device_count() >= 1 and not self.cfg.parallel

        if multi_gpu:
            print("Number of GPUs:", torch.cuda.device_count())
            for tgt_model in target:
                self.models[tgt_model] = nn.DataParallel(self.models[tgt_model])
                self.models[tgt_model].to(devices[0])

        elif single_gpu:
            print("Number of GPUs:", torch.cuda.device_count())
            for tgt_model in target:
                self.models[tgt_model].to(devices[0])

    def set_train(self, target: list):
        """
        Set models to train mode.

        Args:
            target: List of strings containing names of models to be set to train mode
        """
        for tgt_model in target:
            self.models[tgt_model].train()

    def set_eval(self, target: list):
        """
        Set models to eval mode.

        Args:
            target: List of strings containing names of models to be set to eval mode
        """
        for tgt_model in target:
            self.models[tgt_model].eval()

    def print_model_size(self, target: list):
        """
        Print model size and parameter count for models in target

        Args:
            target: List of strings containing names of models to inspect size
        """
        for tgt_model in target:
            print(
                "Number of model parameters in {}: {}".format(
                    tgt_model,
                    sum(
                        [p.data.nelement() for p in self.models[tgt_model].parameters()]
                    ),
                )
            )
            print(
                "Model size of {}: {} MB".format(
                    tgt_model,
                    round(
                        4
                        * sum(
                            [
                                p.data.nelement()
                                for p in self.models[tgt_model].parameters()
                            ]
                        )
                        / 1024**2
                    ),
                )
            )

    def set_requires_grad(self, target: list, requires_grad: bool):
        """
        Set requires_grad value for models in target

        Args:
            target: List of strings containing names of models to inspect size
            requires_grad: Value to set for requires_grad
        """
        for tgt_model in target:
            for param in self.models[tgt_model].parameters():
                param.requires_grad = False

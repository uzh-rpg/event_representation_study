from abc import ABC, abstractmethod
from base.models.model_container import ModelContainer
from base.data.data_container import DataContainer
import torch


class Trainer(ABC):
    """
    Abstract class defining trainer. It is responsible for model training, testing, savig, and logging.
    A typical trainer will encompass all the utilities related to model training.
    Note that cfg contains all the neccessary data needed to create a model container.
    If multiple optimizers/schedulers are needed, write init_optmizer, init_scheduler such that self.optimizer, self.scheduler is a dictionary.
    """

    def __init__(
        self,
        cfg,
        model_container: ModelContainer,
        data_container: DataContainer,
        **kwargs,
    ):
        """
        Initialize Trainer instance

        Args:
            cfg: Parsed config file
            model_container: ModelContainer instance containing all the models used for training
            data_container: DataContainer instance containing all the datasets/dataloaaders used for training
        """
        self.cfg = cfg
        self.model_container = model_container
        self.data_container = data_container
        self.writer = None
        self.devices = []
        self.exp_save_dir = None
        self.use_cuda = not self.cfg.no_cuda and torch.cuda.is_available()
        torch.manual_seed(self.cfg.seed)
        if self.use_cuda:
            torch.cuda.manual_seed(self.cfg.seed)

    @abstractmethod
    def init_env(self, **kwargs):
        """
        Using contents from self.cfg, initialize variables related to training environment: self.writer, self.devices, self.exp_save_dir
        """
        pass

    @abstractmethod
    def init_optimizer(self, **kwargs):
        """
        Initialize self.optimizer using self.cfg.
        """
        pass

    @abstractmethod
    def init_scheduler(self, **kwargs):
        """
        Initialize self.scheduler using self.cfg.
        """
        pass

    @abstractmethod
    def train(self, data_dict, **kwargs):
        """
        Unit operation for training. Could be further wrapped for epoch-wise, batch-wise training.

        Args:
            data_dict: Dictionary containing data used for training. Possibly it may include input data, labels, etc.
        """
        pass

    @abstractmethod
    def test(self, data_dict, **kwargs):
        """
        Unit operation for testing. Could be further wrapped for epoch-wise, batch-wise testing.

        Args:
            data_dict: Dictionary containing data used for testing. Possibly it may include input data, labels, etc.
        """
        pass

    def write(self, total_iter, write_dict: dict, **kwargs):
        """
        Write to logger. Default behavior is write scalar. May be extended to write other modalities as well.

        Args:
            total_iter: Number of total iterations
            write_dict: Dictionary containing data to write on tensorboard
        """
        for key in write_dict.keys():
            self.writer.add_scalar(key, write_dict[key], total_iter)

    def print_state(self, print_dict: dict):
        """
        Print current training state using values from print_dict.

        Args:
            print_dict: Dictionary containing arguments to print
        """
        print_str = ""
        for idx, key in enumerate(print_dict.keys()):
            if idx == len(print_dict.keys()) - 1:
                if type(print_dict[key]) == float:
                    print_str += f"{key} = {print_dict[key]:.4f}"
                else:
                    print_str += f"{key} = {print_dict[key]}"
            else:
                if type(print_dict[key]) == float:
                    print_str += f"{key} = {print_dict[key]:.4f}, "
                else:
                    print_str += f"{key} = {print_dict[key]}, "

        print(print_str)

    def save_model(self, total_epoch, total_iter, total_val_iter, **kwargs):
        """
        Save model on self.exp_save_dir.

        Args:
            total_epoch: Current total epoch
            total_iter: Current total number of iterations
            total_val_iter: Current total number of validation iterations
        """
        """
        Save model on self.exp_save_dir.

        Args:
            total_epoch: Current total epoch
            total_iter: Current total number of iterations
            total_val_iter: Current total number of validation iterations
        """

        save_mode = self.cfg.save_by
        multi_gpu = torch.cuda.device_count() >= 1 and self.cfg.parallel

        model = self.model_container.models["model"]
        if multi_gpu:
            if save_mode == "iter":
                if total_iter % self.cfg.save_every == 0:
                    save_dir = (
                        self.exp_save_dir
                        / "model_log"
                        / f"checkpoint_epoch_{total_epoch}_iter_{total_iter}_val_{total_val_iter}.tar"
                    )
                    torch.save(
                        {
                            "iter": total_iter,
                            "state_dict": model.module.state_dict(),
                        },
                        save_dir,
                    )

            elif save_mode == "epoch":
                if total_epoch % self.cfg.save_every == 0:
                    save_dir = (
                        self.exp_save_dir
                        / "model_log"
                        / f"checkpoint_epoch_{total_epoch}_iter_{total_iter}_val_{total_val_iter}.tar"
                    )
                    torch.save(
                        {
                            "epoch": total_epoch,
                            "state_dict": model.module.state_dict(),
                        },
                        save_dir,
                    )
        else:
            if save_mode == "iter":
                if total_iter % self.cfg.save_every == 0:
                    save_dir = (
                        self.exp_save_dir
                        / "model_log"
                        / f"checkpoint_epoch_{total_epoch}_iter_{total_iter}_val_{total_val_iter}.tar"
                    )
                    torch.save(
                        {
                            "iter": total_iter,
                            "state_dict": model.state_dict(),
                        },
                        save_dir,
                    )

            elif save_mode == "epoch":
                if total_epoch % self.cfg.save_every == 0:
                    save_dir = (
                        self.exp_save_dir
                        / "model_log"
                        / f"checkpoint_epoch_{total_epoch}_iter_{total_iter}_val_{total_val_iter}.tar"
                    )
                    torch.save(
                        {
                            "epoch": total_epoch,
                            "state_dict": model.state_dict(),
                        },
                        save_dir,
                    )

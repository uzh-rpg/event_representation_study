import torch
from base.models.model_container import ModelContainer
from base.train.metrics import accuracy
from base.data.data_container import DataContainer
from base.train.common_trainer import CommonTrainer
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random


class CNNTrainer(CommonTrainer):
    def __init__(
        self,
        cfg,
        model_container: ModelContainer,
        data_container: DataContainer,
        **kwargs,
    ):
        super(CNNTrainer, self).__init__(cfg, model_container, data_container)
        print(f"Initializing trainer {self.__class__.__name__}...")
        self.init_env()
        self.init_optimizer()
        self.init_scheduler()
        self.prep_train()
        self.loss_func = nn.CrossEntropyLoss()
        self.debug = getattr(self.cfg, "debug", False)
        self.debug_input = getattr(self.cfg, "debug_input", False)
        self.debug_labels = getattr(self.cfg, "debug_labels", False)

    def init_optimizer(self, **kwargs):
        """
        Initialize self.optimizer using self.cfg.
        """

        # Choose optimizer
        model = self.model_container.models["model"]
        try:
            opt_type = self.cfg.optimizer
            freeze = getattr(self.cfg, "freeze", False) or getattr(
                self.cfg, "train_classifier", False
            )

            if opt_type == "SGD":
                print("Using SGD as optimizer")
                if freeze:
                    print("Freezing weights!")
                    self.optimizer = optim.SGD(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=self.cfg.learning_rate,
                        momentum=self.cfg.momentum,
                        weight_decay=self.cfg.weight_decay,
                    )
                else:
                    self.optimizer = optim.SGD(
                        model.parameters(),
                        lr=self.cfg.learning_rate,
                        momentum=self.cfg.momentum,
                        weight_decay=self.cfg.weight_decay,
                    )
            elif opt_type == "Adam":
                print("Using Adam as optimizer")
                if freeze:
                    print("Freezing weights!")
                    self.optimizer = optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=self.cfg.learning_rate,
                        weight_decay=self.cfg.weight_decay,
                    )
                else:
                    self.optimizer = optim.Adam(
                        model.parameters(),
                        lr=self.cfg.learning_rate,
                        weight_decay=self.cfg.weight_decay,
                    )

        except AttributeError:
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=self.cfg.learning_rate,
                momentum=self.cfg.momentum,
                weight_decay=self.cfg.weight_decay,
            )

    def train(self, data_dict, **kwargs):
        input_data = data_dict["input_data"]
        label = data_dict["label"]

        self.model_container.set_train(["model"])

        if self.use_cuda:
            input_data, label = input_data.to(self.devices[0]), label.to(
                self.devices[0]
            )

        self.optimizer.zero_grad()

        pred = self.model_container.infer("model", input_data, False)

        loss = self.loss_func(pred, label)

        acc_1, acc_5 = accuracy(
            pred.cpu(), label.cpu(), topk=(1, min(5, pred.shape[-1]))
        )
        loss.backward()
        self.optimizer.step()

        if self.debug:
            if self.debug_input:
                self.inspect_input(input_data)
            if self.debug_labels:
                self.inspect_labels(pred, label, acc_1)

        return loss.item(), acc_1, acc_5

    def test(self, data_dict, **kwargs):
        input_data = data_dict["input_data"]
        label = data_dict["label"]

        self.model_container.set_eval(["model"])

        if self.use_cuda:
            input_data, label = input_data.to(self.devices[0]), label.to(
                self.devices[0]
            )

        pred = self.model_container.infer("model", input_data, True)
        tot_num = len(pred)

        acc_1, acc_5 = accuracy(
            pred.cpu(), label.cpu(), topk=(1, min(5, pred.shape[-1]))
        )

        num_correct = int(acc_1 * tot_num)

        if self.debug:
            if self.debug_input:
                self.inspect_input(input_data)
            if self.debug_labels:
                self.inspect_labels(pred, label, acc_1)

        return acc_1, acc_5, num_correct, tot_num

    def save_model(self, total_epoch, total_iter, total_val_iter, **kwargs):
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

    def prep_train(self):
        """
        Funcion called before training begins. Auxiliary function for initializng different training
        configurations such as model parallelism and weight freezing.
        """
        # Prepare for data parallelism
        self.model_container.load_saved()
        self.model_container.parallelize(["model"], self.devices)
        self.model_container.print_model_size(["model"])

    def inspect_labels(self, pred, label, acc_1):
        # Debugging utility for inspecting ground truth / predicted labels
        thres = getattr(self.cfg, "acc_threshold", 0.1)
        if acc_1 < thres:
            unq, cnt = torch.unique(pred.argmax(-1), return_counts=True)
            print(
                "GT label: "
                + self.data_container.dataset[self.cfg.mode].labels[label[0]]
                + f" {label[0]}"
            )
            print(
                "Most predicted: "
                + self.data_container.dataset[self.cfg.mode].labels[unq[cnt.argmax()]]
                + f" {unq[cnt.argmax()]}"
            )

    def inspect_input(self, input_data):
        inspect_channel = getattr(self.cfg, "inspect_channel", 0)
        inspect_index = getattr(self.cfg, "inspect_index", 0)

        if inspect_index == "random":
            inspect_index = random.randint(0, len(input_data) - 1)

        # Debugging utility for visualizing input data
        if type(inspect_channel) is int:
            tmp = input_data[inspect_index].permute(1, 2, 0)[:, :, inspect_channel]
            plt.imshow(255 * tmp.cpu().numpy())
            plt.show()

        if inspect_channel == "all":
            fig = plt.figure(figsize=(50, 50))

            tmp = input_data[inspect_index].permute(1, 2, 0)
            num_channel = tmp.shape[-1]

            for i in range(num_channel):
                fig.add_subplot(num_channel // 2, 2, i + 1)
                plt.imshow(255 * tmp[..., i].cpu().numpy())

            plt.show()

from datetime import datetime
import pathlib
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch
from time import time
from base.train.mini_batch_trainer import MiniBatchTrainer
from abc import abstractmethod
from base.utils.tracker import MiniBatchTracker


class CommonTrainer(MiniBatchTrainer):
    """
    Abstract class for commonly used trainer. Implements several abstract methods to facilitate code reuse.
    """

    def init_env(self, **kwargs):
        """
        Using contents from self.cfg, initialize variables related to training environment: self.writer, self.devices, self.exp_save_dir
        """
        assert self.cfg.mode in ["train", "test"]
        time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        config_name = (
            self.cfg.name.replace("/", "@").replace("=", "-").replace(",", "&")
        )  # Accout for cases where '/, =, \,' is inside config name
        config_name = config_name[:200]  # Account for file names too long
        self.exp_save_dir = pathlib.Path(self.cfg.save_root_dir) / config_name
        self.writer = SummaryWriter(
            pathlib.Path(self.exp_save_dir)
            / f"{config_name}_{self.cfg.mode}_{time_stamp}"
        )
        self.devices = [
            torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
        ]
        self.devices[0] = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.tracker = MiniBatchTracker()

    def init_optimizer(self, **kwargs):
        """
        Initialize self.optimizer using self.cfg.
        """

        # Choose optimizer
        model = self.model_container.models["model"]
        try:
            opt_type = self.cfg.optimizer
            if opt_type == "SGD":
                print("Using SGD as optimizer")
                self.optimizer = optim.SGD(
                    model.parameters(),
                    lr=self.cfg.learning_rate,
                    momentum=self.cfg.momentum,
                    weight_decay=self.cfg.weight_decay,
                )
            elif opt_type == "Adam":
                print("Using Adam as optimizer")
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

    def init_scheduler(self, **kwargs):
        """
        Initialize self.scheduler using self.cfg.
        """
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "max", verbose=True, patience=3
        )

    def run(self):
        if self.cfg.mode == "train":
            start_full_time = time()

            if not self.exp_save_dir.is_dir():
                self.exp_save_dir.mkdir(parents=True, exist_ok=True)
            if not (self.exp_save_dir / "model_log").is_dir():
                (self.exp_save_dir / "model_log").mkdir(parents=True, exist_ok=True)

            self.tracker.init_run()

            for epoch in range(1, self.cfg.epochs + 1):
                self.tracker.set_epoch(epoch)
                self.run_epoch()

            full_training_time = (time() - start_full_time) / 3600
            print(f"Full training time = {full_training_time:.2f} HR")

        elif self.cfg.mode == "test":
            if not self.exp_save_dir.is_dir():
                self.exp_save_dir.mkdir(parents=True, exist_ok=True)
            self.run_test()

    def run_epoch(self):
        # Train
        print(f"This is {self.tracker.epoch}-th epoch.")

        self.tracker.init_epoch(mode="train")

        skip_train = getattr(self.cfg, "skip_train", False)

        if not skip_train:
            self.train_epoch()

            print(
                f"Epoch {self.tracker.epoch} training loss = {self.tracker.total_train_loss.get_avg():.4f}"
            )

        # Validate
        print(f"Epoch {self.tracker.epoch} validation!")

        self.tracker.init_epoch(mode="val")
        self.validate_epoch()
        print(f"Total validation accuracy = {(self.tracker.get_val_acc()):.4f}")

        # Update scheduler
        self.scheduler.step(self.tracker.get_val_acc())

        # Save model by epoch (depends on cfg)
        if self.cfg.save_by == "epoch":
            self.save_model(
                self.tracker.epoch, self.tracker.total_iter, self.tracker.total_val_iter
            )

    def run_test(self):
        total_test = 0
        total_test_correct = 0
        total_test_num = 0
        total_test_acc_5 = 0
        load_start = time()

        print("Begin test!")
        for batch_idx, data_dict in enumerate(self.data_container.dataloader["test"]):
            load_time = time() - load_start
            total_test += 1
            test_acc_1, test_acc_5, test_correct, test_num = self.test(data_dict)
            total_test_acc_5 += test_acc_5
            write_dict = {
                "top 1 test accuracy": test_acc_1,
                "top 5 test accuracy": test_acc_5,
            }
            print_dict = {
                "Iter": batch_idx,
                "top 1 test acc": test_acc_1,
                "top 5 test acc": test_acc_5,
                "load time": load_time,
            }
            self.write(total_test, write_dict)
            self.print_state(print_dict)

            total_test_correct += test_correct
            total_test_num += test_num
            load_start = time()

        avg_top_1 = total_test_correct / total_test_num
        avg_top_5 = total_test_acc_5 / len(self.data_container.dataloader["test"])

        print(f"Average top 1 accuracy: {avg_top_1}")
        print(f"Average top 5 accuracy: {avg_top_5}")

        save_dir = self.exp_save_dir / "test_result.tar"
        torch.save(
            {
                "test_acc": total_test_correct / total_test_num,
            },
            save_dir,
        )

    def train_epoch(self):
        self.tracker.start_load_timing()
        for batch_idx, data_dict in enumerate(self.data_container.dataloader["train"]):
            self.tracker.set_batch(batch_idx)
            self.tracker.end_load_timing()
            self.train_batch(data_dict)
            self.tracker.start_load_timing()
        self.tracker.end_load_timing()

    def validate_epoch(self):
        self.tracker.start_load_timing()
        for batch_idx, data_dict in enumerate(self.data_container.dataloader["val"]):
            self.tracker.set_batch(batch_idx)
            self.tracker.end_load_timing()
            self.validate_batch(data_dict)
            self.tracker.start_load_timing()
        self.tracker.end_load_timing()

    def train_batch(self, data_dict):
        self.tracker.init_batch(mode="train")
        self.tracker.start_infer_timing()

        loss, acc_1, acc_5 = self.train(data_dict)
        self.tracker.end_infer_timing()

        write_dict = {
            "training loss": loss,
            "top 1 training accuracy": acc_1,
            "top 5 training accuracy": acc_5,
            "load time": self.tracker.load_time,
        }
        print_dict = {
            "Iter": self.tracker.batch_idx,
            "training loss": loss,
            "top 1 training acc": acc_1,
            "top 5 training acc": acc_5,
            "infer time": self.tracker.infer_time,
            "load time": self.tracker.load_time,
        }
        self.print_state(print_dict)
        self.write(self.tracker.total_iter, write_dict)

        self.tracker.total_train_loss.update(loss)
        if self.cfg.save_by == "iter":
            self.save_model(
                self.tracker.epoch, self.tracker.total_iter, self.tracker.total_val_iter
            )

    def validate_batch(self, data_dict):
        self.tracker.init_batch(mode="val")
        val_acc_1, val_acc_5, val_correct, val_num = self.test(data_dict)
        write_dict = {
            "top 1 validation accuracy": val_acc_1,
            "top 5 validation accuracy": val_acc_5,
        }
        print_dict = {
            "Iter": self.tracker.batch_idx,
            "top 1 validation acc": val_acc_1,
            "top 5 validation acc": val_acc_5,
            "load time": self.tracker.load_time,
        }
        self.print_state(print_dict)
        self.write(self.tracker.total_val_iter, write_dict)
        self.tracker.total_val_correct.accumulate(val_correct)
        self.tracker.total_val_num.accumulate(val_num)

    @abstractmethod
    def prep_train(self):
        """
        Funcion called before training begins. Auxiliary function for initializng different training
        configurations such as model parallelism and weight freezing.
        """
        pass


class CommonChunkTrainer(CommonTrainer):
    """
    Abstract class for commonly used trainer, where loading is based on caching chunks in RAM. Implements several abstract
    methods to facilitate code reuse.
    """

    def train_epoch(self):
        self.tracker.start_load_timing()
        self.data_container.refresh_chunk("train")
        num_chunks = self.data_container.dataset["train"].num_chunks
        batch_idx = 0
        for chunk_idx in range(num_chunks):
            self.data_container.create_chunk(batch_idx, "train")
            for _, data_dict in enumerate(self.data_container.dataloader["train"]):
                self.tracker.set_batch(batch_idx)
                self.tracker.end_load_timing()
                self.train_batch(data_dict)
                self.tracker.start_load_timing()
                batch_idx += 1
        self.tracker.end_load_timing()
        self.data_container.release_chunk("train")

    def validate_epoch(self):
        self.tracker.start_load_timing()
        self.data_container.refresh_chunk("val")
        num_chunks = self.data_container.dataset["val"].num_chunks
        batch_idx = 0
        for chunk_idx in range(num_chunks):
            self.data_container.create_chunk(batch_idx, "val")
            for _, data_dict in enumerate(self.data_container.dataloader["val"]):
                self.tracker.set_batch(batch_idx)
                self.tracker.end_load_timing()
                self.validate_batch(data_dict)
                self.tracker.start_load_timing()
                batch_idx += 1
        self.tracker.end_load_timing()
        self.data_container.release_chunk("val")

import torch
from base.models.model_container import ModelContainer
from base.data.data_container import DataContainer
from base.train.mini_batch_trainer import MiniBatchTrainer
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import torch.nn.functional as F
from base.utils.time_utils import Timer
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import pathlib
from base.utils.tracker import SequenceTracker


class CaptionTrainer(MiniBatchTrainer):
    def __init__(
        self,
        cfg,
        model_container: ModelContainer,
        data_container: DataContainer,
        **kwargs,
    ):
        super(CaptionTrainer, self).__init__(cfg, model_container, data_container)
        if self.cfg.checkpoint is not None:
            self.checkpoint = torch.load(self.cfg.checkpoint)
        self.tracker = SequenceTracker()

    def load_status(self):
        if self.cfg.checkpoint is not None:
            self.tracker.start_epoch = self.checkpoint["epoch"] + 1
            self.tracker.epochs_since_improvement = self.checkpoint[
                "epochs_since_improvement"
            ]
            self.tracker.best_bleu4.set_val(self.checkpoint["bleu-4"])

    def init_env(self):
        self.devices = [
            torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
        ]
        self.devices[0] = torch.device("cuda:0" if self.use_cuda else "cpu")
        time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        config_name = self.cfg.name
        self.exp_save_dir = pathlib.Path(self.cfg.save_root_dir) / config_name
        self.writer = SummaryWriter(
            pathlib.Path(self.exp_save_dir)
            / f"{config_name}_{self.cfg.mode}_{time_stamp}"
        )

    def run(self):
        if not self.exp_save_dir.is_dir():
            self.exp_save_dir.mkdir(parents=True, exist_ok=True)
        if not (self.exp_save_dir / "model_log").is_dir():
            (self.exp_save_dir / "model_log").mkdir(parents=True, exist_ok=True)

        if self.cfg.mode == "train":
            # Epochs
            self.tracker.init_run()

            self.load_status()
            for epoch in range(self.tracker.start_epoch, self.cfg.epochs + 1):
                print(f"This is {epoch}-th epoch")
                self.tracker.set_epoch(epoch)
                self.run_epoch()

                if self.tracker.epochs_since_improvement == 20:
                    break

        elif self.cfg.mode == "test":
            self.model_container.set_eval(["encoder", "decoder"])
            self.run_test()

    def run_epoch(self):
        # Scheduler
        try:
            self.scheduler()
        except AttributeError:
            pass

        # One epoch's training

        self.train_epoch()

        # One epoch's validation
        recent_bleu4 = self.validate_epoch()

        # Check if there was an improvement
        is_best = recent_bleu4 > self.tracker.best_bleu4.get_val()
        self.tracker.best_bleu4.accumulate(recent_bleu4)
        if not is_best:
            self.tracker.epochs_since_improvement += 1
            print(
                "\nEpochs since last improvement: %d\n"
                % (self.tracker.epochs_since_improvement)
            )
        else:
            self.tracker.epochs_since_improvement = 0

        if self.cfg.save_by == "epoch":
            self.save_model(
                self.tracker.epoch,
                self.tracker.total_iter,
                self.tracker.total_val_iter,
                epochs_since_improvement=self.tracker.epochs_since_improvement,
                bleu_4=self.tracker.best_bleu4.get_val(),
            )

        elif self.cfg.save_by == "best" and is_best:
            self.save_model(
                self.tracker.epoch,
                self.tracker.total_iter,
                self.tracker.total_val_iter,
                epochs_since_improvement=self.tracker.epochs_since_improvement,
                bleu_4=self.tracker.best_bleu4.get_val(),
            )

    def train_epoch(self):
        """
        Performs one epoch's training.
        """
        train_loader = self.data_container.dataloader["train"]

        self.model_container.set_train(["encoder", "decoder"])

        self.tracker.init_epoch("train")
        self.tracker.start_load_timing()
        # Batches
        for batch_idx, data_dict in enumerate(train_loader):
            self.tracker.set_batch(batch_idx)
            self.tracker.end_load_timing()
            self.train_batch(data_dict)
            self.tracker.start_load_timing()
        self.tracker.end_load_timing()

        avg_top_5 = self.tracker.total_top_5.get_avg()
        avg_train_loss = self.tracker.total_train_loss.get_avg()

        print_dict = {
            "Epoch": self.tracker.epoch,
            "training loss": avg_train_loss,
            "top 5 accuracy": avg_top_5,
        }

        self.print_state(print_dict)

    def train_batch(self, data_dict):
        self.tracker.init_batch("train")

        self.tracker.start_infer_timing()

        loss, top5 = self.train(data_dict)

        self.tracker.end_infer_timing()

        # Update losses and top 5 accuracy
        self.tracker.total_train_loss.update(loss)
        self.tracker.total_top_5.update(top5)

        print_dict = {
            "Iter": self.tracker.batch_idx,
            "training loss": loss,
            "top 5 training acc": top5,
            "infer time": self.tracker.infer_time,
            "load time": self.tracker.load_time,
        }
        write_dict = {
            "training_loss": loss,
            "top 5 train accuracy": top5,
            "load_time": self.tracker.load_time,
        }
        self.print_state(print_dict)
        self.write(self.tracker.total_iter, write_dict)

        if self.cfg.save_by == "iter":
            self.save_model(
                self.tracker.epoch,
                self.tracker.total_iter,
                self.tracker.total_val_iter,
                epochs_since_improvement=self.tracker.epochs_since_improvement,
                bleu_4=self.tracker.best_bleu4.get_val(),
            )

    def validate_epoch(self):
        """
        Performs one epoch's validation.

        :param val_loader: DataLoader for validation data.
        :param encoder: encoder model
        :param decoder: decoder model
        :param criterion: loss layer
        :return: BLEU-4 score
        """
        val_loader = self.data_container.dataloader["val"]

        self.model_container.set_eval(["encoder", "decoder"])

        self.tracker.init_epoch("val")

        self.tracker.start_load_timing()
        for batch_idx, data_dict in enumerate(val_loader):
            self.tracker.set_batch(batch_idx)
            self.tracker.end_load_timing()
            self.validate_batch(data_dict)
            self.tracker.start_load_timing()
        self.tracker.end_load_timing()

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(self.tracker.references, self.tracker.hypotheses)

        avg_top_5 = self.tracker.total_top_5.get_avg()
        print_dict = {
            "Epoch": self.tracker.epoch,
            "top 5 accuracy": avg_top_5,
            "bleu 4": bleu4,
        }
        self.print_state(print_dict)

        return bleu4

    def validate_batch(self, data_dict):
        self.tracker.init_batch("val")
        self.tracker.start_infer_timing()

        top5, img_captions, preds = self.test(data_dict)

        self.tracker.end_infer_timing()

        self.tracker.total_top_5.update(top5)
        # Store references (true captions), and hypothesis (prediction) for each image
        # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
        # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

        self.tracker.references.extend(img_captions)
        self.tracker.hypotheses.extend(preds)

        assert len(self.tracker.references) == len(self.tracker.hypotheses)

        print_dict = {
            "Iter": self.tracker.batch_idx,
            "top 5 validation acc": top5,
            "infer time": self.tracker.infer_time,
            "load time": self.tracker.load_time,
        }
        write_dict = {"top 5 validation accuracy": top5}
        self.print_state(print_dict)
        self.write(self.tracker.total_val_iter, write_dict)

    def run_test(self):
        """
        Note that this function is data-dependent, hence could not be implemented in the base class level.
        """
        raise NotImplementedError(
            "Please implement run_test for classes that inherit CaptionTrainer!"
        )

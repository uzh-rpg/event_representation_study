from base.utils.time_utils import Timer


class Tracker:
    """
    Base class that keeps track of global values such as epoch, batch index that arise in typical training scripts
    """

    def __init__(self):
        self.load_timer = Timer()
        self.infer_timer = Timer()
        self.load_time = 0.0
        self.infer_time = 0.0

    def start_load_timing(self):
        self.load_timer.start()

    def end_load_timing(self):
        self.load_time = self.load_timer.stop()

    def start_infer_timing(self):
        self.infer_timer.start()

    def end_infer_timing(self):
        self.infer_time = self.infer_timer.stop()


class MiniBatchTracker(Tracker):
    """
    Tracker for tracking values in MiniBatchTrainer
    """

    def __init__(self, **kwargs):
        super(MiniBatchTracker, self).__init__()
        self.total_iter = 0
        self.total_val_iter = 0
        self.epoch = 0
        self.batch_idx = 0
        self.total_train_loss = AverageMeter()
        self.total_val_correct = Accumulator()
        self.total_val_num = Accumulator()

    def init_run(self):
        # Set tracking parameters at the beginning of run
        self.total_iter = 0
        self.total_val_iter = 0

    def set_epoch(self, epoch):
        # Set current epoch
        self.epoch = epoch

    def init_epoch(self, mode):
        # Set tracking parameters at the beginning of run_epoch
        assert mode == "train" or mode == "val"
        if mode == "train":
            self.total_train_loss.reset()
        elif mode == "val":
            self.total_val_correct.reset()
            self.total_val_num.reset()

    def set_batch(self, batch_idx):
        # Set current batch
        self.batch_idx = batch_idx

    def init_batch(self, mode):
        # Set tracking parameters at the beginning of train_batch and val_batch
        assert mode == "train" or mode == "val"
        if mode == "train":
            self.total_iter += 1
        elif mode == "val":
            self.total_val_iter += 1

    def get_val_acc(self):
        return float(self.total_val_correct.get_val()) / float(
            self.total_val_num.get_val()
        )


class SequenceTracker(MiniBatchTracker):
    """
    Tracker for tracking values in SequenceTrainer
    """

    def __init__(self, **kwargs):
        super(SequenceTracker, self).__init__()
        self.best_bleu4 = Accumulator(accu_method="max")
        self.epochs_since_improvement = 0
        self.start_epoch = 1
        self.total_top_5 = AverageMeter()
        self.references = (
            list()
        )  # references (true captions) for calculating BLEU-4 score
        self.hypotheses = list()  # hypotheses (predictions)

    def init_run(self):
        super(SequenceTracker, self).init_run()
        self.best_bleu4.reset()
        self.epochs_since_improvement = 0
        self.start_epoch = 1

    def init_epoch(self, mode):
        super(SequenceTracker, self).init_epoch(mode)
        if mode == "train":
            self.total_top_5.reset()
        elif mode == "val":
            self.total_top_5.reset()
            self.references = list()
            self.hypotheses = list()


class AverageMeter:
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_sum(self):
        return self.sum

    def get_val(self):
        return self.val

    def get_count(self):
        return self.count

    def get_avg(self):
        return self.avg


class Accumulator:
    """
    Accumulates counts. May be used for counting number of labels that are predicted correctly.
    """

    def __init__(self, val=0, accu_method="add"):
        # Method can be one of 'add', 'sub', 'max', 'min'
        self.val = val
        self.accu_method = accu_method

    def accumulate(self, target):
        if self.accu_method == "add":
            self.val += target
        elif self.accu_method == "sub":
            self.val -= target
        elif self.accu_method == "max":
            self.val = max(self.val, target)
        elif self.accu_method == "min":
            self.val = min(self.val, target)

    def get_val(self):
        return self.val

    def reset(self):
        self.val = 0

    def set_val(self, val):
        self.val = val

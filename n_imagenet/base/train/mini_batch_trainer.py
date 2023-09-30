from abc import abstractmethod
from base.models.model_container import ModelContainer
from base.data.data_container import DataContainer
from base.train.trainer import Trainer


class MiniBatchTrainer(Trainer):
    """
    Abstract class for trainers dedicated to mini-batch training.
    """

    def __init__(
        self,
        cfg,
        model_container: ModelContainer,
        data_container: DataContainer,
        **kwargs
    ):
        """
        Initialize Trainer instance

        Args:
            cfg: Parsed config file
            model_container: ModelContainer instance containing all the models used for training
            data_container: DataContainer instance containing all the datasets/dataloaaders used for training
            exp_save_dir: Directory in which all the experiment results will be saved
            devices: List of torch.device(s) used for parallelizing models
        """
        super(MiniBatchTrainer, self).__init__(cfg, model_container, data_container)

    @abstractmethod
    def train_batch(self, global_dict, data_dict):
        """
        Wrapping function for self.train(). Tracks global state and manages operations within each batch.

        Args:
            global_dict: Dictionary keeping track of global variables used throughout training
            data_dict: Dictionary containing neccessary data/labels used for training
        """
        pass

    @abstractmethod
    def train_epoch(self, global_dict):
        """
        Wrapping function for self.train_batch(). Tracks global state and manages operations within each epoch.

        Args:
            global_dict: Dictionary keeping track of global variables used throughout training
        """
        pass

    @abstractmethod
    def validate_batch(self, global_dict, data_dict):
        """
        Wrapping function for self.test(). Method dedicated for validation operations in the batch level.

        Args:
            global_dict: Dictionary keeping track of global variables used throughout training
            data_dict: Dictionary containing neccessary data/labels used for training
        """
        pass

    @abstractmethod
    def validate_epoch(self, global_dict):
        """
        Wrapping function for self.validate_batch(). Method dedicated for validation operations in the epoch level.

        Args:
            global_dict: Dictionary keeping track of global variables used throughout training
        """
        pass

    @abstractmethod
    def run_epoch(self, global_dict):
        """
        Run a unit training epoch. This will consist of calling self.train_epoch() and self.validate_epoch().

        Args:
            global_dict: Dictionary keeping track of global variables used throughout training
        """
        pass

    @abstractmethod
    def run_test(self):
        """
        Evaluate model over test data and save the results.
        """
        pass

    @abstractmethod
    def run(self):
        """
        Uppermost method dedicated to running trainer. This will consist of initializing global_dict and
        calling self.run_epoch(), self.run_test().
        """
        pass

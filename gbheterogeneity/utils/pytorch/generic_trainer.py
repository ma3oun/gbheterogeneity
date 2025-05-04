import abc


class GenericTrainer(abc.ABC):
    """Build a generic trainer class."""

    def __init__(self) -> None:
        super().__init__()
        self.best_model_path = None
        self.best_model_score = None

    @abc.abstractmethod
    def training_step(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def validation_step(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def training_epoch_end(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def validation_epoch_end(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def save_model(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def resume_training(self, *args, **kwargs):
        pass

"""Provide common configurations for algorithms."""
import abc
import dataclasses
import pathlib
import hashlib
import json
from few_shot_nlp.utils.constants import ProblemType


class Config(abc.ABC):
    """Base configuration for experiments."""

    @abc.abstractproperty
    def key(self) -> dict:
        """The primary key for this config."""

    @abc.abstractmethod
    def get_output_folder(self, parent_dir: str) -> pathlib.Path:
        """Returns the path to the desired output folder.

        :param parent_dir: The parent directory for the output folder.

        :note: This method does not guarantee that the result exists.
        """


@dataclasses.dataclass(kw_only=True)
class SharedConfig(Config):
    train_split: str
    dataset_name: str
    dataset_seed: str
    model_folder: str
    model_path: str
    max_length: int
    learning_rate: float
    batch_size: int
    datasets_path: str = "data"
    class_col: str = "Class"

    @property
    def key(self) -> dict:
        return {
            "train_split": self.train_split,
            "dataset_name": self.dataset_name,
            "dataset_seed": self.dataset_seed,
            "model_path": self.model_path,
            "max_length": self.max_length,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
        }

    def get_output_folder(self, parent_dir: str) -> pathlib.Path:
        return (
            pathlib.Path(parent_dir)
            / self.dataset_name
            / self.dataset_seed
            / self.model_path
            / self.train_split
        )


@dataclasses.dataclass(kw_only=True)
class SetFitConfig(Config):
    shared_config: SharedConfig
    num_iterations: int
    num_epochs: int

    def __post_init__(self):
        if self.num_iterations is None:
            raise ValueError("num_iterations cannot be None.")
        if self.num_epochs is None:
            raise ValueError("num_epochs cannot be None.")

    @property
    def key(self) -> dict:
        return self.shared_config.key | {
            "algorithm": "setfit",
            "setfit_num_iterations": self.num_iterations,
            "setfit_num_epochs": self.num_epochs,
        }

    def get_output_folder(self, parent_dir: str) -> pathlib.Path:
        folder_hash = hashlib.md5(json.dumps(self.key).encode("utf8")).hexdigest()
        return self.shared_config.get_output_folder(parent_dir) / folder_hash


@dataclasses.dataclass(kw_only=True)
class FineTuningConfig(Config):
    shared_config: SharedConfig
    gradient_accumulation_steps: int
    max_steps: int
    problem_type: ProblemType

    def __post_init__(self):
        if self.gradient_accumulation_steps is None:
            raise ValueError("gradient_accumulation_steps cannot be None.")
        if self.max_steps is None:
            raise ValueError("max_steps cannot be None.")
        if self.problem_type is None:
            raise ValueError("problem_type cannot be None.")

    @property
    def key(self) -> dict:
        return self.shared_config.key | {
            "algorithm": "fine-tuning",
            "fine_tuning_gradient_accumulation_steps": self.gradient_accumulation_steps,
            "fine_tuning_max_steps": self.max_steps,
            "fine_tuning_problem_type": self.problem_type.value,
        }

    def get_output_folder(self, parent_dir: str) -> pathlib.Path:
        folder_hash = hashlib.md5(json.dumps(self.key).encode("utf8")).hexdigest()
        return self.shared_config.get_output_folder(parent_dir) / folder_hash

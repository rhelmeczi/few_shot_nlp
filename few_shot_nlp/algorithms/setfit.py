import numpy as np
import datasets
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitTrainer, SetFitModel
from few_shot_nlp.utils.config import SetFitConfig


def train_and_predict_proba(
    config: SetFitConfig,
    train_dataset: datasets.Dataset,
    test_dataset: datasets.Dataset,
) -> np.ndarray:
    model = SetFitModel.from_pretrained(config.shared_config.model_path)
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        loss_class=CosineSimilarityLoss,
        num_epochs=config.num_epochs,
        num_iterations=config.num_iterations,
        batch_size=config.shared_config.batch_size,
        learning_rate=config.shared_config.learning_rate,
    )
    trainer.train(max_length=config.shared_config.max_length)

    return model.predict_proba(test_dataset["text"])

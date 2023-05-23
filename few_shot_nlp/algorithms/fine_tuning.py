import numpy as np
import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import few_shot_nlp.utils.config


class SequenceClassifierTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        return (outputs.loss, outputs) if return_outputs else outputs.loss


def train_and_predict_proba(
    config: few_shot_nlp.utils.config.FineTuningConfig,
    output_parent_dir: str,
    train_dataset: datasets.Dataset,
    test_dataset: datasets.Dataset,
    num_labels: int,
) -> np.ndarray:
    args = TrainingArguments(
        output_dir=config.get_output_folder(output_parent_dir),
        per_device_train_batch_size=config.shared_config.batch_size,
        per_device_eval_batch_size=config.shared_config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_steps=config.max_steps,
        weight_decay=0.01,
        load_best_model_at_end=False,
        save_steps=250,
        save_strategy="no",
        learning_rate=config.shared_config.learning_rate,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.shared_config.model_path,
        num_labels=num_labels,
        problem_type=config.problem_type.value,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.shared_config.model_path, max_len=config.shared_config.max_length
    )

    trainer = SequenceClassifierTrainer(
        model,
        args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    return trainer.predict(test_dataset).predictions

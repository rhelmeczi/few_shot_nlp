import pathlib
import datasets
from transformers import AutoTokenizer
import few_shot_nlp.utils.config
from few_shot_nlp.utils.constants import Algorithm


def load_dataset(
    *,
    dataset_name: str,
    datasets_path: str,
    seed: str,
    split: str,
):
    return datasets.load_from_disk(pathlib.Path(datasets_path) / dataset_name / seed)[
        split
    ]


def prepare_dataset(
    dataset: datasets.Dataset, model_path: str, max_length: int, algorithm: Algorithm
) -> datasets.Dataset:
    """As SetFit won't let us pass in pairs, we need to reformulate to pairs for it."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if algorithm is Algorithm.SETFIT:
        if not ("Text1" in dataset.features and "Text2" in dataset.features):
            return dataset

        dataset = dataset.map(
            lambda row: {
                "text": tokenizer.decode(
                    tokenizer(
                        row["Text1"],
                        row["Text2"],
                        max_length=max_length,
                        truncation="longest_first",
                        padding=True,
                    )["input_ids"][1:-1],
                )
            },
            remove_columns={"Text1", "Text2"},
        )
    elif algorithm is Algorithm.FINE_TUNING:

        def tokenize(batch):
            return tokenizer(
                batch["Text1"],
                batch["Text2"],
                padding=True,
                truncation="longest_first",
                max_length=max_length,
            )

        dataset = dataset.map(tokenize, batched=True)
    return dataset


def prepare_labels(
    dataset: datasets.Dataset, label2id: dict, class_col: str
) -> datasets.Dataset:
    return dataset.map(lambda row: {"label": label2id[row[class_col]]})


def load_and_prepare_train_test_data(
    config: few_shot_nlp.utils.config.SharedConfig,
    algorithm: Algorithm,
) -> tuple[datasets.Dataset, datasets.Dataset]:
    train_dataset = load_dataset(
        dataset_name=config.dataset_name,
        seed=config.dataset_seed,
        datasets_path=config.datasets_path,
        split=config.train_split,
    )

    test_dataset = load_dataset(
        dataset_name=config.dataset_name,
        seed=config.dataset_seed,
        datasets_path=config.datasets_path,
        split="test",
    )
    train_dataset = prepare_dataset(
        train_dataset,
        config.model_path,
        config.max_length,
        algorithm=algorithm,
    )
    test_dataset = prepare_dataset(
        test_dataset,
        config.model_path,
        config.max_length,
        algorithm=algorithm,
    )
    label_list = test_dataset.unique(config.class_col)
    id2label = {idx: label for idx, label in enumerate(label_list)}
    label2id = {label: idx for idx, label in id2label.items()}

    train_dataset = prepare_labels(train_dataset, label2id, config.class_col)
    test_dataset = prepare_labels(test_dataset, label2id, config.class_col)
    return train_dataset, test_dataset, id2label

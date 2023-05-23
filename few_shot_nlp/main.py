from typing import Optional, Annotated
import json
import typer
from rich import print
from sklearn.metrics import classification_report
from few_shot_nlp.utils.constants import Algorithm, ProblemType
import few_shot_nlp.algorithms.setfit
import few_shot_nlp.algorithms.fine_tuning
import few_shot_nlp.utils.preprocessing
from few_shot_nlp.utils.config import SharedConfig, SetFitConfig, FineTuningConfig

app = typer.Typer(rich_help_panel=True)


@app.command(help="Train and evaluate a few-shot learning algorithm.")
def main(
    dataset_name: Annotated[
        str, typer.Argument(help="The name of the dataset to use.")
    ],
    dataset_seed: Annotated[
        str,
        typer.Argument(help="The random seed used to create the target training data."),
    ],
    train_split: Annotated[
        str, typer.Argument(help="The name of the training split to train on.")
    ],
    algorithm: Annotated[
        Algorithm, typer.Argument(help="The few-shot learning approach to use.")
    ],
    model_path: Annotated[
        str,
        typer.Argument(
            help=(
                "The path to the Hugging Face model to use. This can be the name of a"
                " model on the Hugging Face hub."
            )
        ),
    ],
    max_length: Annotated[
        int, typer.Argument(help="The maximum sequence length to use.", min=1)
    ],
    learning_rate: Annotated[
        float, typer.Argument(help="The learning rate to use in training.")
    ],
    batch_size: Annotated[
        int, typer.Argument(help="The training and evaluation batch sizes.", min=1)
    ],
    model_folder: Optional[str] = Annotated[
        str,
        typer.Argument(
            help=(
                "The folder containing the Hugging Face model. If provided, this"
                " argument will be prefixed to `model_path`."
            )
        ),
    ],
    # shared optional arguments
    output_parent_dir: Annotated[
        str, typer.Argument(help="The folder to save results to.")
    ] = "output",
    datasets_path: Annotated[
        str, typer.Option(help="The folder containing the dataset to load.")
    ] = "data",
    class_col: Annotated[
        str,
        typer.Option(
            help="The name of the column in the dataset containing the labels."
        ),
    ] = "Class",
    print_results: Annotated[
        bool,
        typer.Option(
            help=(
                "Whether or not to print the final results (primary key and report) to"
                " the terminal."
            )
        ),
    ] = True,
    # SetFit optional arguments
    num_iterations: Annotated[
        Optional[int],
        typer.Option(
            help="SetFit only. The number of text pairs to generate.",
            rich_help_panel="SetFit options",
        ),
    ] = None,
    num_epochs: Annotated[
        Optional[int],
        typer.Option(
            help="SetFit only. The number of training epochs.",
            rich_help_panel="SetFit options",
        ),
    ] = None,
    # Fine-tuning optional arguments
    gradient_accumulation_steps: Annotated[
        Optional[int],
        typer.Option(
            help="Fine-tuning only. Gradient accumulation steps.",
            rich_help_panel="Fine-tuning options",
        ),
    ] = None,
    max_steps: Annotated[
        Optional[int],
        typer.Option(
            help="Fine-tuning only. The number of training steps.",
            rich_help_panel="Fine-tuning options",
        ),
    ] = None,
    problem_type: Annotated[
        ProblemType,
        typer.Option(
            help=(
                "Fine-tuning only. The specifies single or multi-label classification."
            ),
            rich_help_panel="Fine-tuning options",
        ),
    ] = ProblemType.SINGLE_LABEL_CLASSIFICATION.value,
):
    shared_config = SharedConfig(
        train_split=train_split,
        dataset_name=dataset_name,
        dataset_seed=dataset_seed,
        model_folder=model_folder,
        model_path=model_path,
        max_length=max_length,
        learning_rate=learning_rate,
        batch_size=batch_size,
        datasets_path=datasets_path,
        class_col=class_col,
    )
    (
        train_dataset,
        test_dataset,
        id2label,
    ) = few_shot_nlp.utils.preprocessing.load_and_prepare_train_test_data(
        shared_config, algorithm=algorithm
    )

    match algorithm:
        case Algorithm.SETFIT:
            config = SetFitConfig(
                shared_config=shared_config,
                num_iterations=num_iterations,
                num_epochs=num_epochs,
            )
            y_pred_proba = few_shot_nlp.algorithms.setfit.train_and_predict_proba(
                config, train_dataset=train_dataset, test_dataset=test_dataset
            )
        case Algorithm.FINE_TUNING:
            config = FineTuningConfig(
                shared_config=shared_config,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_steps=max_steps,
                problem_type=problem_type,
            )
            y_pred_proba = few_shot_nlp.algorithms.fine_tuning.train_and_predict_proba(
                config,
                output_parent_dir=output_parent_dir,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                num_labels=len(id2label),
            )
        case _:
            raise ValueError(f"unsupported algorithm: {algorithm}")

    y_pred_id = y_pred_proba.argmax(axis=1)
    y_pred_label = [id2label[int(idx)] for idx in y_pred_id]
    y_true = test_dataset["Class"]

    report = classification_report(y_true=y_true, y_pred=y_pred_label, output_dict=True)

    if print_results:
        print(json.dumps(config.key, indent=4))
        print(json.dumps(report, indent=4))

    output_dir = config.get_output_folder(output_parent_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(output_dir / "metadata.json", "w", encoding="utf8") as fp:
        json.dump(config.key, fp, indent=4)

    prediction_results = {
        "id2label": id2label,
        "pred_proba": [[float(prob) for prob in row] for row in y_pred_proba],
        "true_labels": y_true,
    }

    with open(output_dir / "prediction_results.json", "w", encoding="utf8") as fp:
        json.dump(prediction_results, fp)

    with open(output_dir / "report.json", "w", encoding="utf8") as fp:
        json.dump(report, fp, indent=4)


if __name__ == "__main__":
    app()

from typing import Optional, Annotated
import pathlib
import datasets
import typer
import numpy as np

app = typer.Typer(
    rich_help_panel=True,
)


@app.command(
    help=(
        "Download and generate the data files for testing this repository. "
        + typer.style(
            (
                "WARNING: if you change any of the default arguments, the commands"
                " provided in the README will need to be adjusted."
            ),
            fg=typer.colors.YELLOW,
            bold=True,
        )
    ),
)
def main(
    path: Annotated[
        str,
        typer.Option(
            help=(
                "The path to the dataset. This can be the name of a dataset on the"
                " Hugging Face hub."
            )
        ),
    ] = "super_glue",
    name: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "The name of the dataset configuration to use. If not provided, then"
                " the dataset should be fully specified by the `path` argument."
            )
        ),
    ] = "boolq",
    num_iterations: Annotated[
        int,
        typer.Option(
            help=(
                "The number of distict datasets to create. These datasets are created"
                " using random seeds and allow for averaging results."
            )
        ),
    ] = 3,
    train_sizes: Annotated[
        list[int], typer.Option(help="The desired sizes of the training datasets.")
    ] = [25, 50, 100],
    test_size: Annotated[
        int, typer.Option(help="The desired size of the generated test datasets.")
    ] = 2000,
    output_path: Annotated[
        pathlib.Path, typer.Option(help="The path to save the generated data files to.")
    ] = pathlib.Path("data"),
    text1_col: Annotated[
        str,
        typer.Option(
            help=(
                "The column in the dataset that corresponds to the first sentence in"
                " the sentence pair."
            )
        ),
    ] = "question",
    text2_col: Annotated[
        str,
        typer.Option(
            help=(
                "The column in the dataset that corresponds to the second sentence in"
                " the sentence pair."
            )
        ),
    ] = "passage",
    class_col: Annotated[
        str,
        typer.Option(
            help="The column in the dataset that corresponds to the class name."
        ),
    ] = "label",
    entropy: Annotated[
        int,
        typer.Option(
            help=(
                "The entropy to use for shuffling data and creating test/training data."
            )
        ),
    ] = 1245330007238731,
):
    train_split = datasets.load_dataset(path, name, split="train")
    validation_split = datasets.load_dataset(path, name, split="validation")
    train_split = train_split.rename_columns(
        {text1_col: "Text1", text2_col: "Text2", class_col: "Class"}
    )
    validation_split = validation_split.rename_columns(
        {text1_col: "Text1", text2_col: "Text2", class_col: "Class"}
    )
    if name:
        output_path /= name
    output_path.mkdir(parents=True, exist_ok=True)

    for seed in np.random.SeedSequence(entropy, pool_size=max(4, num_iterations)).pool[
        :num_iterations
    ]:
        dataset_dict = datasets.DatasetDict()
        dataset_dict["test"] = validation_split.shuffle(seed=seed).select(
            range(test_size)
        )
        for train_size in train_sizes:
            dataset_dict[f"train-{train_size}"] = train_split.shuffle(seed=seed).select(
                range(train_size)
            )
        dataset_dict.select_columns(["Text1", "Text2", "Class"]).save_to_disk(
            output_path / str(seed)
        )


if __name__ == "__main__":
    app()

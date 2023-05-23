"""There are a lot of example scripts to run, and their format sometimes varies between
operating systems. Accordingly, we'll write a script generator here."""
from typing import Annotated
import pathlib
import datasets
import typer
from few_shot_nlp.utils.constants import Algorithm

app = typer.Typer(
    rich_help_panel=True,
)


@app.command(help="Generate scripts for training.")
def main(
    dataset_name: Annotated[
        str,
        typer.Option(
            help=(
                "The path to the dataset parent directory. The contents of this folder"
                " should be the seed folders containing the datasets."
            )
        ),
    ] = "boolq",
    datasets_path: Annotated[
        str, typer.Option(help="The folder containing the dataset to load.")
    ] = "data",
    main_script_path: Annotated[
        str, typer.Option(help="The path to the main file.")
    ] = "./few_shot_nlp/main.py",
    algorithms: Annotated[
        list[Algorithm], typer.Option(help="The algorithms to use.")
    ] = [Algorithm.FINE_TUNING.value, Algorithm.SETFIT.value],
    output_script_path: Annotated[
        pathlib.Path, typer.Option(help="The path to save the resulting scripts to.")
    ] = pathlib.Path("training_scripts.sh"),
):
    all_scripts = []
    for seed_folder in (pathlib.Path(datasets_path) / dataset_name).iterdir():
        dataset: datasets.DatasetDict = datasets.load_from_disk(seed_folder)
        for split in dataset:
            if split == "test":
                continue
            else:
                for algorithm in algorithms:
                    script_args = [
                        "python",
                        main_script_path,
                        dataset_name,
                        seed_folder.name,
                        split,
                        algorithm.value,
                        "bert-base-uncased",
                        "128",
                        "1e-5",
                        "4",
                    ]
                    if algorithm is Algorithm.FINE_TUNING:
                        script_args += [
                            "--gradient-accumulation-steps",
                            "1",
                            "--max-steps",
                            "250",
                        ]
                    elif algorithm is Algorithm.SETFIT:
                        script_args += ["--num-epochs", "1", "--num-iterations", "20"]
                    all_scripts.append(script_args)
        with open(output_script_path, "w", encoding="utf8") as fp:
            for script in all_scripts:
                fp.write(" ".join(script) + "\n")


if __name__ == "__main__":
    app()

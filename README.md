# few_shot_nlp

This repository contains some of the code from my Master's thesis.

## Usage with Google Colab

You can upload the notebook at `examples/colab_notebook.ipynb` and follow the instructions there to get started with training and evaluation for free with accelerated hardware.
You can refer back to this README for usage on your local computer or for some more in depth instructions.

## Getting Started

Add the project home folder to your `PYTHONPATH`:

```console
export PYTHONPATH=/path/to/project:$PYTHONPATH
```

On Windows:
1. In the search bar, type "Edit the system environment variables" and select that option.
2. Click on "Environment Variables".
3. Under system variables, find `PYTHONPATH` or create it if it does not exist.
4. Add the path the project home folder to `PYTHONPATH`.

Then, install the required packages:

```
pip install -r requirements.txt
```

**Note that by default, these requirements install torch for CPU. If you have a GPU, be sure to include CUDA in your torch installation.**

## Running the code

There are a few example scripts setup to make it easy to run this code.
The main entrance point here is `few_shot_nlp/main.py`.
That is where you'll run experiments from.
You can type

```
python few_shot_nlp/main.py --help
```

to get a list of all of the options available to you.

## Code Examples

As the data structure and scripts can be a bit difficult to grasp, you'll find some examples
in the `examples` folder to get started.
First, run

```
python examples/prepare_test_data.py
```

to generate example data (this will appear in the `data` folder).
This data generation file has some options that you can play with, so feel free 
to run 

```
python examples/prepare_test_data.py --help
```

**For the purposes of following the example and the subsequent scripts in this README, do not change from the default arguments in `prepare_test_data.py`**

Now that we've generated our data, we want to run some scripts.
We can generate scripts that call the main function using

```
python examples/generate_training_scripts.py
```

This command creates a bash script at the top level of this directory.
Running this script will take some time, and will populate your output folder.
The `--help` command is available if you'd like to make some changes of your own.

## Analyzing the results

The last step in running these experiments is, of course, analyzing them!
For this, see `examples/analysis.ipynb` for some aggregation and plotting commands.

## Running custom experiments 

You **do not** need to run all of the generated scripts.
To make sure this repository is working for you, you can largely rely on running
a script for both the setfit algorithm and the fine-tuning algorithm:

```
python ./few_shot_nlp/main.py boolq 1379487842 train-25 fine-tuning bert-base-uncased 128 1e-5 4 --gradient-accumulation-steps 1 --max-steps 250
python ./few_shot_nlp/main.py boolq 1379487842 train-25 setfit bert-base-uncased 128 1e-5 4 --num-epochs 1 --num-iterations 20
```

If you just want to make sure the code works, consider reducing `num-iterations`
to 1 for SetFit and `max-steps` to 1 for fine-tuning. The results will have very poor results
but if the code runs start to finish its a good sign that the installation went well.
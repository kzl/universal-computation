# universal-computation

## Overview
Official codebase for [Pretrained Transformers as Universal Computation Engines](https://arxiv.org/abs/2103.05247).
Contains demo notebook and scripts to reproduce experiments.

### Project Demo

For a minimal demonstration of frozen pretrained transformers, see ```demo.ipynb```.
You can run the notebook which reproduces the Bit XOR experiment in a couple minutes, and visualizes the learned
attention maps.


### Status
Project is released but will receive updates soon.

Currently the repo supports the following tasks:
```
['bit-memory', 'bit-xor', 'listops', 'mnist', 'cifar10', 'cifar10-gray']
```

Note that CIFAR-10 LRA is ```cifar10-gray``` with a patch size of 1.

## Usage

### Installation

1. Install Anaconda environment:
    ```
    $ conda env create -f environment.yml
    ```

2. Add ```universal-computation/``` to your PYTHONPATH, i.e. add this line to your ```~/.bashrc```:
    ```
    export PYTHONPATH=~/universal-computation:$PYTHONPATH
    ```

### Downloading datasets

Datasets are stored in ```data/```.
MNIST and CIFAR-10 are automatically downloaded by Pytorch upon starting experiment.

#### Listops

Download the files for Listops from [Long Range Arena](https://github.com/google-research/long-range-arena).
Move the ```.tsv``` files into ```data/listops```.
There should be three files: ```basic_test, basic_train, basic_val```.
The script evaluates on the validation set by default.

#### Remote homology

Support coming soon.

### Running experiments

You can run experiments with:
```
python scripts/run.py
```

Adding ```-w True``` will log results to Weights and Biases.

## Citation

```
@article{lu2021fpt,
  title={Pretrained Transformers as Universal Computation Engines},
  author={Kevin Lu and Aditya Grover and Pieter Abbeel and Igor Mordatch},
  journal={arXiv preprint arXiv:2103.05247},
  year={2021}
}
```

## License

MIT

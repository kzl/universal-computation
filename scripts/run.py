from universal_computation.experiment import run_experiment
from argparse import ArgumentParser
import sys


if __name__ == '__main__':
    parser = ArgumentParser(description='Pick the task to be run.')
    parser.add_argument('name', help='the name of the experiment')
    parser.add_argument('task', help='the name of the task to be run')
    parser.add_argument('--model', default='gpt2', help='the model to use')
    args = parser.parse_args()

    experiment_name = args.name

    experiment_params = dict(
        task=args.task,
        n=1000,                # ignored if not a bit task
        num_patterns=5,        # ignored if not a bit task
        patch_size=16,

        model_name=args.model,
        pretrained=True,       # if vit this is forced to true, if lstm this is forced to false

        freeze_trans=True,     # if False, we don't check arguments other than in and out
        freeze_in=False,
        freeze_pos=False,
        freeze_ln=False,
        freeze_attn=True,
        freeze_ff=True,
        freeze_out=False,

        in_layer_sizes=None,   # not in paper, but can specify layer sizes for an MLP,
        out_layer_sizes=None,  # ex. [32, 32] creates a 2-layer MLP with dimension 32

        learning_rate=1e-3,
        batch_size=4,
        dropout=0.1,
        orth_gain=1.41,        # orthogonal initialization of input layer
    )

    sys.argv = ['']  # clear args since run_experiment also has an argparser
    run_experiment(experiment_name, experiment_params)

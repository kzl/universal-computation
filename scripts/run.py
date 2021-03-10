from universal_computation.experiment import run_experiment


if __name__ == '__main__':

    experiment_name = 'fpt'

    experiment_params = dict(
        task='bit-memory',
        n=1000,                # ignored if not a bit task
        num_patterns=5,        # ignored if not a bit task
        patch_size=50,

        model_name='gpt2',
        pretrained=True,

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
        batch_size=2,
        dropout=0.1,
        orth_gain=1.41,
    )

    run_experiment(experiment_name, experiment_params)

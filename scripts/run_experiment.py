from experiment_utils.base_experiment import run_experiment


if __name__ == '__main__':

    experiment_name = 'bit_mem_gpt2'

    sweep_values = dict(
        task=['bit-memory'],
        n=[1000],
        num_patterns=[5],
        patch_size=[50],

        batch_size=[2],

        model_type=['transformer'],
        model_name=['gpt2'],
        embedding_size=[768],  # ignore this if we specify model_name
        num_layers=[None],
        pretrained=[True],

        freeze_middle=[True],
        freeze_in=[False],
        freeze_out=[False],
        freeze_ln=[False],
        freeze_pos=[False],

        in_layer_sizes=[None],
        out_layer_sizes=[None],
        use_adapter=[False],
        freeze_adapter=[True],

        learning_rate=[1e-3],
        dropout=[0.1],
        orth_gain=[1.41],
    )

    run_experiment(experiment_name, sweep_values)

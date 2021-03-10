import multiprocessing
import os
import random
import time

from doodad.easy_sweep.hyper_sweep import run_sweep_doodad, Sweeper

ctx = multiprocessing.get_context("spawn")
multiprocessing = ctx


def run_sweep_multi_gpu(
        run_method,
        params,
        repeat=1,
        num_cpu=multiprocessing.cpu_count(),
        num_gpu=3,
        exps_per_gpu=1,
        min_gpu=0,
):
    sweeper = Sweeper(params, repeat, include_name=False)
    gpu_frac = 0.9 / exps_per_gpu
    num_runs = num_gpu * exps_per_gpu
    cpu_per_gpu = num_cpu / num_gpu
    exp_args = []
    for config in sweeper:
        exp_args.append((config, run_method))
    random.shuffle(exp_args)
    processes = [None] * num_runs
    run_info = [(i, (i * cpu_per_gpu, (i + 1) * cpu_per_gpu)) for i in range(num_gpu)] * exps_per_gpu
    for kwarg, run in exp_args:
        launched = False
        while not launched:
            for idx in range(num_runs):
                if processes[idx] is None or not processes[idx].is_alive():
                    # kwarg['gpu_frac'] = gpu_frac
                    p = multiprocessing.Process(target=run, kwargs=kwarg)
                    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (min_gpu+run_info[idx][0])
                    os.system("taskset -p -c %d-%d %d" % (run_info[idx][1] + (os.getpid(),)))
                    p.start()
                    processes[idx] = p
                    launched = True
                    break
            if not launched:
                time.sleep(10)

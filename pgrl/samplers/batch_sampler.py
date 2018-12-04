import logging

import numpy as np
import time as timer
import pgrl.samplers.base_sampler as base_sampler
import pgrl.samplers.trajectory_sampler as trajectory_sampler
from pgrl.utils.gym_utils import get_environment

logging.disable(logging.CRITICAL)

def sample_paths(N,
    policy,
    T=1e6,
    env=None,
    env_name=None,
    seed=None,
    num_cpu='max',
    paths_per_call=5,
    mode='sample'):

    if num_cpu == 1:
        return sample_paths_one_core(N, policy, T, env, env_name, seed, mode)
    else:
        start_time = timer.time()
        print("####### Gathering Samples #######")
        sampled_so_far = 0
        paths_so_far = 0
        paths = []
        while sampled_so_far <= N:
            if seed is None:
                new_paths = trajectory_sampler.sample_paths_parallel(paths_per_call,
                            policy, T, env_name, seed, num_cpu, suppress_print=True, mode=mode)
            else:
                seed += paths_so_far
                new_paths = trajectory_sampler.sample_paths_parallel(paths_per_call,
                            policy, T, env_name, seed, num_cpu, suppress_print=True, mode=mode)

            for path in new_paths:
                paths.append(path)
            paths_so_far += paths_per_call
            new_samples = np.sum([len(p['rewards']) for p in new_paths])
            sampled_so_far += new_samples
        print("======= Samples Gathered  ======= | >>>> Time taken = %f " % (timer.time()-start_time))
        print("................................. | >>>> # samples = %i # trajectories = %i "
              % (sampled_so_far, paths_so_far))
        return paths

def sample_paths_one_core(N,
    policy,
    T=1e6,
    env=None,
    env_name=None,
    seed=None,
    mode='sample'):

    if env_name is None and env is None:
        print("No environment specified! Error will be raised")
    if env is None: env = get_environment(env_name)
    #if seed is not None: env.env.seed(seed)
    T = min(T, env.horizon)

    start_time = timer.time()

    print("####### Gathering Samples #######")
    sampled_so_far = 0
    paths = []
    seed_sample = seed if seed is not None else 0

    while sampled_so_far < N:
        if mode == 'sample':
            this_path = base_sampler.do_rollout(1, policy, T, env, env_name, seed_sample)
        elif mode == 'evaluation':
            this_path = eval_sampler.do_evaluation_rollout(1, policy, env, env_name, seed_sample)
        else:
            print("Mode has to be either 'sample' for training time or 'evaluation' for test time performance")
            break
        paths.append(this_path[0])
        seed_sample += 1
        sampled_so_far += len(this_path[0]["rewards"])

    print("======= Samples Gathered  ======= | >>>> Time taken = %f " % (timer.time()-start_time))
    print("................................. | >>>> # samples = %i # trajectories = %i "
          % (sampled_so_far, len(paths)))
    return paths

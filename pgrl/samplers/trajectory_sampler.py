import logging
logging.disable(logging.CRITICAL)

import numpy as np
import copy
import dill
import multiprocessing as mp
import time as timer
import pgrl.samplers.base_sampler as base_sampler
import pgrl.samplers.evaluation_sampler as eval_sampler

def sample_paths(N,
                 T=1e6,
                 env=None,
                 env_name=None,
                 seed='None',
                 mode='sample',
                 render=False):

    global _global_policy_info
    policy = _global_policy_info
    if mode == 'sample':
        return base_sampler.do_rollout(N,
                                       policy,
                                       T,
                                       env,
                                       env_name,
                                       seed,
                                       stochastic=True)
    elif mode == 'evaluation':
        return eval_sampler.do_evaluation_rollout(N,
                                                  policy,
                                                  T,
                                                  env,
                                                  env_name,
                                                  seed,
                                                  stochastic=False,
                                                  render=render)
    else:
        raise NotImplementedError()

def sample_paths_star(*args_list):
    return sample_paths(*args_list)

def sample_paths_parallel(N,
                          policy,
                          T=1e6,
                          env_name=None,
                          seed=None,
                          num_cpu='max',
                          max_process_time=100,
                          max_timeouts=4,
                          suppress_print=False,
                          mode='sample',
                          render=False):

    global _global_policy_info
    _global_policy_info = policy

    if num_cpu is None or num_cpu == 'max':
        num_cpu = mp.cpu_count()
    elif num_cpu == 1:
        return sample_paths(N,
                            T,
                            None,
                            env_name,
                            seed,
                            mode=mode,
                            render=render)
    else:
        num_cpu = min(mp.cpu_count(), num_cpu)

    if not(mode == 'sample' or mode == 'evaluation'):
        raise NotImplementedError()

    paths_per_cpu = int(np.ceil(N/num_cpu))
    args_list = []
    for i in range(num_cpu):
        if seed is None:
            args_list_cpu = [paths_per_cpu, T,
                             None, env_name, None,
                             mode]
        else:
            args_list_cpu = [paths_per_cpu, T, None,
                             env_name, seed + i*paths_per_cpu,
                             mode]
        args_list.append(args_list_cpu)

    if suppress_print is False:
        start_time = timer.time()
        print("####### Gathering Samples #######")

    results = _try_multiprocess(args_list, num_cpu,
                                max_process_time, max_timeouts, mode)

    paths = []
    for result in results:
        for path in result:
            paths.append(path)

    if suppress_print is False:
        print("======= Samples Gathered  ======= | >>>> Time taken = %f "
              % (timer.time()-start_time))

    return paths

def _try_multiprocess(args_list, num_cpu, max_process_time, max_timeouts, mode):

    if max_timeouts == 0:
        return None

    pool = mp.Pool(processes=num_cpu, maxtasksperchild=1)

    if mode is 'sample':
        parallel_runs = [pool.apply_async(sample_paths_star,
                                          args=(args_list[i]))
                         for i in range(num_cpu)]
    elif mode is 'evaluation':
        parallel_runs = [pool.apply_async(sample_paths_star,
                                          args=(args_list[i]))
                         for i in range(num_cpu)]

    try:
        results = [p.get(timeout=max_process_time) for p in parallel_runs]
    except Exception as e:
        print(str(e))
        print("Timeout Error raised... Trying again")
        pool.close()
        pool.terminate()
        pool.join()
        return _try_multiprocess(args_list,
                                 num_cpu,
                                 max_process_time,
                                 max_timeouts-1,
                                 mode=mode)

    pool.close()
    pool.terminate()
    pool.join()
    return results

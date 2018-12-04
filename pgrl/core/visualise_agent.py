import logging

from tabulate import tabulate
from pgrl.utils.gym_utils import GymEnv
import pgrl.utils.tf_utils as tfu
import pgrl.samplers.evaluation_sampler as eval_sampler
import numpy as np
import tensorflow as tf
import pickle
import time as timer
import os
import copy

logging.disable(logging.CRITICAL)

def visualise_agent(job_name,
                    agent,
                    seed=0,
                    num_cpu=1,
                    model_file=None,
                    stochastic=False,
                    num_traj=10):

    sess = tfu.make_session(num_cpu=num_cpu)
    sess.__enter__()
    sess.run(tf.global_variables_initializer())
    if model_file is None:
        print('Please load model file!')
        quit()
    else:
        tfu.load_state(model_file)      # Saving best policy
    env = GymEnv(agent.env.env_id)

    print("......................................................................................")
    print("Performing evaluation rollouts ........")
    eval_paths = eval_sampler.do_evaluation_rollout(N=num_traj,
                                                    policy=agent.pi,
                                                    env_name=env.env_id,
                                                    stochastic=stochastic,
                                                    render=True)
    mean_pol_perf = np.mean([np.sum(path['rewards']) for path in eval_paths])
    print('Mean performance : {}'.format(mean_pol_perf))

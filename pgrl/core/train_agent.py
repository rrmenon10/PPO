import logging

from tabulate import tabulate
from pgrl.utils.gym_utils import GymEnv
import pgrl.utils.tf_utils as tfu
from pgrl.samplers.trajectory_sampler import sample_paths_parallel
import numpy as np
import tensorflow as tf
import pickle
import dill
import time as timer
import os
import copy
import pickle

logging.disable(logging.CRITICAL)

def train_agent(job_name,
                agent,
                seed=0,
                niter=501,
                nsamples=None,
                gamma=0.995,
                gae_lambda=None,
                num_cpu=1,
                sample_mode='trajectories',
                num_traj=50,
                num_samples=50000,  # has precedence
                save_freq=10,
                evaluation_rollouts=None,
                plot_keys=['stoc_pol_mean', 'surr_improvement']):

    sess = tfu.make_session(num_cpu=num_cpu)
    sess.__enter__()
    sess.run(tf.global_variables_initializer())
    if os.path.isdir(job_name) is False:
        os.makedirs(job_name)
    previous_dir = os.getcwd()
    os.chdir(job_name)  # important! we are now in the directory to save data
    if os.path.isdir('iterations') is False: os.mkdir('iterations')
    if os.path.isdir('logs') is False and agent.save_logs is True: os.mkdir('logs')
    best_model_file = os.path.join("logs", "best")
    tfu.save_state(best_model_file)      # Saving best policy
    best_perf = -1e8
    train_curve = best_perf*np.ones(niter if nsamples is None else int(nsamples/num_samples)+1)
    mean_pol_perf = 0.0
    env = GymEnv(agent.env.env_id)
    nstopper = niter if nsamples is None else nsamples
    iter_num = 0
    i = 0

    while i < nstopper:

        print("......................................................................................")
        print("ITERATION : %i " % iter_num)
        if train_curve[iter_num-1] > best_perf:
            tfu.save_state(best_model_file)
            best_perf = train_curve[iter_num-1]
        N = num_traj if sample_mode == 'trajectories' else num_samples
        args = dict(N=N,
                    sample_mode=sample_mode,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    num_cpu=num_cpu,
                    nsamples=nsamples)
        stats = agent.train_step(**args)
        train_curve[iter_num] = stats[0]
        i += stats[-1]
        if evaluation_rollouts is not None and evaluation_rollouts > 0:
            print("Performing evaluation rollouts ........")
            eval_paths = sample_paths_parallel(N=10,
                                               policy=agent.pi,
                                               num_cpu=num_cpu,
                                               env_name=env.env_id,
                                               mode='evaluation')
            mean_pol_perf = np.mean([np.sum(path['rewards']) for path in eval_paths])
            if agent.save_logs:
                agent.logger.log_kv('eval_score', mean_pol_perf)
        if iter_num % save_freq == 0 and iter_num > 0:
            if agent.save_logs:
                agent.logger.save_log('logs/')
                agent.logger.make_train_plots(keys=plot_keys, save_loc='logs/')
            model_file = os.path.join("iterations", "{}".format(iter_num+1))
            tfu.save_state(model_file)
        # print results to console
        if iter_num == 0:
            result_file = open('results.txt', 'w')
            print("Iter | Stoc Pol | Mean Pol | Best (Stoc) \n")
            result_file.write("Iter | Sampling Pol | Evaluation Pol | Best (Sampled) \n")
            result_file.close()
        print("[ %s ] %4i %5.2f %5.2f %5.2f " % (timer.asctime(timer.localtime(timer.time())),
                                                 iter_num, train_curve[iter_num], mean_pol_perf, best_perf))
        result_file = open('results.txt', 'a')
        result_file.write("%4i %5.2f %5.2f %5.2f \n" % (iter_num, train_curve[iter_num], mean_pol_perf, best_perf))
        result_file.close()
        if agent.save_logs:
            print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
                                       agent.logger.get_current_log().items()))
            print(tabulate(print_data))

        iter_num += 1

    # final save
    tfu.save_state(model_file)
    if agent.save_logs:
        agent.logger.save_log('logs/')
        agent.logger.make_train_plots(keys=plot_keys, save_loc='logs/')
    os.chdir(previous_dir)

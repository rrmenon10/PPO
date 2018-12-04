import logging

import numpy as np
import scipy as sp
import scipy.sparse.linalg as spLA
import copy
import time as timer
import tensorflow as tf

# samplers
import pgrl.samplers.trajectory_sampler as trajectory_sampler
import pgrl.samplers.batch_sampler as batch_sampler

# utility functions
import pgrl.utils.tf_utils as tfu
import pgrl.utils.process_samples as process_samples
from pgrl.utils.logger import DataLog
from pgrl.utils.conjugate_gradient import conjugate_gradient as cg_solve
from pgrl.algos.batch_reinforce import BatchREINFORCE

logging.disable(logging.CRITICAL)

class KLPPO(BatchREINFORCE):

    def __init__(self,
                 env,
                 policy_fn,
                 kl_targ=0.01,
                 beta_init=1,
                 epochs=2,
                 mb_size=64,
                 learn_rate=3e-4,
                 vf_learn_rate=3e-4,
                 vf_iters=10,
                 batch_size=64,
                 seed=None,
                 schedule='linear',
                 save_logs=False):

        self.__name__ = 'KLPPO'
        self.env = env
        self.observation_space = self.env.env.observation_space
        self.action_space = self.env.env.action_space
        self.pi = policy_fn("pi",
                            ob_space=self.observation_space,
                            ac_space=self.action_space,
                            hidden_sizes=(64, 64))
        self.old_pi = policy_fn("old_pi",
                               ob_space=self.observation_space,
                               ac_space=self.action_space,
                               hidden_sizes=(64, 64))
        self.learn_rate = learn_rate
        self.alpha_vf = vf_learn_rate
        self.vf_iters = vf_iters
        self.batch_size = batch_size
        self.seed = seed
        self.save_logs = save_logs
        self.beta = beta_init
        self.kl_targ = kl_targ
        self.schedule = schedule
        self.epochs = epochs
        self.mb_size = mb_size
        self.running_score = None
        if save_logs: self.logger = DataLog()
        if seed is not None: np.random.seed(seed)
        self.tot_samples = 0
        self.build_graph()
        self.PPO_surrogate, self.ppo_optim = self.ppo_surrogate()

    def ppo_surrogate(self):

        obs = tfu.get_placeholder_cached(name="ob")
        actions = tfu.get_placeholder_cached(name="actions")
        adv = tfu.get_placeholder_cached(name="advantages")
        kl = self.old_pi.pd.kl(self.pi.pd)
        meankl = tf.reduce_mean(kl)
        ratio = tf.exp(self.pi.pd.logp(actions) - self.old_pi.pd.logp(actions))
        surr = tf.reduce_mean(ratio * adv)
        optim = tf.train.AdamOptimizer(self.learn_rate).minimize(-tf.reduce_mean(surr) + self.beta * meankl,
                                                                 var_list=self.pol_var_list)

        surr_fun = tfu.function([obs, actions, adv], surr)
        optim_fun = tfu.function([obs, actions, adv], optim)

        return surr_fun, optim_fun

    def train_from_paths(self, paths, nsamples=None):

        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        # Advantage whitening
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)
        # NOTE : advantage should be zero mean in expectation
        # normalized step size invariant to advantage scaling,
        # but scaling can help with least squares

        # Distribution specific operations
        if self.pi.pd.__name__ is 'DiagGaussian':
            self.pi.obs_norm.update(observations)
        elif self.pi.pd.__name__ is 'Categorical':
            n = self.pi.ac_shape
            actions = np.eye(n)[actions]

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]
        self.running_score = mean_return if self.running_score is None else \
            0.9*self.running_score + 0.1*mean_return  # approx avg of last 10 iters
        num_samples = observations.shape[0]
        if self.save_logs: self.log_rollout_statistics(paths)

        self.tot_samples += num_samples

        # Optimization algorithm
        # --------------------------
        surr_before = self.CPI_surrogate(observations, actions, advantages)
        self.assign_old_new()

        ts = timer.time()
        num_samples = observations.shape[0]
        for ep in range(self.epochs):
            for mb in range(int(num_samples / self.mb_size)):
                rand_idx = np.random.choice(num_samples, size=self.mb_size)
                obs = observations[rand_idx]
                act = actions[rand_idx]
                adv = advantages[rand_idx]
                self.ppo_optim(obs, act, adv)

        surr_after = self.CPI_surrogate(observations, actions, advantages)
        kl_dist = self.kl_old_new(observations, actions).ravel()[0]
        t_opt = timer.time() - ts

        # Beta Multiplier Condition
        # --------------------------
        if kl_dist > self.kl_targ * 1.5:
            self.beta *= 2
        elif kl_dist < (self.kl_targ / 1.5):
            self.beta *= 1./2
        self.beta = np.clip(self.beta, 1e-4, 10)

        # Log information
        if self.save_logs:
            self.logger.log_kv('t_opt', t_opt)
            self.logger.log_kv('kl_dist', kl_dist)
            self.logger.log_kv('surr_improvement', surr_after - surr_before)
            self.logger.log_kv('running_score', self.running_score)
            self.logger.log_kv('beta', self.beta)

        return base_stats

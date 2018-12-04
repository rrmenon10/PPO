import logging
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spLA
import time as timer
import tensorflow as tf

# samplers
import pgrl.samplers.trajectory_sampler as trajectory_sampler
import pgrl.samplers.batch_sampler as batch_sampler

# utilities
import pgrl.utils.tf_utils as tfu
import pgrl.utils.process_samples as process_samples
from pgrl.utils.logger import DataLog

logging.disable(logging.CRITICAL)

class BatchREINFORCE(object):

    def __init__(self,
                 env,
                 policy_fn,
                 learn_rate=0.01,
                 vf_learn_rate=3e-4,
                 batch_size=64,
                 vf_iters=3,
                 seed=None,
                 save_logs=False):

        self.__name__ = 'BatchREINFORCE'
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.pi = policy_fn("pi",
                            ob_space=self.observation_space,
                            ac_space=self.action_space,
                            hidden_sizes=(64,64))
        self.old_pi = policy_fn("old_pi",
                                ob_space=self.observation_space,
                                ac_space=self.action_space,
                                hidden_sizes=(64,64))
        self.alpha = learn_rate
        self.alpha_vf = vf_learn_rate
        self.vf_iters = vf_iters
        self.batch_size = 64
        self.seed = seed
        self.save_logs = save_logs
        self.running_score = None
        if save_logs: self.logger = DataLog()
        if seed is not None: np.random.seed(seed)
        self.tot_samples = 0
        self.build_graph()

    def build_graph(self):

        # Define all placeholders and get all variable lists
        actions = tfu.get_placeholder(name="actions",
                                      dtype=tf.float32,
                                      shape=[None, self.pi.ac_shape])
        obs = tfu.get_placeholder_cached(name="ob")
        adv = tfu.get_placeholder(name="advantages",
                                  dtype=tf.float32,
                                  shape=[None])
        ret = tfu.get_placeholder(name="returns",
                                  dtype=tf.float32,
                                  shape=[None])
        self.all_var_list = self.pi.get_trainable_variables()
        self.pol_var_list = [v for v in self.all_var_list
                             if v.name.startswith("pi/pol")
                             or v.name.startswith("pi/logstd")]
        self.vf_var_list = [v for v in self.all_var_list
                            if v.name.startswith("pi/vff")]
        if self.pi.pd.__name__ is 'DiagGaussian':
            assert len(self.pol_var_list) == len(self.vf_var_list) + 1
        elif self.pi.pd.__name__ is 'Categorical':
            assert len(self.pol_var_list) == len(self.vf_var_list)

        # Define operations
        kl = self.old_pi.pd.kl(self.pi.pd)
        mean_kl = tf.reduce_mean(kl)
        entropy = self.pi.pd.entropy()
        mean_ent = tf.reduce_mean(entropy)
        cpi_surr = self.surrogate()
        flat_vpg = tfu.flatgrad(cpi_surr, self.pol_var_list)
        vferr = tf.reduce_mean(tf.square(self.pi.vpred - ret))
        vfoptim = tf.train.AdamOptimizer(learning_rate=self.alpha_vf).minimize(vferr,
                                                                 var_list=self.vf_var_list)

        # Define functions for operations
        self.get_flat = tfu.GetFlat(self.pol_var_list)
        self.set_from_flat = tfu.SetFromFlat(self.pol_var_list)
        self.vf_get_flat = tfu.GetFlat(self.vf_var_list)
        self.vf_set_from_flat = tfu.SetFromFlat(self.vf_var_list)
        self.kl_old_new = tfu.function([obs, actions], mean_kl)
        self.entropy = tfu.function([obs, actions], mean_ent)
        self.CPI_surrogate = tfu.function([obs, actions, adv], cpi_surr)
        self.flat_vpg = tfu.function([obs, actions, adv], flat_vpg)
        self.vf_loss = tfu.function([obs, ret], vferr)
        self.vf_optim = tfu.function([obs, ret], vfoptim)

        self.assign_old_new = tfu.function([],
                                           [],
                                           updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in zip(self.old_pi.get_variables(),
                                                                            self.pi.get_variables())])

    def surrogate(self):

        actions = tfu.get_placeholder_cached(name="actions")
        adv = tfu.get_placeholder_cached(name="advantages")
        ratio = tf.exp(self.pi.pd.logp(actions) - self.old_pi.pd.logp(actions))
        cpi_surr = tf.reduce_mean(ratio * adv)
        return cpi_surr

    #############

    def train_step(self,
                   N,
                   sample_mode='trajectories',
                   env_name=None,
                   T=1e6,
                   gamma=0.995,
                   gae_lambda=0.98,
                   num_cpu='max',
                   nsamples=None):

        if env_name is None: env_name = self.env.env_id
        if sample_mode != 'trajectories' and sample_mode != 'samples':
            print("sample_mode in NPG must be either in 'trajectories' or in 'samples'")
            quit()

        ts = timer.time()

        if sample_mode == 'trajectories':
            paths = trajectory_sampler.sample_paths_parallel(N,
                                                             self.pi,
                                                             T,
                                                             env_name,
                                                             self.seed,
                                                             num_cpu)
        elif sample_mode == 'samples':
            paths = batch_sampler.sample_paths(N,
                                               self.pi,
                                               T,
                                               env_name=env_name,
                                               seed=self.seed,
                                               num_cpu=num_cpu)

        if self.save_logs:
            self.logger.log_kv('time_sampling', timer.time() - ts)

        self.seed = self.seed + N if self.seed is not None else self.seed

        # compute returns
        process_samples.compute_returns(paths, gamma)
        # compute advantages
        process_samples.compute_advantages(paths,
                                           self.pi,
                                           gamma,
                                           gae_lambda)
        # train from paths
        eval_statistics = self.train_from_paths(paths, nsamples)
        eval_statistics.append(N)
        # fit baseline
        if self.save_logs:
            ts = timer.time()
            observations = np.concatenate([path["observations"] for path in paths])
            returns = np.concatenate([path["returns"] for path in paths])
            num_samples = observations.shape[0]
            error_before = num_samples * self.vf_loss(observations, returns) / (np.sum(returns**2) + 1e-8)
            for _ in range(self.vf_iters):
                rand_idx = np.random.permutation(num_samples)
                for mb in range(int(num_samples / self.batch_size) - 1):
                    idx = rand_idx[mb*self.batch_size:(mb+1)*self.batch_size]
                    obs = observations[idx]
                    ret = returns[idx]
                    self.vf_optim(obs, ret)
            error_after = num_samples * self.vf_loss(observations, returns) / (np.sum(returns**2) + 1e-8)
            self.logger.log_kv('num_samples', self.tot_samples)
            self.logger.log_kv('time_VF', timer.time()-ts)
            self.logger.log_kv('VF_error_before', error_before)
            self.logger.log_kv('VF_error_after', error_after)
        else:
            self.vf_optim(paths["observations"],
                          paths["returns"])

        return eval_statistics

    def train_from_paths(self, paths, nsamples=None):

        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        num_samples = observations.shape[0]
        # Advantage whitening
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages)
                                                           + 1e-6)
        self.pi.obs_norm.update(observations)
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return, num_samples]
        self.running_score = mean_return if self.running_score is None else \
                             0.9*self.running_score + 0.1*mean_return  # approx avg of last 10 iters
        if self.save_logs: self.log_rollout_statistics(paths)

        # Keep track of times for various computations
        t_gLL = 0.0

        # Optimization algorithm
        # --------------------------
        surr_before = self.CPI_surrogate(observations, actions, advantages)

        # VPG
        ts = timer.time()
        vpg_grad = self.flat_vpg(observations, actions, advantages)
        t_gLL += timer.time() - ts

        # Policy update
        # --------------------------
        pi_after = self.get_flat() + self.alpha * vpg_grad
        self.set_from_flat(pi_after)
        surr_after = self.CPI_surrogate(observations,
                                        actions,
                                        advantages).ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).ravel()[0]
        self.assign_old_new()

        # Log information
        if self.save_logs:
            self.logger.log_kv('alpha', self.alpha)
            self.logger.log_kv('time_vpg', t_gLL)
            self.logger.log_kv('kl_dist', kl_dist)
            self.logger.log_kv('surr_improvement', surr_after - surr_before)
            self.logger.log_kv('surr_before', surr_before)
            self.logger.log_kv('running_score', self.running_score)

        return base_stats

    def log_rollout_statistics(self, paths):
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        self.logger.log_kv('stoc_pol_mean', mean_return)
        self.logger.log_kv('stoc_pol_std', std_return)
        self.logger.log_kv('stoc_pol_max', max_return)
        self.logger.log_kv('stoc_pol_min', min_return)

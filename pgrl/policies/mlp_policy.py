import tensorflow as tf
import numpy as np

import pgrl.utils.tf_utils as tfu
import pgrl.policies.distributions as dist
import gym.spaces

class MlpPolicy(object):

    def __init__(self,
                 name,
                 reuse=False,
                 *args,
                 **kwargs):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            self.scope = tf.get_variable_scope().name
            self._init(*args, **kwargs)

    def _init(self,
              ob_space,
              ac_space,
              hidden_sizes):

        self.ob_space = ob_space
        self.ac_space = ac_space
        self.hidden_sizes = hidden_sizes

        ob = tfu.get_placeholder(name="ob",
                                 dtype=tf.float32,
                                 shape=[None] + list(self.ob_space.shape))

        self.obs_norm = tfu.RunningMeanStd(shape=self.ob_space.shape)
        obs = tf.clip_by_value((ob - self.obs_norm.mean) / self.obs_norm.std,
                               -5.0,
                               5.0)

        out = obs
        for i, hidden_size in enumerate(hidden_sizes):
            out = tf.nn.tanh(tfu.dense(out,
                                       hidden_size,
                                       "vffc%i" % (i+1),
                                       weight_init=tfu.normc_initializer(1.0)))
        self.vpred = tfu.dense(out,
                               1,
                               "vffinal",
                               weight_init=tfu.normc_initializer(1.0))[:, 0]

        out = obs
        for i, hidden_size in enumerate(hidden_sizes):
            out = tf.nn.tanh(tfu.dense(out,
                                       hidden_size,
                                       "policyfc%i" % (i+1),
                                       weight_init=tfu.normc_initializer(1.0)))

        if isinstance(self.ac_space, gym.spaces.discrete.Discrete):
            self.ac_shape = ac_shape = ac_space.n
            dist_fn = dist.CategoricalPd
            pdparam = tfu.dense(out,
                                ac_shape,
                                "polfinal",
                                tfu.normc_initializer(0.01))
        elif isinstance(self.ac_space, gym.spaces.box.Box):
            self.ac_shape = ac_shape = ac_space.shape[0]
            dist_fn = dist.DiagGaussianPd
            mean = tfu.dense(out,
                             ac_shape,
                             "polfinal",
                             tfu.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd",
                                     shape=[1, ac_shape],
                                     initializer=tf.zeros_initializer())
            pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)

        self.pd = dist_fn(pdparam)

        stochastic = tfu.get_placeholder(name="stochastic",
                                         dtype=tf.bool, shape=())
        ac = tfu.switch(stochastic, self.pd.sample(), self.pd.mode())
        self.ac = ac
        self._act = tfu.function([stochastic, ob], [ac, self.vpred])
        self._predict = tfu.function([ob], self.vpred)

        # For pickle
        self.flatvars = tfu.GetFlat(self.get_trainable_variables())
        self.unflatvars = tfu.SetFromFlat(self.get_trainable_variables())

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def predict(self, ob):
        vpred = self._predict(ob)
        return vpred

    def get_variables(self, scope=None):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 self.scope if scope is None else scope)

    def get_trainable_variables(self, scope=None):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 self.scope if scope is None else scope)

    def __getstate__(self):
        d = {}
        d['scope'] = self.scope
        d['obs_norm'] = self.obs_norm.__getstate__()
        d['ob_space'] = self.ob_space
        d['ac_space'] = self.ac_space
        d['hidden_sizes'] = self.hidden_sizes
        d['net_params'] = self.flatvars()
        return d

    def __setstate__(self, dict_):
        self.scope = dict_['scope']
        self.obs_norm.__setstate(dict_['obs_norm'])
        self.ob_space = dict_['ob_space']
        self.ac_space = dict_['ac_space']
        self.hidden_sizes = dict_['hidden_sizes']
        self.unflatvars(dict_['net_params'])

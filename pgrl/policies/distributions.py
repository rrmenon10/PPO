import tensorflow as tf
import numpy as np

class DiagGaussianPd(object):

    def __init__(self, flat):
        self.__name__ = 'DiagGaussian'
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape)-1,
                                num_or_size_splits=2,
                                value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def flatparam(self):
        return self.flat

    def mode(self):
        return self.mean

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
               + tf.reduce_sum(self.logstd, axis=-1)

    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) \
               + tf.square(self.mean - other.mean)) / (2.0*tf.square(other.std)) \
               - 0.5, axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.logstd + .5*np.log(2.0*np.pi*np.e), axis=-1)

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

    def logp(self, x):
        return - self.neglogp(x)


class CategoricalPd(object):

    def __init__(self, logits):
        self.__name__ = 'Categorical'
        self.logits = logits

    def flatparam(self):
        return self.logits

    def mode(self):
        return tf.argmax(self.logits, axis=-1)

    @property
    def mean(self):
        return tf.nn.softmax(self.logits)

    def neglogp(self, x):
        if x.dtype in {tf.uint8, tf.int32, tf.int64}:
            x_shape_list = x.shape.as_list()
            logits_shape_list = self.logits.get_shape().as_list()[:-1]
            for xs, ls in zip(x_shape_list, logits_shape_list):
                if xs is not None and ls is not None:
                    assert xs == ls, 'shape mismatch: {} ' + \
                            'in x vs {} in logits'.format(xs, ls)
            x = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        else:
            assert x.shape.as_list() == self.logits.shape.as_list()

        return tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits,
                labels=x)

    def kl(self, other):
        assert isinstance(other, CategoricalPd)
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keep_dims=True)
        a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keep_dims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keep_dims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits), dtype=self.logits.dtype)
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)

    def logp(self, x):
        return - self.neglogp(x)

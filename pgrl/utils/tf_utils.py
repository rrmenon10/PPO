import numpy as np
import tensorflow as tf
import pickle
import os
import csv
import collections
import copy
import multiprocessing

SEED = 0

def set_random_seed(seed):
    np.random.seed(seed)
    SEED = seed

def flattenallbut0(x):
    return tf.reshape(x, [-1, int(np.prod(x.get_shape().as_list()[1:]))])

def flatten(inputs):

    input_shape = inputs.get_shape().as_list()
    return tf.reshape(x,
                      [-1, int(np.prod(input_shape[1:]))])

def var_shape(x):
    out = x.get_shape().as_list()
    assert all(isinstance(a, int) for a in out)
    return out

def numel(x):
    return int(np.prod(var_shape(x)))

def flatgrad(loss, var_list):

    grads = tf.gradients(loss, var_list)
    return tf.concat(axis=0,
                     values=[tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)])
                     for (v, grad) in zip(var_list, grads)])

def switch(condition, then_expression, else_expression):

    x_shape = copy.copy(then_expression.get_shape())
    x = tf.cond(tf.cast(condition, 'bool'),
                lambda: then_expression,
                lambda: else_expression)
    x.set_shape(x_shape)
    return x

_PLACEHOLDER_CACHE = {}

def get_placeholder(name, dtype, shape):

    if name in _PLACEHOLDER_CACHE:
        out, dtype1, shape1 = _PLACEHOLDER_CACHE[name]
        assert dtype1 == dtype and shape1 == shape
        return out
    else:
        out = tf.placeholder(dtype=dtype, shape=shape, name=name)
        _PLACEHOLDER_CACHE[name] = (out, dtype, shape)
        return out

def get_placeholder_cached(name):
    return _PLACEHOLDER_CACHE[name][0]

class SetFromFlat(object):
    def __init__(self, var_list, dtype=tf.float32):
        assigns = []
        shapes = list(map(var_shape, var_list))
        total_size = np.sum(
                [int(np.prod(shape)) for shape in shapes])

        self.theta = theta = tf.placeholder(
                                            dtype,
                                            [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = int(np.prod(shape))
            assigns.append(tf.assign(v,
                                     tf.reshape(
                                                theta[start:start + size],
                                                shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        tf.get_default_session().run(self.op,
                                     feed_dict={self.theta: theta})

class GetFlat(object):
    def __init__(self, var_list):
        self.op = tf.concat(axis=0,
                            values=[tf.reshape(v,
                                               [numel(v)]) for v in var_list])

    def __call__(self):
        return tf.get_default_session().run(self.op)

def load_state(fname):
    saver = tf.train.Saver()
    saver.restore(tf.get_default_session(), fname)

def save_state(fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    saver = tf.train.Saver()
    saver.save(tf.get_default_session(), fname)


def function(inputs, outputs, updates=None, givens=None):

    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        f = _Function(inputs, outputs.values(), updates, givens=givens)
        return lambda *args, **kwargs: type(outputs)(zip(outputs.keys(), f(*args, **kwargs)))
    else:
        f = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *args, **kwargs: f(*args, **kwargs)[0]

class _Function(object):

    def __init__(self, inputs, outputs, updates, givens):
        for inpt in inputs:
            if not hasattr(inpt, 'make_feed_dict') and not (type(inpt) is tf.Tensor and len(inpt.op.inputs) == 0):
                assert False, "inputs should all be placeholders, constants, or have a make_feed_dict method"
        self.inputs = inputs
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens

    def _feed_input(self, feed_dict, inpt, value):
        if hasattr(inpt, 'make_feed_dict'):
            feed_dict.update(inpt.make_feed_dict(value))
        else:
            feed_dict[inpt] = value

    def __call__(self, *args):
        assert len(args) <= len(self.inputs), "Too many arguments provided"
        feed_dict = {}
        # Update the args
        for inpt, value in zip(self.inputs, args):
            self._feed_input(feed_dict, inpt, value)
        # Update feed dict with givens.
        for inpt in self.givens:
            feed_dict[inpt] = feed_dict.get(inpt, self.givens[inpt])
        results = tf.get_default_session().run(self.outputs_update, feed_dict=feed_dict)[:-1]
        return results

def normc_initializer(std=1.0, axis=0):
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return tf.constant(out)
    return _initializer

def dense(x,
          size,
          name,
          weight_init=None,
          bias_init=0,
          weight_loss_dict=None,
          reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        assert (len(tf.get_variable_scope().name.split('/')) == 2)
        w = tf.get_variable("w", [x.get_shape()[1], size], initializer=weight_init)
        b = tf.get_variable("b", [size], initializer=tf.constant_initializer(bias_init))
        weight_decay_fc = 3e-4

        if weight_loss_dict is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(w), weight_decay_fc, name='weight_decay_loss')
            if weight_loss_dict is not None:
                weight_loss_dict[w] = weight_decay_fc
                weight_loss_dict[b] = 0.0

            tf.add_to_collection(tf.get_variable_scope().name.split('/')[0] + '_' + 'losses', weight_decay)

        return tf.nn.bias_add(tf.matmul(x, w), b)


def conv2d(x,
           num_filters,
           name,
           filter_size=(3, 3),
           stride=(1, 1),
           pad="SAME",
           dtype=tf.float32,
           collections=None,
           summary_tag=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0],
                        filter_size[1],
                        int(x.get_shape()[3]), num_filters]

        fan_in = int(np.prod(filter_shape[:3]))
        fan_out = int(np.prod(filter_shape[:2])) * num_filters
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W",
                            filter_shape,
                            dtype,
                            tf.random_uniform_initializer(-w_bound, w_bound, seed=SEED),
                            collections=collections)
        b = tf.get_variable("b",
                            [1, 1, 1, num_filters],
                            initializer=tf.zeros_initializer(),
                            collections=collections)

        if summary_tag is not None:
            tf.summary.image(summary_tag,
                             tf.transpose(tf.reshape(w, [filter_size[0],
                                                         filter_size[1],
                                                         -1,
                                                         1]), [2, 0, 1, 3]),
                             max_images=10)

        return tf.nn.conv2d(x, w, stride_shape, pad) + b



def make_session(num_cpu=None, make_default=False, graph=None):

    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu)
    if make_default:
        return tf.InteractiveSession(config=tf_config, graph=graph)
    else:
        return tf.Session(config=tf_config, graph=graph)

class RunningMeanStd(object):

    def __init__(self, epsilon=1e-2, shape=()):

        self._sum = tf.get_variable(dtype=tf.float32,
                                    shape=shape,
                                    initializer=tf.constant_initializer(0.0),
                                    name="runningsum", trainable=False)

        self._sumsq = tf.get_variable(dtype=tf.float32,
                                      shape=shape,
                                      initializer=tf.constant_initializer(epsilon),
                                      name="runningsumsq", trainable=False)

        self._count = tf.get_variable(dtype=tf.float32,
                                      shape=(),
                                      initializer=tf.constant_initializer(epsilon),
                                      name="count", trainable=False)

        self.shape = shape
        self.mean = tf.to_float(self._sum / self._count)
        self.std = tf.sqrt(tf.maximum(tf.to_float(self._sumsq / self._count)
                           - tf.square(self.mean), 1e-2))

        newsum = get_placeholder(shape=self.shape, dtype=tf.float32, name='sum')
        newsumsq = get_placeholder(shape=self.shape, dtype=tf.float32, name='var')
        newcount = get_placeholder(shape=[], dtype=tf.float32, name='count')
        self.incfiltparams = function([newsum, newsumsq, newcount], [],
                                      updates=[tf.assign_add(self._sum, newsum),
                                               tf.assign_add(self._sumsq, newsumsq),
                                               tf.assign_add(self._count, newcount)])

    def update(self, x):
        x = x.astype('float32')
        sums = x.sum(axis=0).ravel()
        sums_sq = np.square(x).sum(axis=0).ravel()
        count = x.shape[0]
        self.incfiltparams(sums.reshape(self.shape),
                           sums_sq.reshape(self.shape),
                           count)

    def __getstate__(self):
        d = {}
        d['shape'] = self.shape
        d['mean'], d['std'], \
            d['_sum'], d['_sumsq'], \
            d['_count'] = tf.get_default_session().run([self.mean,
                                                        self.std,
                                                        self._sum,
                                                        self._sumsq,
                                                        self._count])
        return d

    def __setstate__(self, d):
        ops = []
        ops.append(tf.assign(self.mean, d['mean']))
        ops.append(tf.assign(self.std, d['std']))
        ops.append(tf.assign(self._sum, d['_sum']))
        ops.append(tf.assign(self._sumsq, d['_sumsq']))
        ops.append(tf.assign(self._count, d['_count']))
        tf.get_default_session().run(ops)

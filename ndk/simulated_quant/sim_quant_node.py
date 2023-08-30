#!\usr\bin\python
#-*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import datetime

func_name = "SimQuant"

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8)) + str(datetime.datetime.now().microsecond)

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def np_sim_quant(x, quant_params, floor):
    x_splits = np.split(x, quant_params.shape[0], axis=1)

    x_sim_qs = []
    for idx, x_split in enumerate(x_splits):
        n, d = list(quant_params[idx])
        d = float(d)
        det = pow(2, d)
        x_int_max = (1 << (n - 1)) - 1
        x_int_min = -(1 << (n - 1))
        if floor:
            x_q = np.floor(x_split/det)
        else:
            x_q = np.round(x_split/det)
        x_clip = np.clip(x_q, x_int_min, x_int_max)
        x_sim_q = x_clip * det
        x_sim_qs.append(x_sim_q)
    x_sim_qs = np.concatenate(x_sim_qs, axis=1)

    return x_sim_qs

def compute_gradient(x_split, n, d):
    g_split = np.ones_like(x_split)
    d = float(d)
    det = pow(2, d)
    x_int_max = (1 << (n - 1)) - 1
    x_int_min = -(1 << (n - 1))
    x_max = x_int_max * det
    x_min = x_int_min * det
    g_split[x_split < x_min] = 0.
    g_split[x_split > x_max] = 0.
    return g_split

def np_sim_quant_grad(x, quant_params):
    nb_dims = len(x.shape)
    g = np.ones_like(x)
    x_splits = np.split(x, quant_params.shape[0], axis=1)

    if quant_params.shape[0] > 1:
        for idx, x_split in enumerate(x_splits):
            n, d = list(quant_params[idx])
            g_split = compute_gradient(x_split, n, d)
            if nb_dims == 1:
                g = g_split
            elif nb_dims == 2:
                if len(g_split.shape) == 2:
                    g[:, idx] = np.squeeze(g_split, axis=1)
                else:
                    g[:, idx] = g_split
            elif nb_dims == 3:
                if len(g_split.shape) == 3:
                    g[idx, :, :] = np.squeeze(g_split, axis=1)
                else:
                    g[idx, :, :] = g_split
            elif nb_dims == 4:
                if len(g_split.shape) == 4:
                    g[:, idx, :, :] = np.squeeze(g_split, axis=1)
                else:
                    g[:, idx, :, :] = g_split
            else:
                raise ValueError("The dimensions of the input tensor is not supported.")
    else:
        x_split = x_splits[0]
        n, d = quant_params[0]
        g_split = compute_gradient(x_split, n, d)
        g = g_split

    return g


def tf_sim_quant_grad(x, quant_params, name=None):
    with ops.name_scope(name, func_name + "Grad", [x]) as name:
        y = tf.py_func(np_sim_quant_grad,
                       [x, quant_params],
                       [tf.float32],
                       name=name,
                       stateful=False)
        y[0].set_shape(x.get_shape())
        g_q_p = tf.zeros_like(quant_params)
        g_floor = tf.constant(0.)
        return [y[0], g_q_p, g_floor]

def sim_quant_gradient(op, grad):
    x = op.inputs[0]
    quant_params = op.inputs[1]
    n_gr, g_q_p, g_floor = tf_sim_quant_grad(x, quant_params)

    return [grad * n_gr, g_q_p, g_floor]

def np_sim_quant_node(x, quant_params, floor=False, name=None):
    with ops.name_scope(name, func_name, [x]) as name:
        y = py_func(np_sim_quant,
                    [x, quant_params, floor],
                    [tf.float32],
                    name=name,
                    grad=sim_quant_gradient)
        y[0].set_shape(x.get_shape())
        return y[0]

if __name__ == '__main__':
    a = tf.random_normal(shape=[2, 4, 3, 1])
    quant_params = [(8, -6)] * 4
    b = np_sim_quant_node(a, quant_params, floor=False)
    ga = tf.gradients(b, a)
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        r_a, r_b, r_ga = sess.run([a, b, ga])
        print("a:", r_a, "b:", r_b, "r_ga:", r_ga)

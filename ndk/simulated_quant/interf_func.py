#!\usr\bin\python
#-*- coding:utf-8 -*-
from functools import partial

import tensorflow as tf
from tensorflow.python.ops import array_ops
from ndk.simulated_quant.sim_quant_node import np_sim_quant_node

_SPEED_UP = True
_USE_CUSTOM_OP = False

class SimQuantInt(object):
    def __init__(self, n, d, isWeight=False):
        d = float(d)
        self._min_var = -(2 ** (n - 1) - 1) * (2 ** d)
        self._max_var = (2 ** (n - 1) - 1) * (2 ** d)
        self._narrow_range = True
        self._num_bits = n
        self.subtract_num = 0.5 * pow(2, d)
        self.isWeight = isWeight

    def __call__(self, inputs):
        if self.isWeight:
           subtracted_inputs=inputs
        else:
           subtracted_inputs = tf.subtract(inputs, self.subtract_num)
        return array_ops.fake_quant_with_min_max_vars(subtracted_inputs, self._min_var, self._max_var,
                                                      num_bits=self._num_bits, narrow_range=self._narrow_range)


class SimQuantIntPerChannel(object):
    def __init__(self, quant_params, isWeight=False):
        n = quant_params[0][0]
        ds = [float(quant_param[1]) for quant_param in quant_params]
        self._min_var = [-(2 ** (n - 1) - 1) * (2 ** d) for d in ds]
        self._max_var = [(2 ** (n - 1) - 1) * (2 ** d) for d in ds]
        self._narrow_range = True
        self._num_bits = n
        self.subtract_num = [0.5 * pow(2, d) for d in ds]
        self.isWeight = isWeight

    def __call__(self, inputs):
        if self.isWeight:
           subtracted_inputs=inputs
        else:
           subtracted_inputs = tf.subtract(inputs, self.subtract_num)
        return array_ops.fake_quant_with_min_max_vars_per_channel(subtracted_inputs, self._min_var, self._max_var,
                                                      num_bits=self._num_bits, narrow_range=self._narrow_range)

def get_function_tf(n, d, isWeight=False):
    if _SPEED_UP and n >= 2 and n <= 8:
        return SimQuantInt(n, d, isWeight)
    else:
        floor = not isWeight
        return partial(np_sim_quant_node, quant_params=[(n, d)], floor=floor)

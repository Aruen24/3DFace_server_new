# -*- coding: utf-8 -*-
"""
Copyright(c) 2018 by Ningbo XiTang Information Technologies, Inc and
WuQi Technologies, Inc. ALL RIGHTS RESERVED.

This Information is proprietary to XiTang and WuQi, and MAY NOT be copied by
any method or incorporated into another program without the express written
consent of XiTang and WuQi. This Information or any portion thereof remains
the property of XiTang and WuQi. The Information contained herein is believed
to be accurate and XiTang and WuQi assumes no responsibility or liability for
its use in any way and conveys no license or title under any patent or copyright
and makes no representation or warranty that this Information is free from patent
or copyright infringement.
"""

import numpy as np
import abc

import os
import sys
ndk_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ndk_dir)

from ndk.quant_tools.quant_func import float2quant, quant2float, quantized_value

ENABLE_FLOOR_FEATURE = {8:True, 16:True}
ENABLE_FLOOR_WEIGHT = {8:False, 16:False}

try:
    import torch
    TF_FOUND = False
    TORCH_FOUND = True
except ImportError:
    TORCH_FOUND = False
    try:
        import tensorflow as tf
        TF_FOUND = True
    except ImportError:
        TF_FOUND = False

# TF_FOUND = False

_sigmoid_look_up_file_name = 'sigmoid_sim_out.dat'
_softmax_look_up_file_name = 'soft_out_result'

def read_vector_from_file_hex(filename):
    table = {}
    contents = []
    with open(filename, 'r') as f:
        contents = f.read().rstrip().split('\n')
    for i in range(len(contents)):
        if i < 32768:
            table[i] = float(int(contents[i], 16)) / 32768
        else:
            table[i - 65536] = float(int(contents[i], 16)) / 32768
    return table

cur_dirname = os.path.dirname(os.path.realpath(__file__))
_sigmoid_lut = read_vector_from_file_hex(os.path.join(cur_dirname, _sigmoid_look_up_file_name))
_softmax_lut = read_vector_from_file_hex(os.path.join(cur_dirname, _softmax_look_up_file_name))

class QuantLayer(metaclass=abc.ABCMeta):
    def __init__(self, name=None):
        self.name = name
        self.quantized = False

    def set_quant_param(self, bitwidth, in_frac, out_frac, input_signed=True):
        assert type(bitwidth)==int, 'Bit-width should be an integer'
        assert bitwidth==8 or bitwidth==16, 'Bit-width should be either 8 or 16.'
        if not input_signed:
            assert bitwidth==8, 'Only 8-bit support unsigned input.'
        self.bitwidth = bitwidth
        self.in_frac = in_frac
        self.out_frac = out_frac
        self.quantized = True
        self.input_signed = input_signed

    def run(self, data_in, quant=True, use_ai_framework=True, quant_weight_only=False, quant_feature_only=False, **kwargs):
        if (TF_FOUND or TORCH_FOUND) and use_ai_framework:
            self.use_ai_framework = True
        else:
            self.use_ai_framework = False
        if quant and not quant_weight_only:
            assert self.quantized, 'This layer has not be quantized, please call set_quant_param method first.'
            return self.run_quant(data_in)
        else:
            return self.run_float_point(data_in)

    @abc.abstractmethod
    def run_quant(self, data_in):
        pass

    @abc.abstractmethod
    def run_float_point(self, data_in):
        pass


class QuantInput(QuantLayer):
    def __init__(self, bitwidth=None, out_frac=None, input_signed=True, name=None):
        super().__init__(name=name)
        if type(bitwidth)!=type(None) and type(out_frac)!=type(None):
            self.set_quant_param(bitwidth=bitwidth, in_frac=None, out_frac=out_frac, input_signed=input_signed)

    def run_quant(self, data_in):
        return quantized_value(data_in, bitwidth=self.bitwidth, frac=self.out_frac, signed=self.input_signed, floor=ENABLE_FLOOR_FEATURE[self.bitwidth])

    def run_float_point(self, data_in):
        return data_in


class QuantSigmoid(QuantLayer):
    def __init__(self, bitwidth=None, in_frac=None, out_frac=None, input_signed=True, name=None):
        super().__init__(name=name)
        if type(bitwidth)!=type(None):
            if type(in_frac)==type(None) or type(out_frac)==type(None):
                if bitwidth==8:
                    in_frac, out_frac = 4, 7
                else:
                    in_frac, out_frac = 12, 15
            self.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)

    def set_quant_param(self, bitwidth, in_frac, out_frac, input_signed=True):
        if bitwidth==8:
            assert in_frac==4 and out_frac==7, 'Sigmoid - only (in_frac, out_frac)=(4,7) allowed for 8-bit quantization, but get ({},{}).'.format(in_frac, out_frac)
        else:
            assert in_frac==12 and out_frac==15, 'Sigmoid - only (in_frac, out_frac)=(12,15) allowed for 16-bit quantization, but get ({},{}).'.format(in_frac, out_frac)
        assert input_signed==True, 'input data must be signed.'
        super().set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)

    def run_quant(self, data_in):
        data_in = quantized_value(data_in, bitwidth=self.bitwidth, frac=self.in_frac, signed=self.input_signed, floor=ENABLE_FLOOR_FEATURE[self.bitwidth])
        data_out = np.zeros(data_in.shape)
        n, c, h, w = data_in.shape
        for idx_n in range(n):
            for idx_c in range(c):
                for idx_h in range(h):
                    for idx_w in range(w):
                        x = data_in[idx_n, idx_c, idx_h, idx_w]
                        index = int(x * 4096)
                        if index < -32768:
                            data_out[idx_n, idx_c, idx_h, idx_w] = _sigmoid_lut[-32768]
                        elif index > 32767:
                            data_out[idx_n, idx_c, idx_h, idx_w] = _sigmoid_lut[32767]
                        else:
                            data_out[idx_n, idx_c, idx_h, idx_w] = _sigmoid_lut[index]
        return quantized_value(data_out, bitwidth=self.bitwidth, frac=self.out_frac, floor=ENABLE_FLOOR_FEATURE[self.bitwidth])

    def run_float_point(self, data_in):
        return 1/(1+np.exp(-data_in))


class QuantReLU(QuantLayer):
    def __init__(self, bitwidth=None, in_frac=None, out_frac=None, negative_slope=0, quant_negative_slope=None, input_signed=True, name=None):
        super().__init__(name=name)
        self.negative_slope = negative_slope
        if type(bitwidth)!=type(None) and type(in_frac)!=type(None) and type(out_frac)!=type(None):
            if self.negative_slope==0 and type(quant_negative_slope)==type(None):
                quant_negative_slope = 0
            self.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, quant_negative_slope=quant_negative_slope, input_signed=input_signed)

    def set_quant_param(self, bitwidth, in_frac, out_frac, quant_negative_slope=None, input_signed=True):
        if type(quant_negative_slope)==type(None):
            if self.negative_slope == 0:
                quant_negative_slope = 0
            else:
                assert type(quant_negative_slope)!=type(None), 'quant_negative_slope must be assigned.'
        assert in_frac==out_frac, 'Input and output of ReLU share the same quantization scheme, but get ({},{}).'.format(in_frac, out_frac)
        self.quant_negative_slope = quantized_value(quant_negative_slope, bitwidth=8, frac=6) # fixed quant scheme for slope
        assert input_signed==True, 'input data must be signed.'
        super().set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)

    def run_quant(self, data_in):
        data_in = quantized_value(data_in, bitwidth=self.bitwidth, frac=self.in_frac, signed=self.input_signed, floor=ENABLE_FLOOR_FEATURE[self.bitwidth])
        if self.quant_negative_slope==0:
            return np.clip(data_in, 0, np.inf)
        else:
            data_out = np.clip(data_in, 0, np.inf) + np.clip(data_in, -np.inf, 0) * self.quant_negative_slope
            return quantized_value(data_out, bitwidth=self.bitwidth, frac=self.out_frac, floor=ENABLE_FLOOR_FEATURE[self.bitwidth])

    def run_float_point(self, data_in):
        return np.clip(data_in, 0, np.inf) + np.clip(data_in, -np.inf, 0) * self.negative_slope


class QuantReLU6(QuantLayer):
    def __init__(self, bitwidth=None, in_frac=None, out_frac=None, input_signed=True, name=None):
        super().__init__(name=name)
        if type(bitwidth)!=type(None):
            if type(in_frac)==type(None) or type(out_frac)==type(None):
                if bitwidth==8:
                    in_frac, out_frac = 4, 4
                else:
                    in_frac, out_frac = 12, 12
            self.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)

    def set_quant_param(self, bitwidth, in_frac, out_frac, input_signed=True):
        if bitwidth==8:
            assert in_frac==4 and out_frac==4, 'ReLU6 - only (in_frac, out_frac)=(4,4) allowed for 8-bit quantization, but get ({},{}).'.format(in_frac, out_frac)
        else:
            assert in_frac==12 and out_frac==12, 'ReLU6 - only (in_frac, out_frac)=(12,12) allowed for 16-bit quantization, but get ({},{}).'.format(in_frac, out_frac)
        assert input_signed==True, 'input data must be signed.'
        super().set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)

    def run_quant(self, data_in):
        data_in = quantized_value(data_in, bitwidth=self.bitwidth, frac=self.in_frac, signed=self.input_signed, floor=ENABLE_FLOOR_FEATURE[self.bitwidth])
        data_out = np.clip(data_in, 0, 6)
        return quantized_value(data_out, bitwidth=self.bitwidth, frac=self.out_frac, floor=ENABLE_FLOOR_FEATURE[self.bitwidth])

    def run_float_point(self, data_in):
        return np.clip(data_in, 0, 6)


class QuantTanH(QuantLayer):
    def __init__(self, bitwidth=None, in_frac=None, out_frac=None, input_signed=True, name=None):
        super().__init__(name=name)
        if type(bitwidth)!=type(None):
            if type(in_frac)==type(None) or type(out_frac)==type(None):
                if bitwidth==8:
                    in_frac, out_frac = 5, 7
                else:
                    in_frac, out_frac = 13, 15
            self.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)

    def set_quant_param(self, bitwidth, in_frac, out_frac, input_signed=True):
        if bitwidth==8:
            assert in_frac==5 and out_frac==7, 'TanH - only (in_frac, out_frac)=(5,7) allowed for 8-bit quantization, but get ({},{}).'.format(in_frac, out_frac)
        else:
            assert in_frac==13 and out_frac==15, 'TanH - only (in_frac, out_frac)=(13,15) allowed for 16-bit quantization, but get ({},{}).'.format(in_frac, out_frac)
        assert input_signed==True, 'input data must be signed.'
        super().set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)

    def run_quant(self, data_in):
        data_in = quantized_value(data_in, bitwidth=self.bitwidth, frac=self.in_frac, signed=self.input_signed, floor=ENABLE_FLOOR_FEATURE[self.bitwidth])
        data_out = np.zeros(data_in.shape)
        n, c, h, w = data_in.shape
        for idx_n in range(n):
            for idx_c in range(c):
                for idx_h in range(h):
                    for idx_w in range(w):
                        x = data_in[idx_n, idx_c, idx_h, idx_w]
                        index = int(x * 8192)
                        if index < -32768:
                            data_out[idx_n, idx_c, idx_h, idx_w] = _sigmoid_lut[-32768]*2-1
                        elif index > 32767:
                            data_out[idx_n, idx_c, idx_h, idx_w] = _sigmoid_lut[32767]*2-1
                        else:
                            data_out[idx_n, idx_c, idx_h, idx_w] = _sigmoid_lut[index]*2-1
        return quantized_value(data_out, bitwidth=self.bitwidth, frac=self.out_frac, floor=ENABLE_FLOOR_FEATURE[self.bitwidth])

    def run_float_point(self, data_in):
        return np.tanh(data_in)


class QuantConv2D(QuantLayer):
    def __init__(self, c_in, c_out, kernel_size_h, kernel_size_w,
                 weight, bias=None,
                 quant_weight=None, quant_bias=None,
                 stride_h=1, stride_w=1, dilation_h=1, dilation_w=1, group=1,
                 pad_n=0, pad_s=0, pad_w=0, pad_e=0, bias_term=True,
                 bitwidth=None, in_frac=None, out_frac=None, input_signed=True, name=None):
        super().__init__(name=name)
        if c_in % group != 0:
            raise Exception(
                "input channel should be multiple of group, here input channel is {}, group is {}".format(c_in, group))
        if c_out % group != 0:
            raise Exception(
                "output channel should be multiple of group, here output channel is {}, group is {}".format(c_out,
                                                                                                            group))
        self.c_in_g = int(c_in / group)
        self.c_out_g = int(c_out / group)
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.group = group
        self.pad_n = pad_n
        self.pad_s = pad_s
        self.pad_w = pad_w
        self.pad_e = pad_e
        self.bias_term = bias_term

        assert pad_n >= 0, 'pad_n must be non-negative, but get {}'.format(pad_n)
        assert pad_w >= 0, 'pad_w must be non-negative, but get {}'.format(pad_w)
        assert len(weight.shape) == 4, 'weight must be a 4-dim array, but get {}'.format(weight.shape)
        co, cig, k_h, k_w = weight.shape
        assert co == self.c_out_g * self.group, 'weight shape is not matched with the layer parameter'
        assert cig == self.c_in_g, 'weight shape is not matched with the layer parameter'
        assert kernel_size_h==k_h and kernel_size_w==k_w, 'weight shape is not matched with the layer parameter'
        self.fp_weight = weight.copy()
        if bias_term:
            assert type(bias)!=type(None), 'bias must be given when bias_term set True'
            assert type(bias) == np.ndarray, 'invalid data type {}, it must be numpy.ndarray'.format(type(bias))
            assert len(bias.shape) == 1, 'bias must be a 1-dim array'
            assert c_out == len(bias), 'bias length is not matched with the layer parameter'
            self.fp_bias = bias.copy()
        else:
            self.fp_bias = None
        if type(bitwidth)!=type(None) and type(in_frac)!=type(None) and type(out_frac)!=type(None):
            self.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac,
                                 quant_weight=quant_weight, quant_bias=quant_bias, input_signed=input_signed)


    def set_quant_param(self, bitwidth, in_frac, out_frac, quant_weight, quant_bias=None, input_signed=True):
        assert type(quant_weight)==np.ndarray, 'invalid type for quant_weight, but get {}'.format(type(quant_weight))
        assert quant_weight.shape == self.fp_weight.shape, 'quant_weight {} must share the same shape with weight {}'.format(quant_weight, self.fp_weight.shape)
        self.quant_weight = quant_weight.copy()
        if self.bias_term:
            assert type(quant_bias)!=type(None), 'quant_bias must be given when bias_term set True'
            assert type(quant_bias)==np.ndarray, 'invalid type for quant_bias, but get {}'.format(type(quant_bias))
            assert quant_bias.shape == self.fp_bias.shape, 'quant_bias {} must share the same shape with bias {}'.format(quant_bias, self.fp_bias.shape)
            self.quant_bias = quant_bias.copy()
        else:
            self.quant_bias = None
        super().set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)

    def run(self, data_in, quant=True, use_ai_framework=True, quant_weight_only=False, quant_feature_only=False, **kwargs):
        if (TF_FOUND or TORCH_FOUND) and use_ai_framework:
            self.use_ai_framework = True
        else:
            self.use_ai_framework = False

        if quant:
            assert not (quant_weight_only and quant_feature_only), 'quant_weight_only and quant_feature_only cannot be True at the same time.'

        # tensorflow cannot support this case
        if self.use_ai_framework and TF_FOUND and np.any(np.array([self.stride_h, self.stride_w]) > 1) and np.any(np.array([self.dilation_h, self.dilation_w]) > 1):
            print('\033[0;31m Warning: Layer {}: Tensorflow cannot support strides > 1 in conjunction with dialtion_rate > 1, numpy operations will be used instead. \033[0m'.format(self.name))
            self.use_ai_framework = False

        if self.use_ai_framework and TF_FOUND and not hasattr(self, 'sess'):
            if self.name == None:
                name = 'conv' + str(np.random.randint(0, 100000000))
            else:
                name = self.name
            self.g = tf.Graph()
            with self.g.as_default():
                with tf.variable_scope(name):
                    n, c_in, h_in, w_in = data_in.shape
                    self.tf_padded = tf.placeholder(tf.float64, shape=[None, h_in + self.pad_s + self.pad_n, w_in + self.pad_w + self.pad_e, c_in])
                    outputs_g = []
                    for idx in range(int(self.group)):
                        with tf.variable_scope('group' + str(idx)):
                            input_g = self.tf_padded[:, :, :, idx * self.c_in_g: (idx + 1) * self.c_in_g]
                            outputs_g.append(tf.layers.conv2d(input_g, filters=self.c_out_g,
                                                              kernel_size=(self.kernel_size_h, self.kernel_size_w),
                                                              strides=(self.stride_h, self.stride_w),
                                                              data_format='channels_last',
                                                              dilation_rate=(self.dilation_h, self.dilation_w),
                                                              trainable=False,
                                                              use_bias=self.bias_term,
                                                              activation=None))
                    self.tf_outputs = tf.concat(outputs_g, axis=3)

                    self.tf_weight = tf.placeholder(tf.float64, shape=[self.c_out_g*self.group, self.c_in_g, self.kernel_size_h, self.kernel_size_w])
                    self.tf_bias = tf.placeholder(tf.float64, shape=[self.c_out_g * self.group])
                    self.update_weight = []
                    self.update_bias = []
                    tf_kernel = []
                    tf_bias = []
                    tf_weight_slices = tf.split(self.tf_weight, self.group, axis=0)
                    tf_bias_slices = tf.split(self.tf_bias, self.group, axis=0)

                    self.tf_kernel = []
                    self.tf_biases = []

                    for g in range(self.group):
                        for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                            var_lyr_name_list = variable.name.split('/')
                            var_lyr_name = ''
                            for i in range(len(var_lyr_name_list)-3):
                                var_lyr_name += var_lyr_name_list[i] + '/'
                            var_lyr_name = var_lyr_name[:-1]
                            if var_lyr_name == name and 'kernel' in variable.name and 'group' + str(g) + r'/' in variable.name:
                                tf_kernel.append(variable)
                                self.update_weight.append(tf.assign(variable, tf.transpose(tf_weight_slices[g],[2,3,1,0])))
                                self.tf_kernel.append(variable)
                            if var_lyr_name == name and 'bias' in variable.name and 'group' + str(g) + r'/' in variable.name:
                                tf_bias.append(variable)
                                self.update_bias.append(tf.assign(variable, tf_bias_slices[g]))
                                self.tf_biases.append(variable)

            self.sess = tf.Session(graph=self.g)
            self.sess.run(tf.variables_initializer(var_list=self.tf_kernel))
            if self.bias_term:
                self.sess.run(tf.variables_initializer(var_list=self.tf_biases))

        if quant and not quant_feature_only:
            assert self.quantized, 'This layer has not be quantized, please call set_quant_param method first.'
            if self.use_ai_framework and TF_FOUND:
                self.sess.run(self.update_weight, feed_dict={self.tf_weight: self.quant_weight})
                if self.bias_term:
                    self.sess.run(self.update_bias, feed_dict={self.tf_bias: self.quant_bias})
            else:
                self.weight = self.quant_weight # switch weight to quantized version
                self.bias = self.quant_bias # switch bias to quantized version
        else:
            if self.use_ai_framework and TF_FOUND:
                self.sess.run(self.update_weight, feed_dict={self.tf_weight: self.fp_weight})
                if self.bias_term:
                    self.sess.run(self.update_bias, feed_dict={self.tf_bias: self.fp_bias})
            else:
                self.weight = self.fp_weight # switch weight to floating-point version
                self.bias = self.fp_bias # switch bias to floating-point version

        if quant and not quant_weight_only:
            data_out = self.run_quant(data_in)
        else:
            data_out = self.run_float_point(data_in)
        return data_out

    def run_quant(self, data_in):
        data_in = np.asarray(data_in, dtype=np.float64)
        data_in = quantized_value(data_in, bitwidth=self.bitwidth, frac=self.in_frac, signed=self.input_signed, floor=ENABLE_FLOOR_FEATURE[self.bitwidth])
        data_out = self.run_float_point(data_in)
        return quantized_value(data_out, bitwidth=self.bitwidth, frac=self.out_frac, floor=ENABLE_FLOOR_FEATURE[self.bitwidth])

    def run_float_point(self, data_in):
        data_in = np.asarray(data_in, dtype=np.float64)
        assert len(data_in.shape) == 4, 'input data must be a 4-dim array in NCHW format'
        n, c_in, h_in, w_in = data_in.shape
        assert c_in == self.c_in_g*self.group, 'input channel number ({}) is not aligned with layer input channel number ({})'.format(c_in, self.c_in_g*self.group)
        assert (h_in + self.pad_s + self.pad_n - (
                    self.kernel_size_h - 1) * self.dilation_h - 1) % self.stride_h == 0, 'need more padding in height?'
        assert (w_in + self.pad_w + self.pad_e - (
                    self.kernel_size_w - 1) * self.dilation_w - 1) % self.stride_w == 0, 'need more padding in width?'

        padded = np.zeros((n, c_in, h_in + self.pad_s + self.pad_n, w_in + self.pad_w + self.pad_e), dtype=np.float64)
        if self.pad_s < 0:
            data_in = data_in[:,:,:self.pad_s,:]
        if self.pad_e < 0:
            data_in = data_in[:,:,:,:self.pad_e]
        padded[:, :, self.pad_n: self.pad_n + h_in, self.pad_w: self.pad_w + w_in] = data_in

        if self.use_ai_framework:
            if TF_FOUND:
            # data_out = self.sess.run(self.tf_outputs, feed_dict={self.tf_padded: padded})
                data_out = self.sess.run(self.tf_outputs, feed_dict={self.tf_padded: np.transpose(padded,[0,2,3,1])}).transpose([0,3,1,2])
            elif TORCH_FOUND:
                data_out_torch = torch.nn.functional.conv2d(torch.from_numpy(padded).double(), \
                                                      torch.from_numpy(self.weight).double(), \
                                                      torch.from_numpy(self.bias).double() if self.bias_term else None, \
                                                      stride = (self.stride_h, self.stride_w), \
                                                      dilation = (self.dilation_h, self.dilation_w), \
                                                      groups = self.group)
                data_out = data_out_torch.data.numpy()
        else:
            h_out = int(
                (h_in + self.pad_s + self.pad_n - (self.kernel_size_h - 1) * self.dilation_h - 1) / self.stride_h) + 1
            w_out = int(
                (w_in + self.pad_w + self.pad_e - (self.kernel_size_w - 1) * self.dilation_w - 1) / self.stride_w) + 1
            data_out = np.zeros((n, self.c_out_g * self.group, h_out, w_out), dtype=np.float64)
            for g in range(self.group):
                weight = self.weight[g * self.c_out_g: (g + 1) * self.c_out_g, :, :, :]
                if self.bias_term:
                    bias = self.bias[g * self.c_out_g: (g + 1) * self.c_out_g]
                else:
                    bias = 0

                # for k in range(n):
                #     for i in range(h_out):
                #         for j in range(w_out):
                #             data_out[k, g * self.c_out_g: (g + 1) * self.c_out_g, i, j] += bias
                #             for c in range(self.c_out_g):
                #                 data_out[k, g*self.c_out_g+c, i, j] += np.sum(
                #                     padded[k, :, i*self.stride_h:i*self.stride_h+self.kernel_size_h*self.dilation_h:self.dilation_h,
                #                                 j*self.stride_w:j*self.stride_w+self.kernel_size_w*self.dilation_w:self.dilation_w]
                #                     * weight[c,:]
                #                 )

                for k in range(n):
                    for i in range(h_out):
                        for j in range(w_out):
                            data_out[k, g * self.c_out_g: (g + 1) * self.c_out_g, i, j] += bias
                            index_h = i * self.stride_h
                            for h in range(self.kernel_size_h):
                                index_w = j * self.stride_w
                                for w in range(self.kernel_size_w):
                                    data_out[k, g * self.c_out_g: (g + 1) * self.c_out_g, i, j] += np.dot(
                                        weight[:, :, h, w],
                                        padded[k, g * self.c_in_g: (g + 1) * self.c_in_g, index_h, index_w])
                                    index_w += self.dilation_w
                                index_h += self.dilation_h
        return data_out


class QuantInnerProduct(QuantConv2D):
    def __init__(self, c_in, c_out, weight, bias=None,
                 h_in=1, w_in=1,
                 quant_weight=None, quant_bias=None,
                 bias_term=True, bitwidth=None, in_frac=None, out_frac=None, name=None, input_signed=True):
        assert type(weight)==np.ndarray, 'invalid data type {}, it must be numpy.ndarray'.format(type(weight))
        if bias_term:
            assert type(bias)==np.ndarray, 'invalid data type {}, it must be numpy.ndarray'.format(type(bias))
        assert len(weight.shape)==4, 'invalid weight shape {}, it must be 4D array'.format(weight.shape)
        cout, cin, h, w = weight.shape
        assert cout==c_out and cin==c_in and h==h_in and w==w_in, 'invalid weight shape {}, while input shape is {}'.format(weight.shape, (c_out, c_in, h_in, w_in))
        if bias_term:
            assert len(bias.shape)==1, 'invalid bias shape {}, it must be 1D array'.format(bias.shape)
            assert len(bias)==cout, 'invalid bias shape {}, it\'s length must align with c_out {}'.format(bias.shape, c_out)
        super().__init__(c_in=c_in, c_out=c_out, kernel_size_h=h_in, kernel_size_w=w_in,
                 weight=weight, bias=bias, quant_weight=quant_weight, quant_bias=quant_bias, bias_term=bias_term,
                 bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, name=name, input_signed=input_signed)
        self.num_input = c_in * h_in * w_in

    def run(self, data_in, quant=True, **kwargs):
        assert type(data_in)==np.ndarray, 'invalid data type {}, it must be numpy.ndarray'.format(type(data_in))
        _,c,h,w = data_in.shape
        assert c*h*w==self.num_input, 'mis-matched input data size {} to InnerProduct with input sized {}'.format(c*h*w, self.num_input)
        return super().run(data_in=data_in, quant=quant, **kwargs)


class QuantMaxPool2D(QuantLayer):
    def __init__(self, kernel_size_h, kernel_size_w, stride_h = 1, stride_w = 1, dilation_h = 1, dilation_w = 1,
                 pad_n = 0, pad_s = 0, pad_w = 0, pad_e = 0, ceil_mode = False,
                 bitwidth=None, in_frac=None, out_frac=None, input_signed=True, name=None):
        super().__init__(name=name)
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.pad_n = pad_n
        self.pad_s = pad_s
        self.pad_w = pad_w
        self.pad_e = pad_e
        self.ceil_mode = ceil_mode
        assert pad_n >= 0, 'pad_n must be non-negative, but get {}'.format(pad_n)
        assert pad_w >= 0, 'pad_w must be non-negative, but get {}'.format(pad_w)
        if type(bitwidth)!=type(None) and type(in_frac)!=type(None) and type(out_frac)!=type(None):
            self.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)

    def set_quant_param(self, bitwidth, in_frac, out_frac, input_signed=True):
        assert in_frac==out_frac, 'MaxPool - in_frac({}) should be the same with out_frac({})'.format(in_frac, out_frac)
        # assert input_signed == True, 'input data must be signed.'
        super().set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)

    def run_quant(self, data_in):
        data_in = np.asarray(data_in, dtype=np.float64)
        data_in = quantized_value(data_in, bitwidth=self.bitwidth, frac=self.in_frac, signed=self.input_signed, floor=ENABLE_FLOOR_FEATURE[self.bitwidth])
        data_out = self.run_float_point(data_in)
        return quantized_value(data_out, bitwidth=self.bitwidth, frac=self.out_frac, floor=ENABLE_FLOOR_FEATURE[self.bitwidth])

    def run_float_point(self, data_in):
        data_in = np.asarray(data_in, dtype = np.float64)
        assert len(data_in.shape) == 4, 'input data must be a 4-dim array in NCHW format'
        n, c_in, h_in, w_in = data_in.shape
        assert (h_in + self.pad_s + self.pad_n - (
                    self.kernel_size_h - 1) * self.dilation_h - 1) % self.stride_h == 0, 'need more padding in height?'
        assert (w_in + self.pad_w + self.pad_e - (
                    self.kernel_size_w - 1) * self.dilation_w - 1) % self.stride_w == 0, 'need more padding in width?'
        if self.pad_s < 0:
            data_in = data_in[:,:,:self.pad_s,:]
        if self.pad_e < 0:
            data_in = data_in[:,:,:,:self.pad_e]
        padded = np.zeros((n, c_in, h_in + self.pad_s + self.pad_n, w_in + self.pad_w + self.pad_e), dtype = np.float64)
        padded -= np.inf
        padded[:, :, self.pad_n : self.pad_n + h_in, self.pad_w : self.pad_w + w_in] = data_in


        if self.use_ai_framework and TF_FOUND and np.any(np.array([self.dilation_h, self.dilation_w]) > 1):
            print('\033[0;31m Warning: Layer {}: Tensorflow pooling cannot support dialtion > 1, numpy operations will be used instead. \033[0m'.format(self.name))
            self.use_ai_framework = False

        if self.use_ai_framework and TF_FOUND and not hasattr(self, 'sess'):
            if self.name == None:
                name = 'maxpool' + str(np.random.randint(0, 100000000))
            else:
                name = self.name

            self.g = tf.Graph()
            with self.g.as_default():
                with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                    self.tf_padded = tf.placeholder(tf.float64, shape=[None, h_in + self.pad_s + self.pad_n, w_in + self.pad_w + self.pad_e, c_in])

                    self.tf_outputs = tf.layers.max_pooling2d(self.tf_padded,
                                                              pool_size=(self.kernel_size_h, self.kernel_size_w),
                                                              strides=(self.stride_h, self.stride_w),
                                                              data_format='channels_last')
                self.sess = tf.Session(graph=self.g)

        if self.use_ai_framework and TORCH_FOUND:
            data_out = torch.nn.MaxPool2d((self.kernel_size_h, self.kernel_size_w), \
                                          stride = (self.stride_h, self.stride_w), \
                                          dilation = (self.dilation_h, self.dilation_w))(torch.from_numpy(padded)).data.numpy()
        elif self.use_ai_framework and self.dilation_h==1 and self.dilation_w==1:
            data_out = self.sess.run(self.tf_outputs, feed_dict={self.tf_padded: np.transpose(padded,[0,2,3,1])}).transpose([0,3,1,2])
        else:
            h_out = int((h_in + self.pad_s + self.pad_n - (self.kernel_size_h - 1) * self.dilation_h - 1) / self.stride_h) + 1
            w_out = int((w_in + self.pad_w + self.pad_e - (self.kernel_size_w - 1) * self.dilation_w - 1) / self.stride_w) + 1
            if self.ceil_mode:
                h_out = int((h_in + self.pad_s + self.pad_n - (self.kernel_size_h - 1) * self.dilation_h - 2) / self.stride_h) + 2
                w_out = int((w_in + self.pad_w + self.pad_e - (self.kernel_size_w - 1) * self.dilation_w - 2) / self.stride_w) + 2
                if (h_in + self.pad_n) % self.stride_h == 0:
                    h_out = min(h_out, int((h_in + self.pad_n) / self.stride_h))
                if (w_in + self.pad_w) % self.stride_w == 0:
                    w_out = min(w_out, int((w_in + self.pad_w) / self.stride_w))
            data_out = np.zeros((n, c_in, h_out, w_out), dtype = np.float64) - np.inf
            for k in range(n):
                for c in range(c_in):
                    for i in range(h_out):
                        for j in range(w_out):
                            index_h = i * self.stride_h
                            for h in range(self.kernel_size_h):
                                index_w = j * self.stride_w
                                for w in range(self.kernel_size_w):
                                    data_out[k, c, i, j] = max(data_out[k, c, i, j], padded[k, c, index_h, index_w])
                                    index_w += self.dilation_w
                                index_h += self.dilation_h
        return data_out


class QuantAvgPool2D(QuantLayer):
    def __init__(self, kernel_size_h, kernel_size_w, stride_h = 1, stride_w = 1, dilation_h = 1, dilation_w = 1,
                 pad_n = 0, pad_s = 0, pad_w = 0, pad_e = 0, ceil_mode = False,
                 bitwidth=None, in_frac=None, out_frac=None, input_signed=True, name=None):
        super().__init__(name=name)
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.pad_n = pad_n
        self.pad_s = pad_s
        self.pad_w = pad_w
        self.pad_e = pad_e
        self.ceil_mode = ceil_mode
        assert pad_n >= 0, 'pad_n must be non-negative, but get {}'.format(pad_n)
        assert pad_w >= 0, 'pad_w must be non-negative, but get {}'.format(pad_w)
        if type(bitwidth)!=type(None) and type(in_frac)!=type(None) and type(out_frac)!=type(None):
            self.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)

    def run(self, data_in, quant=True, hw_aligned=False, use_ai_framework=True, quant_weight_only=False, quant_feature_only=False, **kwargs):
        assert type(data_in)==np.ndarray, 'invalid data type {}'.format(type(data_in))
        n, c_in, h_in, w_in = data_in.shape

        if (TF_FOUND or TORCH_FOUND) and use_ai_framework:
            self.use_ai_framework = True
        else:
            self.use_ai_framework = False

        if self.use_ai_framework and np.any(np.array([self.dilation_h, self.dilation_w]) > 1):
            print('\033[0;31m Warning: Layer {}: Tensorflow pooling cannot support dialtion > 1, numpy operations will be used instead. \033[0m'.format(self.name))
            self.use_ai_framework = False


        if self.use_ai_framework and TF_FOUND and not hasattr(self, 'sess') :
            if self.name == None:
                name = 'avgpool' + str(np.random.randint(0, 100000000))
            else:
                name = self.name

            self.g = tf.Graph()
            with self.g.as_default():
                with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                    self.tf_padded = tf.placeholder(tf.float64, shape=[None, h_in + self.pad_s + self.pad_n, w_in + self.pad_w + self.pad_e, c_in])

                    self.tf_outputs = tf.layers.average_pooling2d(self.tf_padded,
                                                              pool_size=(self.kernel_size_h, self.kernel_size_w),
                                                              strides=(self.stride_h, self.stride_w),
                                                              data_format='channels_last')
                self.sess = tf.Session(graph=self.g)

        data_out, elem_cnt = super().run(data_in=data_in, quant=quant, use_ai_framework=self.use_ai_framework, quant_weight_only=quant_weight_only, quant_feature_only=quant_feature_only)

        if hw_aligned:
            # if hardware aligned, the element cnt is determined by location directly.
            if self.pad_s < 0:
                h_in += self.pad_s
            if self.pad_e < 0:
                w_in += self.pad_e
            # center
            active_h = min(self.kernel_size_h, h_in)
            active_w = min(self.kernel_size_w, w_in)
            elem_cnt = np.ones(elem_cnt.shape) * active_h * active_w
            # left edge
            active_h = min(self.kernel_size_h, h_in)
            active_w = min(self.kernel_size_w-self.pad_w, w_in)
            elem_cnt[:,0] = active_h * active_w
            # upper edge
            active_h = min(self.kernel_size_h-self.pad_n, h_in)
            active_w = min(self.kernel_size_w, w_in)
            elem_cnt[0,:] = active_h * active_w
            # lower edge
            active_h = min(self.kernel_size_h-max(self.pad_s,0), h_in)
            active_w = min(self.kernel_size_w, w_in)
            elem_cnt[-1,:] = active_h * active_w
            # upper-left corner
            active_h = min(self.kernel_size_h-self.pad_n, h_in)
            active_w = min(self.kernel_size_w-self.pad_w, w_in)
            elem_cnt[0, 0] = active_h * active_w
            # lower-left corner
            active_h = min(self.kernel_size_h-max(self.pad_s,0), h_in)
            active_w = min(self.kernel_size_w-self.pad_w, w_in)
            elem_cnt[-1, 0] = active_h * active_w

        scale = 1 / elem_cnt
        if quant and not quant_weight_only:
            min_elem_cnt = np.min(elem_cnt)
            if self.bitwidth==8:
                # when 8-bit allows unsigned number
                scale_frac = self.bitwidth + int(np.ceil(np.log2(min_elem_cnt))) - 1
                scale = quantized_value(scale, bitwidth=self.bitwidth, frac=scale_frac, floor=False, signed=False)
            else:
                scale_frac = self.bitwidth + int(np.ceil(np.log2(min_elem_cnt))) - 2
                scale = quantized_value(scale, bitwidth=self.bitwidth, frac=scale_frac, floor=False, signed=True)
        # print('scale: \n{}'.format(scale))
        for idx_n in range(n):
            for idx_c in range(c_in):
                data_out[idx_n,idx_c,:,:] = data_out[idx_n,idx_c,:,:] * scale
        if quant and not quant_weight_only:
            data_out = quantized_value(data_out, bitwidth=self.bitwidth, frac=self.out_frac, floor=ENABLE_FLOOR_FEATURE[self.bitwidth])

        return data_out


    def run_quant(self, data_in):
        data_in = np.asarray(data_in, dtype=np.float64)
        data_in = quantized_value(data_in, bitwidth=self.bitwidth, frac=self.in_frac, signed=self.input_signed, floor=ENABLE_FLOOR_FEATURE[self.bitwidth])
        data_out, elem_cnt = self.run_float_point(data_in)
        return data_out, elem_cnt

    def run_float_point(self, data_in):
        data_in = np.asarray(data_in, dtype=np.float64)
        assert len(data_in.shape) == 4, 'input data must be a 4-dim array in NCHW format'
        n, c_in, h_in, w_in = data_in.shape
        assert (h_in + self.pad_s + self.pad_n - (
                    self.kernel_size_h - 1) * self.dilation_h - 1) % self.stride_h == 0, 'need more padding in height?'
        assert (w_in + self.pad_w + self.pad_e - (
                    self.kernel_size_w - 1) * self.dilation_w - 1) % self.stride_w == 0, 'need more padding in width?'
        if self.pad_s < 0:
            data_in = data_in[:,:,:self.pad_s,:]
        if self.pad_e < 0:
            data_in = data_in[:,:,:,:self.pad_e]
        # padded = np.zeros((n, c_in, h_in + self.pad_s + self.pad_n + self.kernel_size_h, w_in + self.pad_w + self.pad_e + self.kernel_size_w), dtype = np.float64)
        padded = np.zeros((n, c_in, h_in + self.pad_s + self.pad_n, w_in + self.pad_w + self.pad_e), dtype = np.float64)
        padded[:, :, self.pad_n : self.pad_n + h_in, self.pad_w : self.pad_w + w_in] = data_in

        if self.use_ai_framework:
            mask = np.ones((1,c_in,h_in, w_in))
            if self.pad_s < 0:
                mask = mask[:, :, :self.pad_s, :]
            if self.pad_e < 0:
                mask = mask[:, :, :, :self.pad_e]
            mask_padded = np.zeros((1, c_in, h_in + self.pad_s + self.pad_n, w_in + self.pad_w + self.pad_e), dtype = np.float64)
            mask_padded[:, :, self.pad_n: self.pad_n + h_in, self.pad_w: self.pad_w + w_in] = mask
            if TF_FOUND:
                data_out = self.sess.run(self.tf_outputs, feed_dict={self.tf_padded: np.transpose(padded,[0,2,3,1])}).transpose([0,3,1,2])
                data_out *= self.kernel_size_w*self.kernel_size_h
                mask_out = self.sess.run(self.tf_outputs, feed_dict={self.tf_padded: np.transpose(mask_padded,[0,2,3,1])}).transpose([0,3,1,2])
            else:
                data_out = torch.nn.AvgPool2d((self.kernel_size_h, self.kernel_size_w), \
                                              stride=(self.stride_h, self.stride_w))(torch.from_numpy(padded)).data.numpy()
                data_out *= self.kernel_size_w*self.kernel_size_h
                mask_out = torch.nn.AvgPool2d((self.kernel_size_h, self.kernel_size_w), \
                                              stride=(self.stride_h, self.stride_w))(torch.from_numpy(mask_padded)).data.numpy()
            elem_cnt = mask_out[0,0]*self.kernel_size_w*self.kernel_size_h
        else:
            h_out = int((h_in + self.pad_s + self.pad_n - (self.kernel_size_h - 1) * self.dilation_h - 1) / self.stride_h) + 1
            w_out = int((w_in + self.pad_w + self.pad_e - (self.kernel_size_w - 1) * self.dilation_w - 1) / self.stride_w) + 1
            if self.ceil_mode:
                h_out = int((h_in + self.pad_s + self.pad_n - (self.kernel_size_h - 1) * self.dilation_h - 2) / self.stride_h) + 2
                w_out = int((w_in + self.pad_w + self.pad_e - (self.kernel_size_w - 1) * self.dilation_w - 2) / self.stride_w) + 2
                if (h_in + self.pad_n) % self.stride_h == 0:
                    h_out = min(h_out, int((h_in + self.pad_n) / self.stride_h))
                if (w_in + self.pad_w) % self.stride_w == 0:
                    w_out = min(w_out, int((w_in + self.pad_w) / self.stride_w))
            data_out = np.zeros((n, c_in, h_out, w_out), dtype = np.float64)
            elem_cnt = np.zeros((h_out, w_out), dtype= np.float64)
            for k in range(n):
                for c in range(c_in):
                    for i in range(h_out):
                        for j in range(w_out):
                            index_h = i * self.stride_h
                            for h in range(self.kernel_size_h):
                                index_w = j * self.stride_w
                                for w in range(self.kernel_size_w):
                                    if c==0 and k==0 and index_h in range(self.pad_n, h_in + self.pad_n) and index_w in range(self.pad_w, w_in + self.pad_w):
                                        elem_cnt[i, j] += 1
                                    data_out[k, c, i, j] += padded[k, c, index_h, index_w]
                                    index_w += self.dilation_w
                                index_h += self.dilation_h
        return data_out, elem_cnt

class QuantScaleByTensor(QuantLayer):
    def __init__(self, bitwidth=None, in_frac=None, out_frac=None, input_signed=(True, True), name=None):
        super().__init__(name=name)
        if type(bitwidth)!=type(None) and type(in_frac)!=type(None) and type(out_frac)!=type(None):
            self.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)

    def set_quant_param(self, bitwidth, in_frac, out_frac, input_signed=(True, True)):
        super().set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)

    def run(self, data_in, quant=True, **kwargs):
        if quant:
            data_out = self.run_quant(data_in)
        else:
            data_out = self.run_float_point(data_in)
        return data_out

    def run_quant(self, data_ins):
        for i in range(2):
            data_in = data_ins[i]
            data_in = np.asarray(data_in, dtype=np.float64)
            data_in = quantized_value(data_in, bitwidth=self.bitwidth, frac=self.in_frac[i], signed=self.input_signed[i], floor=ENABLE_FLOOR_FEATURE[self.bitwidth])
        data_out = self.run_float_point(data_ins)
        return quantized_value(data_out, bitwidth=self.bitwidth, frac=self.out_frac, floor=ENABLE_FLOOR_FEATURE[self.bitwidth])

    def run_float_point(self, data_ins):
        for data_in in data_ins:
            data_in = np.asarray(data_in, dtype=np.float64)
            assert type(data_in)==np.ndarray, 'invalid type for input data, get {}'.format(type(data_in))
            assert len(data_in.shape)==4, 'input data must be a 4-dim array in NCHW format, but get {}'.format(data_in.shape)
        data_out = data_ins[0].copy()
        c = data_ins[1].shape[1]
        for k in range(c):
            data_out[:,k,:,:] *= data_ins[1][:,k,:,:]
        return data_out


class QuantScale(QuantLayer):
    def __init__(self, c, weight, bias, bias_term=True, bitwidth=None, in_frac=None, out_frac=None, quant_weight=None, quant_bias=None, input_signed=True, name=None):
        super().__init__(name=name)
        assert isinstance(c, int), 'channel number must be an integer, but get a {}'.format(type(c))
        assert c>0, 'channel number must be larger than zero, but get {}'.format(c)
        self.c = c
        self.bias_term = bias_term
        if isinstance(weight, (int, float)):
            self.fp_weight = weight * np.ones(c)
        else:
            assert type(weight)==np.ndarray, 'invalid type for weight, but get {}'.format(type(weight))
            assert len(weight.shape) == 1, 'weight must be a 1-dim array'
            assert len(weight) == c, 'weight size({}) must be the same with channel number({}).'.format(len(weight), c)
            self.fp_weight = weight.copy()
        if bias_term:
            assert type(bias)!=type(None), 'bias must be given when bias_term set True'
            if isinstance(bias, (int, float)):
                self.fp_bias = bias * np.ones(c)
            else:
                assert type(bias)==np.ndarray, 'invalid type for bias, but get {}'.format(type(bias))
                assert len(bias.shape) == 1, 'bias must be a 1-dim array'
                assert len(bias) == c, 'weight size({}) must be the same with channel number({}).'.format(len(bias), c)
                self.fp_bias = bias.copy()
        else:
            self.fp_bias = None
        if type(bitwidth)!=type(None) and type(in_frac)!=type(None) and type(out_frac)!=type(None):
            self.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, quant_weight=quant_weight, quant_bias=quant_bias, input_signed=input_signed)

    def set_quant_param(self, bitwidth, in_frac, out_frac, quant_weight=None, quant_bias=None, input_signed=True):
        if isinstance(quant_weight, (int, float)):
            quant_weight = quant_weight * np.ones(self.c)
        assert type(quant_weight)==np.ndarray, 'invalid type for quant_weight, get {}'.format(type(quant_weight))
        assert quant_weight.shape == self.fp_weight.shape, 'quant_weight must share the same shape with weight'
        self.quant_weight = quant_weight.copy()
        if self.bias_term:
            assert type(quant_bias)!=type(None), 'quant_bias must be given when bias_term set True'
            if isinstance(quant_bias, (int, float)):
                quant_bias = quant_bias * np.ones(self.c)
            assert type(quant_bias)==np.ndarray, 'invalid type for quant_bias, get {}'.format(type(quant_bias))
            assert quant_bias.shape == self.fp_bias.shape, 'quant_weight must share the same shape with weight'
            self.quant_bias = quant_bias.copy()
        else:
            self.quant_bias = None
        super().set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)

    def run(self, data_in, quant=True, quant_weight_only=False, quant_feature_only=False, **kwargs):
        if quant:
            assert not (quant_weight_only and quant_feature_only), 'quant_weight_only and quant_feature_only cannot be True at the same time.'

        if quant and not quant_feature_only:
            assert self.quantized, 'This layer has not be quantized, please call set_quant_param method first.'
            self.weight = self.quant_weight # switch weight to quantized version
            self.bias = self.quant_bias # switch bias to quantized version
        else:
            self.weight = self.fp_weight # switch weight to floating-point version
            self.bias = self.fp_bias # switch bias to floating-point version

        if quant and not quant_weight_only:
            data_out = self.run_quant(data_in)
        else:
            data_out = self.run_float_point(data_in)
        return data_out

    # def run_quant(self, data_in):
    #     assert type(data_in)==np.ndarray, 'invalid type for input data, get {}'.format(type(data_in))
    #     assert len(data_in.shape)==4, 'input data must be a 4-dim array in NCHW format'
    #     _, c_data, _, _ = data_in.shape
    #     assert c_data==self.c, 'not matched number of input channels ({}) to this layer, assuming it is {}'.format(c_data, self.c)
    #     data_out = quantized_value(data_in, bitwidth=self.bitwidth, frac=self.in_frac, signed=self.input_signed)
    #     for k in range(self.c):
    #         data_out[:,k,:,:] *= self.weight[k]
    #         if self.bias_term:
    #             data_out[:, k, :, :] += self.bias[k]
    #     return quantized_value(data_out, bitwidth=self.bitwidth, frac=self.out_frac)


    def run_quant(self, data_in):
        data_in = np.asarray(data_in, dtype=np.float64)
        data_in = quantized_value(data_in, bitwidth=self.bitwidth, frac=self.in_frac, signed=self.input_signed, floor=ENABLE_FLOOR_FEATURE[self.bitwidth])
        data_out = self.run_float_point(data_in)
        return quantized_value(data_out, bitwidth=self.bitwidth, frac=self.out_frac, floor=ENABLE_FLOOR_FEATURE[self.bitwidth])

    def run_float_point(self, data_in):
        data_in = np.asarray(data_in, dtype=np.float64)
        assert type(data_in)==np.ndarray, 'invalid type for input data, get {}'.format(type(data_in))
        assert len(data_in.shape)==4, 'input data must be a 4-dim array in NCHW format'
        _, c_data, _, _ = data_in.shape
        assert c_data==self.c, 'not matched number of input channels ({}) to this layer, assuming it is {}'.format(c_data, self.c)
        data_out = data_in.copy()
        for k in range(self.c):
            data_out[:,k,:,:] *= self.weight[k]
            if self.bias_term:
                data_out[:, k, :, :] += self.bias[k]
        return data_out


class QuantBatchNorm(QuantScale):
    def __init__(self, c, weight, bias, bitwidth=None, in_frac=None, out_frac=None, quant_weight=None, quant_bias=None, input_signed=True, name=None):
        super().__init__(c=c, weight=weight, bias=bias, bias_term=True, bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, quant_weight=quant_weight, quant_bias=quant_bias, input_signed=input_signed, name=name)


class QuantBias(QuantScale):
    def __init__(self, c, bias, bitwidth=None, in_frac=None, out_frac=None, quant_weight=None, quant_bias=None, input_signed=True, name=None):
        super().__init__(c=c, weight=1, bias=bias, bias_term=True, bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, quant_weight=quant_weight, quant_bias=quant_bias, input_signed=input_signed, name=name)

    def set_quant_param(self, bitwidth, in_frac, out_frac, quant_bias=None, input_signed=True):
        assert out_frac <= in_frac, 'Bias - out_frac ({}) must no larger than in_frac ({})'.format(out_frac, in_frac)
        super().set_quant_param(bitwidth, in_frac, out_frac, quant_weight=1, quant_bias=quant_bias, input_signed=input_signed)


class QuantSlice(QuantLayer):
    def __init__(self, slice_point, axis=1, bitwidth=None, in_frac=None, out_frac=None, input_signed=True, name=None):
        super().__init__(name=name)
        assert isinstance(slice_point, list), 'Slice - slice_point should be a list of integers, but get a {}'.format(type(slice_point))
        for sp_idx in range(len(slice_point)):
            assert isinstance(slice_point[sp_idx], int), 'Slice - slice_point should be a list of integers, but get the {}-th element is {}'.format(sp_idx, type(slice_point[sp_idx]))
            assert slice_point[sp_idx]>0, 'Slice - invalid slice_point[{}]={}, which must be larger than 0'.format(sp_idx, slice_point[sp_idx])
        self.slice_point = slice_point
        self.axis = axis
        if type(bitwidth)!=type(None) and type(in_frac)!=type(None) and type(out_frac)!=type(None):
            self.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)

    def set_quant_param(self, bitwidth, in_frac, out_frac, input_signed=True):
        assert in_frac==out_frac, 'Slice - in_frac ({}) must be the same with out_frac ({})'.format(in_frac, out_frac)
        # assert input_signed==True, 'input data must be signed.'
        super().set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)

    def run_quant(self, data_in):
        assert type(data_in)==np.ndarray, 'Slice - input should be a numpy.array, but get a {}'.format(type(data_in))
        data_in = quantized_value(data_in, bitwidth=self.bitwidth, frac=self.in_frac, signed=self.input_signed, floor=ENABLE_FLOOR_FEATURE[self.bitwidth])
        return self.run_float_point(data_in)

    def run_float_point(self, data_in):
        assert type(data_in)==np.ndarray, 'Slice - input should be a numpy.array, but get a {}'.format(type(data_in))
        assert len(data_in.shape)==4, 'input data should be 4-dim numpy.array'
        _, c_in, _, _ = data_in.shape
        for sp_idx in range(len(self.slice_point)):
            assert self.slice_point[sp_idx]>0 and self.slice_point[sp_idx]<c_in, 'Slice - cannot apply slice_point[{}]={}, which must be larger than 0 and smaller than {}'.format(sp_idx, self.slice_point[sp_idx], c_in)
        list_data_out = []
        for sp_idx in range(len(self.slice_point)):
            if sp_idx==0:
                list_data_out.append(data_in[:, :self.slice_point[sp_idx], :, :])
            else:
                list_data_out.append(data_in[:, self.slice_point[sp_idx-1]:self.slice_point[sp_idx], :, :])
        list_data_out.append(data_in[:, self.slice_point[-1]:, :, :])
        return list_data_out


class QuantConcat(QuantLayer):
    def __init__(self, bitwidth=None, in_frac=None, out_frac=None, axis=1, input_signed=True, name=None):
        super().__init__(name=name)
        self.axis = axis
        if type(bitwidth)!=type(None) and type(in_frac)!=type(None) and type(out_frac)!=type(None):
            self.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)

    def set_quant_param(self, bitwidth, in_frac, out_frac, input_signed=True):
        assert in_frac==out_frac, 'Concat - in_frac ({}) must be the same with out_frac ({})'.format(in_frac, out_frac)
        # assert input_signed==True, 'input data must be signed.'
        super().set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)

    def run_quant(self, list_data_in):
        assert isinstance(list_data_in, list), 'Concat - input should be a list of numpy.array, but get a {}'.format(type(list_data_in))
        for i in range(len(list_data_in)):
            list_data_in[i] = quantized_value(list_data_in[i], bitwidth=self.bitwidth, frac=self.in_frac, signed=self.input_signed, floor=ENABLE_FLOOR_FEATURE[self.bitwidth])
        return np.concatenate(list_data_in, axis=1)

    def run_float_point(self, list_data_in):
        assert isinstance(list_data_in, list), 'Concat - input should be a list of numpy.array, but get a {}'.format(type(list_data_in))
        return np.concatenate(list_data_in, axis=1)


class QuantEltwiseSum(QuantLayer):
    def __init__(self, bitwidth=None, in_frac=None, out_frac=None, input_signed=True, name=None):
        super().__init__(name=name)
        if type(bitwidth)!=type(None) and type(in_frac)!=type(None) and type(out_frac)!=type(None):
            self.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)

    def set_quant_param(self, bitwidth, in_frac, out_frac, input_signed=True):
        assert in_frac>=out_frac, 'EltwiseSum - in_frac ({}) should be no smaller than out_frac ({})'.format(in_frac, out_frac)
        super().set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)

    def run_quant(self, list_data_in):
        assert isinstance(list_data_in, list), 'EltwiseSum - input should be a list of numpy.array, but get a {}'.format(type(list_data_in))
        for i in range(len(list_data_in)):
            list_data_in[i] = quantized_value(list_data_in[i], bitwidth=self.bitwidth, frac=self.in_frac, signed=self.input_signed, floor=ENABLE_FLOOR_FEATURE[self.bitwidth])
        return quantized_value(np.sum(list_data_in, axis=0), bitwidth=self.bitwidth, frac=self.out_frac, signed=True, floor=ENABLE_FLOOR_FEATURE[self.bitwidth])

    def run_float_point(self, list_data_in):
        assert isinstance(list_data_in, list), 'EltwiseSum - input should be a list of numpy.array, but get a {}'.format(type(list_data_in))
        return np.sum(list_data_in, axis=0)


class QuantLogSoftmax(QuantLayer):
    def __init__(self, bitwidth=None, in_frac=None, out_frac=None, input_signed=True, name=None):
        super().__init__(name=name)
        if type(bitwidth)!=type(None):
            if type(in_frac)==type(None) or type(out_frac)==type(None):
                if bitwidth==8:
                    in_frac, out_frac = 4, 4
                else:
                    in_frac, out_frac = 12, 12
            self.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)

    def set_quant_param(self, bitwidth, in_frac, out_frac, input_signed=True):
        if bitwidth==8:
            assert in_frac==4 and out_frac==4, 'LogSoftmax - only (in_frac, out_frac)=(4,4) allowed for 8-bit quantization, but get ({},{}).'.format(in_frac, out_frac)
        else:
            assert in_frac==12 and out_frac==12, 'LogSoftmax - only (in_frac, out_frac)=(12,12) allowed for 16-bit quantization, but get ({},{}).'.format(in_frac, out_frac)
        assert input_signed==True, 'input data must be signed.'
        super().set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)

    def run_quant(self, data_in):
        data_in = quantized_value(data_in, bitwidth=self.bitwidth, frac=self.in_frac, signed=self.input_signed, floor=ENABLE_FLOOR_FEATURE[self.bitwidth])
        assert len(data_in.shape) == 4, 'input data must be a 4-dim array in NCHW format'
        n, c, h, w = data_in.shape
        data_out = np.zeros(data_in.shape)
        for idx_n in range(n):
            for idx_h in range(h):
                for idx_w in range(w):
                    # vec_x = data_in[idx_n, :, idx_h, idx_w]
                    sum_exp = -8.0
                    for idx_c in range(c):
                        x = data_in[idx_n, idx_c, idx_h, idx_w]
                        diff = np.abs(sum_exp - x)
                        index = int(diff * 4096)
                        if index > 32767:
                            index = 32767
                        rem_term = _softmax_lut[index] / 2
                        if sum_exp > x:
                            sum_exp = sum_exp + rem_term
                        else:
                            sum_exp = x + rem_term
                        if sum_exp < -8:
                            sum_exp = -8.0
                    sum_exp = quantized_value(sum_exp, bitwidth=self.bitwidth, frac=self.out_frac, floor=False)
                    data_out[idx_n, :, idx_h, idx_w] = data_in[idx_n, :, idx_h, idx_w] - sum_exp
        return quantized_value(data_out, bitwidth=self.bitwidth, frac=self.out_frac, floor=ENABLE_FLOOR_FEATURE[self.bitwidth])

    def run_float_point(self, data_in):
        data_out = np.zeros(data_in.shape)
        assert len(data_in.shape) == 4, 'input data must be a 4-dim array in NCHW format'
        n, c, h, w = data_in.shape
        for idx_n in range(n):
            for idx_h in range(h):
                for idx_w in range(w):
                    sum_exp = np.sum(np.exp(data_in[idx_n, :, idx_h, idx_w]))
                    data_out[idx_n, :, idx_h, idx_w] = np.exp(data_in[idx_n, :, idx_h, idx_w]) / sum_exp
        return np.log(data_out)


class QuantShuffleChannel(QuantLayer):
    def __init__(self, group, bitwidth=None, in_frac=None, out_frac=None, input_signed=True, name=None):
        super().__init__(name=name)
        self.group = group
        if type(bitwidth)!=type(None) and type(in_frac)!=type(None) and type(out_frac)!=type(None):
            self.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)

    def set_quant_param(self, bitwidth, in_frac, out_frac, input_signed=True):
        assert in_frac==out_frac, 'ShuffleChannel - in_frac ({}) must be the same with out_frac ({})'.format(in_frac, out_frac)
        # assert input_signed==True, 'input data must be signed.'
        super().set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)

    def run_quant(self, data_in):
        assert type(data_in)==np.ndarray, 'ShuffleChannel - input should be numpy.array, but get a {}'.format(type(data_in))
        data_in = quantized_value(data_in, bitwidth=self.bitwidth, frac=self.in_frac, signed=self.input_signed, floor=ENABLE_FLOOR_FEATURE[self.bitwidth])
        return self.run_float_point(data_in)

    def run_float_point(self, data_in):
        assert type(data_in)==np.ndarray, 'ShuffleChannel - input should be numpy.array, but get a {}'.format(type(data_in))
        assert len(data_in.shape) == 4, 'input data must be a 4-dim array in NCHW format'
        _, c_in, _, _ = data_in.shape
        assert c_in%self.group==0, 'ShuffleChannel - number of input channels ({}) should be multiple of group ({})'.format(c_in, self.group)
        c_idx = np.array(range(c_in)).reshape((self.group, -1)).transpose().reshape(-1)
        return data_in[:, c_idx, :, :]


if __name__=='__main__':

    # a = np.array([[[[3.7]], [[3.3]], [[2.2]], [[1.1]], [[-1]]]])

    # Quantization parameters could be given when creating the layer object
    # quant_lyr = QuantReLU(bitwidth=8, in_frac=4, out_frac=4, name='layer1')
    # f = quant_lyr.run(a, quant=False) # floating-point mode
    # q = quant_lyr.run(a, quant=True) # fix-point mode

    # Or, creating the layer object without setting quantization parameters, and setting them later
    # quant_lyr = QuantSigmoid(name='layer1')
    # f = quant_lyr.run(a, quant=False) # floating-point mode is usable even before assigning the quantization parameters
    # q = quant_lyr.run(a, quant=True) # fix-point mode will raise an error if the quantization parameters are not set
    # quant_lyr.set_quant_param(bitwidth=8, in_frac=4, out_frac=7) # set quantization parameters
    # q = quant_lyr.run(a, quant=True) # fix-point mode now works :)

    # test
    # q = quant_lyr.run(a, quant=True)
    # f = quant_lyr.run(a, quant=False)
    # print('Quant:')
    # print(q)
    # print('Float:')
    # print(f)
    # print('Error:')
    # print(f-q)

    ## test convlution
    H = 5
    W = 5
    C = 4
    h = 3
    w = 3
    Cout = 2
    # a = np.ones((2,C,H,W))
    # weight = np.zeros((Cout, C, h, w))
    # bias = np.array([1, 0])
    n = 2
    a = np.zeros((n,C,H,W))
    a[0,0] = np.array([[1,1,2,2,1],[1,1,1,2,1],[2,1,1,0,2],[2,1,0,1,2],[2,1,2,2,2]])
    a[0,1] = np.array([[0,1,2,0,1],[2,2,1,1,0],[2,1,0,0,2],[1,0,0,0,2],[0,1,0,1,2]])
    a[0,2] = np.array([[2,2,0,1,2],[0,0,2,1,2],[2,1,0,2,1],[1,1,0,0,0],[0,0,1,1,1]])
    a[0,3] = np.array([[1,-1,0,2,2],[1,0,1,1,1],[0,1,0,1,0],[1,0,1,0,1],[1,0,1,1,1]])
    a[1,0] = np.array([[1,1,2,2,1],[1,1,1,2,1],[2,1,1,0,2],[2,1,0,1,2],[2,1,2,2,2]])
    a[1,1] = np.array([[0,1,2,0,1],[2,2,1,1,0],[2,1,0,0,2],[1,0,0,0,2],[0,1,0,1,2]])
    a[1,2] = np.array([[2,2,0,1,2],[0,0,2,1,2],[2,1,0,2,1],[1,1,0,0,0],[0,0,1,1,1]])
    a[1,3] = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
    weight = np.zeros((Cout,int(C/2),h,w))
    weight[0,0] = np.array([[1,1,1],[-1,-1,0],[-1,1,0]])
    weight[0,1] = np.array([[-1,-1,1],[-1,1,0],[-1,1,0]])
    # weight[0,2] = np.array([[1,0,-1],[0,0,0],[1,-1,-1]])
    # weight[0,3] = np.array([[1,1,-1],[1,0,1],[0,-1,1]])
    weight[1,0] = np.array([[1,1,1],[1,1,1],[1,1,1]])
    weight[1,1] = np.array([[0,0,0],[0,0,0],[0,0,0]])
    # weight[1,2] = np.array([[0,0,0],[0,0,0],[0,0,0]])
    # weight[1,3] = np.array([[1,0,1],[0,1,0],[1,0,1]])
    bias = np.array([0,1])
    print("weight")
    print(weight)

    # quant_lyr = QuantConv2D(bitwidth=8, in_frac=5, out_frac=4, c_in=C, c_out=Cout, kernel_size_h=h, kernel_size_w=w,
    #                        stride_h = 1, stride_w = 1,
    #                        dilation_h = 1, dilation_w = 1,
    #                        group = 2,
    #                        pad_n = 0, pad_s = -1, pad_w = 0, pad_e = -1,
    #                        bias_term = False, weight=weight, bias=bias, quant_weight=weight, quant_bias=bias)

    # quant_lyr = QuantAvgPool2D(kernel_size_h=h, kernel_size_w=w,
    #                        stride_h = 1, stride_w = 1,
    #                        dilation_h = 1, dilation_w = 1,
    #                        pad_n = 1, pad_s = 1, pad_w = 1, pad_e = 1)
    # quant_lyr.set_quant_param(bitwidth=8, in_frac=6, out_frac=6, quant_weight=weight, quant_bias=bias)
    # quant_lyr.set_quant_param(bitwidth=8, in_frac=6, out_frac=6)
    # when running avgpool, one can configure quantized or not, and hardware aligned or not.

    a = np.load('data1.npy')-8
    a = a.reshape((1,-1,1,1))
    quant_lyr = QuantLogSoftmax()
    quant_lyr.set_quant_param(bitwidth=8, in_frac=4, out_frac=4)
    q = quant_lyr.run(a, quant=True, hw_aligned=True).reshape(-1)
    f = quant_lyr.run(a, quant=False, hw_aligned=False).reshape(-1)

    print('Input:')
    print(a)
    print('Quant:')
    print(q)
    print('Float:')
    print(f)
    print('Error:')
    print(f-q)
    np.savetxt('res.csv', [f.reshape(-1), q.reshape(-1), (f-q).reshape(-1)], delimiter=',')
    a = a.reshape(-1)
    print(max(a))
    print(sorted(a)[-5:])
    print(max(f))
    print(max(q))


    ## test InnerProduct - 2D
    # H = 3
    # W = 3
    # C = 3
    # Cout = 2
    # a = np.zeros((1,C,H,W))
    # a[0,0] = np.array([[1,2,2],[1,1,2],[1,1,0]])
    # a[0,1] = np.array([[1,2,0],[2,1,1],[1,0,0]])
    # a[0,2] = np.array([[2,0,1],[0,2,1],[1,0,2]])
    # weight = np.zeros((Cout,C,H,W))
    # weight[0,0] = np.array([[1,1,1],[-1,-1,0],[-1,1,0]])
    # weight[0,1] = np.array([[-1,-1,1],[-1,1,0],[-1,1,0]])
    # weight[0,2] = np.array([[1,0,-1],[0,0,0],[1,-1,-1]])
    # weight[1,0] = np.array([[0,0,-1],[-1,1,1],[0,0,0]])
    # weight[1,1] = np.array([[0,0,1],[1,0,1],[0,-1,-1]])
    # weight[1,2] = np.array([[-1,1,1],[0,1,1],[1,-1,1]])
    # bias = np.array([1, 0])
    #
    # quant_lyr = QuantInnerProduct(bitwidth=16, in_frac=12, out_frac=11, c_in=C, c_out=Cout, h_in=H, w_in=W,
    #                        bias_term = False, weight=weight, bias=bias, quant_weight=weight, quant_bias=bias)
    # q = quant_lyr.run(a, quant=True)
    # f = quant_lyr.run(a, quant=False)
    #
    # print('Quant:')
    # print(q)
    # print('Float:')
    # print(f)
    # print('Error:')
    # print(f-q)

    # test InnerProduct - 1D
    # K = 2
    # N = 4
    # a = np.random.rand(K)
    # weight = np.random.rand(N,K)
    # bias = np.random.rand(N)
    # # a = np.array([1,2,3])
    # # weight = np.array([[1,2,3],[-1,-2,-3]])
    # # bias = np.random.rand(N)
    # quant_lyr = QuantInnerProduct(bitwidth=16, in_frac=12, out_frac=11, c_in=K, c_out=N,
    #                        bias_term = True, weight=weight.reshape(N,K,1,1), bias=bias, quant_weight=weight.reshape(N,K,1,1), quant_bias=bias)
    # q = quant_lyr.run(a.reshape(-1,K,1,1), quant=True)
    # f = quant_lyr.run(a.reshape(-1,K,1,1), quant=False)
    #
    # print('Quant:')
    # print(q)
    # print('Float:')
    # print(f)
    # print('Manual')
    # print(np.matmul(weight, a)+bias)
    # print('Error:')
    # print(f-q)

    #
    # H = 5
    # W = 5
    # C = 2
    # h = 3
    # w = 3
    # n = 2
    # Cout = 2
    # a = np.zeros((n,C,H,W))
    # a[0,0] = np.array([[1,1,2,2,1],[1,1,1,2,1],[2,1,1,0,2],[2,1,0,1,2],[2,1,2,2,2]])
    # a[0,1] = np.array([[0,1,2,0,1],[2,2,1,1,0],[2,1,0,0,2],[1,0,0,0,2],[0,1,0,1,2]])
    #
    # b = np.zeros((n,C,H,W))
    # b[n-1,0] = np.array([[1,1,2,2,1],[1,1,1,2,1],[2,1,1,0,2],[2,1,0,1,2],[2,1,2,2,2]])+2
    # b[n-1,1] = np.array([[0,1,2,0,1],[2,2,1,1,0],[2,1,0,0,2],[1,0,0,0,2],[0,1,0,1,2]])+2
    #
    # quant_lyr = QuantEltwiseSum()
    # quant_lyr.set_quant_param(bitwidth=8, in_frac=4, out_frac=4)
    # c = quant_lyr.run([a,b], quant=False)
    # print('array A:')
    # print(a)
    # print('array B:')
    # print(b)
    # print('array C:')
    # print(c)

    # test ShuffleChannel
    # n = 1
    # C = 12
    # H = 5
    # W = 5
    # g = 3
    # a = np.ones((n,C,H,W))
    # quant_lyr = QuantShuffleChannel(group=g)
    # for idx_c in range(C):
    #     a[:, idx_c, :, :] *= idx_c/10
    # f = quant_lyr.run(a, quant=False)
    # quant_lyr.set_quant_param(bitwidth=8, in_frac=3, out_frac=3)
    # q = quant_lyr.run(a, quant=True)
    # print('Quant:')
    # print(q)
    # print('Float:')
    # print(f)
    # print('Manual')
    # print(np.matmul(weight, a)+bias)
    # print('Error:')
    # print(f-q)

    # quant_lyr = QuantAvgPool2D(kernel_size_h=h, kernel_size_w=w,
    #                        stride_h = 1, stride_w = 1,
    #                        dilation_h = 1, dilation_w = 1,
    #                        pad_n = 1, pad_s = 1, pad_w = 1, pad_e = 1)
    # quant_lyr.set_quant_param(bitwidth=8, in_frac=6, out_frac=6, quant_weight=weight, quant_bias=bias)
    # quant_lyr.set_quant_param(bitwidth=8, in_frac=6, out_frac=6)
    # when running avgpool, one can configure quantized or not, and hardware aligned or not.
    # q = quant_lyr.run(a, quant=True, hw_aligned=True)
    # f = quant_lyr.run(a, quant=False, hw_aligned=False)

    # print('Input:')
    # print(a)
    # print('Quant:')
    # print(q)
    # print('Float:')
    # print(f)
    # print('Error:')
    # print(f-q)

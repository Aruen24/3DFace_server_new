import sys
import os
import pdb
from functools import partial

from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variable_scope
from tensorflow.contrib import graph_editor
from tensorflow.python.training.moving_averages import assign_moving_average

from ndk.simulated_quant.interf_func import get_function_tf, _SPEED_UP, _USE_CUSTOM_OP, SimQuantIntPerChannel
from ndk.layers import Layer
from ndk.simulated_quant.py_func import py_func
from ndk.simulated_quant.sim_quant_node import np_sim_quant_node
if (sys.platform == 'linux' or sys.platform == 'linux2') and _USE_CUSTOM_OP:
    libfake_quant_with_d_module = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                                  'fake_quant_with_d_per_out_channel.so'))
    import ndk.simulated_quant._fake_quant_with_d_grad

_hdw_fix_rs_dict = {'sigmoid': {8: {'tensor_in_frac': 4, 'tensor_out_frac': 7},
                                           16: {'tensor_in_frac': 12, 'tensor_out_frac': 15}},
                               'relu6': {8: {'tensor_in_frac': 4, 'tensor_out_frac': 4},
                                         16: {'tensor_in_frac': 12, 'tensor_out_frac': 12}},
                               'tanh': {8: {'tensor_in_frac': 5, 'tensor_out_frac': 7},
                                        16: {'tensor_in_frac': 13, 'tensor_out_frac': 15}},
                               'Softmax': {8: {'tensor_in_frac': 4, 'tensor_out_frac': 4},
                                           16: {'tensor_in_frac': 12, 'tensor_out_frac': 12}},
                               }

def _getRSNb(bitwidth):
    assert bitwidth in [8, 16], 'Now can only support 8 or 16 bit quantization.'
    if bitwidth == 8:
        return 15
    else:
        return 31

def get_weight(shape, trainable=True, name=None, initializer=tf.contrib.slim.xavier_initializer(), regularizer=None):
    weight = tf.get_variable(shape=shape,
                             dtype=tf.float32,
                             initializer=initializer,
                             regularizer=regularizer,
                             trainable=trainable,
                             name=name)
    return weight

def get_conv_kernel_shape(inputs: tf.Tensor, kernel_size: tuple, num_output: int,
                          data_format: str = 'channels_last') -> list:
    assert data_format in ['channels_last', 'channels_first'], "data_format can only be channels_first or channels_last"
    if not isinstance(inputs, tf.Tensor):
        raise TypeError('The inputs of the layer must be a tensor.')
    if data_format == 'channels_last':
        n, h, w, c = [shape.value for shape in inputs.shape]
    else:
        n, c, h, w = [shape.value for shape in inputs.shape]

    filter_h = kernel_size[0]
    filter_w = kernel_size[1]
    nb_in_channels = c
    nb_out_channels = num_output

    return [filter_h, filter_w, nb_in_channels, nb_out_channels]

def get_dense_kernel_shape(inputs, num_output):
    cin = inputs.shape[1].value
    cout = num_output
    return [cin, cout]

def conv2d_q_1_g(inputs, num_output, kernel_size, strides, dilation_rate=(1, 1), trainable=True, use_bias=True,
                 name=None):

    dilation_rate = [1, 1, dilation_rate[0], dilation_rate[1]]
    strides = [1, 1, strides[0], strides[1]]
    kernel_shape = get_conv_kernel_shape(inputs=inputs, kernel_size=kernel_size, num_output=num_output, data_format='channels_first')
    kernel = get_weight(kernel_shape, trainable=trainable, name=name+'_weight', initializer=tf.contrib.slim.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(0.0005))
    w_mul_x = tf.nn.conv2d(input=inputs, filter=kernel, strides=strides,
                           padding='VALID', dilations=dilation_rate, name=name, data_format='NCHW')

    if use_bias:
        bias = get_weight(shape=[num_output], trainable=trainable, name=name+'_bias', initializer=tf.zeros_initializer())
        bias_add = tf.nn.bias_add(w_mul_x, bias, data_format="NCHW")
        return kernel, bias, bias_add
    else:
        bias_add = w_mul_x
        return kernel, None, bias_add

def _check_pad(inputs, kernel_size, strides, pad, dilation_rate):
    n, c , h, w= [shape.value for shape in inputs.shape]
    kernel_size_h, kernel_size_w = kernel_size
    stride_h, stride_w = strides
    pad_n, pad_s, pad_w, pad_e = pad
    dilation_h, dilation_w = dilation_rate
    assert (h + pad_s + pad_n - (
            kernel_size_h - 1) * dilation_h - 1) % stride_h == 0, 'need more padding in height?'
    assert (w + pad_w + pad_e - (
            kernel_size_w - 1) * dilation_w - 1) % stride_w == 0, 'need more padding in width?'
    assert pad_n >= 0, 'pad_n must be non-negative, but get {}'.format(pad_n)
    assert pad_w >= 0, 'pad_w must be non-negative, but get {}'.format(pad_w)

def _hdw_align_by_pad(inputs, pad):
    pad_n, pad_s, pad_w, pad_e = pad
    if pad_s < 0:
        inputs = inputs[:, :, :pad_s, :]
        pad_s = 0
    if pad_e < 0:
        inputs = inputs[:, :, :, :pad_e]
        pad_e = 0
    return inputs, (pad_n, pad_s, pad_w, pad_e)

def _pool_pad_number(pooling_type):
    assert isinstance(pooling_type, str), "pooling_type must be a string"
    if pooling_type.lower() == 'max':
        return float('-inf')
    else:
        return 0

def add_sim_quant_per_outchannels(tensor, quant_params, isWeight):
    '''Spliting the weights or bias along the out dimension, then add the sim quant
      node along each dimension.
      Inputs:
          tensor:the weights or bias
          quant_params: a list of (n,d) tuples
      Return:
          The concat node of all sim quant nodes.
     '''
#    pdb.set_trace()
    if not _SPEED_UP:
        floor = not isWeight
        sim_quant = np_sim_quant_node(tensor, quant_params, floor=floor)
    else:
        if (sys.platform == 'linux' or sys.platform == 'linux2') and _USE_CUSTOM_OP:
            if not isWeight:
                tensor = tf.transpose(tensor, perm=(0, 2, 3, 1))
            _bitwidth = quant_params[0][0]
            ds = [quant_param[1] for quant_param in quant_params]
            floor = 0. if isWeight else 1.
            sim_quant = libfake_quant_with_d_module.fake_quant_with_d(tensor, ds, _bitwidth, floor)
            if not isWeight:
                sim_quant = tf.transpose(sim_quant, perm=(0, 3, 1, 2))
        else:
            if not isWeight:
                tensor = tf.transpose(tensor, perm=(0, 2, 3, 1))
            sim_quant_func = SimQuantIntPerChannel(quant_params, isWeight=isWeight)
            sim_quant = sim_quant_func(tensor)
            if not isWeight:
                sim_quant = tf.transpose(sim_quant, perm=(0, 3, 1, 2))
            # nb_out = tensor.shape[-1].value
            # splits = tf.split(value=tensor, num_or_size_splits=nb_out, axis=-1)
            # assert len(splits) == len(quant_params), 'unmatched shape of quant_params and weights(kernels or bias).'
            # quant_weights = []
            # for (n, d), split in zip(quant_params, splits):
            #     quant_weight = get_function_tf(n, d, isWeight=isWeight)(split)
            #     quant_weights.append(quant_weight)
            # sim_quant = tf.concat(values=quant_weights, axis=-1)
    return sim_quant

def inser_sim_quant_op(g, producer, name, quant_tensor):
    '''Insert the quant_tensor between the producer tensor and its consumer.
  Inputs:
    g: the graph to be modified
    producer: a tensor which will be quantized
    name:layer name
    quant_tensor:quantized tensor
  '''
    tf.logging.debug('layer:{}, node:{}\n'.format(name, producer.name))
    consumers = []
    for (op_name, op) in g._nodes_by_name.items():
        if producer in op.inputs._inputs and name in op_name:
            consumers.append(op)
    assert len(consumers) != 0, "Can not find the consumers of node {}\n".format(producer.name)
    with variable_scope.variable_scope(name + '/SimQuant'):
        graph_editor.reroute_ts([quant_tensor], [producer], can_modify=consumers)
    assert len(quant_tensor.consumers()) != 0, "Insert sim quant node failed, layer:{}, tensor: {}\n".format(name,
                                                                                                             producer.name)

def add_sim_quant_per_layer(tensor, quant_param, isWeight):
    (n, d) = quant_param[0]
    sim_quant = get_function_tf(n, d, isWeight=isWeight)(tensor)
    return sim_quant

def add_sim_quant(tensor, quant_param, perchannel=False, isWeight=False):
    if perchannel:
        return add_sim_quant_per_outchannels(tensor, quant_param, isWeight)
    else:
        return add_sim_quant_per_layer(tensor, quant_param, isWeight)

def _is_top_frac_in_param_dict(layer, param_dict):
    if isinstance(layer.top, str):
        if layer.top + '_frac' not in param_dict:
            return False
    else:
        for top in layer.top:
            if top + '_frac' not in param_dict:
                return False
    return True

def _is_bottom_frac_in_param_dict(layer, param_dict):
    if isinstance(layer.bottom, str):
        if layer.bottom + '_frac' not in param_dict:
            return False
    else:
        for bottom in layer.bottom:
            if bottom + '_frac' not in param_dict:
                return False
    return True

def _check_is_satisfy_hdw_fix_rs(layer, bitwidth, param_dict):

    if param_dict is None:
        raise ValueError("param_dict should not be None when checking the quant paramters.")
    if not (_is_top_frac_in_param_dict(layer, param_dict) and _is_bottom_frac_in_param_dict(layer, param_dict)):
        raise ValueError("param_dict should contain the layer's bottom and top fracs when checking quant parameters.")

    bottom_frac = param_dict[layer.bottom + '_frac']
    top_frac = param_dict[layer.top + '_frac']

    if layer.type.lower() in ['sigmoid', 'softmax', 'tanh', 'relu6']:
        if bottom_frac != _hdw_fix_rs_dict[layer.type.lower()][bitwidth]['tensor_in_frac']:
            raise ValueError(
                "The layer {}, which type is {}, is not satisfy the hardware restriction. "
                "The frac should be {} when {} bit. The {}_frac is {}".format(
                    layer.name, layer.type, _hdw_fix_rs_dict[layer.type.lower()][bitwidth]['tensor_in_frac'],
                    bitwidth, layer.bottom, bottom_frac))
        if top_frac != _hdw_fix_rs_dict[layer.type.lower()][bitwidth]['tensor_out_frac']:
            raise ValueError(
                "The layer {}, which type is {}, is not satisfy the hardware restriction. "
                "The frac should be {} when {} bit. The {}_frac is {}".format(
                    layer.name, layer.type, _hdw_fix_rs_dict[layer.type.lower()][bitwidth]['tensor_out_frac'],
                    bitwidth, layer.top, top_frac))

def _check_is_satify_hdw_same_rs(layer, param_dict):
    if param_dict is None:
        raise ValueError("param_dict should not be None when checking the quant paramters.")
    if not (_is_top_frac_in_param_dict(layer, param_dict) and _is_bottom_frac_in_param_dict(layer, param_dict)):
        raise ValueError("param_dict should contain the layer's bottom and top fracs when checking quant parameters.")

    if layer.type.lower() in ['relu', 'shufflechannel']:
        bottom_frac = param_dict[layer.bottom + '_frac']
        top_frac = param_dict[layer.top + '_frac']
        if bottom_frac != top_frac:
            raise ValueError("The layer {}, which type is {}, is not satisfy the hardware restriction. "
                             "The {}_frac is {}, the {}_frac is {}, which should be the same.".format(
                layer.name, layer.type, layer.bottom, bottom_frac, layer.top, top_frac
            ))
    elif layer.type.lower() in ['pooling']:
        if layer.pool.lower() == 'max':
            bottom_frac = param_dict[layer.bottom + '_frac']
            top_frac = param_dict[layer.top + '_frac']
            if bottom_frac != top_frac:
                raise ValueError("The layer {}, which type is {}, is not satisfy the hardware restriction. "
                                 "The {}_frac is {}, the {}_frac is {}, which should be the same.".format(
                    layer.name, layer.type, layer.bottom, bottom_frac, layer.top, top_frac
                ))
    elif layer.type.lower() == 'concat':
        top_frac = param_dict[layer.top + '_frac']
        for bottom in layer.bottom:
            bottom_frac = param_dict[bottom + '_frac']
            if bottom_frac != top_frac:
                raise ValueError("The layer {}, which type is {}, is not satisfy the hardware restriction. "
                                 "The {}_frac is {}, the {}_frac is {}, which should be the same.".format(
                    layer.name, layer.type, bottom, bottom_frac, layer.top, top_frac
                ))
    elif layer.type.lower() == 'slice':
        bottom_frac = param_dict[layer.bottom + '_frac']
        for top in layer.top:
            top_frac = param_dict[top + '_frac']
            if bottom_frac != top_frac:
                raise ValueError("The layer {}, which type is {}, is not satisfy the hardware restriction. "
                                 "The {}_frac is {}, the {}_frac is {}, which should be the same.".format(
                    layer.name, layer.type, layer.bottom, bottom_frac, top, top_frac
                ))
    elif layer.type.lower() == 'eltwise':
        bottom_fracs = []
        for idx, bottom in enumerate(layer.bottom):
            if idx == 0:
                bottom_fracs.append(param_dict[bottom + '_frac'])
            else:
                bottom_fracs.append(param_dict[bottom + '_frac'])
                if bottom_fracs[idx] != bottom_fracs[idx-1]:
                    raise ValueError("The layer {}, which type is {}, is not satisfy the hardware restriction. "
                                 "The {}_frac is {}, the {}_frac is {}, which should be the same.".format(
                    layer.name, layer.type, layer.bottom[idx], bottom_fracs[idx], layer.bottom[idx-1], bottom_fracs[idx-1]
                ))

        if layer.operation == 'max':
            bottom_frac = param_dict[layer.bottom + '_frac']
            top_frac = param_dict[layer.top + '_frac']
            if bottom_frac != top_frac:
                raise ValueError("The layer {}, which type is {}, is not satisfy the hardware restriction. "
                                 "the {}_frac is {}, the {}_frac is {}, which should be the same,"
                                 "When the operation is max.".format(
                    layer.name, layer.type, layer.bottom, bottom_frac, layer.top, top_frac
                ))

def _check_is_satisfy_hdw_inequal_rs(layer, bitwidth, param_dict):
    if param_dict is None:
        raise ValueError("param_dict should not be None when checking the quant paramters.")
    if not (_is_top_frac_in_param_dict(layer, param_dict) and _is_bottom_frac_in_param_dict(layer, param_dict)):
        raise ValueError("param_dict should contain the layer's bottom and top fracs when checking quant parameters.")
    top_frac = param_dict[layer.top + '_frac']
    if layer.type.lower() != 'eltwise':
        bottom_frac = param_dict[layer.bottom + '_frac']

    if layer.type.lower() in ['convolution','innerproduct','batchnorm','scale']:
        if layer.name + '_frac_weight' not in param_dict:
            raise ValueError("param_dict should contain the layer's weight frac when checking quant parameters.")
        if layer.type.lower() == 'batchnorm' or layer.bias_term:
            if layer.name + '_frac_bias' not in param_dict:
                raise ValueError("param_dict should contain the layer's bias frac when checking quant parameters.")
            bias_frac = param_dict[layer.name + '_frac_bias']

        kernel_frac = param_dict[layer.name + '_frac_weight']
        if isinstance(kernel_frac, int):
            if top_frac > (kernel_frac + bottom_frac) or top_frac < ( kernel_frac + bottom_frac - _getRSNb(bitwidth)):
                raise ValueError("input_fraction + weight_fraction - output_fraction of conv layer should be "
                                 "in [0, 15] in 8bit mode , [0, 31] in 16bit mode, "
                                 "when weight fraction or bias fraction differ on output channel. "
                                 "layer:{}, top tensor:{}. input_fraction:{},output_fraction:{},"
                                 "weight_fraction:{}".format(layer.name, layer.top, bottom_frac, top_frac, kernel_frac))
        else:
            for k_frac in list(kernel_frac):
                if top_frac > (k_frac + bottom_frac) or top_frac < (
                        k_frac + bottom_frac - _getRSNb(bitwidth)):
                    raise ValueError("input_fraction + weight_fraction - output_fraction of conv layer should be "
                                     "in [0, 15] in 8bit mode , [0, 31] in 16bit mode, "
                                     "when weight fraction or bias fraction differ on output channel. "
                                     "layer:{}, top tensor:{}.input_fraction:{},output_fraction:{},"
                                     "weight_fraction:{}".format(layer.name, layer.top, bottom_frac, top_frac, k_frac))
        if layer.type.lower() == 'batchnorm' or layer.bias_term:
            if isinstance(bias_frac, int):
                if bias_frac > (kernel_frac + bottom_frac) or bias_frac < (kernel_frac + bottom_frac - _getRSNb(bitwidth)):
                    raise ValueError("input_fraction + weight_fraction - bias_fraction of conv layer"
                                     " should be in [0, 15] in 8bit mode, [0, 31] in 16bit mode, "
                                     "when weight fraction or bias fraction differ on output channel. "
                                     "layer:{}. weight_fraction:{}, input_fraction:{}, "
                                     "bias_fraction:{}".format(layer.name, kernel_frac, bottom_frac, bias_frac))
            else:
                for k_frac,b_frac in zip(list(kernel_frac), list(bias_frac)):
                    if b_frac > (k_frac + bottom_frac) or b_frac < (k_frac + bottom_frac - _getRSNb(bitwidth)):
                        raise ValueError("input_fraction + weight_fraction - bias_fraction of conv layer"
                                         " should be in [0, 15] in 8bit mode, [0, 31] in 16bit mode, "
                                         "when weight fraction or bias fraction differ on output channel. "
                                         "layer:{}.weight_fraction:{}, input_fraction:{}, "
                                         "bias_fraction:{}".format(layer.name, k_frac, bottom_frac, b_frac))
    elif layer.type.lower() == 'bias':
        bias_frac = param_dict[layer.name + '_frac_bias']
        if isinstance(bias_frac, int):
            if bias_frac > bottom_frac or bias_frac < (bottom_frac - _getRSNb(bitwidth)):
                raise ValueError("input_fraction - bias_fraction of conv layer"
                                 " should be in [0, 15] in 8bit mode, [0, 31] in 16bit mode, "
                                 "when weight fraction or bias fraction differ on output channel. "
                                 "layer:{}. input_fraction:{},"
                                 "bias_fraction ".format(layer.name, bottom_frac, bias_frac))
        else:
            for b_frac in list(bias_frac):
                if b_frac > bottom_frac or b_frac < (bottom_frac - _getRSNb(bitwidth)):
                    raise ValueError("input_fraction - bias_fraction of conv layer"
                                     " should be in [0, 15] in 8bit mode, [0, 31] in 16bit mode, "
                                     "when weight fraction or bias fraction differ on output channel. "
                                     "layer:{}.input_fraction:{},"
                                     "bias_fraction ".format(layer.name, bottom_frac, b_frac))
    elif layer.type.lower() == 'eltwise':
        if layer.operation.lower() == 'sum':
            for bottom in layer.bottom:
                bottom_frac = param_dict[bottom + '_frac']
                if top_frac > bottom_frac:
                    raise ValueError("output_fraction should be grater than input_frac when layer type is {},"
                                     "and operation is sum.".format(layer.type))

def convert_frac_to_quant_param(fracs, bitwidth):
    quant_param = []
    if type(fracs) == int:
        quant_param.append((bitwidth, -fracs))
    else:
        if len(list(fracs)) == 1:
            quant_param.append((bitwidth, -fracs))
        else:
            for frac in list(fracs):
                quant_param.append((bitwidth, int(-frac)))
    return quant_param

def get_bitwidth(tensorname, bitwidth, param_dict):
    if param_dict is None:
        return bitwidth
    elif bitwidth is 8 and tensorname + '_signed' in param_dict:
        if not param_dict[tensorname + '_signed']:
            return 9
        else:
            return bitwidth
    else:
        return bitwidth

def _add_simq_op_af_output(ndklayer, output, bitwidth, param_dict):
    if param_dict is None:
        return output
    if isinstance(ndklayer.top, str):
        if ndklayer.top + '_frac' in param_dict:
            frac = param_dict[ndklayer.top + '_frac']
            bitwidth = get_bitwidth(ndklayer.top, bitwidth=bitwidth, param_dict=param_dict)
            output = get_function_tf(bitwidth, -frac)(output)
    else:
        for idx, top in enumerate(ndklayer.top):
            if top + '_frac' in param_dict:
                frac = param_dict[top + '_frac']
                bitwidth = get_bitwidth(top, bitwidth=bitwidth, param_dict=param_dict)
                output[idx] = get_function_tf(bitwidth, -frac)(output[idx])
    return output

class SimQuantLayerBase(metaclass=ABCMeta):
    def __init__(self, input, param_dict=None, bitwidth=8):
        self.input = input
        self.input_tensor = input[0]
        self.param_dict = param_dict
        self.bitwidth = bitwidth

    def _get_bottom_ts_name(self, input):
        if isinstance(input, tf.Tensor):

            bottom = input.name
        elif isinstance(input, list):
            bottom = []
            for sub_input in input:
                if isinstance(sub_input, tf.Tensor):
                    bottom.append(sub_input.name)
                else:
                    raise TypeError("The input of the layer must be tf.Tensor object.")
        else:
            raise TypeError("The input of the layer must be tf.Tensor object.")

        return bottom

    @abstractmethod
    def _construct_ndk_layer(self):
        pass

    @abstractmethod
    def _construct_layer(self):
        pass


class SimQuantLayerInput(SimQuantLayerBase):
    def __init__(self,
                 input,
                 param_dict=None,
                 ndklayer=None,
                 bitwidth=8,
                 name=None,
                 dim=None):
        super(SimQuantLayerInput, self).__init__(input, param_dict=param_dict,  bitwidth=bitwidth)

        # self.input_tensor = tf.transpose(self.input_tensor, perm=[0,2,3,1])
        if ndklayer is not None:
            self.ndklayer = ndklayer
        else:
            if name is None:
                raise ValueError(
                    'When ndklayer argument is not given,the name argument should not be None, which will be used to construct ndklayer.')
            self.ndklayer = self._construct_ndk_layer(name=name,
                                                      top=name+'_out:0',
                                                      dim=dim)

        self.output = (self._construct_layer(), self.ndklayer.top)
        self.output_tensor = self.output[0]

    def _construct_ndk_layer(self,
                             name=None,
                             top=None,
                             dim=None):
        ndklayer = Layer()
        ndklayer.set_input_layer(name=name,
                                 top=top,
                                 dim=dim)
        return ndklayer

    def _construct_layer(self):
        output = _add_simq_op_af_output(self.ndklayer, self.input_tensor, self.bitwidth, self.param_dict)
        return output


class SimQuantLayerSigmoid(SimQuantLayerBase):
    def __init__(self,
                 input,
                 param_dict=None,
                 ndklayer=None,
                 bitwidth=8,
                 name=None):
        super(SimQuantLayerSigmoid, self).__init__(input, param_dict=param_dict,  bitwidth=bitwidth)
        if ndklayer is not None:
            self.ndklayer = ndklayer
        else:
            if name is None:
                raise ValueError(
                    'When ndklayer argument is not given,the name argument should not be None, which will be used to construct ndklayer.')
            bottom = self.input[1]
            self.ndklayer = self._construct_ndk_layer(name=name,
                                                      top=name+'_out:0',
                                                      bottom=bottom)

        self.output = (self._construct_layer(), self.ndklayer.top)
        self.output_tensor = self.output[0]

    def _construct_ndk_layer(self,
                             name=None,
                             top=None,
                             bottom=None,
                             negative_slope=None):
        ndklayer = Layer()
        ndklayer.set_sigmoid_layer(name=name,
                               bottom=bottom,
                               top=top)
        return ndklayer

    def _construct_layer(self):

        # output = tf.nn.sigmoid(self.input_tensor, name=self.ndklayer.name)
        output = tf.keras.activations.sigmoid(self.input_tensor)
        if self.param_dict is not None and self.ndklayer.top + '_frac' in self.param_dict:
            _check_is_satisfy_hdw_fix_rs(self.ndklayer, self.bitwidth, self.param_dict)
        output = _add_simq_op_af_output(self.ndklayer, output, self.bitwidth, self.param_dict)

        return output


class SimQuantLayerRelu(SimQuantLayerBase):
    def __init__(self,
                 input,
                 param_dict=None,
                 ndklayer=None,
                 bitwidth=8,
                 name=None,
                 negative_slope=0.):
        super(SimQuantLayerRelu, self).__init__(input, param_dict=param_dict,  bitwidth=bitwidth)
        if ndklayer is not None:
            self.ndklayer = ndklayer
        else:
            if name is None:
                raise ValueError(
                    'When ndklayer argument is not given,the name argument should not be None, which will be used to construct ndklayer.')
            bottom = self.input[1]
            self.ndklayer = self._construct_ndk_layer(name=name,
                                                      top=name+'_out:0',
                                                      bottom=bottom,
                                                      negative_slope=negative_slope)

        self.output = (self._construct_layer(), self.ndklayer.top)
        self.output_tensor = self.output[0]

    def _construct_ndk_layer(self,
                             name=None,
                             top=None,
                             bottom=None,
                             negative_slope=None):
        ndklayer = Layer()
        ndklayer.set_relu_layer(name=name,
                               bottom=bottom,
                               top=top,
                               negative_slope=negative_slope)
        return ndklayer

    def _construct_layer(self):
        if self.ndklayer.negative_slope != 0:
            alpha = tf.convert_to_tensor(self.ndklayer.negative_slope)
            alpha = get_function_tf(self.bitwidth, -6)(alpha)
        else:
            alpha=0

        if self.param_dict is not None and self.ndklayer.top + '_frac' in self.param_dict:
            _check_is_satify_hdw_same_rs(self.ndklayer, self.param_dict)
        output = tf.keras.layers.LeakyReLU(alpha=alpha, name=self.ndklayer.name)(self.input_tensor)
        # output = tf.nn.leaky_relu(self.input_tensor, alpha=alpha, name=self.ndklayer.name)
        output = _add_simq_op_af_output(self.ndklayer, output, self.bitwidth, self.param_dict)

        return output


class SimQuantLayerRelu6(SimQuantLayerBase):
    def __init__(self,
                 input,
                 param_dict=None,
                 ndklayer=None,
                 bitwidth=8,
                 name=None):
        super(SimQuantLayerRelu6, self).__init__(input, param_dict=param_dict,  bitwidth=bitwidth)
        if ndklayer is not None:
            self.ndklayer = ndklayer
        else:
            if name is None:
                raise ValueError(
                    'When ndklayer argument is not given,the name argument should not be None, which will be used to construct ndklayer.')
            bottom = self.input[1]
            self.ndklayer = self._construct_ndk_layer(name=name,
                                                      top=name+'_out:0',
                                                      bottom=bottom)

        self.output = (self._construct_layer(), self.ndklayer.top)
        self.output_tensor = self.output[0]

    def _construct_ndk_layer(self,
                             name=None,
                             top=None,
                             bottom=None,
                             negative_slope=None):
        ndklayer = Layer()
        ndklayer.set_relu6_layer(name=name,
                               bottom=bottom,
                               top=top)
        return ndklayer

    def _construct_layer(self):

        output = tf.nn.relu6(self.input_tensor, name=self.ndklayer.name)
        if self.param_dict is not None and self.ndklayer.top + '_frac' in self.param_dict:
            _check_is_satisfy_hdw_fix_rs(self.ndklayer, self.bitwidth, self.param_dict)
        output = _add_simq_op_af_output(self.ndklayer, output, self.bitwidth, self.param_dict)

        return output


class SimQuantLayerTanh(SimQuantLayerBase):
    def __init__(self,
                 input,
                 param_dict=None,
                 ndklayer=None,
                 bitwidth=8,
                 name=None):
        super(SimQuantLayerTanh, self).__init__(input, param_dict=param_dict,  bitwidth=bitwidth)
        if ndklayer is not None:
            self.ndklayer = ndklayer
        else:
            if name is None:
                raise ValueError(
                    'When ndklayer argument is not given,the name argument should not be None, which will be used to construct ndklayer.')
            bottom = self.input[1]
            self.ndklayer = self._construct_ndk_layer(name=name,
                                                      top=name+'_out:0',
                                                      bottom=bottom)

        self.output = (self._construct_layer(), self.ndklayer.top)
        self.output_tensor = self.output[0]

    def _construct_ndk_layer(self,
                             name=None,
                             top=None,
                             bottom=None):
        ndklayer = Layer()
        ndklayer.set_tanh_layer(name=name,
                               bottom=bottom,
                               top=top)
        return ndklayer

    def _construct_layer(self):

        # output = tf.nn.tanh(self.input_tensor, name=self.ndklayer.name)
        output = tf.keras.activations.tanh(self.input_tensor)
        if self.param_dict is not None and self.ndklayer.top + '_frac' in self.param_dict:
            _check_is_satisfy_hdw_fix_rs(self.ndklayer, self.bitwidth, self.param_dict)
        output = _add_simq_op_af_output(self.ndklayer, output, self.bitwidth, self.param_dict)

        return output

class SimQuantLayerSoftmax(SimQuantLayerBase):
    def __init__(self,
                 input,
                 param_dict=None,
                 ndklayer=None,
                 bitwidth=8,
                 name=None):
        super(SimQuantLayerSoftmax, self).__init__(input, param_dict=param_dict,  bitwidth=bitwidth)
        if ndklayer is not None:
            self.ndklayer = ndklayer
        else:
            if name is None:
                raise ValueError(
                    'When ndklayer argument is not given,the name argument should not be None, which will be used to construct ndklayer.')
            bottom = self.input[1]
            self.ndklayer = self._construct_ndk_layer(name=name,
                                                      top=name+'_out:0',
                                                      bottom=bottom)

        self.output = (self._construct_layer(), self.ndklayer.top)
        self.output_tensor = self.output[0]

    def _construct_ndk_layer(self,
                             name=None,
                             top=None,
                             bottom=None):
        ndklayer = Layer()
        ndklayer.set_logsoftmax_layer(name=name,
                               bottom=bottom,
                               top=top)
        return ndklayer

    def _construct_layer(self):

        # output = tf.nn.softmax(self.input_tensor, name=self.ndklayer.name)
        output = tf.keras.activations.softmax(self.input_tensor, axis=1)
        if self.param_dict is not None and self.ndklayer.top + '_frac' in self.param_dict:
            _check_is_satisfy_hdw_fix_rs(self.ndklayer, self.bitwidth, self.param_dict)
        output = _add_simq_op_af_output(self.ndklayer, output, self.bitwidth, self.param_dict)

        return output

class SimQuantLayerDense(SimQuantLayerBase):
    def __init__(self,
                 input,
                 param_dict=None,
                 ndklayer=None,
                 bitwidth=8,
                 name=None,
                 trainable=True,
                 perchannel=True,
                 num_output=None,
                 use_bias=True):
        super(SimQuantLayerDense, self).__init__(input, param_dict=param_dict, bitwidth=bitwidth)
        self.perchannel = perchannel
        self.trainable = trainable
        if ndklayer is not None:
            self.ndklayer = ndklayer
        else:
            if name is None:
                raise ValueError(
                    'When ndklayer argument is not given,the name argument should not be None, which will be used to construct ndklayer.')
            bottom = self.input[1]
            self.ndklayer = self._construct_ndk_layer(name=name,
                                                      top=name + '_out:0',
                                                      bottom=bottom,
                                                      num_output=num_output,
                                                      bias_term=use_bias)

        self.output = (self._construct_layer(), self.ndklayer.top)
        self.output_tensor = self.output[0]

    def _construct_ndk_layer(self,
                             name=None,
                             top=None,
                             bottom=None,
                             num_output=None,
                             bias_term=True):
        ndk_layer = Layer()
        ndk_layer.set_innerproduct_layer(name=name,
                                         bottom=bottom,
                                         top=top,
                                         num_output=num_output,
                                         bias_term=bias_term)
        return ndk_layer

    def _construct_layer(self):
        if len(self.input_tensor.shape) > 2:
            input = tf.layers.flatten(self.input_tensor)
        else:
            input = self.input_tensor

        kernel_shape = get_dense_kernel_shape(input, self.ndklayer.num_output)
        with variable_scope.variable_scope(self.ndklayer.name, reuse=tf.AUTO_REUSE):
            _kernel = get_weight(kernel_shape, trainable=self.trainable, name='dense_weight', regularizer=tf.contrib.slim.l2_regularizer(0.0005))
            matmul = tf.matmul(input, _kernel)
            if self.ndklayer.bias_term:
                _bias = get_weight([self.ndklayer.num_output], trainable=self.trainable, name='dense_bias', initializer=tf.zeros_initializer())
                _output = tf.add(matmul, _bias)
            else:
                _output = matmul

        g = tf.get_default_graph()

        if self.param_dict is not None and self.ndklayer.name + '_frac_weight' in self.param_dict:
            _check_is_satisfy_hdw_inequal_rs(self.ndklayer, self.bitwidth, self.param_dict)

        # Kernel
        if self.param_dict is not None and self.ndklayer.name + "_frac_weight" in self.param_dict and self.param_dict[
            self.ndklayer.name + "_frac_weight"] is not None:
            producer = _kernel.value()
            _quant_kernel = add_sim_quant(
                producer,
                convert_frac_to_quant_param(self.param_dict[self.ndklayer.name + '_frac_weight'], self.bitwidth),
                perchannel=self.perchannel, isWeight=True)
            inser_sim_quant_op(g, producer, self.ndklayer.name, _quant_kernel)

        # Bias
        if self.ndklayer.bias_term and self.param_dict is not None and self.ndklayer.name + "_frac_bias" in self.param_dict and \
                self.param_dict[self.ndklayer.name + "_frac_bias"] is not None:
            producer = _bias.value()
            _quant_bias = add_sim_quant(
                producer,
                convert_frac_to_quant_param(self.param_dict[self.ndklayer.name + '_frac_bias'], self.bitwidth),
                perchannel=self.perchannel, isWeight=True)
            inser_sim_quant_op(g, producer, self.ndklayer.name, _quant_bias)

        self.kernel = _kernel
        if self.ndklayer.bias_term:
            self.bias = _bias
        output = _add_simq_op_af_output(self.ndklayer, _output, self.bitwidth, self.param_dict)

        return output


class SimQuantLayerConv2d(SimQuantLayerBase):
    def __init__(self,
                 input,
                 param_dict=None,
                 ndklayer=None,
                 bitwidth=8,
                 name=None,
                 trainable=True,
                 perchannel=True,
                 # weight_initializer=tf.contrib
                 num_output=None,
                 kernel_size=None,
                 stride=None,
                 pad=None,
                 use_bias=True,
                 dilation_rate=1,
                 group=1):
        super(SimQuantLayerConv2d, self).__init__(input, param_dict=param_dict,  bitwidth=bitwidth)
        self.perchannel = perchannel
        self.trainable = trainable
        if ndklayer is not None:
            self.ndklayer = ndklayer
        else:
            if name is None:
                raise ValueError(
                    'When ndklayer argument is not given,the name argument should not be None, which will be used to construct ndklayer.')
            bottom = self.input[1]
            self.ndklayer = self._construct_ndk_layer(name=name,
                                                      top=name+'_out:0',
                                                      bottom=bottom,
                                                      num_output=num_output,
                                                      kernel_size=kernel_size,
                                                      stride=stride,
                                                      pad=pad,
                                                      bias_term=use_bias,
                                                      dilation=dilation_rate,
                                                      group=group)
        self.output = (self._construct_layer(), self.ndklayer.top)
        self.output_tensor = self.output[0]

    def _construct_ndk_layer(self,
                             name=None,
                             top=None,
                             bottom=None,
                             num_output=None,
                             kernel_size=None,
                             stride=None,
                             pad=None,
                             bias_term=True,
                             dilation=1,
                             group=1):
        if num_output is None:
            raise ValueError('The num_output argument is required.')
        ndklayer = Layer()
        ndklayer.set_convolution_layer(name=name,
                                       top=top,
                                       bottom=bottom,
                                       num_output=num_output,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       pad=pad,
                                       bias_term=bias_term,
                                       dilation=dilation,
                                       group=group)
        return ndklayer

    def _construct_layer(self):
        _check_pad(self.input_tensor, self.ndklayer.kernel_size, self.ndklayer.stride, self.ndklayer.pad,
                   self.ndklayer.dilation)
        input, pad = _hdw_align_by_pad(self.input_tensor, self.ndklayer.pad)
        assert type(pad) == tuple and len(pad) == 4, "pad argument must be a tuple of 4 elements"
        pad = [[pad[0], pad[1]], [pad[2], pad[3]]]
        pad = [[0, 0]] + [[0, 0]] + pad
        input = tf.pad(input, paddings=pad)

        assert self.ndklayer.group != 0, "Groups should not be zero!"
        assert input.shape[1].value % self.ndklayer.group == 0, "The input channels of convolution should be exact divided by group"
        assert self.ndklayer.num_output % self.ndklayer.group == 0, "The output channels of convolution should be exact divided by groups."
        num_cin_channel_per_group = int(input.shape[1].value / self.ndklayer.group)
        num_cout_channel_per_group = int(self.ndklayer.num_output / self.ndklayer.group)
        _outputs = []
        _kernels = []
        if self.ndklayer.bias_term:
            _biases = []
        for idx in range(int(self.ndklayer.group)):
            with variable_scope.variable_scope(self.ndklayer.name + '/group' + str(idx), reuse=tf.AUTO_REUSE):
                input_g = input[:, idx * num_cin_channel_per_group: (idx + 1) * num_cin_channel_per_group, :, :]
                kernel_g, bias_g, output_g = conv2d_q_1_g(input_g, num_output=num_cout_channel_per_group,
                                                          kernel_size=self.ndklayer.kernel_size,
                                                          strides=self.ndklayer.stride,
                                                          dilation_rate=self.ndklayer.dilation,
                                                          trainable=self.trainable,
                                                          use_bias=self.ndklayer.bias_term,
                                                          name='conv2d')
                _kernels.append(kernel_g)
                _outputs.append(output_g)
                if self.ndklayer.bias_term:
                    _biases.append(bias_g)
        output = tf.concat(_outputs, axis=1)

        g = tf.get_default_graph()

        if self.param_dict is not None and self.ndklayer.name + '_frac_weight' in self.param_dict:
            _check_is_satisfy_hdw_inequal_rs(self.ndklayer, self.bitwidth, self.param_dict)

        # quant kernel
        for idx in range(self.ndklayer.group):
            if self.param_dict is not None and self.ndklayer.name + "_frac_weight" in self.param_dict and self.param_dict[self.ndklayer.name + "_frac_weight"] is not None:
                producer = _kernels[idx].value()
                if self.perchannel:
                    quant_kernel_g = add_sim_quant(
                        producer,
                        convert_frac_to_quant_param(self.param_dict[self.ndklayer.name + '_frac_weight'], self.bitwidth)[
                        idx * num_cout_channel_per_group: (idx + 1) * num_cout_channel_per_group],
                        self.perchannel, isWeight=True)
                else:
                    quant_kernel_g = add_sim_quant(
                        producer,
                        convert_frac_to_quant_param(self.param_dict[self.ndklayer.name + '_frac_weight'], self.bitwidth),
                        self.perchannel, isWeight=True)
                inser_sim_quant_op(g, producer, self.ndklayer.name, quant_kernel_g)

        # quant bias
        for idx in range(self.ndklayer.group):
            if self.ndklayer.bias_term and self.param_dict is not None and self.ndklayer.name + "_frac_bias" in self.param_dict and self.param_dict[self.ndklayer.name + "_frac_bias"] is not None:
                producer = _biases[idx].value()
                if self.perchannel:
                    quant_bias_g = add_sim_quant(
                        producer,
                        convert_frac_to_quant_param(self.param_dict[self.ndklayer.name + '_frac_bias'], self.bitwidth)[
                        idx * num_cout_channel_per_group: (idx + 1) * num_cout_channel_per_group],
                        self.perchannel, isWeight=True)
                else:
                    quant_bias_g = add_sim_quant(
                        producer,
                        convert_frac_to_quant_param(self.param_dict[self.ndklayer.name + '_frac_bias'], self.bitwidth),
                        self.perchannel,isWeight=True)
                inser_sim_quant_op(g, producer, self.ndklayer.name, quant_bias_g)

        output = _add_simq_op_af_output(self.ndklayer, output, self.bitwidth, self.param_dict)
        self.kernel = _kernels
        if self.ndklayer.bias_term:
            self.bias = _biases

        return output


class AvgPool2DHwAligned():
    def __init__(self, pool_size, padding, stride, name=None):
        self.pool_size = pool_size
        self.kernel_size_h, self.kernel_size_w = self.pool_size
        self.padding = padding
        self.pad_n, self.pad_s, self.pad_w, self.pad_e = self.padding
        self.stride = stride
        self.stride_h, self.stride_w = stride
        self.name = name

    def adj_elem_cnt_by_hw(self, elem_cnt):
        # center
        active_h = min(self.kernel_size_h, self.h_in)
        active_w = min(self.kernel_size_w, self.w_in)
        elem_cnt = np.ones(elem_cnt.shape) * active_h * active_w
        # left edge
        active_h = min(self.kernel_size_h, self.h_in)
        active_w = min(self.kernel_size_w-self.pad_w, self.w_in)
        elem_cnt[:, 0] = active_h * active_w
        # upper edge
        active_h = min(self.kernel_size_h-self.pad_n, self.h_in)
        active_w = min(self.kernel_size_w, self.w_in)
        elem_cnt[0, :] = active_h * active_w
        # lower edge
        active_h = min(self.kernel_size_h-max(self.pad_s,0), self.h_in)
        active_w = min(self.kernel_size_w, self.w_in)
        elem_cnt[-1, :] = active_h * active_w
        # upper-left corner
        active_h = min(self.kernel_size_h-self.pad_n, self.h_in)
        active_w = min(self.kernel_size_w-self.pad_w, self.w_in)
        elem_cnt[0, 0] = active_h * active_w
        # lower-left corner
        active_h = min(self.kernel_size_h-max(self.pad_s,0), self.h_in)
        active_w = min(self.kernel_size_w-self.pad_w, self.w_in)
        elem_cnt[-1, 0] = active_h * active_w

        return elem_cnt

    def __call__(self, input):
        self.input = input
        self.n, self.c, self.h_in, self.w_in = input.shape
        self.output, self.elem_cnt = self._np_pooling_2d()
        self.elem_cnt = self.adj_elem_cnt_by_hw(self.elem_cnt)

        scale = 1 / self.elem_cnt
        for idx_n in range(self.n):
            for idx_c in range(self.c):
                self.output[idx_n, idx_c, :, :] = self.output[idx_n, idx_c, :, :] * scale

        self.output = self.output.astype(np.float32)
        return self.output

    def _np_pooling_2d(self):
        h_out = int(
            (self.h_in  - self.kernel_size_h) / self.stride_h) + 1
        w_out = int(
            (self.w_in - self.kernel_size_w) / self.stride_w) + 1
        data_out = np.zeros((self.n, self.c, h_out, w_out), dtype=np.float64)
        elem_cnt = np.zeros((h_out, w_out), dtype=np.float64)
        for k in range(self.n):
            for c in range(self.c):
                for i in range(h_out):
                    for j in range(w_out):
                        index_h = i * self.stride_h
                        for h in range(self.kernel_size_h):
                            index_w = j * self.stride_w
                            for w in range(self.kernel_size_w):
                                if c == 0 and k == 0 and index_h in range(self.pad_n,self.h_in + self.pad_n) \
                                        and index_w in range(self.pad_w, self.w_in + self.pad_w):
                                    elem_cnt[i, j] += 1
                                data_out[k, c, i, j] += self.input[k, c, index_h, index_w]
                                index_w += 1
                            index_h += 1

        return data_out, elem_cnt

    def gradient(self, grad):
        self.n_out, self.c_out , self.h_out, self.w_out= [shape.value for shape in grad.shape]

        grad_prev = np.zeros(shape=(self.n, self.c, self.h_in, self.w_in))
        with tf.Session() as sess:
            grad = grad.eval()
        for idx_n in range(self.n_out):
            for idx_c in range(self.c_out):
                for idx_h in range(self.h_out):
                    for idx_w in range(self.w_out):
                        if self.elem_cnt[idx_h, idx_w] <= 0:
                            raise ValueError('pad:{},stride:{},pool_size:{}. '
                                             'This confiugration is not support by hardwere'.format(
                                self.padding, self.stride, self.pool_size))
                        average_grad_loc = grad[idx_n, idx_c, idx_h, idx_w]/ self.elem_cnt[idx_h, idx_w]
                        grad_prev[idx_n,
                        idx_c,
                        idx_h*self.stride_h:idx_h*self.stride_h+self.kernel_size_h,
                        idx_w*self.stride_w:idx_w*self.stride_w+self.kernel_size_w] += \
                            np.ones((self.kernel_size_h, self.kernel_size_w))*average_grad_loc
        grad_prev = tf.convert_to_tensor(grad_prev)
        return grad_prev


def tf_avg_pool2d_hw_aligned_generator(pool_size, padding, stride):
    avg_pool2d_hw_aligned = AvgPool2DHwAligned(pool_size, padding, stride)

    def ave_pool2d_hw_aligned_gradient(op, grad):
        n, c, h_in, w_in = [shape.value for shape in op.inputs[0].shape]
        avg_pool2d_hw_aligned.n = n
        avg_pool2d_hw_aligned.h_in = h_in
        avg_pool2d_hw_aligned.w_in = w_in
        avg_pool2d_hw_aligned.c = c
        elem_cnt = tf.constant(
            avg_pool2d_hw_aligned.kernel_size_w*avg_pool2d_hw_aligned.kernel_size_h,
            shape=[grad.shape[1].value, grad.shape[2].value]
        )
        avg_pool2d_hw_aligned.elem_cnt = avg_pool2d_hw_aligned.adj_elem_cnt_by_hw(elem_cnt)

        return avg_pool2d_hw_aligned.gradient(grad)

    def tf_ave_pool2d_hw_aligned(input, name):
        y = py_func(avg_pool2d_hw_aligned, [input], [tf.float32], name, grad=ave_pool2d_hw_aligned_gradient)
        return y[0]

    return tf_ave_pool2d_hw_aligned


class SimQuantLayerPool(SimQuantLayerBase):
    def __init__(self,
                 input,
                 param_dict=None,
                 ndklayer=None,
                 bitwidth=8,
                 name=None,
                 kernel_size=None,
                 stride=None,
                 pad=None,
                 dilation_rate=1,
                 pooling_type='max',
                 hdw_aligned=False):
        super(SimQuantLayerPool, self).__init__(input, param_dict=param_dict,  bitwidth=bitwidth)
        self.hdw_aligned = hdw_aligned
        if ndklayer is not None:
            self.ndklayer = ndklayer
        else:
            if name is None:
                raise ValueError(
                    'When ndklayer argument is not given,the name argument should not be None, which will be used to construct ndklayer.')
            bottom = self.input[1]
            self.ndklayer = self._construct_ndk_layer(name=name,
                                                      top=name+'_out:0',
                                                      bottom=bottom,
                                                      kernel_size=kernel_size,
                                                      stride=stride,
                                                      pad=pad,
                                                      dilation=dilation_rate,
                                                      pool=pooling_type)

        self.output = (self._construct_layer(), self.ndklayer.top)
        self.output_tensor = self.output[0]

    def _construct_ndk_layer(self,
                             name=None,
                             top=None,
                             bottom=None,
                             kernel_size=None,
                             stride=None,
                             pad=None,
                             dilation=1,
                             pool='max'):
        ndklayer = Layer()
        ndklayer.set_pooling_layer(name=name,
                                   top=top,
                                   bottom=bottom,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   pad=pad,
                                   dilation=dilation,
                                   pool=pool)
        return ndklayer

    def _construct_layer(self):
        if self.ndklayer.dilation != None:
            if self.ndklayer.dilation[0] > 1 or self.ndklayer.dilation[1] > 1:
                raise ValueError('pooling layer do not support dilatrion_rate bigger than 1 in quantization training.')

        _check_pad(self.input_tensor, self.ndklayer.kernel_size, self.ndklayer.stride, self.ndklayer.pad,
                   self.ndklayer.dilation)
        input, pad = _hdw_align_by_pad(self.input_tensor, self.ndklayer.pad)
        pool_pad_number = _pool_pad_number(self.ndklayer.pool)

        if pad != None:
            assert type(pad) == tuple and len(pad) == 4, "pad argument must be a tuple of 4 elements"
            pad = [[pad[0], pad[1]], [pad[2], pad[3]]]
            pad = [[0, 0]] + [[0, 0]] + pad
            input = tf.pad(input, paddings=pad, constant_values=pool_pad_number)

        assert self.ndklayer.pool in ['max', 'ave'], 'Not supported pooling type: {}!\n'.format(self.ndklayer.pool)
        if self.ndklayer.pool.lower() == 'max':
            # output = tf.layers.max_pooling2d(input, pool_size=self.ndklayer.kernel_size,
            #                                                strides=self.ndklayer.stride, name=self.ndklayer.name, data_format="channels_first")
            output = tf.keras.layers.MaxPooling2D(pool_size=self.ndklayer.kernel_size,
                                                           strides=self.ndklayer.stride, name=self.ndklayer.name, data_format="channels_first")(input)

        elif self.ndklayer.pool.lower() == 'ave':
            if self.hdw_aligned:
                avg_pool2d_fn = tf_avg_pool2d_hw_aligned_generator(pool_size=self.ndklayer.kernel_size,
                                                                   padding=pad, stride=self.ndklayer.stride)
                output = avg_pool2d_fn(input=input, name=self.ndklayer.name)
            else:
                # output = tf.layers.average_pooling2d(input, pool_size=self.ndklayer.kernel_size,
                #                                      strides=self.ndklayer.stride, name=self.ndklayer.name, data_format="channels_first")
                output = tf.keras.layers.AveragePooling2D(pool_size=self.ndklayer.kernel_size,
                                                          strides=self.ndklayer.stride, name=self.ndklayer.name, data_format="channels_first")(input)
        else:
            raise TypeError("pooling type:{}, plearse be careful about spelling.".format(self.ndklayer.pool))

        if self.param_dict is not None and self.ndklayer.top + '_frac' in self.param_dict:
            _check_is_satify_hdw_same_rs(self.ndklayer, self.param_dict)
        output = _add_simq_op_af_output(self.ndklayer, output, self.bitwidth, self.param_dict)

        return output


def scale_base(inputs, kernel, bias=None, name=None):
    '''channel first inputs, nchw, kernel c dimension, bias c dimension.
  '''
    with tf.name_scope(name):
        # if len(inputs.shape) == 4:
        #     cout = inputs.shape[1].value
        #     splits = tf.split(axis=1, value=inputs, num_or_size_splits=cout)
        # else:
        cout = inputs.shape[1].value
        splits = tf.split(axis=1, value=inputs, num_or_size_splits=cout)
        if bias != None:
            bias_splits = tf.split(axis=-1, value=bias, num_or_size_splits=cout)
        kernel_splits = tf.split(axis=-1, value=kernel, num_or_size_splits=cout)

    scaled = []
    for idx, split in enumerate(splits):
        if bias != None:
            scaled.append(tf.add(tf.multiply(split, kernel_splits[idx]), bias_splits[idx]))
        else:
            scaled.append(tf.multiply(split, kernel_splits[idx]))

    scaled = tf.concat(scaled, 1)

    return scaled

def bias_base(inputs, bias, name):
    '''channel first inputs, nchw, kernel c dimension.
  '''
    with tf.name_scope(name):
        # if len(inputs.shape) == 4:
        #     cout = inputs.shape[3].value
        #     splits = tf.split(axis=3, value=inputs, num_or_size_splits=cout)
        # else:
        cout = inputs.shape[1].value
        splits = tf.split(axis=1, value=inputs, num_or_size_splits=cout)
        bias_splits = tf.split(axis=-1, value=bias, num_or_size_splits=cout)

    biased = []
    for idx, split in enumerate(splits):
        biased.append(tf.add(split, bias_splits[idx]))

    # if len(inputs.shape) == 4:
    #     biased = tf.concat(biased, 3)
    # else:
    biased = tf.concat(biased, 1)

    return biased


class SimQuantLayerBatchNorm(SimQuantLayerBase):
    def __init__(self,
                 input,
                 param_dict=None,
                 ndklayer=None,
                 bitwidth=8,
                 name=None,
                 perchannel=True,
                 trainable=False):
        super(SimQuantLayerBatchNorm, self).__init__(input, param_dict=param_dict,  bitwidth=bitwidth)
        self.decay = 0.9
        self.trainable = trainable
        self.perchannel = perchannel
        if ndklayer is not None:
            self.ndklayer = ndklayer
        else:
            if name is None:
                raise ValueError(
                    'When ndklayer argument is not given,the name argument should not be None, which will be used to construct ndklayer.')
            bottom = self.input[1]
            self.ndklayer = self._construct_ndk_layer(name=name,
                                                      top=name+'_out:0',
                                                      bottom=bottom)

        self.output = (self._construct_layer(), self.ndklayer.top)
        self.output_tensor = self.output[0]

    def _construct_ndk_layer(self,
                             name=None,
                             top=None,
                             bottom=None):
        ndklayer = Layer()
        ndklayer.set_batchnorm_layer(name=name,
                                     bottom=bottom,
                                     top=top)
        return ndklayer

    def _construct_layer(self):
        num_output = self.input_tensor.shape[1].value
        self.ndklayer.num_output = num_output
        with variable_scope.variable_scope(self.ndklayer.name, reuse=tf.AUTO_REUSE):
            _bias = tf.get_variable(name='bias', shape=[num_output], initializer=tf.zeros_initializer, trainable=False)
            _kernel = tf.get_variable(name='kernel', shape=[num_output], initializer=tf.ones_initializer, trainable=False)
            moving_variance = tf.get_variable(name='moving_variance', shape=[num_output], initializer=tf.ones_initializer, trainable=False)
            moving_mean = tf.get_variable(name='moving_mean', shape=[num_output], initializer=tf.zeros_initializer, trainable=False)


        # def kernel_bias_with_update(moving_variance, moving_mean):
        def kernel_bias_with_update():
            with tf.control_dependencies([moving_variance.assign(1/(_kernel + 1e-10)), moving_mean.assign(-_bias/(_kernel + 1e-10))]):
                if len(self.input_tensor.shape.as_list()) == 2:
                    bn_shape = [0]
                elif len(self.input_tensor.shape.as_list()) == 4:
                    bn_shape = [0, 2, 3]
                mean, variance = tf.nn.moments(self.input_tensor, axes=bn_shape, name='moments')
                with tf.control_dependencies([assign_moving_average(moving_mean, mean, self.decay),
                                              assign_moving_average(moving_variance, variance, self.decay)]):
                    kernel_updated = 1 / (variance + 1e-10)
                    bias_updated = -mean / (variance + 1e-10)
                    with tf.control_dependencies([_kernel.assign(kernel_updated), _bias.assign(bias_updated)]):
                        return tf.identity(kernel_updated), tf.identity(bias_updated)

        # kernel_updated, bias_updated = tf.cond(tf.convert_to_tensor(self.trainable), partial(kernel_bias_with_update, moving_variance=moving_variance, moving_mean=moving_mean), lambda: (_kernel, _bias))
        kernel_updated, bias_updated = tf.cond(tf.convert_to_tensor(self.trainable), kernel_bias_with_update, lambda: (_kernel, _bias))

        _output = scale_base(self.input_tensor, kernel_updated, bias_updated, name=self.ndklayer.name)

        g = tf.get_default_graph()

        if self.param_dict is not None and self.ndklayer.name + '_frac_weight' in self.param_dict:
            _check_is_satisfy_hdw_inequal_rs(self.ndklayer, self.bitwidth, self.param_dict)

        # Kernel
        if self.param_dict is not None and self.ndklayer.name + "_frac_weight" in self.param_dict and self.param_dict[self.ndklayer.name + "_frac_weight"] is not None:
            # producer = _kernel.value()
            producer = kernel_updated
            _quant_kernel = add_sim_quant(
                producer,
                convert_frac_to_quant_param(self.param_dict[self.ndklayer.name + '_frac_weight'], self.bitwidth),
                self.perchannel, isWeight=True)
            inser_sim_quant_op(g, producer, self.ndklayer.name, _quant_kernel)

        # Bias
        if self.param_dict is not None and self.ndklayer.name + "_frac_bias" in self.param_dict and self.param_dict[self.ndklayer.name + "_frac_bias"] is not None:
            # producer = _bias.value()
            producer = bias_updated
            _quant_bias = add_sim_quant(producer,
                                        convert_frac_to_quant_param(self.param_dict[self.ndklayer.name + '_frac_bias'], self.bitwidth),
                                        self.perchannel, isWeight=True)
            inser_sim_quant_op(g, producer, self.ndklayer.name, _quant_bias)

        # Output
        output = _add_simq_op_af_output(self.ndklayer, _output, self.bitwidth, self.param_dict)
        self.kernel = _kernel
        self.bias = _bias

        return output


class SimQuantLayerBias(SimQuantLayerBase):
    def __init__(self,
                 input,
                 param_dict=None,
                 ndklayer=None,
                 bitwidth=8,
                 name=None,
                 perchannel=True):
        super(SimQuantLayerBias, self).__init__(input, param_dict=param_dict, bitwidth=bitwidth)
        self.perchannel = perchannel
        if ndklayer is not None:
            self.ndklayer = ndklayer
        else:
            if name is None:
                raise ValueError(
                    'When ndklayer argument is not given,the name argument should not be None, which will be used to construct ndklayer.')
            bottom = self.input[1]
            self.ndklayer = self._construct_ndk_layer(name=name,
                                                      top=name+'_out:0',
                                                      bottom=bottom)

        self.output = (self._construct_layer(), self.ndklayer.top)
        self.output_tensor = self.output[0]

    def _construct_ndk_layer(self,
                             name=None,
                             top=None,
                             bottom=None):
        ndklayer = Layer()
        ndklayer.set_bias_layer(name=name,
                                 bottom=bottom,
                                 top=top)
        return ndklayer

    def _construct_layer(self):
        num_output = self.input_tensor.shape[1].value

        self.ndklayer.num_output = num_output
        with variable_scope.variable_scope(self.ndklayer.name, reuse=tf.AUTO_REUSE):
            _bias = tf.Variable(tf.random_uniform(shape=[num_output]), trainable=False, name='bias')

        self.bias = _bias
        if self.param_dict is not None and self.ndklayer.name + '_frac_bias' in self.param_dict:
            _check_is_satisfy_hdw_inequal_rs(self.ndklayer, self.bitwidth, self.param_dict)

        _output = bias_base(self.input_tensor, _bias, name=self.ndklayer.name)

        g = tf.get_default_graph()

        # Bias
        if self.param_dict is not None and self.ndklayer.name + "_frac_bias" in self.param_dict and self.param_dict[self.ndklayer.name + "_frac_bias"] is not None:
            producer = _bias.value()
            _quant_bias = add_sim_quant(producer,
                                        convert_frac_to_quant_param(self.param_dict[self.ndklayer.name + '_frac_bias'], self.bitwidth),
                                        self.perchannel, isWeight=True)
            inser_sim_quant_op(g, producer, self.ndklayer.name, _quant_bias)

        output = _add_simq_op_af_output(self.ndklayer, _output, self.bitwidth, self.param_dict)

        return output


class SimQuantLayerScale(SimQuantLayerBase):
    def __init__(self,
                 input,
                 param_dict=None,
                 ndklayer=None,
                 bitwidth=8,
                 name=None,
                 perchannel=True,
                 use_bias=True,
                 trainable=False):
        super(SimQuantLayerScale, self).__init__(input, param_dict=param_dict, bitwidth=bitwidth)
        self.perchannel = perchannel
        self.trainable = trainable
        if ndklayer is not None:
            self.ndklayer = ndklayer
        else:
            if name is None:
                raise ValueError(
                    'When ndklayer argument is not given,the name argument should not be None, which will be used to construct ndklayer.')
            bottom = self.input[1]
            self.ndklayer = self._construct_ndk_layer(name=name,
                                                      top=name+'_out:0',
                                                      bottom=bottom,
                                                      bias_term=use_bias)

        self.output = (self._construct_layer(), self.ndklayer.top)
        self.output_tensor = self.output[0]

    def _construct_ndk_layer(self,
                             name=None,
                             top=None,
                             bottom=None,
                             bias_term=True):
        ndklayer = Layer()
        ndklayer.set_scale_layer(name=name,
                                 bottom=bottom,
                                 top=top,
                                 bias_term=bias_term)
        return ndklayer

    def _construct_layer(self):
        # if len(self.input_tensor.shape) == 4:
        #     num_output = self.input_tensor.shape[3].value
        # else:
        num_output = self.input_tensor.shape[1].value
        self.ndklayer.num_output = num_output
        with variable_scope.variable_scope(self.ndklayer.name, reuse=tf.AUTO_REUSE):
            _kernel = tf.Variable(tf.random_uniform(shape=[num_output]), trainable=self.trainable, name='kernel')
            if self.ndklayer.bias_term:
                _bias = tf.Variable(tf.random_uniform(shape=[num_output]), trainable=self.trainable, name='bias')
            else:
                _bias = tf.zeros([self.ndklayer.num_output])

        _output = scale_base(self.input_tensor, _kernel, _bias, name=self.ndklayer.name)

        g = tf.get_default_graph()

        if self.param_dict is not None and self.ndklayer.name + '_frac_weight' in self.param_dict:
            _check_is_satisfy_hdw_inequal_rs(self.ndklayer, self.bitwidth, self.param_dict)

        # Kernel
        if self.param_dict is not None and self.ndklayer.name + "_frac_weight" in self.param_dict and self.param_dict[self.ndklayer.name + "_frac_weight"] is not None:
            producer = _kernel.value()
            _quant_kernel = add_sim_quant(
                producer,
                convert_frac_to_quant_param(self.param_dict[self.ndklayer.name + '_frac_weight'], self.bitwidth),
                self.perchannel, isWeight=True)
            inser_sim_quant_op(g, producer, self.ndklayer.name, _quant_kernel)

        # Bias
        if self.ndklayer.bias_term and self.param_dict is not None and self.ndklayer.name + "_frac_bias" in self.param_dict and self.param_dict[self.ndklayer.name + "_frac_bias"] is not None:
            producer = _bias.value()
            _quant_bias = add_sim_quant(producer,
                                        convert_frac_to_quant_param(self.param_dict[self.ndklayer.name + '_frac_bias'], self.bitwidth),
                                        self.perchannel, isWeight=True)
            inser_sim_quant_op(g, producer, self.ndklayer.name, _quant_bias)

        # Output
        output = _add_simq_op_af_output(self.ndklayer, _output, self.bitwidth, self.param_dict)
        self.kernel = _kernel
        if self.ndklayer.bias_term:
            self.bias = _bias

        return output

class SimQuantLayerScaleByTensor(SimQuantLayerBase):
    def __init__(self,
                 input,
                 param_dict=None,
                 ndklayer=None,
                 bitwidth=8,
                 name=None,
                 trainable=False):
        super(SimQuantLayerScaleByTensor, self).__init__(input, param_dict=param_dict, bitwidth=bitwidth)
        self.trainable = trainable
        if ndklayer is not None:
            self.ndklayer = ndklayer
        else:
            if name is None:
                raise ValueError(
                    'When ndklayer argument is not given,the name argument should not be None, which will be used to construct ndklayer.')
            bottom = self.input[1]
            self.ndklayer = self._construct_ndk_layer(name=name,
                                                      top=name+'_out:0',
                                                      bottom=bottom)

        self.output = (self._construct_layer(), self.ndklayer.top)
        self.output_tensor = self.output[0]

    def _construct_ndk_layer(self,
                             name=None,
                             top=None,
                             bottom=None):
        ndklayer = Layer()
        ndklayer.set_scale_by_tensor_layer(name=name, bottom=bottom, top=top)
        return ndklayer

    def _construct_layer(self):
        input1 = tf.expand_dims(self.input_tensor[1], -1)
        input2 = tf.expand_dims(input1, -1)
        _outputs = tf.multiply(self.input_tensor[0], input2)
        output = _add_simq_op_af_output(self.ndklayer, _outputs, self.bitwidth, self.param_dict)

        return output

class SimQuantLayerSlice(SimQuantLayerBase):
    def __init__(self,
                 input,
                 param_dict=None,
                 ndklayer=None,
                 bitwidth=8,
                 name=None,
                 axis=1,
                 slice_point=None
                 ):
        super(SimQuantLayerSlice, self).__init__(input, param_dict=param_dict,  bitwidth=bitwidth)
        if ndklayer is not None:
            self.ndklayer = ndklayer
        else:
            if name is None:
                raise ValueError(
                    'When ndklayer argument is not given,the name argument should not be None, which will be used to construct ndklayer.')
            if slice_point is None:
                raise ValueError(
                    'Please give the slice_point argument, which should not be None.'
                )
            bottom = self.input[1]
            self.ndklayer = self._construct_ndk_layer(name=name,
                                                      top=[name+'_out:{}'.format(idx) for idx in range(len(slice_point)+1)],
                                                      bottom=bottom,
                                                      axis=axis,
                                                      slice_point=slice_point)

        self.output = (self._construct_layer(), self.ndklayer.top)
        self.output_tensor = self.output[0]

    def _construct_ndk_layer(self,
                             name=None,
                             top=None,
                             bottom=None,
                             axis=None,
                             slice_point=None):
        ndklayer = Layer()
        ndklayer.set_slice_layer(name=name,
                                 bottom=bottom,
                                 top=top,
                                 axis = axis,
                                 slice_point=slice_point)
        return ndklayer

    def _construct_layer(self):

        outputs = []
        with tf.name_scope(self.ndklayer.name):
            for sp_idx in range(len(self.ndklayer.slice_point)):
                if sp_idx == 0:
                    outputs.append(self.input_tensor[:, :self.ndklayer.slice_point[sp_idx], :, :])
                else:
                    outputs.append(self.input_tensor[:, self.ndklayer.slice_point[sp_idx - 1]:self.ndklayer.slice_point[sp_idx], :, :])
            outputs.append(self.input_tensor[:, self.ndklayer.slice_point[-1]:, :, :])
        if self.param_dict is not None and self.ndklayer.bottom + '_frac' in self.param_dict:
            _check_is_satify_hdw_same_rs(self.ndklayer, self.param_dict)
        outputs = _add_simq_op_af_output(self.ndklayer, outputs, self.bitwidth, self.param_dict)

        return outputs


class SimQuantLayerConcat(SimQuantLayerBase):
    def __init__(self,
                 input,
                 param_dict=None,
                 ndklayer=None,
                 bitwidth=8,
                 name=None,
                 axis=1):
        super(SimQuantLayerConcat, self).__init__(input, param_dict=param_dict,  bitwidth=bitwidth)
        if ndklayer is not None:
            self.ndklayer = ndklayer
        else:
            if name is None:
                raise ValueError(
                    'When ndklayer argument is not given,the name argument should not be None, which will be used to construct ndklayer.')
            bottom = self.input[1]
            self.ndklayer = self._construct_ndk_layer(name=name,
                                                      top=name+'_out:0',
                                                      bottom=bottom,
                                                      axis=axis)

        self.output = (self._construct_layer(), self.ndklayer.top)
        self.output_tensor = self.output[0]

    def _construct_ndk_layer(self,
                             name=None,
                             top=None,
                             bottom=None,
                             axis=None):
        ndklayer = Layer()
        ndklayer.set_concat_layer(name=name,
                               bottom=bottom,
                               top=top,
                                axis=axis)
        return ndklayer

    def _construct_layer(self):
        for idx, input_ts in enumerate(self.input_tensor):
            if idx == 0:
                prev_ts_shape_len = len(input_ts.shape)
            if len(self.input_tensor[idx-1].shape) != prev_ts_shape_len:
                raise ValueError('The input tensor of concat must have the same number of dimensions.')

        with tf.name_scope(self.ndklayer.name):
            # if len(self.input_tensor[0].shape) == 4:
            #     output = tf.concat(self.input_tensor, 3)
            # else:
            output = tf.concat(self.input_tensor, 1)
        if self.param_dict is not None and self.ndklayer.top + '_frac' in self.param_dict:
            _check_is_satify_hdw_same_rs(self.ndklayer, self.param_dict)
        output = _add_simq_op_af_output(self.ndklayer, output, self.bitwidth, self.param_dict)

        return output


class SimQuantLayerEltwise(SimQuantLayerBase):
    def __init__(self,
                 input,
                 param_dict=None,
                 ndklayer=None,
                 bitwidth=8,
                 name=None,
                 operation='sum'):
        super(SimQuantLayerEltwise, self).__init__(input, param_dict=param_dict,  bitwidth=bitwidth)
        if ndklayer is not None:
            self.ndklayer = ndklayer
        else:
            if name is None:
                raise ValueError(
                    'When ndklayer argument is not given,the name argument should not be None, which will be used to construct ndklayer.')
            bottom = self.input[1]
            self.ndklayer = self._construct_ndk_layer(name=name,
                                                      top=name+'_out:0',
                                                      bottom=bottom,
                                                      operation=operation)

        self.output = (self._construct_layer(), self.ndklayer.top)
        self.output_tensor = self.output[0]

    def _construct_ndk_layer(self,
                             name=None,
                             top=None,
                             bottom=None,
                             operation=None):
        ndklayer = Layer()
        ndklayer.set_eltwise_layer(name=name,
                               bottom=bottom,
                               top=top,
                                operation=operation)
        return ndklayer

    def _construct_layer(self):

        assert self.ndklayer.operation.lower() in [
            'sum'], "Please supply the supported operation of eltwise, the operation is {}\n".format(
            self.ndklayer.operation)

        if self.ndklayer.operation.lower() == 'sum':
            _outputs = tf.zeros_like(self.input_tensor[0])
            for input_ts in self.input_tensor:
                _outputs = tf.add(_outputs, input_ts)
        if self.param_dict is not None and self.ndklayer.bottom[0] + '_frac' in self.param_dict:
            _check_is_satisfy_hdw_inequal_rs(self.ndklayer, self.bitwidth, self.param_dict)
        output = _add_simq_op_af_output(self.ndklayer, _outputs, self.bitwidth, self.param_dict)

        return output


class SimQuantLayerShuffleChannel(SimQuantLayerBase):
    def __init__(self,
                 input,
                 param_dict=None,
                 ndklayer=None,
                 bitwidth=8,
                 name=None,
                 group=1):
        super(SimQuantLayerShuffleChannel, self).__init__(input, param_dict=param_dict, bitwidth=bitwidth)
        if ndklayer is not None:
            self.ndklayer = ndklayer
        else:
            if name is None:
                raise ValueError(
                    'When ndklayer argument is not given,the name argument should not be None, which will be used to construct ndklayer.')
            bottom = self.input[1]
            self.ndklayer = self._construct_ndk_layer(name=name,
                                                      top=name + '_out:0',
                                                      bottom=bottom,
                                                      group=group)

        self.output = (self._construct_layer(), self.ndklayer.top)
        self.output_tensor = self.output[0]

    def _construct_ndk_layer(self,
                             name=None,
                             top=None,
                             bottom=None,
                             group=None):
        ndklayer = Layer()
        ndklayer.set_shufflechannel_layer(name=name,
                                   bottom=bottom,
                                   top=top,
                                   group=group)
        return ndklayer

    def _construct_layer(self):

        _, c_in, _, _ = self.input_tensor.shape
        assert c_in % self.ndklayer.group == 0, 'ShuffleChannel - number of input channels ({}) should be multiple of group ({})'.format(
            c_in, self.ndklayer.group)
        c_idx = np.array(range(c_in)).reshape((self.ndklayer.group, -1)).transpose().reshape(-1)
        _outputs = []
        with tf.name_scope(self.ndklayer.name):
            for idx in c_idx:
                _outputs.append(self.input_tensor[:, idx:idx + 1, :, :])
            _outputs = tf.concat(_outputs, 1)
        if self.param_dict is not None and self.ndklayer.bottom + '_frac' in self.param_dict:
            _check_is_satify_hdw_same_rs(self.ndklayer, self.param_dict)
        output = _add_simq_op_af_output(self.ndklayer, _outputs, self.bitwidth, self.param_dict)

        return output
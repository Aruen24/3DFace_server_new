# -*- coding: utf-8 -*-
"""
Created on Tue May 9 16:01:57 2019

@author: qing.chuandong
"""
import os
import sys
import time
from abc import ABCMeta, abstractmethod

import tensorflow as tf
import numpy as np

ndk_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ndk_dir)

from ndk.quant_tools.quant_func import quant2float, float2quant
from ndk.simulated_quant.sim_quant_layers import SimQuantLayerConv2d, SimQuantLayerDense, SimQuantLayerPool,\
    SimQuantLayerBatchNorm, SimQuantLayerScale, SimQuantLayerScaleByTensor, SimQuantLayerInput, SimQuantLayerRelu, SimQuantLayerConcat, \
    SimQuantLayerShuffleChannel, SimQuantLayerRelu6, SimQuantLayerEltwise, SimQuantLayerBias, SimQuantLayerSigmoid, \
    SimQuantLayerSlice, SimQuantLayerTanh
from ndk.utils import print_log

class SimQuantModel:
    def __init__(self,
                 layer_list=None,
                 bitwidth=8,
                 param_dict=None,
                 quant_layer_names=None,
                 trainable_layer_names=None,
                 perlayer_quant_layer_names=None):

        if layer_list is None:
            self.layer_list = []
        else:
            self.layer_list = layer_list

        if perlayer_quant_layer_names is None:
            self._perlayer_quant_layer_names = set()
        else:
            if not self._is_set_of_str(perlayer_quant_layer_names):
                raise TypeError("The layer_names of per layer quantization must be a set of layer name or None")
            self._perlayer_quant_layer_names = perlayer_quant_layer_names

        if param_dict is None:
            self.param_dict = {}
        else:
            self.param_dict = param_dict

        if quant_layer_names is None:
            self._quant_layer_names = set()
        else:
            if not self._is_set_of_str(quant_layer_names):
                raise TypeError("quant_layer_names can only be a set of str or None")
            self._quant_layer_names = quant_layer_names

        if trainable_layer_names is None:
            self._trainable_layer_names = set()
        else:
            if not self._is_set_of_str(trainable_layer_names):
                raise TypeError(
                    "trainable_layer_names can only be a set of str.")
            self._trainable_layer_names = trainable_layer_names

        if not isinstance(bitwidth, int):
            raise TypeError("Target bit must be a int.")
        if not bitwidth in [8, 16]:
            raise ValueError('''bitwidth must be either 8 or 16''')
        self.bitwidth = bitwidth

        self.dict_of_weight_bias_variable = {}
        self.dict_of_feature_map_tensor = {}

        self.layertype_layerclass = {'input': SimQuantLayerInput,
                                     'convolution': SimQuantLayerConv2d,
                                     'innerproduct': SimQuantLayerDense,
                                     'pooling':SimQuantLayerPool,
                                     'relu': SimQuantLayerRelu,
                                     'relu6': SimQuantLayerRelu6,
                                     'sigmoid':SimQuantLayerSigmoid,
                                     'tanh':SimQuantLayerTanh,
                                     'concat':SimQuantLayerConcat,
                                     'slice':SimQuantLayerSlice,
                                     'eltwise':SimQuantLayerEltwise,
                                     'shufflechannel':SimQuantLayerShuffleChannel,
                                     'bias':SimQuantLayerBias,
                                     'scale':SimQuantLayerScale,
                                     'scalebytensor':SimQuantLayerScaleByTensor,
                                     'batchnorm':SimQuantLayerBatchNorm}

        if isinstance(self.layer_list, list) and len(self.layer_list) != 0:
            self._init_with_layer_list()

    def _init_with_layer_list(self):
        self.order_lys = self._construct_order_layers()
        self._check_not_surport_layer_type(self.order_lys)
        self._is_quant_layer_names_in_layer_names(self._quant_layer_names)
        self._is_trainable_layer_names_in_layer_names(self._trainable_layer_names)

    def set_quant_layer_names(self, quant_layer_names):
        if quant_layer_names is None:
            self._quant_layer_names = set()
        else:
            if not self._is_set_of_str(quant_layer_names):
                raise TypeError("quant_layers can only be a set of str or None")
            self._is_quant_layer_names_in_layer_names(quant_layer_names)
            self._quant_layer_names = quant_layer_names

    def set_trainable_layer_names(self, trainable_layer_names):

        if trainable_layer_names is None:
            self._trainable_layer_names = set()
        else:
            if not self._is_set_of_str(trainable_layer_names):
                raise TypeError(
                        "trainable_layers can only be a set of str or None")
            self._is_trainable_layer_names_in_layer_names(trainable_layer_names)
            self._trainable_layer_names = trainable_layer_names

    def set_perlayer_quant_layer_names(self, perlayer_quant_layer_names):

        if perlayer_quant_layer_names is None:
            self._perlayer_quant_layer_names = set()
        else:
            if not self._is_set_of_str(perlayer_quant_layer_names):
                raise TypeError(
                        "trainable_layers can only be a set of str or None")
            self._perlayer_quant_layer_names = perlayer_quant_layer_names

    def construct_network_graph(self, input):
        self.order_lys = self._construct_order_layers()
        # construct the input layer_names
        layer = self.input_layers[0]
        print_log('''constructing layer {}, {}/{}'''.format(layer.name, 1, len(self.order_lys)))
        if layer.type.lower() == 'input':
            constructed_layer = self._construct_one_layer(layer, (input, input.name))
            self._add_feature_map_tensors_a_layer(constructed_layer)
            self._add_weight_bias_variable_a_layer(constructed_layer)
            self._check_parm_dict(constructed_layer)

        for idx, layer in enumerate(self.order_lys):
            if layer in self.input_layers:  # self.order_lys also contain the input layer_names
                continue
            print_log('''constructing layer {}, {}/{}'''.format(layer.name, idx + 1, len(self.order_lys)))
            if layer.type.lower() in ['concat', 'eltwise', 'scalebytensor']:
                bottoms = [self.dict_of_feature_map_tensor[bottom] for bottom in layer.bottom]
                constructed_layer = self._construct_one_layer(
                    layer,
                    (bottoms, layer.bottom),
                )
            elif layer.type.lower() in ['slice']:
                constructed_layer = self._construct_one_layer(
                    layer,
                    (self.dict_of_feature_map_tensor[layer.bottom], layer.bottom),
                )
            else:
                constructed_layer = self._construct_one_layer(
                    layer,
                    (self.dict_of_feature_map_tensor[layer.bottom], layer.bottom)
                )
            self._add_feature_map_tensors_a_layer(constructed_layer)
            self._add_weight_bias_variable_a_layer(constructed_layer)
            self._check_parm_dict(constructed_layer)

        # gather the output
        output = []
        for output_layer in self.output_layers:
            if output_layer.type.lower() in ['slice']:
                for top in layer.top:
                    output.append(self.dict_of_feature_map_tensor[top])
            else:
                output.append(self.dict_of_feature_map_tensor[output_layer.top])

        if len(output) == 1:
            return output[0]
        else:
            return output

    def add_layer(self, simquantlayer):
        if simquantlayer.ndklayer.type.lower() == 'input':
            if len(self.layer_list) > 0:
                raise ValueError("Can not add input layer when the layer list is not empty.")
        if simquantlayer.ndklayer in self.layer_list:
            raise ValueError("The layer {} is already in layer_list.".format(simquantlayer.ndklayer))
        else:
            self.layer_list.append(simquantlayer.ndklayer)
            self._add_param_dict_a_layer(simquantlayer)
        self._check_parm_dict(simquantlayer)
        self._add_weight_bias_variable_a_layer(simquantlayer)
        self._add_feature_map_tensors_a_layer(simquantlayer)

    def restore_weight_bias(self, sess, from_quant=False):
        self._construct_weight_bias_restoring_operations(from_quant=from_quant)
        sess.run(self.restoring_operations)

    def export_quant_param_dict(self, sess):
        self._generate_weight_bias_tensors_whose_dimension_in_param_dict_order()
        _trained_weight_bias = sess.run(self._weight_bias_tensors_whose_dimension_in_param_dict_order)
        _quant_weight_bias = self._get_quant_weight_bias(_trained_weight_bias, self.param_dict)
        self.param_dict.update(_trained_weight_bias)
        self.param_dict.update(_quant_weight_bias)
        _quant_layer_list = self.layer_list
        _quant_param_dict = self.param_dict
        return _quant_layer_list, _quant_param_dict

    def _add_param_dict_a_layer(self, sim_quant_layer):
        if sim_quant_layer.param_dict is not None:
            self._add_float_weight_bias_a_layer_in_param_dict(sim_quant_layer)
            self._add_quant_weight_bias_a_layer_in_param_dict(sim_quant_layer)
            self._add_fracs_a_layer_in_param_dict(sim_quant_layer)

    def _add_float_weight_bias_a_layer_in_param_dict(self, sim_quant_layer):
        if sim_quant_layer.ndklayer.name + '_weight' in sim_quant_layer.param_dict:
            self.param_dict[sim_quant_layer.ndklayer.name + '_weight'] = \
                sim_quant_layer.param_dict[sim_quant_layer.ndklayer.name + '_weight']
        if sim_quant_layer.ndklayer.name + '_bias' in sim_quant_layer.param_dict:
            self.param_dict[sim_quant_layer.ndklayer.name + '_bias'] = \
                sim_quant_layer.param_dict[sim_quant_layer.ndklayer.name + '_bias']

    def _add_quant_weight_bias_a_layer_in_param_dict(self, sim_quant_layer):
        if sim_quant_layer.ndklayer.name + '_quant_weight' in sim_quant_layer.param_dict:
            self.param_dict[sim_quant_layer.ndklayer.name + '_quant_weight'] = \
                sim_quant_layer.param_dict[sim_quant_layer.ndklayer.name + '_quant_weight']
        if sim_quant_layer.ndklayer.name + '_quant_bias' in sim_quant_layer.param_dict:
            self.param_dict[sim_quant_layer.ndklayer.name + '_quant_bias'] = \
                sim_quant_layer.param_dict[sim_quant_layer.ndklayer.name + '_quant_bias']

    def _add_fracs_a_layer_in_param_dict(self, sim_quant_layer):
        if sim_quant_layer.ndklayer.name + '_frac_weight' in sim_quant_layer.param_dict:
            self.param_dict[sim_quant_layer.ndklayer.name + '_frac_weight'] = \
                sim_quant_layer.param_dict[sim_quant_layer.ndklayer.name + '_frac_weight']
        if sim_quant_layer.ndklayer.name + '_frac_bias' in sim_quant_layer.param_dict:
            self.param_dict[sim_quant_layer.ndklayer.name + '_frac_bias'] = \
                sim_quant_layer.param_dict[sim_quant_layer.ndklayer.name + '_frac_bias']
        if isinstance(sim_quant_layer.ndklayer.top, str):
            if sim_quant_layer.ndklayer.top + '_frac' in sim_quant_layer.param_dict:
                self.param_dict[sim_quant_layer.ndklayer.top + '_frac'] = \
                    sim_quant_layer.param_dict[sim_quant_layer.ndklayer.top + '_frac']
        else:
            for top in sim_quant_layer.ndklayer.top:
                if top + '_frac' in sim_quant_layer.param_dict:
                    self.param_dict[top + '_frac'] = sim_quant_layer.param_dict[top + '_frac']

    def _check_parm_dict(self, simquantlayer):
        if simquantlayer.ndklayer.name in self._quant_layer_names:
            if simquantlayer.ndklayer.type.lower() in ['convolution', 'innerproduct','batchnorm','scale']:
                if simquantlayer.ndklayer.name + "_frac_weight" not in simquantlayer.param_dict:
                    raise ValueError("Please give the weight frac of layer {}".format(simquantlayer.ndklayer.name))
                if simquantlayer.ndklayer.name + "_frac_bias" not in simquantlayer.param_dict:
                    if simquantlayer.ndklayer.bias_term:
                        raise ValueError("Please give the bias frac of layer {}".format(simquantlayer.ndklayer.name))
            elif simquantlayer.ndklayer.type.lower() in ['bias']:
                if simquantlayer.ndklayer.name + "_frac_bias" not in simquantlayer.param_dict:
                    raise ValueError("Please give the bias frac of layer {}".format(simquantlayer.ndklayer.name))
            else:
                pass

            if simquantlayer.ndklayer.type.lower() == 'slice':
                for top in simquantlayer.ndklayer.top:
                    if top + "_frac" not in simquantlayer.param_dict:
                        raise ValueError("Please give the frac of tensor {}.".format(top))
            else:
                if simquantlayer.ndklayer.top + '_frac'  not in simquantlayer.param_dict:
                    raise ValueError("Please give the frac of tensor {}.".format(simquantlayer.ndklayer.top))
        else:
            if simquantlayer.param_dict is not None:
                if simquantlayer.ndklayer.type.lower() in ['convolution', 'innerproduct','batchnorm','scale']:
                    if simquantlayer.ndklayer.name + "_frac_weight" in simquantlayer.param_dict:
                        raise ValueError("Please don't give the weight frac of layer {},"
                                         "which will not be quantized according to quant_layer_names.".format(simquantlayer.ndklayer.name))
                    if simquantlayer.ndklayer.name + "_frac_bias" in simquantlayer.param_dict:
                        raise ValueError("Please don't give the bias frac of layer {}, "
                                         "which will not be quantized according to quant_layer_names.".format(simquantlayer.ndklayer.name))
                elif simquantlayer.ndklayer.type.lower() in ['bias']:
                    if simquantlayer.ndklayer.name + "_frac_bias" in simquantlayer.param_dict:
                        raise ValueError("Please don't give the bias frac of layer {},"
                                         "which will not be quantized according to quant_layer_names.".format(simquantlayer.ndklayer.name))
                else:
                    pass

                if simquantlayer.ndklayer.type.lower() == 'slice':
                    for top in simquantlayer.ndklayer.top:
                        if top + "_frac" in simquantlayer.param_dict:
                            raise ValueError("Please don't give the frac of tensor {}, "
                                             "which will not be quantized according to quant_layer_names.".format(top))
                else:
                    if simquantlayer.ndklayer.top + '_frac'  in simquantlayer.param_dict:
                        raise ValueError("Please don't give the frac of tensor {}, "
                                         "which will not be quantized according to quant_layer_names.".format(simquantlayer.ndklayer.top))

    def _add_feature_map_tensors_a_layer(self, simquant_layer):
        if simquant_layer.ndklayer.type.lower() in ['concat', 'eltwise']:
            self.dict_of_feature_map_tensor[simquant_layer.ndklayer.top] = simquant_layer.output[0]
        elif simquant_layer.ndklayer.type.lower() in ['slice']:
            for idx, top in enumerate(simquant_layer.ndklayer.top):
                self.dict_of_feature_map_tensor[top] = simquant_layer.output[0][idx]
        else:
            self.dict_of_feature_map_tensor[simquant_layer.ndklayer.top] = simquant_layer.output[0]

    def _add_weight_bias_variable_a_layer(self, simquant_layer):
        if hasattr(simquant_layer, 'kernel'):
            self.dict_of_weight_bias_variable[simquant_layer.ndklayer.name + '_weight'] = simquant_layer.kernel
        if hasattr(simquant_layer, 'bias'):
            self.dict_of_weight_bias_variable[simquant_layer.ndklayer.name + '_bias'] = simquant_layer.bias
        else:
            pass

    def _get_ly_by_name(self, name):
        for layer in self.layer_list:
            if layer.name == name:
                return layer

    def _adj_param_dict_weight_bias_dimension_order(self, key_name, param_dict_weight):
        if 'weight' in key_name:
            layer_name = key_name[:-7]
            layer = self._get_ly_by_name(layer_name)
            if layer.type.lower() in ['batchnorm', 'scale']:
                return param_dict_weight
            elif layer.type.lower() in ['convolution']:
                return np.transpose(param_dict_weight, axes=(2, 3, 1, 0))
            elif layer.type.lower() in ['innerproduct']:
                if len(param_dict_weight.shape) > 2:
                    cout = param_dict_weight.shape[0]
                    return np.reshape(np.transpose(param_dict_weight, axes=(1, 2, 3,  0)), newshape=(-1, cout)) #NCHW
                else:
                    return np.transpose(param_dict_weight, axes=(1, 0))
        else:
            return param_dict_weight

    def _get_layer_name_by_weight_bias_key(self, weight_bias_key):
        if 'weight' in weight_bias_key:
            layer_name = weight_bias_key[:-7]
        elif 'bias' in weight_bias_key:
            layer_name = weight_bias_key[:-5]
        else:
            raise ValueError("You can only supply the weight_bias_key to get the layer_name")
        return layer_name

    def _get_restore_weight_bias(self, weight_bias_key,  param_dict, from_quant):
        layer_name = self._get_layer_name_by_weight_bias_key(weight_bias_key)
        if from_quant and layer_name in self._quant_layer_names:
            if 'weight' in weight_bias_key:
                if layer_name + '_quant_weight' in param_dict:
                    restore_np_weight_bias = quant2float(param_dict[layer_name + '_quant_weight'],
                                                      self.bitwidth,
                                                      self.param_dict[layer_name+'_frac_weight'],
                                                              floor=False)
                else:
                    raise ValueError("Can not find {}_quant_weight in param_dict to restore from quant weight.".format(layer_name))
            elif 'bias' in weight_bias_key:
                if layer_name + '_quant_bias' in param_dict:
                    restore_np_weight_bias = quant2float(param_dict[layer_name + '_quant_bias'],
                                                      self.bitwidth,
                                                      self.param_dict[layer_name+'_frac_bias'],
                                                      floor=False)
                else:
                    raise ValueError("Can not find {}_quant_bias in param_dict to restore from quant bias.".format(layer_name))
            else:
                raise ValueError("You can only supply the weight_bias_key to get the layer_name")
        else:
            if weight_bias_key in param_dict:
                restore_np_weight_bias = param_dict[weight_bias_key]
            else:
                raise ValueError("Can not find {} in param_dict to restore from float weight and bias.".format(weight_bias_key))

        return restore_np_weight_bias

    def _construct_weight_bias_restoring_operations(self, from_quant):
        assert self.param_dict != None, "Please supply the weights dict when restore!"

        self.restoring_operations = []
        for name, variable in self.dict_of_weight_bias_variable.items():
            if isinstance(variable, tf.Variable):
                assert name in self.param_dict, "can not restore from the model, {} is not supplied.".format(
                    name)
                for idx, np_shape in enumerate(self._adj_param_dict_weight_bias_dimension_order(name, self.param_dict[name]).shape):
                    assert variable.shape[
                               idx].value == np_shape, "can not restore from the model, shape of {} is not comform.".format(
                        name)
                restore_np_weight_bias = self._get_restore_weight_bias(name, self.param_dict, from_quant=from_quant)
                restoring_operation = variable.assign(self._adj_param_dict_weight_bias_dimension_order(
                    name, restore_np_weight_bias))
                self.restoring_operations.append(restoring_operation)
            elif isinstance(variable, list):
                for idx, g_variable in enumerate(variable):  # The group convolution.
                    if isinstance(g_variable, tf.Variable):
                        if len(g_variable.shape) == 4:  # kernel
                            gcout = g_variable.shape[3].value  # number of channel of group convolution output.
                        elif len(g_variable.shape) == 1:  # bias
                            gcout = g_variable.shape[0].value  # number of channel of group convolution output.
                        group = len(variable)
                        assert gcout * group == self.param_dict[name].shape[0], "unmatched shape."
                        restore_np_weight_bias = self._get_restore_weight_bias(name, self.param_dict, from_quant=from_quant)
                        restoring_operation = g_variable.assign(
                            self._adj_param_dict_weight_bias_dimension_order(
                                name,
                                restore_np_weight_bias[idx * gcout:(idx + 1) * gcout]
                            ))
                        self.restoring_operations.append(restoring_operation)

    def _adj_dimension_order_of_weight_bias_tensor(self, name, tensor):
        if 'weight' in name:
            if 'sim_quant' in name:
                layer_name = name[:-17]
            else:
                layer_name = name[:-7]
            layer = self._get_ly_by_name(layer_name)
            if layer.type.lower() in ['batchnorm', 'scale']:
                return tensor
            if layer.type.lower() in ['convolution']:
                return tf.transpose(tensor, perm=[3, 2, 0, 1])  # origin kernel variable h,w,cin,cout
            elif layer.type.lower() in ['innerproduct']:
                if len(self.dict_of_feature_map_tensor[layer.bottom].shape) > 2:  # the bottom come from a convolution layer.
                    height = self.dict_of_feature_map_tensor[layer.bottom].shape[2].value
                    width = self.dict_of_feature_map_tensor[layer.bottom].shape[3].value
                    cin = self.dict_of_feature_map_tensor[layer.bottom].shape[1].value #NCHW
                    return tf.transpose(tf.reshape(tensor, shape=[cin, height, width, -1]), perm=[3, 0, 1, 2])
                else:
                    return tf.reshape(tf.transpose(tensor, perm=[1, 0]), \
                                      shape=[tensor.shape[1].value, tensor.shape[0].value, 1,
                                             1])  # Transpose and reshape to 4 dimension
        else:  # bias
            return tensor

    def _generate_weight_bias_tensors_whose_dimension_in_param_dict_order(self):
        self._weight_bias_tensors_whose_dimension_in_param_dict_order = {}
        for name, variable in self.dict_of_weight_bias_variable.items():
            if isinstance(variable, list):
                if 'weight' in name:
                    concated_variable = tf.concat(variable, axis=3)
                elif 'bias' in name:
                    concated_variable = tf.concat(variable, axis=0)
                weight_bias_tensor_whose_dimension_in_param_dict_order = \
                    self._adj_dimension_order_of_weight_bias_tensor(name, concated_variable)
                self._weight_bias_tensors_whose_dimension_in_param_dict_order[name] = \
                    weight_bias_tensor_whose_dimension_in_param_dict_order
            else:
                weight_bias_tensor_whose_dimension_in_param_dict_order = \
                    self._adj_dimension_order_of_weight_bias_tensor(name, variable)
                self._weight_bias_tensors_whose_dimension_in_param_dict_order[name] = \
                    weight_bias_tensor_whose_dimension_in_param_dict_order

    def _quantize_per_channel(self, value, fracs):

        quant_value = np.ones_like(value, dtype=np.int)
        for idx, frac in enumerate(list(fracs)):
            quant_value[idx] = self._quantize(value[idx], self.bitwidth, -frac)

        return quant_value

    def _quantize(self, value, n, d, floor=False):
        return float2quant(value, self.bitwidth, int(-d), floor=floor)

    def _get_quant_weight_bias(self, weight_bias_dict, param_dict):

        quant_weights = {}

        for key, value in weight_bias_dict.items():
            if 'weight' in key and 'frac' not in key:
                layer_name = key[:-7]
                if layer_name+"_frac_weight" in param_dict:
                    frac = param_dict[layer_name+"_frac_weight"]

                    if len(frac) == 1:
                        quant_value = self._quantize(value, self.bitwidth, -frac[0])
                    else:
                        quant_value = self._quantize_per_channel(value, fracs=frac)

                    quant_weights[layer_name + '_quant_weight'] = quant_value
            elif 'bias' in key and 'frac' not in key:
                layer_name = key[:-5]
                if layer_name + '_frac_bias' in param_dict:
                    frac = param_dict[layer_name + '_frac_bias']
                    if len(frac) == 1:
                        quant_value = self._quantize(value, self.bitwidth, -frac[0])
                    else:
                        quant_value = self._quantize_per_channel(value, fracs=frac)

                    quant_weights[layer_name + '_quant_bias'] = quant_value
            else:
                continue

        return quant_weights

    def _convert_fracs_to_param_dict_format(self, frac_dict):
        new_frac_dict = {}
        for key, fracs in frac_dict.items():
            if 'frac' in key:
                if isinstance(fracs, int):
                    new_frac_dict[key] = fracs
                elif isinstance(fracs, list):
                    if len(fracs) == 1:
                        new_frac_dict[key] = int(fracs[0])
                    else:
                        new_frac_dict[key] = np.array(fracs)
                elif isinstance(fracs, np.ndarray):
                    if len(fracs.shape) == 0:
                        new_frac_dict[key] = int(fracs)
                    else:
                        pass
                else:
                    pass
        return new_frac_dict

    def _is_set_of_str(self, str_set):
        """Check if the input is a set consist of string components.
        """
        if type(str_set) != set:
            return False

        if len(str_set) == 0:
            return True
        else:
            for value in str_set:
                if type(value) != str:
                    return False

        return True

    def _is_quant_layer_names_in_layer_names(self, quant_layer_names):

        assert len(self.order_lys) != 0, "The layer_list should not be empty."
        layer_names = [layer.name for layer in self.order_lys]

        for layer_name in quant_layer_names:
            assert layer_name in layer_names, "layer {} of quant_layer_names not in layer_list.".format(layer_name)

    def _is_trainable_layer_names_in_layer_names(self, trainable_layers):

        assert len(self.order_lys) != 0, "The layer_list should not be empty."
        layer_names = [layer.name for layer in self.order_lys]

        for layer_name in trainable_layers:
            assert layer_name in layer_names, "layer {} of trainable_layers not in layer_list.".format(layer_name)

    def _check_not_surport_layer_type(self, layer_list):
        for layer in layer_list:
            if layer.type.lower() not in ['input', "sigmoid", "relu", "relu6", "tanh", "innerproduct", "convolution",
                                          "pooling", "batchnorm", "bias", "scale", "slice", "concat", "eltwise",
                                          "softmax", "shufflechannel", "scalebytensor"]:
                raise TypeError('layer type {} is not support in quantization training.'.format(layer.type))

    def _construct_order_layers(self):  # construct order
        self.bottoms = []
        for layer in self.layer_list:
            if layer.type.lower() in ['input']:
                continue
            if type(layer.bottom) == str:
                self.bottoms.append(layer.bottom)
                assert layer.bottom not in ['kernel',
                                            'bias'], "The tensor name should not be kernel or bias, which is reserved internal."
            else:
                for bottom in layer.bottom:
                    self.bottoms.append(bottom)
                    assert layer.bottom not in ['kernel',
                                                'bias'], "The tensor name should not be kernel or bias, which is reserved internal."

        self.tops = []
        for layer in self.layer_list:
            if type(layer.top) == str:
                self.tops.append(layer.top)
                assert layer.top not in ['kernel',
                                         'bias'], "The tensor name should not be kernel or bias, which is reserved internal."
            else:
                for top in layer.top:
                    self.tops.append(top)
                    assert layer.top not in ['kernel',
                                             'bias'], "The tensor name should not be kernel or bias, which is reserved internal."

        # inputs
        self.input_layers = []
        for layer in self.layer_list:
            if layer.type.lower() in ['input']:
                self.input_layers.append(layer)
                continue
            if type(layer.bottom) == str:
                if layer.bottom not in self.tops:
                    self.input_layers.append(layer)
                    print('Warning:The {} will be the input tensor.'.format(layer.bottom))
            else:
                for bottom in layer.bottom:
                    if bottom not in self.tops:
                        self.input_layers.append(layer)
                        print('Warning:The {} will be the input tensor.'.format(bottom))

        assert len(
            self.input_layers) == 1, "Can only support one network input, please recheck the input of network or the layer_list."

        # output_layers
        self.output_layers = []
        for layer in self.layer_list:
            if type(layer.top) == str:
                if layer.top not in self.bottoms:
                    self.output_layers.append(layer)
            else:
                for top in layer.top:
                    if top not in self.bottoms:
                        self.output_layers.append(layer)
                        # break
        # assert len(
        #     self.output_layers) == 1, "Can only support one network output, please recheck the output of network or the layer_list."

        lys_will_be_sim_constructed = []
        lys_sim_constructed = []

        # sim_construct the input layer:
        blobs_sim_constructed = [layer.top for layer in self.input_layers]
        for layer in self.layer_list:
            if layer in self.input_layers:
                lys_sim_constructed.append(layer)
                continue
            lys_will_be_sim_constructed.append(layer)

        def sim_construct_lys_one_loop(lys_will_be_sim_constructed, lys_sim_constructed, blobs_sim_constructed):
            '''
          lys_will_be_sim_constructed:list
          blobs_sim_constructed:list
          '''
            sim_constructed_lys_addition = []
            for layer in lys_will_be_sim_constructed:
                if type(layer.bottom) == str:
                    if layer.bottom in blobs_sim_constructed:
                        lys_sim_constructed.append(layer)
                        if type(layer.top) == str:
                            blobs_sim_constructed.append(layer.top)
                        else:
                            for top in layer.top:
                                blobs_sim_constructed.append(top)
                        sim_constructed_lys_addition.append(layer)
                else:
                    is_b_in_bsc = True  # if bottom in bottoms constucted.
                    for bottom in layer.bottom:
                        is_b_in_bsc = is_b_in_bsc and (bottom in blobs_sim_constructed)
                    if is_b_in_bsc:
                        lys_sim_constructed.append(layer)
                        if type(layer.top) == str:
                            blobs_sim_constructed.append(layer.top)
                        else:
                            for top in layer.top:
                                blobs_sim_constructed.append(top)
                        sim_constructed_lys_addition.append(layer)

            return sim_constructed_lys_addition, lys_sim_constructed, blobs_sim_constructed

        while True:
            sim_constructed_lys_addition, lys_sim_constructed, blobs_sim_constructed = sim_construct_lys_one_loop(
                lys_will_be_sim_constructed, lys_sim_constructed, blobs_sim_constructed)

            for layer in sim_constructed_lys_addition:
                lys_will_be_sim_constructed.remove(layer)

            if len(lys_will_be_sim_constructed) == 0:
                break

        return lys_sim_constructed

    def _is_layer_top_in_frac_dict(self, layer, frac_dict):
        if frac_dict is None:
            return False
        if isinstance(layer.top, str):
            if layer.top+'_frac' not in frac_dict:
                return False
        else:
            for top in layer.top:
                if top+'_frac' not in frac_dict:
                    return False
        return True

    def _extract_fracs_a_layer(self, layer, frac_dict):
        extracted_frac_dict = {}
        if layer.type.lower() in ['convolution','innerproduct','scale']:
            extracted_frac_dict[layer.name + '_frac_weight'] = frac_dict[layer.name + '_frac_weight']
            if layer.bias_term:
                extracted_frac_dict[layer.name + '_frac_bias'] = frac_dict[layer.name + '_frac_bias']
        elif layer.type.lower() in ['batchnorm']:
            extracted_frac_dict[layer.name + '_frac_weight'] = frac_dict[layer.name + '_frac_weight']
            extracted_frac_dict[layer.name + '_frac_bias'] = frac_dict[layer.name + '_frac_bias']
        elif layer.type.lower() in ['bias']:
            extracted_frac_dict[layer.name + '_frac_bias'] = frac_dict[layer.name + '_frac_bias']

        if isinstance(layer.top, str):
            extracted_frac_dict[layer.top+'_frac'] = frac_dict[layer.top+'_frac']
            if layer.top + '_signed' in frac_dict:
                extracted_frac_dict[layer.top + '_signed'] = frac_dict[layer.top + '_signed']
        else:
            for top in layer.top:
                extracted_frac_dict[top+'_frac'] = frac_dict[top+'_frac']
                if top + '_signed' in frac_dict:
                    extracted_frac_dict[top + '_signed'] = frac_dict[top + '_signed']
        return extracted_frac_dict

    def _construct_one_layer(self, layer, x):

        if layer.name in self._trainable_layer_names:
            _trainable = True
        else:
            _trainable = False

        if layer.name in self._perlayer_quant_layer_names:
            _perchannel = False
        else:
            _perchannel = True
        if layer.name not in self._quant_layer_names:
            _param_dict = None
        else:
            _param_dict = self.param_dict


        if layer.type.lower() in self.layertype_layerclass:
            if layer.type.lower() in ['convolution', 'innerproduct', 'batchnorm', 'scale']:
                return self.layertype_layerclass[layer.type.lower()](x, _param_dict, layer,
                                                                     bitwidth = self.bitwidth,
                                                                     trainable=_trainable,
                                                                     perchannel=_perchannel)
            elif layer.type.lower() in ['bias']:
                return  self.layertype_layerclass[layer.type.lower()](x, _param_dict, layer,
                                                                     bitwidth = self.bitwidth,
                                                                     perchannel=_perchannel)
            else:
                return self.layertype_layerclass[layer.type.lower()](x, _param_dict, layer,
                                                                     bitwidth = self.bitwidth)


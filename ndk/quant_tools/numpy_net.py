# !/usr/bin/env python3
# -*- coding:utf-8 -*- -
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
import time
import copy

import os
import sys
ndk_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ndk_dir)

from ndk.layers import Layer, sort_layers, get_dict_name_to_bottom, get_tensor_name_list, get_layer_name_list
import ndk.quant_tools.quant_layers as qlayers
import ndk.quant_tools.quant_func as qfunc
from ndk.utils import print_log

'''This function will create a numpy layer according to an input ndk.layers.Layer object and parameter dictionary.'''
def create_numpy_layer(layer, param_dict, quant=False, bitwidth=8):
    assert isinstance(layer, Layer), 'input layer must be an object of ndk.layers.Layer, but get a {}'.format(type(layer))
    np_layer = None
    if layer.type == layer.type_dict['sigmoid']:
        np_layer = qlayers.QuantSigmoid(name=layer.name)
        if quant:
            in_frac = param_dict[layer.bottom + '_frac']
            out_frac = param_dict[layer.top + '_frac']
            if layer.bottom + '_signed' in param_dict:
                input_signed = param_dict[layer.bottom + '_signed']
            else:
                input_signed = True
            np_layer.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)
    elif layer.type == layer.type_dict['relu']:
        np_layer = qlayers.QuantReLU(name=layer.name, negative_slope=layer.negative_slope)
        if quant:
            in_frac = param_dict[layer.bottom + '_frac']
            out_frac = param_dict[layer.top + '_frac']
            if layer.bottom + '_signed' in param_dict:
                input_signed = param_dict[layer.bottom + '_signed']
            else:
                input_signed = True
            np_layer.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac,
                                   quant_negative_slope=layer.negative_slope, input_signed=input_signed)
    elif layer.type == layer.type_dict['relu6']:
        np_layer = qlayers.QuantReLU6(name=layer.name)
        if quant:
            in_frac = param_dict[layer.bottom + '_frac']
            out_frac = param_dict[layer.top + '_frac']
            if layer.bottom + '_signed' in param_dict:
                input_signed = param_dict[layer.bottom + '_signed']
            else:
                input_signed = True
            np_layer.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)
    elif layer.type == layer.type_dict['tanh']:
        np_layer = qlayers.QuantTanH(name=layer.name)
        if quant:
            in_frac = param_dict[layer.bottom + '_frac']
            out_frac = param_dict[layer.top + '_frac']
            if layer.bottom + '_signed' in param_dict:
                input_signed = param_dict[layer.bottom + '_signed']
            else:
                input_signed = True
            np_layer.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)
    elif layer.type == layer.type_dict['innerproduct']:
        bias_term = layer.bias_term
        float_weight = param_dict[layer.name + '_weight']
        if bias_term:
            float_bias = param_dict[layer.name + '_bias']
        else:
            float_bias = None
        _, c_in, h_in, w_in = float_weight.shape

        np_layer = qlayers.QuantInnerProduct(name=layer.name, c_in=c_in, c_out=layer.num_output, weight=float_weight,
                                           bias=float_bias,
                                           h_in=h_in, w_in=w_in, bias_term=bias_term)
        if quant:
            in_frac = param_dict[layer.bottom + '_frac']
            out_frac = param_dict[layer.top + '_frac']
            if layer.bottom + '_signed' in param_dict:
                input_signed = param_dict[layer.bottom + '_signed']
            else:
                input_signed = True
            quant_weight = qlayers.quant2float(q=param_dict[layer.name + '_quant_weight'], bitwidth=bitwidth,
                                               frac=param_dict[layer.name + '_frac_weight'])
            if bias_term:
                quant_bias = qlayers.quant2float(q=param_dict[layer.name + '_quant_bias'], bitwidth=bitwidth,
                                                 frac=param_dict[layer.name + '_frac_bias'])
            else:
                quant_bias = None
            np_layer.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, quant_weight=quant_weight,
                                   quant_bias=quant_bias, input_signed=input_signed)
    elif layer.type == layer.type_dict['convolution']:
        bias_term = layer.bias_term
        float_weight = param_dict[layer.name + '_weight']
        if bias_term:
            float_bias = param_dict[layer.name + '_bias']
        else:
            float_bias = None

        _, c_in_g, _, _ = float_weight.shape

        np_layer = qlayers.QuantConv2D(c_in=c_in_g*layer.group, c_out=layer.num_output, kernel_size_h=layer.kernel_size[0],
                                     kernel_size_w=layer.kernel_size[1],
                                     weight=float_weight, bias=float_bias,
                                     stride_h=layer.stride[0], stride_w=layer.stride[1], dilation_h=layer.dilation[0],
                                     dilation_w=layer.dilation[1], group=layer.group,
                                     pad_n=layer.pad[0], pad_s=layer.pad[1], pad_w=layer.pad[2], pad_e=layer.pad[3],
                                     bias_term=bias_term, name=layer.name)

        if quant:
            in_frac = param_dict[layer.bottom + '_frac']
            out_frac = param_dict[layer.top + '_frac']
            if layer.bottom + '_signed' in param_dict:
                input_signed = param_dict[layer.bottom + '_signed']
            else:
                input_signed = True
            quant_weight = qlayers.quant2float(q=param_dict[layer.name + '_quant_weight'], bitwidth=bitwidth,
                                               frac=param_dict[layer.name + '_frac_weight'])
            if bias_term:
                quant_bias = qlayers.quant2float(q=param_dict[layer.name + '_quant_bias'], bitwidth=bitwidth,
                                                 frac=param_dict[layer.name + '_frac_bias'])
            else:
                quant_bias = None
            np_layer.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, quant_weight=quant_weight,
                                   quant_bias=quant_bias, input_signed=input_signed)
    elif layer.type == layer.type_dict['pool']:
        if layer.pool == 'max':
            np_layer = qlayers.QuantMaxPool2D(kernel_size_h=layer.kernel_size[0], kernel_size_w=layer.kernel_size[1],
                                            stride_h=layer.stride[0], stride_w=layer.stride[1],
                                            dilation_h=layer.dilation[0], dilation_w=layer.dilation[1],
                                            pad_n=layer.pad[0], pad_s=layer.pad[1], pad_w=layer.pad[2], pad_e=layer.pad[3],
                                            ceil_mode=False, name=layer.name)
        elif layer.pool == 'ave':
            np_layer = qlayers.QuantAvgPool2D(kernel_size_h=layer.kernel_size[0], kernel_size_w=layer.kernel_size[1],
                                            stride_h=layer.stride[0], stride_w=layer.stride[1],
                                            dilation_h=layer.dilation[0], dilation_w=layer.dilation[1],
                                            pad_n=layer.pad[0], pad_s=layer.pad[1], pad_w=layer.pad[2], pad_e=layer.pad[3],
                                            ceil_mode=False, name=layer.name)
        if quant:
            in_frac = param_dict[layer.bottom + '_frac']
            out_frac = param_dict[layer.top + '_frac']
            if (layer.bottom + '_signed') in param_dict:
                input_signed = param_dict[layer.bottom + '_signed']
            else:
                input_signed = True
            np_layer.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)
    elif layer.type == layer.type_dict['batchnorm']:
        float_weight = param_dict[layer.name + '_weight']
        float_bias = param_dict[layer.name + '_bias']
        num_channel = len(float_weight)
        np_layer = qlayers.QuantBatchNorm(c=num_channel, weight=float_weight, bias=float_bias, name=layer.name)
        if quant:
            in_frac = param_dict[layer.bottom + '_frac']
            out_frac = param_dict[layer.top + '_frac']
            if layer.bottom + '_signed' in param_dict:
                input_signed = param_dict[layer.bottom + '_signed']
            else:
                input_signed = True
            quant_weight = qlayers.quant2float(q=param_dict[layer.name + '_quant_weight'], bitwidth=bitwidth,
                                               frac=param_dict[layer.name + '_frac_weight'])
            quant_bias = qlayers.quant2float(q=param_dict[layer.name + '_quant_bias'], bitwidth=bitwidth,
                                             frac=param_dict[layer.name + '_frac_bias'])
            np_layer.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, quant_weight=quant_weight,
                                   quant_bias=quant_bias, input_signed=input_signed)
    elif layer.type == layer.type_dict['bias']:
        float_bias = param_dict[layer.name + '_bias']
        num_channel = len(float_bias)
        np_layer = qlayers.QuantBias(c=num_channel, bias=float_bias, name=layer.name)
        if quant:
            in_frac = param_dict[layer.bottom + '_frac']
            out_frac = param_dict[layer.top + '_frac']
            if layer.bottom + '_signed' in param_dict:
                input_signed = param_dict[layer.bottom + '_signed']
            else:
                input_signed = True
            quant_bias = qlayers.quant2float(q=param_dict[layer.name + '_quant_bias'], bitwidth=bitwidth,
                                             frac=param_dict[layer.name + '_frac_bias'])
            np_layer.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, quant_bias=quant_bias,
                                   input_signed=input_signed)
    elif layer.type == layer.type_dict['scale']:
        float_weight = param_dict[layer.name + '_weight']
        if layer.bias_term:
            float_bias = param_dict[layer.name + '_bias']
        else:
            float_bias = None
        num_channel = len(float_weight)
        np_layer = qlayers.QuantScale(c=num_channel, weight=float_weight, bias=float_bias, bias_term=layer.bias_term,
                                    name=layer.name)
        if quant:
            in_frac = param_dict[layer.bottom + '_frac']
            out_frac = param_dict[layer.top + '_frac']
            if layer.bottom + '_signed' in param_dict:
                input_signed = param_dict[layer.bottom + '_signed']
            else:
                input_signed = True
            quant_weight = qlayers.quant2float(q=param_dict[layer.name + '_quant_weight'], bitwidth=bitwidth,
                                               frac=param_dict[layer.name + '_frac_weight'])
            if layer.bias_term:
                quant_bias = qlayers.quant2float(q=param_dict[layer.name + '_quant_bias'], bitwidth=bitwidth,
                                                 frac=param_dict[layer.name + '_frac_bias'])
            else:
                quant_bias = None
            np_layer.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, quant_weight=quant_weight,
                                   quant_bias=quant_bias, input_signed=input_signed)
    elif layer.type == layer.type_dict['scalebytensor']:
        np_layer = qlayers.QuantScaleByTensor(name=layer.name)
        if quant:
            in_frac = [0, 0]
            input_signed = [True, True]
            in_frac[0] = param_dict[layer.bottom[0] + '_frac']
            in_frac[1] = param_dict[layer.bottom[1] + '_frac']
            out_frac = param_dict[layer.top + '_frac']
            if layer.bottom[0] + '_signed' in param_dict:
                input_signed[0] = param_dict[layer.bottom[0] + '_signed']
            if layer.bottom[1] + '_signed' in param_dict:
                input_signed[1] = param_dict[layer.bottom[1] + '_signed']
            np_layer.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)
    elif layer.type == layer.type_dict['slice']:
        np_layer = qlayers.QuantSlice(layer.slice_point, axis=layer.axis, name=layer.name)
        if quant:
            in_frac = param_dict[layer.bottom + '_frac']
            for tensor_name in layer.top:
                assert param_dict[layer.top[0] + '_frac'] == param_dict[tensor_name + '_frac'], 'inconsistent quantization schemes'
            out_frac = param_dict[layer.top[0] + '_frac']
            if layer.top[0] + '_signed' in param_dict:
                input_signed = param_dict[layer.top[0] + '_signed']
            else:
                input_signed = True
            np_layer.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)
    elif layer.type == layer.type_dict['concat']:
        np_layer = qlayers.QuantConcat(axis=layer.axis, name=layer.name)
        if quant:
            for tensor_name in layer.bottom:
                assert param_dict[layer.bottom[0] + '_frac'] == param_dict[
                    tensor_name + '_frac'], 'inconsistent quantization schemes'
            in_frac = param_dict[layer.bottom[0] + '_frac']
            out_frac = param_dict[layer.top + '_frac']
            if layer.bottom[0] + '_signed' in param_dict:
                input_signed = param_dict[layer.bottom[0] + '_signed']
            else:
                input_signed = True
            np_layer.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)
    elif layer.type == layer.type_dict['eltwise']:
        if layer.operation == 'sum':
            np_layer = qlayers.QuantEltwiseSum(name=layer.name)
            if quant:
                for tensor_name in layer.bottom:
                    assert param_dict[layer.bottom[0] + '_frac'] == param_dict[
                        tensor_name + '_frac'], 'inconsistent quantization schemes'
                in_frac = param_dict[layer.bottom[0] + '_frac']
                out_frac = param_dict[layer.top + '_frac']
                if layer.bottom[0] + '_signed' in param_dict:
                    input_signed = param_dict[layer.bottom[0] + '_signed']
                else:
                    input_signed = True
                np_layer.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)
        else:
            raise Exception('not supported.')
    elif layer.type == layer.type_dict['logsoftmax']:
        np_layer = qlayers.QuantLogSoftmax(name=layer.name)
        if quant:
            in_frac = param_dict[layer.bottom + '_frac']
            out_frac = param_dict[layer.top + '_frac']
            if layer.bottom + '_signed' in param_dict:
                input_signed = param_dict[layer.bottom + '_signed']
            else:
                input_signed = True
            np_layer.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)
    elif layer.type == layer.type_dict['shufflechannel']:
        np_layer = qlayers.QuantShuffleChannel(name=layer.name, group=layer.group)
        if quant:
            in_frac = param_dict[layer.bottom + '_frac']
            out_frac = param_dict[layer.top + '_frac']
            if layer.bottom + '_signed' in param_dict:
                input_signed = param_dict[layer.bottom + '_signed']
            else:
                input_signed = True
            np_layer.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=input_signed)
    elif layer.type == layer.type_dict['input']:
        np_layer = qlayers.QuantInput(name=layer.name)
        if quant:
            out_frac = param_dict[layer.top + '_frac']
            if layer.top + '_signed' in param_dict:
                input_signed = param_dict[layer.top + '_signed']
            else:
                input_signed = True
            np_layer.set_quant_param(bitwidth=bitwidth, in_frac=None, out_frac=out_frac, input_signed=input_signed)
    else:
        raise Exception('Unknown layer type: {}'.format(layer.type))

    return np_layer


'''This function will build a list of numpy layers according to the Layer list and parameter dictionary.'''
def build_numpy_layers(layer_list, param_dict, quant=False, bitwidth=8):
    numpy_layers = []
    for lyr in layer_list:
        numpy_layers.append(create_numpy_layer(layer=lyr, param_dict=param_dict, quant=quant, bitwidth=bitwidth))
    return numpy_layers


'''
Get a target weight or bias from param_dict, target_tensor_name must end with "_weight" or "_bias";
When getting a quantized version, bitwidth must be determined.
'''
def get_layer_weight_bias(target_tensor_name, param_dict, quant=False, bitwidth=8):
    assert isinstance(target_tensor_name, str), 'target_tensor_name must be a string, but got a {}.'.format(type(target_tensor_name))
    assert target_tensor_name.endswith('_weight') or target_tensor_name.endswith('_bias'), 'target tensor name ({}) must end with \'_weight\' or \'_bias\''.format(target_tensor_name)
    if target_tensor_name.endswith('_weight'):
        layer_name = target_tensor_name[:-7]
        suffix = '_weight'
    else:
        layer_name = target_tensor_name[:-5]
        suffix = '_bias'
    if quant:
        assert type(bitwidth)!=type(None), 'bitwidth must be determined when trying to get a quantized weight/bias.'
        assert layer_name+'_quant'+suffix in param_dict, 'layer named {} has no quantized {} in the param_dict'.format(layer_name, suffix[1:])
        assert layer_name+'_frac'+suffix in param_dict, 'layer named {} has no {} fraction in the param_dict'.format(layer_name, suffix[1:])
        data = qfunc.quant2float(param_dict[layer_name+'_quant'+suffix], bitwidth=bitwidth, frac=param_dict[layer_name+'_frac'+suffix])
    else:
        assert layer_name+suffix in param_dict, 'layer named {} has no quant{} in the param_dict'.format(layer_name, suffix[1:])
        data = param_dict[layer_name+suffix]
    return data


'''This function runs a data batch through the layer_list and return a dict which includes the designated feature tensors.
   The data will go through the ndk.quant_tools.quant_layers.QuantLayer objects in numpy_layers.
   If numpy_layers is not set, i.e., numpy_layers==None, it will build a list of numpy layers first.
   Note that, the elements in layer_list must correspond to those in numpy_layers sequentially.
'''
def run_layers(input_data_batch, layer_list, target_feature_tensor_list, param_dict=None,
               quant=False, bitwidth=8, hw_aligned=False, numpy_layers=None, log_on=False, use_ai_framework=True,
               quant_weight_only=False, quant_feature_only=False):

    assert isinstance(quant, bool), 'quant should be a Bool number, True or False'
    feature_tensor_name_list = get_tensor_name_list(layer_list, feature_only=True)
    for tensor_name in target_feature_tensor_list:
        assert tensor_name in feature_tensor_name_list, 'Feature {} not found.'.format(tensor_name)

    if type(numpy_layers)==type(None):
        # layer_list = copy.deepcopy(layer_list)
        layer_list = sort_layers(layer_list)
        numpy_layers = build_numpy_layers(layer_list=layer_list, param_dict=param_dict, quant=quant, bitwidth=bitwidth)
    else:
        sorted_layer_list = sort_layers(layer_list)
        for lyr_idx in range(len(layer_list)):
            assert layer_list[lyr_idx].name == sorted_layer_list[lyr_idx].name, 'When running using the pre-built numpy_layers (i.e., numpy_layers!=None), layer_list must be sorted using ndk.layers.sort_layers() first'

    error_msg = 'elements in layer list must correspond to those in numpy_layers sequentially'
    assert len(layer_list)==len(numpy_layers), error_msg
    for lyr_idx in range(len(layer_list)):
        assert layer_list[lyr_idx].name == numpy_layers[lyr_idx].name, error_msg+' @ layer[{}] = [{}]-->[{}]'.format(lyr_idx, layer_list[lyr_idx].name, numpy_layers[lyr_idx].name)

    assert layer_list[0].type==layer_list[0].type_dict['input'], 'layer_list must contains an Input layer.'
    assert input_data_batch.shape[1]==layer_list[0].dim[1], 'input channel number:{} is not equal to the that of Input layer:{}'.format(input_data_batch.shape[1], layer_list[0].dim[1])
    assert input_data_batch.shape[2]==layer_list[0].dim[2], 'input height:{} is not equal to the that of Input layer:{}'.format(input_data_batch.shape[2], layer_list[0].dim[2])
    assert input_data_batch.shape[3]==layer_list[0].dim[3], 'input width:{} is not equal to the that of Input layer:{}'.format(input_data_batch.shape[3], layer_list[0].dim[3])

    # analysis how long one tensor result should be kept
    tensor_life_dict = {}
    for i,lyr in enumerate(layer_list):
        if hasattr(lyr, 'bottom'):
            if type(lyr.bottom)==list:
                for tensor_name in lyr.bottom:
                    tensor_life_dict[tensor_name] = i
            else:
                tensor_life_dict[lyr.bottom] = i
        if hasattr(lyr, 'top'):
            if type(lyr.top)==list:
                for tensor_name in lyr.top:
                    tensor_life_dict[tensor_name] = i
            else:
                tensor_life_dict[lyr.top] = i

    tic = time.clock()
    data_dict = {}
    for i,lyr in enumerate(layer_list):
        if log_on:
            print_log('run_layers: running layer {}/{} with name: {}'.format(i+1, len(layer_list), lyr.name))
            print_log('run_layers: '+str(lyr))
        if lyr.type==lyr.type_dict['input']:
            assert isinstance(input_data_batch, np.ndarray), 'input data must be 4-dim numpy.ndarray, but got type {}'.format(type(input_data_batch))
            assert len(input_data_batch.shape)==4, 'input data must be 4-dim numpy.ndarray, but got shape {}'.format(input_data_batch.shape)
            _, c_in, h_in, w_in = input_data_batch.shape
            assert lyr.dim[1]==c_in, 'input data has {} channels while input layer requires {}'.format(c_in, lyr.dim[1])
            assert lyr.dim[2]==h_in, 'input data is of height {} while input layer requires H={}'.format(h_in, lyr.dim[2])
            assert lyr.dim[3]==w_in, 'input data is of width {} while input layer requires W={}'.format(w_in, lyr.dim[3])
            data_out = numpy_layers[i].run(input_data_batch, quant=quant, hw_aligned=hw_aligned, use_ai_framework=use_ai_framework,
                                           quant_weight_only=quant_weight_only, quant_feature_only=quant_feature_only)
            data_dict[lyr.top] = data_out.copy()
        else:
            if type(lyr.bottom)==list:
                data_in = []
                for tensor_name in lyr.bottom:
                    data_in.append(data_dict[tensor_name])
            else:
                data_in = data_dict[lyr.bottom]

            data_out = numpy_layers[i].run(data_in, quant=quant, hw_aligned=hw_aligned, use_ai_framework=use_ai_framework,
                                           quant_weight_only=quant_weight_only, quant_feature_only=quant_feature_only)

            if type(lyr.top)==list:
                for tensor_idx, tensor_name in enumerate(lyr.top):
                    data_dict[tensor_name] = data_out[tensor_idx]
            else:
                data_dict[lyr.top] = data_out

        if log_on:
            print_log('run_layers: Elapsed time: {:.2f} seconds'.format(time.clock()-tic))

        # release the memory of useless results
        cnt_collected_targets = 0
        result_tensor_name_list = list(data_dict.keys())
        for tensor_name in  result_tensor_name_list:
            if tensor_name in target_feature_tensor_list:
                cnt_collected_targets += 1
            else:
                if i >= tensor_life_dict[tensor_name]:
                    data_dict.pop(tensor_name)

        # if all target results have been collected, exit the loop
        if cnt_collected_targets == len(target_feature_tensor_list):
            result_tensor_name_list = list(data_dict.keys())
            for tensor_name in result_tensor_name_list:
                if tensor_name not in target_feature_tensor_list:
                    data_dict.pop(tensor_name)
            break


    return data_dict


if __name__ == '__main__':
    pass



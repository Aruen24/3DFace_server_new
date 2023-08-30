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
import time

import os
import sys
ndk_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ndk_dir)
import ndk
import copy
from ndk.utils import print_log
import ndk.quant_tools.numpy_net as qnet
import ndk.quant_tools.quant_func as qfunc
import time
import shutil

ENABLE_FLOOR_FEATURE = ndk.quant_tools.quant_layers.ENABLE_FLOOR_FEATURE
ENABLE_FLOOR_WEIGHT = ndk.quant_tools.quant_layers.ENABLE_FLOOR_WEIGHT

_FRAC_UPPER_LIMIT = {8: 15, 16: 31}
_NUM_BIN_FOR_KL_JS = 89 # number of bins used to compute KL/JS divergence, it is better not to be 256 or 65536
_IGNORE_WEIGHT_THRESHOLD = {8: 2**10, 16: 2**20}
_IGNORE_BIAS_THRESHOLD = {8: 2**16, 16: 2**32}

_QUANTIZATION_LOG_PATH = None
_QUANTIZATION_LOG_FNAME = None

_MIN_POSITIVE_NUMBER = 2**-64
_MULTI_INPUT_ALLOW_DIFFERENT_FRAC = ['ScaleByTensor']

def set_quantization_log_path(path):
    global _QUANTIZATION_LOG_PATH
    global _QUANTIZATION_LOG_FNAME
    _QUANTIZATION_LOG_PATH = path
    if type(path)!=type(None):
        if os.path.exists(path):
            # print_log('quantize_net: log path:\'{}\' already exists, the original contents will be erased.'.format(path))
            # shutil.rmtree(path)
            # os.mkdir(path)
            pass
        else:
            os.mkdir(path)
        _QUANTIZATION_LOG_FNAME = 'quantize_net_' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log'
        file_path = os.path.join(_QUANTIZATION_LOG_PATH, _QUANTIZATION_LOG_FNAME)
        if os.path.exists(file_path):
            os.remove(file_path)

def get_quantization_log_path():
    return _QUANTIZATION_LOG_PATH

def add_log(msg):
    if type(_QUANTIZATION_LOG_PATH) != type(None):
        assert isinstance(msg, str), 'write_log_file: msg must be a string.'
        log_fname = os.path.join(_QUANTIZATION_LOG_PATH, _QUANTIZATION_LOG_FNAME)
        with open(log_fname, 'a') as f:
            f.write('{}: {}\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), msg))

'''collect float-point parameters into a new param_dict'''
def collect_float_point_param(layer_list, param_dict):
    fp_param_dict = {}
    for layer in layer_list:
        if layer.name + '_weight' in param_dict:
            fp_param_dict[layer.name + '_weight'] = param_dict[layer.name + '_weight']
        if layer.name + '_bias' in param_dict:
            fp_param_dict[layer.name + '_bias'] = param_dict[layer.name + '_bias']
    return fp_param_dict

'''auxilary functions'''
def get_in_frac(layer, param_dict):
    in_frac = None
    if hasattr(layer, 'bottom'):
        if isinstance(layer.bottom, list):
            for tsr_name in layer.bottom:
                if not tsr_name + '_frac' in param_dict:
                    in_frac = None
                    break
                else:
                    cur_in_frac = param_dict[tsr_name + '_frac']
                    if in_frac==None:
                        in_frac = cur_in_frac
                    elif layer.type not in _MULTI_INPUT_ALLOW_DIFFERENT_FRAC:
                        assert in_frac == cur_in_frac, 'Input fractions of layer:{} are not aligned.'.format(layer.name)
        else:
            if layer.bottom + '_frac' in param_dict:
                in_frac = param_dict[layer.bottom + '_frac']
    return in_frac
def get_out_frac(layer, param_dict):
    out_frac = None
    if isinstance(layer.top, list):
        for tsr_name in layer.top:
            if not tsr_name+'_frac' in param_dict:
                out_frac = None
                break
            else:
                out_frac = param_dict[tsr_name+'_frac']
    else:
        if layer.top+'_frac' in param_dict:
            out_frac = param_dict[layer.top+'_frac']
    return out_frac
def get_weight_frac(layer, param_dict):
    weight_frac = None
    if layer.name+'_frac_weight' in param_dict:
        weight_frac = param_dict[layer.name+'_frac_weight']
    return weight_frac
def get_bias_frac(layer, param_dict):
    bias_frac = None
    if layer.name+'_frac_bias' in param_dict:
        bias_frac = param_dict[layer.name+'_frac_bias']
    return bias_frac
def get_signed(layer, param_dict):
    signed = None
    if layer.type=='Input':
        if layer.top+'_signed' in param_dict:
            signed = param_dict[layer.top+'_signed']
        else:
            signed = True
    else:
        if isinstance(layer.bottom, list):
            for tsr_name in layer.bottom:
                if not tsr_name + '_signed' in param_dict:
                    cur_signed = True
                else:
                    cur_signed = param_dict[tsr_name+'_signed']
                if signed == None:
                    signed = cur_signed
                else:
                    assert signed == cur_signed, 'Input signs of layer:{} are not aligned.'.format(layer.name)
        else:
            if layer.bottom + '_signed' in param_dict:
                signed = param_dict[layer.bottom + '_signed']
            else:
                signed = True
    return signed
def get_quant_param(layer, param_dict):
    in_frac = get_in_frac(layer, param_dict)
    out_frac = get_out_frac(layer, param_dict)
    weight_frac = get_weight_frac(layer, param_dict)
    bias_frac = get_bias_frac(layer, param_dict)
    signed = get_signed(layer, param_dict)
    return in_frac, out_frac, weight_frac, bias_frac, signed
def quantize_prob_distribution(bin_vals, float_prob, bitwitdh, fraction, signed):
    quant_bin = qfunc.quantized_value(val=bin_vals, bitwidth=bitwitdh, frac=fraction, floor=ENABLE_FLOOR_FEATURE[bitwitdh], signed=signed)
    quant_prob = copy.deepcopy(float_prob)
    num_bin = len(bin_vals)
    start_idx = 0
    cur_val = quant_bin[0]
    sum_prob = quant_prob[0]
    for end_idx in range(1, num_bin):
        if quant_bin[end_idx] == cur_val:
            sum_prob += quant_prob[end_idx]
        else:
            quant_prob[start_idx:end_idx] = sum_prob/(end_idx-start_idx)
            start_idx = end_idx
            cur_val = quant_bin[start_idx]
            sum_prob = quant_prob[start_idx]
    quant_prob[start_idx:] = sum_prob / (num_bin - start_idx)
    return quant_prob

'''
Group the layer type into 3 classies according to fraction restriction of output tensor:
  'fix' - output tensor fraction is fixed;
  'same' - output tensor fraction must equal to that of input tensor;
  'free' - output tensor fraction can be adjusted according to situations;
'''
def classify_layer_type(layer):
    assert hasattr(layer, 'type'), 'found no type in layer'
    cls = 'unknown'
    if layer.type in ['Sigmoid', 'ReLU6', 'TanH', 'LogSoftmax']:
        cls = 'fix'
    elif layer.type in ['ReLU', 'Slice', 'Concat', 'ShuffleChannel']:
        cls = 'same'
    elif layer.type in ['Input', 'InnerProduct', 'Convolution', 'Scale', 'BatchNorm', 'Bias']:
        cls = 'free'
    elif layer.type == 'Pooling':
        if layer.pool.lower() == 'max':
            cls = 'same'
        elif layer.pool.lower() == 'ave':
            cls = 'free'
    elif layer.type == 'Eltwise':
        if layer.operation.lower() == 'sum':
            cls = 'free'
    elif layer.type == 'ScaleByTensor':
        cls = 'free'
    assert not cls=='unknown', 'cannot classify layer:{} with type: {}'.format(layer.name, layer.type)
    return cls

'''Sort the layer for quantization'''
def sort_layer_list_for_quantize_net(layer_list):
    layer_list = copy.deepcopy(layer_list)
    class2dist = {'fix': 0, 'same': 0, 'free':1}
    # sorted_layer_list = ndk.layers.sort_layers(layer_list)
    layer_list = ndk.layers.sort_layer_by_name(layer_list, reverse=False)
    adjacent_dict = ndk.layers.get_adjacent_layer_dict(layer_list, reverse=True)
    for key in adjacent_dict.keys():
        adjacent_dict[key] = ndk.layers.sort_layer_by_name(adjacent_dict[key], reverse=True)
    output_tensor_list = ndk.layers.get_network_output(layer_list)
    layer_dict_to_top = ndk.layers.get_dict_name_to_top(layer_list)
    output_layer_set = set()
    for tensor_name in output_tensor_list:
        output_layer_set.update(layer_dict_to_top[tensor_name])

    # weight the layers according to the distance to the output
    stack = []
    weighted_layer_list = []
    for lyr in output_layer_set:
        lyr.dist = class2dist[classify_layer_type(lyr)]
        stack.append(lyr)
    assert len(stack) == 1, 'the network must have but only have one output layer, but {} inputs are found'.format(len(stack))
    while len(stack) > 0:
        stack = sorted(stack, key=lambda lyr: lyr.dist, reverse=False)
        cur_lyr = stack.pop()
        adjacent_layers = list(adjacent_dict[cur_lyr.name])
        for prev_lyr in adjacent_layers:
            if not hasattr(prev_lyr, 'dist'): # not visited
                prev_lyr.dist = cur_lyr.dist + class2dist[classify_layer_type(prev_lyr)]
                stack.append(prev_lyr)
            else:
                new_prev_lyr_dist = min(prev_lyr.dist, cur_lyr.dist+class2dist[classify_layer_type(prev_lyr)])
                if prev_lyr.dist != new_prev_lyr_dist:
                    stack.append(prev_lyr)
        if not cur_lyr in weighted_layer_list:
            weighted_layer_list.append(cur_lyr)
    weighted_layer_list = ndk.layers.sort_layer_by_name(weighted_layer_list, reverse=True)
    weighted_layer_list = sorted(weighted_layer_list, key=lambda lyr: lyr.dist, reverse=True)

    # determine the quantization order of the layers
    stack = []
    sorted_layer_list = []
    visited = []
    adjacent_dict = ndk.layers.get_adjacent_layer_dict(layer_list, reverse=False)
    # layer_dict_to_bottom = ndk.layers.layer_dict_to_bottom(layer_list)
    for layer in weighted_layer_list:
        if layer.type == 'Input':
            stack.append(layer)
    while len(stack) > 0:
        current_layer = stack[-1]
        stack_out = True
        to_push_layer = None
        adjacent_layers = list(adjacent_dict[current_layer.name])
        adjacent_layers = sorted(adjacent_layers, key=lambda lyr: lyr.dist, reverse=True)
        for next_layer in adjacent_layers:
            if next_layer.name not in visited:
                visited.append(next_layer.name)
                to_push_layer = next_layer
                stack_out = False
                break
        if stack_out:
            sorted_layer_list.insert(0, current_layer)
            stack.pop()
        else:
            stack.append(to_push_layer)

    return sorted_layer_list

'''Propagete the fraction configurations in the net'''
def _propagate_fraction_config_core(layer_list, param_dict, bitwidth):
    assert bitwidth==8 or bitwidth==16, 'bitwidth must be 8 or 16, but get {}'.format(bitwidth)
    flag_updated = False
    for layer in layer_list:
        tsr_in_frac = None
        tsr_out_frac = None
        if classify_layer_type(layer)=='fix':
            if layer.type == 'Sigmoid':
                if bitwidth==8:
                    tsr_in_frac = 4
                    tsr_out_frac = 7
                else:
                    tsr_in_frac = 12
                    tsr_out_frac = 15
            elif layer.type == 'ReLU6':
                if bitwidth==8:
                    tsr_in_frac = 4
                    tsr_out_frac = 4
                else:
                    tsr_in_frac = 12
                    tsr_out_frac = 12
            elif layer.type == 'TanH':
                if bitwidth==8:
                    tsr_in_frac = 5
                    tsr_out_frac = 7
                else:
                    tsr_in_frac = 13
                    tsr_out_frac = 15
            elif layer.type == 'LogSoftmax':
                if bitwidth==8:
                    tsr_in_frac = 4
                    tsr_out_frac = 4
                else:
                    tsr_in_frac = 12
                    tsr_out_frac = 12
            else:
                raise ValueError('Unknown layer type:{} for class FIX.')
            if not layer.bottom+'_frac' in param_dict:
                param_dict[layer.bottom+'_frac'] = tsr_in_frac
                flag_updated = True
                add_log('Feature {}: frac={}, because it is input to {} layer:{}'.format(layer.bottom, tsr_in_frac, layer.type, layer.name))
            if not layer.top+'_frac' in param_dict:
                param_dict[layer.top+'_frac'] = tsr_out_frac
                flag_updated = True
                add_log('Feature {}: frac={}, because it is output of {} layer:{}'.format(layer.top, tsr_out_frac, layer.type, layer.name))
        elif classify_layer_type(layer)=='same':
            if isinstance(layer.bottom, list):
                for tsr_name in layer.bottom:
                    if tsr_name+'_frac' in param_dict:
                        if tsr_in_frac==None:
                            tsr_in_frac = param_dict[tsr_name+'_frac']
                        else:
                            assert tsr_in_frac==param_dict[tsr_name+'_frac'], 'Error: layer {} with type:{} found different input fractions.'.format(layer.name, layer.type)
            else:
                if layer.bottom+'_frac' in param_dict:
                    tsr_in_frac = param_dict[layer.bottom+'_frac']

            if isinstance(layer.top, list):
                for tsr_name in layer.top:
                    if tsr_name+'_frac' in param_dict:
                        if tsr_out_frac==None:
                            tsr_out_frac = param_dict[tsr_name+'_frac']
                        else:
                            assert tsr_out_frac==param_dict[tsr_name+'_frac'], 'Error: layer {} with type:{} found different output fractions.'.format(layer.name, layer.type)
            else:
                if layer.top+'_frac' in param_dict:
                    tsr_out_frac = param_dict[layer.top+'_frac']

            if tsr_in_frac!=None and tsr_out_frac!=None:
                assert tsr_in_frac==tsr_out_frac, 'Error: layer {} with type:{} found different input fraction:{} and output fraction:{}'.format(layer.name, layer.type, tsr_in_frac, tsr_out_frac)
            elif tsr_in_frac==None:
                tsr_in_frac = tsr_out_frac
            # elif tsr_out_frac==None:
            #     tsr_out_frac = tsr_in_frac

            if tsr_in_frac!=None:
                if isinstance(layer.bottom, list):
                    for tsr_name in layer.bottom:
                        if not tsr_name+'_frac' in param_dict:
                            param_dict[tsr_name+'_frac'] = tsr_in_frac
                            add_log('Feature {}: frac={}, because it is input to {} layer:{} and one of its input/output has frac={}'.format(tsr_name, tsr_in_frac, layer.type, layer.name, tsr_in_frac))
                            flag_updated = True
                else:
                    if not layer.bottom+'_frac' in param_dict:
                        param_dict[layer.bottom+'_frac'] = tsr_in_frac
                        add_log('Feature {}: frac={}, because it is input to {} layer:{} and one of its input/output has frac={}'.format(layer.bottom, tsr_in_frac, layer.type, layer.name, tsr_in_frac))
                        flag_updated = True

                if isinstance(layer.top, list):
                    for tsr_name in layer.top:
                        if not tsr_name+'_frac' in param_dict:
                            param_dict[tsr_name+'_frac'] = tsr_in_frac
                            add_log('Feature {}: frac={}, because it is output of {} layer:{} and one of its input/output has frac={}'.format(tsr_name, tsr_in_frac, layer.type, layer.name, tsr_in_frac))
                            flag_updated = True
                else:
                    if not layer.top+'_frac' in param_dict:
                        param_dict[layer.top+'_frac'] = tsr_in_frac
                        add_log('Feature {}: frac={}, because it is output of {} layer:{} and one of its input/output has frac={}'.format(layer.top, tsr_in_frac, layer.type, layer.name, tsr_in_frac))
                        flag_updated = True
        elif layer.type=='Eltwise' and layer.operation.lower() == 'sum':
            for tsr_name in layer.bottom:
                if tsr_name + '_frac' in param_dict:
                    if tsr_in_frac == None:
                        tsr_in_frac = param_dict[tsr_name + '_frac']
                    else:
                        assert tsr_in_frac == param_dict[tsr_name + '_frac'], 'Error: layer {} with type:{} found different input fractions.'.format(layer.name, layer.type)
            if tsr_in_frac != None:
                if isinstance(layer.bottom, list):
                    for tsr_name in layer.bottom:
                        if not tsr_name + '_frac' in param_dict:
                            param_dict[tsr_name + '_frac'] = tsr_in_frac
                            add_log('Feature {}: frac={}, because it is input of {} layer:{} and one of its inputs has frac={}'.format(tsr_name, tsr_in_frac, layer.type, layer.name, tsr_in_frac))
                            flag_updated = True

    return flag_updated

def propagate_fraction_configuration(layer_list, param_dict, bitwidth, max_iter=None):
    if max_iter == None:
        max_iter = len(layer_list)
    cnt_update = 0
    while _propagate_fraction_config_core(layer_list=layer_list, param_dict=param_dict, bitwidth=bitwidth):
        cnt_update += 1
        if cnt_update > max_iter:
            print_log('quantize_net: exceeds maximum iteration:{} when propagating fraction configurations.'.format(max_iter))
            break
    return param_dict

'''function determines weight fraction, mininize MSE'''
def determine_weight_fraction(layer, param_dict, bitwidth, max_num_candidate=32):
    out_frac = param_dict[layer.top+'_frac']
    in_frac = param_dict[layer.bottom+'_frac']
    float_weight = param_dict[layer.name+'_weight']
    num_channel = float_weight.shape[0]

    if layer.name+'_frac_weight' in param_dict:
        weight_frac = param_dict[layer.name+'_frac_weight']
        if layer.name+'_frac_bias' in param_dict:
            add_log('Layer {}: frac_weight={} is previously determined, with in_frac:{}, out_frac:{}, and frac_bias:{}'.format(layer.name, list(weight_frac), in_frac, out_frac, list(param_dict[layer.name+'_frac_bias'])))
        else:
            add_log('Layer {}: frac_weight={} is previously determined, with in_frac:{}, out_frac:{}'.format(layer.name, list(weight_frac), in_frac, out_frac))

        assert in_frac + np.min(np.array(weight_frac)) - out_frac >= 0, 'failed when quantizing the bias of layer:{} with in_frac={}, out_frac={} and detemined weight_frac={}'.format(layer.name, in_frac, out_frac, np.min(np.array(weight_frac)))
        assert in_frac + np.max(np.array(weight_frac)) - out_frac <= _FRAC_UPPER_LIMIT[bitwidth], 'failed when quantizing the bias of layer:{} with in_frac={}, out_frac={} and detemined weight_frac={}'.format(layer.name, in_frac, out_frac, np.max(np.array(weight_frac)))
        if layer.name + '_frac_bias' in param_dict:
            bias_frac = param_dict[layer.name+'_frac_bias']
            for c in range(num_channel):
                if isinstance(bias_frac, int):
                    bis_frac = bias_frac
                else:
                    bis_frac = bias_frac[c]
                if isinstance(weight_frac, int):
                    wgt_frac = weight_frac
                else:
                    wgt_frac = weight_frac[c]
                assert in_frac + wgt_frac - bis_frac >= 0, 'failed when quantizing the weight of layer:{} with in_frac={}, bias_frac={} and detemined weight_frac={}'.format(layer.name, in_frac, bis_frac, wgt_frac)
                assert in_frac + wgt_frac - bis_frac <= _FRAC_UPPER_LIMIT[bitwidth], 'failed when quantizing the weight of layer:{} with in_frac={}, bias_frac={} and detemined weight_frac={}'.format(layer.name, in_frac, bis_frac, wgt_frac)
    else:
        if layer.name+'_frac_bias' in param_dict:
            add_log('Layer {}: weight fraction determined according to in_frac:{}, out_frac:{}, and frac_bias:{}'.format(layer.name, in_frac, out_frac, list(param_dict[layer.name+'_frac_bias'])))
        else:
            add_log('Layer {}: weight fraction determined according to in_frac:{} and out_frac:{}'.format(layer.name, in_frac, out_frac))
        candidate_frac_list = []
        for offset in range(max_num_candidate):
            candidate_frac = _FRAC_UPPER_LIMIT[bitwidth] + out_frac - in_frac - offset
            if layer.type in ['Convolution', 'InnerProduct', 'Scale', 'BatchNorm']:
                if in_frac + candidate_frac - out_frac >= 0 and in_frac + candidate_frac - out_frac <= _FRAC_UPPER_LIMIT[bitwidth]:
                    candidate_frac_list.append(candidate_frac)
            else:
                pass

        assert len(candidate_frac_list)>0, 'failed when quantizing the weight of layer:{} with in_frac={} and out_frac={}'.format(layer.name, in_frac, out_frac)

        weight_frac = np.zeros(num_channel, dtype=int)
        for c in range(num_channel):
            mse_list = np.inf * np.ones(len(candidate_frac_list))
            for i, cur_frac in enumerate(candidate_frac_list):
                if layer.type in ['BatchNorm'] or (layer.type in ['Convolution', 'InnerProduct', 'Scale'] and layer.bias_term==True):
                    if layer.name+'_frac_bias' in param_dict:
                        if isinstance(param_dict[layer.name+'_frac_bias'], int):
                            bias_frac = param_dict[layer.name+'_frac_bias']
                        else:
                            bias_frac = param_dict[layer.name+'_frac_bias'][c]
                        if in_frac + cur_frac - bias_frac < 0 or in_frac + cur_frac - bias_frac > _FRAC_UPPER_LIMIT[bitwidth]:
                            continue
                mse_list[i] = np.mean(np.square(qfunc.quantized_value(float_weight[c], bitwidth=bitwidth, frac=cur_frac, floor=ENABLE_FLOOR_WEIGHT[bitwidth]) - float_weight[c]))
            if layer.name + '_frac_bias' in param_dict:
                assert np.min(mse_list) != np.inf, 'failed when quantizing the weight of layer:{} with in_frac={}, out_frac={}, bias_frac={}'.format(layer.name, in_frac, out_frac, param_dict[layer.name+'_frac_bias'])
            else:
                assert np.min(mse_list) != np.inf, 'failed when quantizing the weight of layer:{} with in_frac={} and out_frac={}'.format(layer.name, in_frac, out_frac)
            best_idx = np.where(mse_list==np.min(mse_list))[0][0]
            weight_frac[c] = candidate_frac_list[best_idx]
            add_log('Layer {}: weight ch#{}/{}: min={:.3e}, max={:.3e}, frac={}, mse={:.3e}'.format(layer.name, c, num_channel, np.min(float_weight[c]), np.max(float_weight[c]), weight_frac[c], np.min(mse_list)))
        add_log('Layer {}: frac_weight={}'.format(layer.name, list(weight_frac)))
    return weight_frac

'''function determines bias fraction, mininize MSE'''
def determine_bias_fraction(layer, param_dict, bitwidth, max_num_candidate=32):
    out_frac = param_dict[layer.top+'_frac']
    in_frac = param_dict[layer.bottom+'_frac']

    float_bias = param_dict[layer.name+'_bias']
    num_channel = float_bias.shape[0]

    if layer.name+'_frac_bias' in param_dict:
        bias_frac = param_dict[layer.name+'_frac_bias']
        if layer.name+'_frac_weight' in param_dict:
            add_log('Layer {}: bias_frac={} is previously determined, with in_frac:{}, out_frac:{}, and frac_weight:{}'.format(layer.name, list(bias_frac), in_frac, out_frac, list(param_dict[layer.name+'_frac_weight'])))
        else:
            add_log('Layer {}: bias_frac={} is previously determined, with in_frac:{}, out_frac:{}'.format(layer.name, list(bias_frac), in_frac, out_frac))
        if layer.name + '_frac_weight' in param_dict:
            weight_frac = param_dict[layer.name+'_frac_weight']
            for c in range(num_channel):
                if isinstance(bias_frac, int):
                    bis_frac = bias_frac
                else:
                    bis_frac = bias_frac[c]
                if isinstance(weight_frac, int):
                    wgt_frac = weight_frac
                else:
                    wgt_frac = weight_frac[c]
                assert in_frac + wgt_frac - bis_frac >= 0, 'failed when quantizing the bias of layer:{} with in_frac={}, weight_frac={} and detemined bias_frac={}'.format(layer.name, in_frac, wgt_frac, bis_frac)
                assert in_frac + wgt_frac - bis_frac <= _FRAC_UPPER_LIMIT[bitwidth], 'failed when quantizing the bias of layer:{} with in_frac={}, weight_frac={} and detemined bias_frac={}'.format(layer.name, in_frac, wgt_frac, bis_frac)
    else:
        if layer.name+'_frac_weight' in param_dict:
            add_log('Layer {}: bias fraction determined according to in_frac:{}, out_frac:{}, and frac_weight:{}'.format(layer.name, in_frac, out_frac, list(param_dict[layer.name+'_frac_weight'])))
        else:
            add_log('Layer {}: bias fraction determined according to in_frac:{} and out_frac:{}'.format(layer.name, in_frac, out_frac))

        candidate_frac_list = []
        for offset in range(max_num_candidate):
            candidate_frac = out_frac-offset+int(max_num_candidate/2)
            if layer.type=='Bias':
                if in_frac - candidate_frac >= 0 and in_frac - candidate_frac <= _FRAC_UPPER_LIMIT[bitwidth]:
                    candidate_frac_list.append(candidate_frac)
            elif layer.type in ['BatchNorm'] or (layer.type in ['Convolution', 'InnerProduct', 'Scale'] and layer.bias_term==True):
                candidate_frac_list.append(candidate_frac)
            else:
                pass

        assert len(candidate_frac_list)>0, 'failed when quantizing the bias of layer:{} with in_frac={} and out_frac={}'.format(layer.name, in_frac, out_frac)

        bias_frac = np.zeros(num_channel, dtype=int)
        for c in range(num_channel):
            mse_list = np.inf * np.ones(len(candidate_frac_list))
            for i, cur_frac in enumerate(candidate_frac_list):
                if layer.type in ['BatchNorm'] or (layer.type in ['Convolution', 'InnerProduct', 'Scale'] and layer.bias_term==True):
                    if layer.name+'_frac_weight' in param_dict:
                        if isinstance(param_dict[layer.name+'_frac_weight'], int):
                            weight_frac = param_dict[layer.name+'_frac_weight']
                        else:
                            weight_frac = param_dict[layer.name+'_frac_weight'][c]
                        if in_frac + weight_frac - cur_frac < 0 or in_frac + weight_frac - cur_frac > _FRAC_UPPER_LIMIT[bitwidth]:
                            continue
                mse_list[i] = np.square(qfunc.quantized_value(float_bias[c], bitwidth=bitwidth, frac=cur_frac, floor=ENABLE_FLOOR_WEIGHT[bitwidth]) - float_bias[c])
            assert np.min(mse_list) != np.inf, 'failed when quantizing the bias of layer:{} with in_frac={} and out_frac={}'.format(layer.name, in_frac, out_frac)
            best_idx = np.where(mse_list==np.min(mse_list))[0][0]
            bias_frac[c] = candidate_frac_list[best_idx]
            add_log('Layer {}: bias ch#{}/{}: min={:.3e}, max={:.3e}, frac={}, mse={:.3e}'.format(layer.name, c, num_channel, np.min(float_bias[c]), np.max(float_bias[c]), bias_frac[c], np.min(mse_list)))
        add_log('Layer {}: frac_bias={}'.format(layer.name, list(bias_frac)))

    return bias_frac

'''function determines fraction of a feature tensor and it is signed or unsigned'''
def determine_feature_fraction_sign(feature_name, bitwidth, feature_stat_dict, frac_bound, method=None, allow_unsigned=True, max_num_candidate=4):
    if method==None:
        method = 'minmax'

    add_log('Feature {}: fraction determined using method: {}, with bound:{} and allow_unsigned={}'.format(feature_name, method, list(frac_bound), allow_unsigned))

    # find the best fraction
    if allow_unsigned and feature_stat_dict[feature_name + '_min'] >= 0:
        signed = False
        if feature_stat_dict[feature_name + '_max'] <= _MIN_POSITIVE_NUMBER:
            feature_stat_dict[feature_name + '_max'] = _MIN_POSITIVE_NUMBER
            add_log('WARNING: encounter (almost) all-zero feature:{} with max absolute value: {:e}'.format(feature_name, feature_stat_dict[feature_name + '_max']))
        frac = int(np.floor(-np.log2(feature_stat_dict[feature_name + '_max'] / (2 ** bitwidth - 1))))
    else:
        signed = True
        max_abs_val = np.max(np.abs([feature_stat_dict[feature_name + '_min'], feature_stat_dict[feature_name + '_max']]))
        if max_abs_val <= _MIN_POSITIVE_NUMBER:
            max_abs_val = _MIN_POSITIVE_NUMBER
            add_log('WARNING: encounter (almost) all-zero feature:{} with max absolute value: {:e}'.format(feature_name, max_abs_val))
        frac = int(np.floor(-np.log2(max_abs_val / (2 ** (bitwidth - 1) - 1))))

    add_log('Feature {}: min={:.3e}, max={:.3e}'.format(feature_name, feature_stat_dict[feature_name + '_min'], feature_stat_dict[feature_name + '_max']))

    if method.lower() == 'minmax':
        pass
    elif method.lower() in ['mse', 'kl', 'js']:
        list_candidate_frac = []
        for offset in range(max_num_candidate):
            list_candidate_frac.append(frac+offset)

        loss_list = np.zeros(len(list_candidate_frac))
        float_vals = feature_stat_dict[feature_name + '_pd'][0]
        float_probs = feature_stat_dict[feature_name + '_pd'][1]
        compact_bin = np.linspace(float_vals[0], float_vals[-1], _NUM_BIN_FOR_KL_JS)
        compact_float_prbs = ndk.utils.format_prob_distribution(bins=float_vals, probs=float_probs, target_bins=compact_bin)
        add_log('Feature {}: bins of probability distribution: {}'.format(feature_name, list(compact_bin)))
        add_log('Feature {}: prbs of probability distribution: {}'.format(feature_name, list(compact_float_prbs)))
        for i, cur_frac in enumerate(list_candidate_frac):
            quant_vals = qfunc.quantized_value(float_vals, bitwidth=bitwidth, frac=cur_frac, signed=signed, floor=ENABLE_FLOOR_FEATURE[bitwidth])
            if method.lower()=='mse':
                loss_list[i] = np.dot(np.square(quant_vals - float_vals), float_probs)
            elif method.lower() in ['kl', 'js']:
                quant_probs = quantize_prob_distribution(bin_vals=float_vals, float_prob=float_probs, bitwitdh=bitwidth, fraction=cur_frac, signed=signed)
                compact_quant_prbs = ndk.utils.format_prob_distribution(bins=float_vals, probs=quant_probs, target_bins=compact_bin)
                if method.lower()=='kl':
                    loss_list[i] = ndk.utils.KLD(compact_float_prbs, compact_quant_prbs)
                else:
                    loss_list[i] = ndk.utils.JSD(compact_float_prbs, compact_quant_prbs)
            else:
                raise ValueError('Unknown error')
        best_idx = np.where(loss_list==np.min(loss_list))[0][0]
        frac = list_candidate_frac[best_idx]
        add_log('Feature {}: method:{} finds optimal loss={:.3e}'.format(feature_name, method, np.min(loss_list)))
    else:
        raise ValueError('method: {} not supported'.format(method))

    add_log('Feature {}: method:{} proposes candidate frac={}'.format(feature_name, method, frac))
    # adjust the result to meet the bounds
    # print('{}: frac={}, frac_bound[0]={}'.format(feature_name, frac, frac_bound[0]))
    while frac < frac_bound[0]:
        frac += 1
    while frac > frac_bound[1]:
        frac -= 1
    assert frac>=frac_bound[0] and frac<=frac_bound[1], 'failed in quantize feature tensor:{}, whose fraction is supposed to be larger than {} but smaller than {}'.format(feature_name, frac_bound[0], frac_bound[1])

    add_log('Feature {}: frac={}, signed={}'.format(feature_name, frac, signed))
    return frac, signed

'''function determines upper and lower bounds of the fraction at the output of the given layer'''
def get_output_frac_bound(layer, param_dict, bitwidth):
    frac_bound = [-np.inf, np.inf]
    if layer.type in ['Convolution', 'InnerProduct', 'BatchNorm', 'Scale', 'Bias']:
        in_frac = None
        out_frac = None
        weight_frac = None
        bias_frac = None
        if layer.bottom + '_frac' in param_dict:
            in_frac = np.array(param_dict[layer.bottom + '_frac'])
        if layer.top + '_frac' in param_dict:
            out_frac = np.array(param_dict[layer.top + '_frac'])
        if layer.name + '_frac_weight' in param_dict:
            weight_frac = np.array(param_dict[layer.name + '_frac_weight'])
        if layer.name + '_frac_bias' in param_dict:
            bias_frac = np.array(param_dict[layer.name + '_frac_bias'])

        if type(out_frac)!=type(None): # output fraction has been determined
            frac_bound = [out_frac, out_frac]
        elif layer.type=='Bias' and type(in_frac)!=type(None):
            frac_bound = [in_frac - _FRAC_UPPER_LIMIT[bitwidth], in_frac]
        elif layer.type in ['Convolution', 'InnerProduct', 'BatchNorm', 'Scale'] and type(in_frac)!=type(None) and type(weight_frac)!=type(None):
            frac_bound = [in_frac + np.max(weight_frac) - _FRAC_UPPER_LIMIT[bitwidth], in_frac + np.min(weight_frac)]
    elif layer.type in ['Eltwise']:
        in_frac = None
        for i in range(len(layer.bottom)):
            cur_in_frac = None
            if layer.bottom[i] + '_frac' in param_dict:
                cur_in_frac = param_dict[layer.bottom[i] + '_frac']
            if in_frac==None:
                in_frac = cur_in_frac
            elif cur_in_frac==None:
                in_frac = None
            else:
                assert in_frac==cur_in_frac, 'find inconsistent input fractions in layer:{}, where tensor:{} is {} and some other is {}'.format(layer.name, layer.bottom[i], cur_in_frac, in_frac)
        if type(in_frac) != type(None):
            frac_bound[1] = in_frac
    return frac_bound

'''update the given param_dict, if weight/bias of some layer can be quantized: first bias then weight'''
def try_quantize_weight_bias(layer_list, param_dict, bitwidth, bias_first=True):
    assert bitwidth==8 or bitwidth==16, 'bitwidth must be 8 or 16, but get {}'.format(bitwidth)
    for layer in layer_list:
        if layer.type in ['Convolution', 'InnerProduct', 'Scale']:
            in_frac = None
            out_frac = None
            weight_frac = None
            bias_frac = None
            if layer.bottom+'_frac' in param_dict:
                in_frac = param_dict[layer.bottom+'_frac']
            if layer.top+'_frac' in param_dict:
                out_frac = param_dict[layer.top+'_frac']
            if layer.name+'_frac_weight' in param_dict:
                weight_frac = param_dict[layer.name+'_frac_weight']
            if layer.name+'_frac_bias' in param_dict:
                bias_frac = param_dict[layer.name+'_frac_bias']
            if type(in_frac)!=type(None) and type(out_frac)!=type(None):
                if bias_first:
                    if (type(bias_frac)==type(None) or layer.name+'_quant_bias' not in param_dict) and layer.bias_term==True:
                        bias_frac = determine_bias_fraction(layer=layer, param_dict=param_dict, bitwidth=bitwidth)
                        param_dict[layer.name+'_frac_bias'] = bias_frac
                        param_dict[layer.name+'_quant_bias'] = qfunc.float2quant(param_dict[layer.name+'_bias'], bitwidth=bitwidth, frac=bias_frac, floor=ENABLE_FLOOR_WEIGHT[bitwidth])

                    if type(weight_frac)==type(None) or layer.name+'_quant_weight' not in param_dict:
                        weight_frac = determine_weight_fraction(layer=layer, param_dict=param_dict, bitwidth=bitwidth)
                        param_dict[layer.name+'_frac_weight'] = weight_frac
                        param_dict[layer.name+'_quant_weight'] = qfunc.float2quant(param_dict[layer.name+'_weight'], bitwidth=bitwidth, frac=weight_frac, floor=ENABLE_FLOOR_WEIGHT[bitwidth])
                else:
                    if type(weight_frac)==type(None) or layer.name+'_quant_weight' not in param_dict:
                        weight_frac = determine_weight_fraction(layer=layer, param_dict=param_dict, bitwidth=bitwidth)
                        param_dict[layer.name+'_frac_weight'] = weight_frac
                        param_dict[layer.name+'_quant_weight'] = qfunc.float2quant(param_dict[layer.name+'_weight'], bitwidth=bitwidth, frac=weight_frac, floor=ENABLE_FLOOR_WEIGHT[bitwidth])
                    if (type(bias_frac)==type(None) or layer.name+'_quant_bias' not in param_dict) and layer.bias_term==True:
                        bias_frac = determine_bias_fraction(layer=layer, param_dict=param_dict, bitwidth=bitwidth)
                        param_dict[layer.name+'_frac_bias'] = bias_frac
                        param_dict[layer.name+'_quant_bias'] = qfunc.float2quant(param_dict[layer.name+'_bias'], bitwidth=bitwidth, frac=bias_frac, floor=ENABLE_FLOOR_WEIGHT[bitwidth])
        elif layer.type in ['BatchNorm']:
            in_frac = None
            out_frac = None
            weight_frac = None
            bias_frac = None
            if layer.bottom+'_frac' in param_dict:
                in_frac = param_dict[layer.bottom+'_frac']
            if layer.top+'_frac' in param_dict:
                out_frac = param_dict[layer.top+'_frac']
            if layer.name+'_frac_weight' in param_dict:
                weight_frac = param_dict[layer.name+'_frac_weight']
            if layer.name+'_frac_bias' in param_dict:
                bias_frac = param_dict[layer.name+'_frac_bias']
            if type(in_frac)!=type(None) and type(out_frac)!=type(None):
                if bias_first:
                    if type(bias_frac)==type(None) or layer.name+'_quant_bias' not in param_dict:
                        bias_frac = determine_bias_fraction(layer=layer, param_dict=param_dict, bitwidth=bitwidth)
                        param_dict[layer.name+'_frac_bias'] = bias_frac
                        param_dict[layer.name+'_quant_bias'] = qfunc.float2quant(param_dict[layer.name+'_bias'], bitwidth=bitwidth, frac=bias_frac, floor=ENABLE_FLOOR_WEIGHT[bitwidth])
                    if type(weight_frac)==type(None) or layer.name+'_quant_weight' not in param_dict:
                        weight_frac = determine_weight_fraction(layer=layer, param_dict=param_dict, bitwidth=bitwidth)
                        param_dict[layer.name+'_frac_weight'] = weight_frac
                        param_dict[layer.name+'_quant_weight'] = qfunc.float2quant(param_dict[layer.name+'_weight'], bitwidth=bitwidth, frac=weight_frac, floor=ENABLE_FLOOR_WEIGHT[bitwidth])
                else:
                    if type(weight_frac)==type(None) or layer.name+'_quant_weight' not in param_dict:
                        weight_frac = determine_weight_fraction(layer=layer, param_dict=param_dict, bitwidth=bitwidth)
                        param_dict[layer.name+'_frac_weight'] = weight_frac
                        param_dict[layer.name+'_quant_weight'] = qfunc.float2quant(param_dict[layer.name+'_weight'], bitwidth=bitwidth, frac=weight_frac, floor=ENABLE_FLOOR_WEIGHT[bitwidth])
                    if type(bias_frac)==type(None) or layer.name+'_quant_bias' not in param_dict:
                        bias_frac = determine_bias_fraction(layer=layer, param_dict=param_dict, bitwidth=bitwidth)
                        param_dict[layer.name+'_frac_bias'] = bias_frac
                        param_dict[layer.name+'_quant_bias'] = qfunc.float2quant(param_dict[layer.name+'_bias'], bitwidth=bitwidth, frac=bias_frac, floor=ENABLE_FLOOR_WEIGHT[bitwidth])
        elif layer.type in ['Bias']:
            in_frac = None
            out_frac = None
            bias_frac = None
            if layer.bottom+'_frac' in param_dict:
                in_frac = param_dict[layer.bottom+'_frac']
            if layer.top+'_frac' in param_dict:
                out_frac = param_dict[layer.top+'_frac']
            if layer.name+'_frac_bias' in param_dict:
                bias_frac = param_dict[layer.name+'_frac_bias']
            if type(in_frac)!=type(None) and type(out_frac)!=type(None):
                if type(bias_frac)==type(None) or layer.name+'_quant_weight' not in param_dict:
                    bias_frac = determine_bias_fraction(layer=layer, param_dict=param_dict, bitwidth=bitwidth)
                    param_dict[layer.name+'_frac_bias'] = bias_frac
                    param_dict[layer.name+'_quant_bias'] = qfunc.float2quant(param_dict[layer.name+'_bias'], bitwidth=bitwidth, frac=bias_frac, floor=ENABLE_FLOOR_WEIGHT[bitwidth])

    return param_dict

'''Go through the net and update the numpy_layers: if any layer is newly quantized, set the numpy layer quantized'''
def try_set_quant_param_for_numpy_layers(numpy_layers, layer_list, param_dict, bitwidth):
    for idx, np_layer in enumerate(numpy_layers):
        if not np_layer.quantized:
            layer = layer_list[idx]
            in_frac, out_frac, weight_frac, bias_frac, signed = get_quant_param(layer, param_dict)
            if layer.type in ['Input']:
                if type(out_frac) != type(None):
                    np_layer.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=signed)
            elif layer.type in ['Sigmoid', 'ReLU', 'ReLU6', 'TanH', 'Pooling', 'Slice', 'Concat', 'Eltwise', 'LogSoftmax', 'ShuffleChannel']:
                if type(in_frac) != type(None) and type(out_frac) != type(None):
                    if layer.type=='ReLU' and layer.negative_slope!=0:
                        quant_negative_slope = qfunc.quantized_value(layer.negative_slope, bitwidth=8, frac=6, floor=False)
                        np_layer.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, quant_negative_slope=quant_negative_slope, input_signed=signed)
                    else:
                        np_layer.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, input_signed=signed)
            elif layer.type in ['Convolution', 'InnerProduct', 'Scale', 'BatchNorm', 'Bias']:
                if type(in_frac) != type(None) and type(out_frac) != type(None):
                    has_weight = False
                    has_bias = False
                    if layer.type in ['Convolution', 'InnerProduct', 'Scale']:
                        has_weight = True
                        if layer.bias_term==True:
                            has_bias = True
                    elif layer.type in ['BatchNorm']:
                        has_weight = True
                        has_bias = True
                    elif layer.type in ['Bias']:
                        has_bias = True

                    if has_weight and has_bias:
                        if type(weight_frac) != type(None) and type(bias_frac) != type(None):
                            quant_weight = qfunc.quant2float(param_dict[layer.name+'_quant_weight'], bitwidth=bitwidth, frac=weight_frac)
                            quant_bias = qfunc.quant2float(param_dict[layer.name+'_quant_bias'], bitwidth=bitwidth, frac=bias_frac)
                            np_layer.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, quant_weight=quant_weight, quant_bias=quant_bias, input_signed=signed)
                    elif has_weight:
                        if type(weight_frac) != type(None):
                            quant_weight = qfunc.quant2float(param_dict[layer.name+'_quant_weight'], bitwidth=bitwidth, frac=weight_frac)
                            np_layer.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, quant_weight=quant_weight, input_signed=signed)
                    elif has_bias:
                        if type(bias_frac) != type(None):
                            quant_bias = qfunc.quant2float(param_dict[layer.name+'_quant_bias'], bitwidth=bitwidth, frac=bias_frac)
                            np_layer.set_quant_param(bitwidth=bitwidth, in_frac=in_frac, out_frac=out_frac, quant_bias=quant_bias, input_signed=signed)
                    else:
                        raise ValueError('unknown structure of layer:{}'.format(layer.name))
    return numpy_layers

'''Run the net. If try_run_quant set to True, quantized parameters will be used when available.'''
def try_run_quant_layers(input_data_batch,
                         layer_list,
                         target_feature_tensor_list,
                         param_dict=None,
                         try_run_quant=False,
                         bitwidth=8,
                         hw_aligned=False,
                         numpy_layers=None,
                         log_on=False,
                         use_ai_framework=True):

    assert isinstance(try_run_quant, bool), 'quant should be a Bool number, True or False'
    feature_tensor_name_list = ndk.layers.get_tensor_name_list(layer_list, feature_only=True)
    for tensor_name in target_feature_tensor_list:
        assert tensor_name in feature_tensor_name_list, 'Feature {} not found.'.format(tensor_name)

    if type(numpy_layers)==type(None):
        # layer_list = copy.deepcopy(layer_list)
        layer_list = ndk.layers.sort_layers(layer_list)
        numpy_layers = qnet.build_numpy_layers(layer_list=layer_list, param_dict=param_dict, quant=False, bitwidth=bitwidth)
    else:
        for lyr_idx in range(len(layer_list)):
            assert layer_list[lyr_idx].name == numpy_layers[lyr_idx].name, 'When running using the pre-built numpy_layers (i.e., numpy_layers!=None), layer_list must be sorted using sort_layer_list_for_quantize_net() first'

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
            if try_run_quant and numpy_layers[i].quantized:
                data_out = numpy_layers[i].run(input_data_batch, quant=True, hw_aligned=hw_aligned, use_ai_framework=use_ai_framework)
            else:
                data_out = numpy_layers[i].run(input_data_batch, quant=False, hw_aligned=hw_aligned, use_ai_framework=use_ai_framework)
            data_dict[lyr.top] = data_out.copy()
        else:
            if type(lyr.bottom)==list:
                data_in = []
                for tensor_name in lyr.bottom:
                    data_in.append(data_dict[tensor_name])
            else:
                data_in = data_dict[lyr.bottom]

            if try_run_quant and numpy_layers[i].quantized:
                data_out = numpy_layers[i].run(data_in, quant=True, hw_aligned=hw_aligned, use_ai_framework=use_ai_framework)
            else:
                data_out = numpy_layers[i].run(data_in, quant=False, hw_aligned=hw_aligned, use_ai_framework=use_ai_framework)

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

'''Get the statistics and distributions of the target feature tensors.'''
def get_feature_distribution(layer_list,
                             target_feature_list,
                             numpy_layers,
                             data_generator,
                             try_run_quant=True,
                             bitwidth=8,
                             num_batch=1000,
                             hw_aligned=True,
                             num_bin=0,
                             minmax_val_dict=None,
                             log_on=False,
                             features_require_pd=None,
                             min_num_samples_for_bin=1000):
    stats_dict = {}

    if features_require_pd==None:
        features_require_pd = []
    if minmax_val_dict==None:
        minmax_val_dict = {}

    # check
    feature_name_list = ndk.layers.get_tensor_name_list(layer_list, feature_only=True)
    for tsr in target_feature_list:
        if not tsr in feature_name_list:
            raise ValueError('Tensor:{} from target_feature_list is not a feature tensor.'.format(tsr))
    for tsr in features_require_pd:
        if not tsr in target_feature_list:
            raise ValueError('Tensor:{} from features_require_pd is not included in target_feature_list.'.format(tsr))

    bin_dict = {}
    if num_bin > 0 and len(features_require_pd) > 0:

        # check whether all the tensor in features_require_pd has determined min/max in minmax_val_dict
        features_no_minmax = []
        for tsr in features_require_pd:
            if tsr+'_min' in minmax_val_dict and tsr+'_max' in minmax_val_dict:
                pass
            else:
                features_no_minmax.append(tsr)

        if len(features_no_minmax) > 0:
            if log_on:
                print_log('quantize_net: get_feature_distribution: determining bins...')
            # run at least 1000 samples to get the maximum absolute value which will be used to determine the bin values
            for tensor_name in features_no_minmax:
                minmax_val_dict[tensor_name] = -np.inf
                minmax_val_dict[tensor_name] = np.inf
            num_samples = 0
            while num_samples < min_num_samples_for_bin:
                data_batch = next(data_generator)
                n,_,_,_ = data_batch['input'].shape
                num_samples += n
                result_dict = try_run_quant_layers(input_data_batch=data_batch['input'],
                                                   layer_list=layer_list,
                                                   target_feature_tensor_list=features_no_minmax,
                                                   param_dict=None,
                                                   try_run_quant=try_run_quant,
                                                   bitwidth=bitwidth,
                                                   hw_aligned=hw_aligned,
                                                   numpy_layers=numpy_layers,
                                                   log_on=False,
                                                   use_ai_framework=True)

                for tensor_name in features_no_minmax:
                    minmax_val_dict[tensor_name] = np.max([minmax_val_dict[tensor_name], np.max(result_dict[tensor_name])])
                    minmax_val_dict[tensor_name] = np.min([minmax_val_dict[tensor_name], np.min(result_dict[tensor_name])])

                print_log('quantize_net: get_feature_distribution: determining bins: {}/{} samples collected'.format(num_samples, min_num_samples_for_bin))

            if log_on:
                print_log('quantize_net: get_feature_distribution: bins determined using {} samples'.format(num_samples))
            add_log('quantize_net: get_feature_distribution: bins determined using {} samples'.format(num_samples))

        # create the bins
        for tensor_name in features_require_pd:
            bin_dict[tensor_name] = np.linspace(minmax_val_dict[tensor_name+'_min'], minmax_val_dict[tensor_name+'_max'], num_bin)

    # process the target features
    cnt_dict = {}
    feat_stats_dict = {}
    for tensor_name in target_feature_list:
        cnt_dict[tensor_name] = np.zeros(num_bin, dtype=int)
        feat_stats_dict[tensor_name+'_min'] = np.inf
        feat_stats_dict[tensor_name+'_max'] = -np.inf

    tic = time.clock() - 3600
    for bidx in range(num_batch):
        data_batch = next(data_generator)
        batch_size,_,_,_=data_batch['input'].shape
        if log_on and (time.clock()-tic)>=10: # print log every 10 seconds
            print_log('quantize_net: get_feature_distribution: processing bacth #{}/{} with size {}'.format(bidx+1, num_batch, batch_size))
            tic = time.clock()
        result_dict = try_run_quant_layers(layer_list=layer_list, target_feature_tensor_list=target_feature_list,
                                            input_data_batch=data_batch['input'],
                                            numpy_layers=numpy_layers, param_dict=None, bitwidth=bitwidth,
                                            try_run_quant=try_run_quant, hw_aligned=hw_aligned, log_on=False)

        for tensor_name in target_feature_list:
            feat_stats_dict[tensor_name+'_min'] = np.min([feat_stats_dict[tensor_name+'_min'], np.min(result_dict[tensor_name])])
            feat_stats_dict[tensor_name+'_max'] = np.max([feat_stats_dict[tensor_name+'_max'], np.max(result_dict[tensor_name])])
            if num_bin > 0 and tensor_name in features_require_pd:
                cnt_dict[tensor_name] += ndk.utils.hist_func(result_dict[tensor_name], bin_dict[tensor_name])

    if num_batch > 0:
        add_log('quantize_net: get_feature_distribution: processing {} bacthes with size {}'.format(num_batch, batch_size))

    # update the stats_dict
    for tensor_name in target_feature_list:
        stats_dict[tensor_name+'_min'] = feat_stats_dict[tensor_name+'_min']
        stats_dict[tensor_name+'_max'] = feat_stats_dict[tensor_name+'_max']
        if num_bin > 0 and tensor_name in features_require_pd:
            stats_dict[tensor_name+'_pd'] = np.array([bin_dict[tensor_name], cnt_dict[tensor_name]/sum(cnt_dict[tensor_name])])

    return stats_dict

'''Find next layer which is in layer_list'''
def find_next_layer_index(cur_layer_index, layer_list):
    layer = layer_list[cur_layer_index]
    next_layer_index = None
    for nxt_idx in range(cur_layer_index+1, len(layer_list)):
        nxt_layer = layer_list[nxt_idx]
        if isinstance(layer.top, list):
            if isinstance(nxt_layer.bottom, list):
                for tsr in nxt_layer.bottom:
                    if tsr in layer.top:
                        next_layer_index = nxt_idx
                        break
            else:
                if nxt_layer.bottom in layer.top:
                    next_layer_index = nxt_idx
        else:
            if isinstance(nxt_layer.bottom, list):
                for tsr in nxt_layer.bottom:
                    if tsr == layer.top:
                        next_layer_index = nxt_idx
                        break
            else:
                if nxt_layer.bottom == layer.top:
                    next_layer_index = nxt_idx
        if next_layer_index!=None:
            break
    return next_layer_index

'''Find the key feature tensor from the current layer index. Used when quantizing feature tensor'''
def find_next_feature_tensor_to_quantize(start_layer_idx, layer_list, param_dict, numpy_layers, bitwidth):
    # calculate the upper and lower bounds for output fraction
    frac_bound = get_output_frac_bound(layer=layer_list[start_layer_idx], param_dict=param_dict, bitwidth=bitwidth)
    # looking forward
    cur_layer_idx = start_layer_idx
    nxt_layer_idx = find_next_layer_index(cur_layer_index=cur_layer_idx, layer_list=layer_list)
    # keep looking forward while: 1. not last layer; 2. not quantized; 3. next layer is in SAME class
    while type(nxt_layer_idx) != type(None) \
            and not numpy_layers[nxt_layer_idx].quantized \
            and classify_layer_type(layer_list[nxt_layer_idx]) == 'same':
        cur_layer_idx = nxt_layer_idx
        nxt_layer_idx = find_next_layer_index(cur_layer_index=cur_layer_idx, layer_list=layer_list)
        # update the bounds of output fraction
        cur_frac_bound = get_output_frac_bound(layer=layer_list[cur_layer_idx], param_dict=param_dict, bitwidth=bitwidth)
        frac_bound[0] = np.max([frac_bound[0], cur_frac_bound[0]])
        frac_bound[1] = np.min([frac_bound[1], cur_frac_bound[1]])

    if nxt_layer_idx == None:  # if current layer is the last layer
        target_feature_tensor = layer_list[cur_layer_idx].top
    else:
        target_feature_tensor = layer_list[nxt_layer_idx].bottom
        if isinstance(target_feature_tensor, list):
            for tsr in layer_list[nxt_layer_idx].bottom:
                if tsr in layer_list[cur_layer_idx].top:
                    target_feature_tensor = tsr
                    break
    # print_log('target feature: {}'.format(target_feature_tensor))
#    assert not target_feature_tensor + '_frac' in param_dict, 'It is impossible.'
    if target_feature_tensor + '_frac' in param_dict:
        print("{} has been determined".format(target_feature_tensor + '_frac'))
        print("layer info")
        print(layer_list[cur_layer_idx])
        assert False, 'It is impossible.'
    if bitwidth==8 and layer_list[cur_layer_idx].type == 'Input' and classify_layer_type(layer_list[cur_layer_idx]) == 'free':
        allow_unsigned = True
    else:
        allow_unsigned = False
    print("determin {}'s frac".format(target_feature_tensor))
    return target_feature_tensor, frac_bound, allow_unsigned

'''Extract the useful information from the usr_param_dict'''
def format_usr_param_dict(usr_param_dict, layer_list, param_dict):
    result_usr_param_dict = {}
    layer_names = ndk.layers.get_layer_name_list(layer_list)
    feature_names = ndk.layers.get_tensor_name_list(layer_list, feature_only=True)
    for key in usr_param_dict:
        if isinstance(key, str) and (
                (key.endswith('_frac_weight') and key[:-12] in layer_names) or
                (key.endswith('_frac_bias') and key[:-10] in layer_names) or
                (key.endswith('_frac') and key[:-5] in feature_names)
        ):
            val = usr_param_dict[key]
            if isinstance(val, list):
                val = np.array(val)
            elif isinstance(val, np.ndarray):
                pass
            else:
                if key.endswith('_frac_weight'):
                    ref_var = param_dict[key[:-12] + '_weight']
                    val = int(val) * np.ones(ref_var.shape[0])
                elif key.endswith('_frac_bias'):
                    ref_var = param_dict[key[:-10] + '_bias']
                    val = int(val) * np.ones(ref_var.shape[0])
                else:
                    val = int(val)
            result_usr_param_dict[key] = val
            add_log('user_param_dict: {}={}'.format(key, val))
    return result_usr_param_dict

'''Special process for LogSoftmax layer: add bias to previous layer in order to avoid overflow'''
def add_bias_to_previous_layer_for_logsoftmax(layer_list, param_dict, data_generator, num_batch, bitwidth, log_on=True):
    tensor_shapes = ndk.layers.get_tensor_shape(layer_list)
    net_output_tensors = ndk.layers.get_network_output(layer_list)
    layer_has_top = ndk.layers.get_dict_name_to_top(layer_list)
    input_tensors_to_logsoftmax = []
    for layer_idx, layer in enumerate(layer_list):
        if layer.type=='LogSoftmax':
            assert layer.top in net_output_tensors, 'Error: find a LogSoftmax layer:{} appears to be NOT an output layer.'.format(layer.name)
            assert layer_has_top[layer.bottom][0].type in ['Convolution', 'InnerProduct'], 'Error: find a LogSoftmax layer:{} which does NOT follow an Convolution/InnerProduct layer.'.format(layer.name)
            input_tensors_to_logsoftmax.append(layer.bottom)
    assert len(input_tensors_to_logsoftmax)<=1, 'Error: find more than one net outputs'

    if len(input_tensors_to_logsoftmax)==1:
        print_log('quantize_net: adding bias to previous layer for LogSoftmax, layer_list and param_dict will be updated...')
        key_tensor_name = input_tensors_to_logsoftmax[0]
        c = tensor_shapes[key_tensor_name][1]
        h = tensor_shapes[key_tensor_name][2]
        w = tensor_shapes[key_tensor_name][3]
        max_val = -np.inf

        print_log('quantize_net: for LogSoftmax: building numpy layers...')
        sorted_layer_list = ndk.layers.sort_layers(layer_list)
        np_layers = qnet.build_numpy_layers(layer_list=sorted_layer_list, param_dict=param_dict, quant=False, bitwidth=bitwidth)

        tic = time.clock() - 3600
        for bidx in range(num_batch):
            data_batch = next(data_generator)
            batch_size,_,_,_=data_batch['input'].shape
            if log_on and (time.clock()-tic)>=10: # print log every 10 seconds:
                print_log('quantize_net: for LogSoftmax: processing bacth #{}/{} with size {}'.format(bidx+1, num_batch, batch_size))
                tic = time.clock()
            result_dict = qnet.run_layers(layer_list=sorted_layer_list, target_feature_tensor_list=[key_tensor_name],
                                          input_data_batch=data_batch['input'],
                                          numpy_layers=np_layers, param_dict=None, bitwidth=bitwidth,
                                          quant=False, hw_aligned=True, log_on=False)
            max_val = np.max([np.max(result_dict[key_tensor_name]), max_val])

        prev_layer = layer_has_top[key_tensor_name][0]
        if prev_layer.bias_term == False:
            prev_layer.bias_term = True
            param_dict[prev_layer.name+'_bias'] = np.zeros(prev_layer.num_output)
        param_dict[prev_layer.name + '_bias'] -= max_val

        print_log('quantize_net: adding {} to bias of layer:{} because of a following LogSoftmax...'.format(-max_val, prev_layer.name))

    return layer_list, param_dict

'''Special process for LogSoftmax layer: delete LogSoftmax layer, and return an updated layer_list'''
def delete_logsoftmax_layer(layer_list):
    net_output_tensors = ndk.layers.get_network_output(layer_list)
    layer_has_top = ndk.layers.get_dict_name_to_top(layer_list)
    index_list_logsoftmax = []
    for layer_idx, layer in enumerate(layer_list):
        if layer.type=='LogSoftmax':
            assert layer.top in net_output_tensors, 'Error: find a LogSoftmax layer:{} appears to be NOT an output layer.'.format(layer.name)
            assert layer_has_top[layer.bottom][0].type in ['Convolution', 'InnerProduct'], 'Error: find a LogSoftmax layer:{} which does NOT follow an Convolution/InnerProduct layer.'.format(layer.name)
            index_list_logsoftmax.append(layer_idx)
    assert len(index_list_logsoftmax)<=1, 'Error: find more than one net outputs'

    modified_layer_list = copy.deepcopy(layer_list)
    deleted_layer = None
    if len(index_list_logsoftmax)==1:
        print_log('quantize_net: delete the last LogSoftmax layer, suggest to use CPU instead of NPU, or adjust the quantization parameters mannually.')
        deleted_layer = modified_layer_list.pop()

    return modified_layer_list, deleted_layer

'''quantize weight without check input and output tensors'''
def directly_determine_weight_fraction(layer_list, param_dict, bitwidth=8, method='mse', max_num_candidate=8):
    num_layer = len(layer_list)
    for lyr_idx in range(num_layer):
        layer = layer_list[lyr_idx]
        if layer.name + '_weight' in param_dict \
                and layer.name + '_frac_weight' not in param_dict \
                and layer.name + '_quant_weight' not in param_dict:
            add_log('Layer {}: weight fraction determined directly using method: {}'.format(layer.name, method))
            fp_weight = param_dict[layer.name + '_weight']
            fp_weight[np.fabs(fp_weight) <= np.max(np.fabs(fp_weight))/_IGNORE_WEIGHT_THRESHOLD[bitwidth]] = 0
            add_log('Layer {}: weight with absolute values smaller than {:.3e} will be set to zeros.'.format(layer.name, np.max(np.fabs(fp_weight))/_IGNORE_WEIGHT_THRESHOLD[bitwidth]))
            num_channel = fp_weight.shape[0]
            frac_weight = np.zeros(num_channel, dtype=int)
            for chn_idx in range(num_channel):
                weights_in_channel = fp_weight[chn_idx]
                max_val = np.max(weights_in_channel)
                min_val = np.min(weights_in_channel)
                max_abs_val = max(abs(max_val), abs(min_val))
                if max_abs_val==0:
                    frac_minmax = -65536
                else:
                    frac_minmax = int(np.floor(-np.log2(max_abs_val/(2**(bitwidth-1)-1))))

                if method=='mse':
                    candidate_frac_list = []
                    for offset in range(max_num_candidate):
                        candidate_frac_list.append(max(frac_minmax + offset, -65000))
                    mse_list = np.inf * np.ones(len(candidate_frac_list))
                    for i, cur_frac in enumerate(candidate_frac_list):
                        if cur_frac > -65000:
                            mse_list[i] = np.mean(np.square(qfunc.quantized_value(frac_weight[chn_idx], bitwidth=bitwidth, frac=cur_frac, floor=ENABLE_FLOOR_WEIGHT[bitwidth]) - frac_weight[chn_idx]))
                    best_idx = np.where(mse_list == np.min(mse_list))[0][0]
                    frac_weight[chn_idx] = candidate_frac_list[best_idx]
                else:
                    frac_weight[chn_idx] = frac_minmax
                add_log('Layer {}: weight ch#{}/{}: min={:.3e}, max={:.3e}, candidate frac={}'.format(layer.name, chn_idx, num_channel, min_val, max_val, frac_weight[chn_idx]))
            frac_weight[frac_weight <= -65000] = np.max(frac_weight)
            frac_weight[frac_weight <= -65000] = 0
            add_log('Layer {}: frac_weight={}'.format(layer.name, list(frac_weight)))
            param_dict[layer.name + '_frac_weight'] = frac_weight

    return layer_list, param_dict

'''quantize bias without check input and output tensors'''
def directly_determine_bias_fraction(layer_list, param_dict, bitwidth=8, method='mse', max_num_candidate=8):
    num_layer = len(layer_list)
    for lyr_idx in range(num_layer):
        layer = layer_list[lyr_idx]
        if layer.name + '_bias' in param_dict \
                and layer.name + '_frac_bias' not in param_dict \
                and layer.name + '_quant_bias' not in param_dict:
            add_log('Layer {}: bias fraction determined directly using method: {}'.format(layer.name, method))
            fp_bias = param_dict[layer.name + '_bias']
            fp_bias[np.fabs(fp_bias) <= np.max(np.fabs(fp_bias))/_IGNORE_BIAS_THRESHOLD[bitwidth]] = 0
            add_log('Layer {}: bias with absolute values smaller than {:.3e} will be set to zeros.'.format(layer.name, np.max(np.fabs(fp_bias))/_IGNORE_WEIGHT_THRESHOLD[bitwidth]))
            num_channel = fp_bias.shape[0]
            frac_bias = np.zeros(num_channel, dtype=int)
            for chn_idx in range(num_channel):
                bias_in_channel = fp_bias[chn_idx]
                max_val = np.max(bias_in_channel)
                min_val = np.min(bias_in_channel)
                max_abs_val = max(abs(max_val), abs(min_val))
                if max_abs_val==0:
                    frac_minmax = -65536
                else:
                    frac_minmax = int(np.floor(-np.log2(max_abs_val/(2**(bitwidth-1)-1))))

                if method=='mse':
                    candidate_frac_list = []
                    for offset in range(max_num_candidate):
                        candidate_frac_list.append(max(frac_minmax + offset, -65000))
                    mse_list = np.inf * np.ones(len(candidate_frac_list))
                    for i, cur_frac in enumerate(candidate_frac_list):
                        if cur_frac > -65000:
                            mse_list[i] = np.square(qfunc.quantized_value(fp_bias[chn_idx], bitwidth=bitwidth, frac=cur_frac, floor=ENABLE_FLOOR_WEIGHT[bitwidth]) - fp_bias[chn_idx])
                    best_idx = np.where(mse_list == np.min(mse_list))[0][0]
                    frac_bias[chn_idx] = candidate_frac_list[best_idx]
                else:
                    frac_bias[chn_idx] = frac_minmax
                add_log('Layer {}: bias ch#{}/{}: min={:.3e}, max={:.3e}, candidate frac={}'.format(layer.name, chn_idx, num_channel, min_val, max_val, frac_bias[chn_idx]))
            frac_bias[frac_bias <= -65000] = np.max(frac_bias)
            frac_bias[frac_bias <= -65000] = 0
            param_dict[layer.name + '_frac_bias'] = frac_bias
            add_log('Layer {}: frac_bias={}'.format(layer.name, list(frac_bias)))

    return layer_list, param_dict

'''compensate the FLOOR operation on output feature to ROUND by adding a value to bias'''
def compensate_output_feature_on_bias(layer_list, param_dict, bitwidth=8):
    max_q = 2.0 ** (bitwidth - 1) - 1
    min_q = -2.0 ** (bitwidth - 1)
    for layer in layer_list:
        if layer.name+'_quant_bias' in param_dict:
            bias_frac = param_dict[layer.name + '_frac_bias']
            out_frac = param_dict[layer.top + '_frac']
            if isinstance(bias_frac, int) and bias_frac > out_frac:
                param_dict[layer.name + '_quant_bias'] += 1 << (bias_frac - out_frac - 1)
                param_dict[layer.name + '_quant_bias'] = int(np.clip(param_dict[layer.name + '_quant_bias'], min_q, max_q))
            else:
                for c in range(len(bias_frac)):
                    if bias_frac[c] > out_frac:
                        param_dict[layer.name + '_quant_bias'][c] += 1 << (bias_frac[c] - out_frac - 1)
                        while param_dict[layer.name + '_quant_bias'][c] > max_q:
                            print('compensate in layer {}, channel {}'.format(layer.name, c))
                            param_dict[layer.name + '_quant_bias'][c] = param_dict[layer.name + '_quant_bias'][c] / 2
                            bias_frac[c] = bias_frac[c] - 1
                        param_dict[layer.name + '_quant_bias'][c] = round(param_dict[layer.name + '_quant_bias'][c])
                        if param_dict[layer.name + '_quant_bias'][c] > max_q: #in case that it equals to max_q + 1
                            param_dict[layer.name + '_quant_bias'][c] = int(param_dict[layer.name + '_quant_bias'][c] / 2)
                            bias_frac[c] = bias_frac[c] - 1
                param_dict[layer.name + '_quant_bias'] = np.clip(param_dict[layer.name + '_quant_bias'], min_q, max_q)

    return layer_list, param_dict

'''Quantize the network: this is the lite version'''
def quantize_net(layer_list,
                 param_dict,
                 bitwidth,
                 data_generator=None,
                 usr_param_dict=None,
                 method_dict=None,
                 aggressive=True,
                 gaussian_approx=False,
                 factor_num_bin=4,
                 num_step_pre=20,
                 num_step=100,
                 drop_logsoftmax=True,
                 priority_mode='fwb',
                 compensate_floor_on_bias=True):
    add_log('quantize net: bitwidth={}, aggressive={}, drop_logsoftmax={}, priority_mode={}, compensate_floor_on_bias'.format(bitwidth, aggressive, drop_logsoftmax, priority_mode, compensate_floor_on_bias))

    print_log('quantize_net: initializing...')
    add_log('quantize_net: initializing...')
    tic = time.clock()
    if usr_param_dict==None:
        usr_param_dict = {}
    if method_dict==None:
        method_dict = {}
    num_bin = 2**(bitwidth+factor_num_bin)
    if aggressive==False and gaussian_approx==True:
        print_log('quantize_net: Warning: Gaussian approximation is not implemented, will turn to run aggressive mode.')
        add_log('quantize_net: Warning: Gaussian approximation is not implemented, will turn to run aggressive mode.')
        aggressive = True

    print_log('quantize_net: step 1/2 - preprocessing the network, will run it with {} batches whenever necessary.'.format(num_step_pre))
    add_log('quantize_net: step 1/2 - preprocessing the network, will run it with {} batches whenever necessary.'.format(num_step_pre))

    # sort the layer list
    quant_layer_list = sort_layer_list_for_quantize_net(layer_list)

    # modify the negative slope of ReLU layers in quant_layer_list
    for lyr_idx in range(len(quant_layer_list)):
        if quant_layer_list[lyr_idx].type == 'ReLU':
            if quant_layer_list[lyr_idx].negative_slope!=0:
                float_negative_slope = quant_layer_list[lyr_idx].negative_slope
                quant_negative_slope = qfunc.quantized_value(float_negative_slope, bitwidth=8, frac=6, floor=False, signed=True)
                print_log('quantize_net: negative_slope of layer:{} is updated from {:.6e} to {}.'.format(quant_layer_list[lyr_idx].name, float_negative_slope, quant_negative_slope))
                add_log('quantize_net: negative_slope of layer:{} is updated from {:.6e} to {}.'.format(quant_layer_list[lyr_idx].name, float_negative_slope, quant_negative_slope))
                quant_layer_list[lyr_idx].negative_slope = quant_negative_slope

    if type(_QUANTIZATION_LOG_PATH)!=type(None):
        ndk.modelpack.save_to_prototxt(layer_list=quant_layer_list, fname=os.path.join(_QUANTIZATION_LOG_PATH, 'net_to_quantize.prototxt'))

    # collect parameters
    quant_param_dict = format_usr_param_dict(usr_param_dict, layer_list, param_dict)
    quant_param_dict.update(collect_float_point_param(layer_list, param_dict))

    bias_first_mode = False
    if priority_mode in ['wbf', 'bwf']:
        if priority_mode == 'wbf':
            print_log('quantize_net: will determine fractions in order: weight-->bias-->feature.')
            add_log('quantize_net: will determine fractions in order: weight-->bias-->feature.')
        else:
            print_log('quantize_net: will determine fractions in order: bias-->weight-->feature.')
            add_log('quantize_net: will determine fractions in order: bias-->weight-->feature.')
        quant_layer_list, quant_param_dict = directly_determine_weight_fraction(layer_list=quant_layer_list, param_dict=quant_param_dict, bitwidth=bitwidth, method='mse')
        quant_layer_list, quant_param_dict = directly_determine_bias_fraction(layer_list=quant_layer_list, param_dict=quant_param_dict, bitwidth=bitwidth, method='mse')
    elif priority_mode in ['wfb']:
        print_log('quantize_net: will determine fractions in order: weight-->feature-->bias.')
        add_log('quantize_net: will determine fractions in order: weight-->feature-->bias.')
        quant_layer_list, quant_param_dict = directly_determine_weight_fraction(layer_list=quant_layer_list, param_dict=quant_param_dict, bitwidth=bitwidth, method='mse')
    elif priority_mode in ['bfw']:
        print_log('quantize_net: will determine fractions in order: bias-->feature-->weight.')
        add_log('quantize_net: will determine fractions in order: bias-->feature-->weight.')
        quant_layer_list, quant_param_dict = directly_determine_bias_fraction(layer_list=quant_layer_list, param_dict=quant_param_dict, bitwidth=bitwidth, method='mse')
    elif priority_mode in ['fwb']:
        print_log('quantize_net: will determine fractions in order: feature-->weight-->bias.')
        add_log('quantize_net: will determine fractions in order: feature-->weight-->bias.')
        bias_first_mode = False
    elif priority_mode in ['fbw']:
        print_log('quantize_net: will determine fractions in order: feature-->bias-->weight.')
        add_log('quantize_net: will determine fractions in order: feature-->bias-->weight.')
        bias_first_mode = True
    else:
        add_log('quantize_net: unknown priority mode: {}'.format(priority_mode))
        raise ValueError('quantize_net: unknown priority mode: {}'.format(priority_mode))

    # special process for LogSoftmax layer
    if drop_logsoftmax:
        quant_layer_list, logsoftmax_layer = delete_logsoftmax_layer(layer_list=quant_layer_list)
        if type(logsoftmax_layer)!=type(None) and logsoftmax_layer.top in method_dict:
            method_dict.pop(logsoftmax_layer.top)
    else:
        quant_layer_list, quant_param_dict = add_bias_to_previous_layer_for_logsoftmax(layer_list=quant_layer_list, param_dict=quant_param_dict, data_generator=data_generator, num_batch=num_step_pre, bitwidth=bitwidth, log_on=True)

    # build the numpy layers
    print_log('quantize_net: building numpy layers...')
    add_log('quantize_net: building numpy layers...')
    numpy_layers = qnet.build_numpy_layers(quant_layer_list, quant_param_dict)
    quant_param_dict = propagate_fraction_configuration(layer_list=quant_layer_list, param_dict=quant_param_dict, bitwidth=bitwidth)
    quant_param_dict = try_quantize_weight_bias(layer_list=quant_layer_list, param_dict=quant_param_dict, bitwidth=bitwidth, bias_first=bias_first_mode)
    try_set_quant_param_for_numpy_layers(numpy_layers=numpy_layers, layer_list=quant_layer_list, param_dict=quant_param_dict, bitwidth=bitwidth)

    # if need to do MSE or KL optimization for features, count the min/max for them
    minmax_val_dict ={}
    feature_name_list = ndk.layers.get_tensor_name_list(quant_layer_list, feature_only=True)
    features_require_pd = []
    for tsr_name in feature_name_list:
        if tsr_name in method_dict and method_dict[tsr_name].lower()!='minmax':
            features_require_pd.append(tsr_name)
    if len(features_require_pd) > 0:
        print_log('quantize_net: determine feature tensor min and max values using {} batches...'.format(num_step_pre))
        add_log('quantize_net: determine feature tensor min and max values using {} batches...'.format(num_step_pre))
        minmax_val_dict = get_feature_distribution(layer_list=quant_layer_list,
                                                   target_feature_list=features_require_pd,
                                                   numpy_layers=numpy_layers,
                                                   data_generator=data_generator,
                                                   try_run_quant=True,
                                                   bitwidth=bitwidth,
                                                   num_batch=num_step_pre,
                                                   hw_aligned=True,
                                                   num_bin=0,  # set num_bin to zero, so it will only count the max and min values
                                                   minmax_val_dict={},
                                                   log_on=False,
                                                   features_require_pd={})

    # processing
    print_log('quantize_net: step 2/2 - determine tensor fractions using {} batches...'.format(num_step))
    add_log('quantize_net: step 2/2 - determine tensor fractions using {} batches...'.format(num_step))

    if aggressive==True:
        print_log('quantize_net: aggressive mode: get all distributions of the feature tensors at the very beginning.')
        add_log('quantize_net: aggressive mode: get all distributions of the feature tensors at the very beginning.')
        feature_stat_dict = get_feature_distribution(layer_list=quant_layer_list,
                                                     target_feature_list=feature_name_list,
                                                     numpy_layers=numpy_layers,
                                                     data_generator=data_generator,
                                                     try_run_quant=True,
                                                     bitwidth=bitwidth,
                                                     num_batch=num_step,
                                                     hw_aligned=True,
                                                     num_bin=num_bin,
                                                     minmax_val_dict=minmax_val_dict,
                                                     log_on=True,
                                                     features_require_pd=features_require_pd)

        # print_log('quantize_net: for each layer, quantize the output features, then quantize weights and biases.')
        for layer_idx, layer in enumerate(quant_layer_list):
            np_layer = numpy_layers[layer_idx]
            if not np_layer.quantized:
                target_feature_tensor, frac_bound, allow_unsigned = find_next_feature_tensor_to_quantize(start_layer_idx=layer_idx,
                                                                                                         layer_list=quant_layer_list,
                                                                                                         param_dict=quant_param_dict,
                                                                                                         numpy_layers=numpy_layers,
                                                                                                         bitwidth=bitwidth)
                if target_feature_tensor in method_dict:
                    method = method_dict[target_feature_tensor]
                else:
                    method = 'minmax'
                frac, signed = determine_feature_fraction_sign(target_feature_tensor, bitwidth, feature_stat_dict, frac_bound=frac_bound, method=method, allow_unsigned=allow_unsigned)
                quant_param_dict[target_feature_tensor+'_frac'] = frac
                quant_param_dict[target_feature_tensor+'_signed'] = signed
                quant_param_dict = propagate_fraction_configuration(layer_list=quant_layer_list, param_dict=quant_param_dict, bitwidth=bitwidth)
                quant_param_dict = try_quantize_weight_bias(layer_list=quant_layer_list, param_dict=quant_param_dict, bitwidth=bitwidth, bias_first=bias_first_mode)
                try_set_quant_param_for_numpy_layers(numpy_layers=numpy_layers, layer_list=quant_layer_list, param_dict=quant_param_dict, bitwidth=bitwidth)
    else:
        print_log('quantize_net: normal mode: get distributions of the feature tensors according to the partially quantized net.')
        add_log('quantize_net: normal mode: get distributions of the feature tensors according to the partially quantized net.')
        for layer_idx, layer in enumerate(quant_layer_list):
            print_log('quantize_net: #{}/{}: processing layer:{}.'.format(layer_idx+1, len(quant_layer_list), layer.name))
            add_log('quantize_net: #{}/{}: processing layer:{}.'.format(layer_idx+1, len(quant_layer_list), layer.name))
            np_layer = numpy_layers[layer_idx]
            if not np_layer.quantized:
                target_feature_tensor, frac_bound, allow_unsigned = find_next_feature_tensor_to_quantize(start_layer_idx=layer_idx,
                                                                                                         layer_list=quant_layer_list,
                                                                                                         param_dict=quant_param_dict,
                                                                                                         numpy_layers=numpy_layers,
                                                                                                         bitwidth=bitwidth)
                if target_feature_tensor in features_require_pd:
                    cur_features_require_pd = [target_feature_tensor]
                else:
                    cur_features_require_pd = []
                feature_stat_dict = get_feature_distribution(layer_list=quant_layer_list,
                                                             target_feature_list=[target_feature_tensor],
                                                             numpy_layers=numpy_layers,
                                                             data_generator=data_generator,
                                                             try_run_quant=True,
                                                             bitwidth=bitwidth,
                                                             num_batch=num_step,
                                                             hw_aligned=True,
                                                             num_bin=num_bin,
                                                             minmax_val_dict=minmax_val_dict,
                                                             log_on=True,
                                                             features_require_pd=cur_features_require_pd)

                if target_feature_tensor in method_dict:
                    method = method_dict[target_feature_tensor]
                else:
                    method = 'minmax'
                frac, signed = determine_feature_fraction_sign(target_feature_tensor, bitwidth, feature_stat_dict, frac_bound=frac_bound, method=method, allow_unsigned=allow_unsigned)
                # print_log('quantize feature:{} into frac:{} with signed:{}'.format(target_feature_tensor, frac, signed))
                quant_param_dict[target_feature_tensor+'_frac'] = frac
                quant_param_dict[target_feature_tensor+'_signed'] = signed
                quant_param_dict = propagate_fraction_configuration(layer_list=quant_layer_list, param_dict=quant_param_dict, bitwidth=bitwidth)
                quant_param_dict = try_quantize_weight_bias(layer_list=quant_layer_list, param_dict=quant_param_dict, bitwidth=bitwidth, bias_first=bias_first_mode)
                try_set_quant_param_for_numpy_layers(numpy_layers=numpy_layers, layer_list=quant_layer_list, param_dict=quant_param_dict, bitwidth=bitwidth)

        # raise ValueError('Not implemented')
    if ENABLE_FLOOR_FEATURE[bitwidth] and compensate_floor_on_bias:
        quant_layer_list, quant_param_dict = compensate_output_feature_on_bias(quant_layer_list, quant_param_dict, bitwidth)

    print_log('quantize_net: quantization completed, eplased time: {:.2f} seconds '.format(time.clock()-tic))
    add_log('quantize_net: quantization completed, eplased time: {:.2f} seconds '.format(time.clock()-tic))

    return quant_layer_list, quant_param_dict


if __name__=='__main__':

    bins = np.linspace(-1,1,11)
    float_prob = np.array([0.04, 0.06, 0.08, 0.10, 0.12, 0.2, 0.12, 0.10, 0.08, 0.06, 0.04])
    print(sum(float_prob))
    print(bins)
    print(float_prob)

    quant_probs = quantize_prob_distribution(bin_vals=bins, float_prob=float_prob, bitwitdh=8, fraction=1, signed=True)
    print(quant_probs)


    # g = ndk.examples.data_generator_mnist.data_generator_mnist(mnist_dirname=r'C:\Users\51740\Desktop\XiTang-AI\K3DAITool\ndk\examples\mnist',
    #                                                            batch_size=5, random_order=True, use_test_set=False)
    #
    # layer_list, param_dict = ndk.modelpack.load_from_file(fname_prototxt=r'C:\Users\51740\Desktop\XiTang-AI\K3DAITool\qcdnet.prototxt',
    #                                                       fname_npz=r'C:\Users\51740\Desktop\XiTang-AI\K3DAITool\qcdnet.npz')
    # # sorted_layer_list = sort_layer_list_for_quantize_net(layer_list)
    # # for idx, lyr in enumerate(sorted_layer_list):
    # #     print('layer#{}: {}, {}'.format(idx, lyr.name, lyr.dist))
    # #     # print(lyr.name)
    #
    # quant_layer_list, quant_param_dict = quantize_net(layer_list=layer_list,
    #                param_dict=param_dict,
    #                bitwidth=8,
    #                data_generator=g,
    #                usr_param_dict={},
    #                method_dict={'pool1': 'JS'},
    #                aggressive=True,
    #                gaussian_approx=False,
    #                factor_num_bin=4,
    #                num_step_pre=20,
    #                num_step=100)
    # model_name = 'test'
    #
    # print('Save to files...')
    # ndk.modelpack.save_to_file(layer_list=quant_layer_list,
    #                            fname_prototxt=model_name,
    #                            param_dict=quant_param_dict,
    #                            fname_npz=model_name)
    #
    # print('Export to binary image...')
    # ndk.modelpack.modelpack_from_file(bitwidth=8,
    #                                   fname_prototxt=model_name,
    #                                   fname_npz=model_name,
    #                                   out_file_path=model_name,
    #                                   model_name=model_name)
    #
    # print('done')

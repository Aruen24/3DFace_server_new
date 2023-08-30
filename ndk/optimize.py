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
import os
import sys
import datetime
ndk_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ndk_dir)
import numpy as np
import copy
import ndk
from ndk.layers import get_dict_name_to_bottom, sort_layers, Layer, multi_input_layer, get_adjacent_layer_dict, get_tensor_shape

"""
to merge batchnorm, scale and bias operation into linear operations before, linear operations are convolution, innerproduct, batchnorm, scale and bias
"""
def merge_bn_scale_after_linear_op(layer_list, param_dict, fname_log=None):
    print("merge_bn_scale_after_linear_op start!")
    if fname_log != None:
        with open(fname_log, 'a') as f:
            f.write("merge_bn_scale_after_linear_op at time {}\n".format(datetime.datetime.now().strftime('%b_%d_%y_%H_%M_%S')))
    linear_op = ["Convolution", "InnerProduct", "BatchNorm", "Scale", "Bias"]
    to_be_merged_op = ["BatchNorm", "Scale", "Bias"]

    unmerged_layer_list = copy.deepcopy(layer_list)
    unmerged_layer_list = sort_layers(unmerged_layer_list)
    assert unmerged_layer_list[0].type == "Input", "First layer should be of type Input, but got {} instead".format(unmerged_layer_list[0].type)
    dict_name_to_bot = get_dict_name_to_bottom(layer_list)

    merged_layer_list = []
    merged_param_dict = {}

    while len(unmerged_layer_list) > 0:
        layer = unmerged_layer_list.pop(0)
        print("get layer {}".format(layer.name))
        if layer.type in to_be_merged_op and len(dict_name_to_bot[layer.bottom]) == 1 and layer.bottom == merged_layer_list[-1].top and merged_layer_list[-1].type in linear_op:
            if fname_log != None:
                with open(fname_log, 'a') as f:
                    f.write("layer {} was merged into {}\n".format(layer.name, merged_layer_list[-1].name))
            else:
                print("layer {} was merged into {}".format(layer.name, merged_layer_list[-1].name))
            weight_linear_op = merged_param_dict[merged_layer_list[-1].name + "_weight"] * 1.0 if merged_layer_list[-1].name + "_weight" in merged_param_dict.keys() else np.array([1])
            bias_linear_op = merged_param_dict[merged_layer_list[-1].name + "_bias"] * 1.0 if merged_layer_list[-1].name + "_bias" in merged_param_dict.keys() else np.array([0])
            weight_to_merge = param_dict[layer.name + "_weight"] * 1.0 if layer.name + "_weight" in param_dict.keys() else np.array([1])
            bias_to_merge = param_dict[layer.name + "_bias"] * 1.0 if layer.name + "_bias" in param_dict.keys() else np.array([0])
            if merged_layer_list[-1].type == "Bias":
                if layer.type == "Bias":
                    merged_param_dict[merged_layer_list[-1].name + "_bias"] = bias_linear_op + bias_to_merge
                else:
                    merged_param_dict[merged_layer_list[-1].name + "_bias"] = bias_linear_op * weight_to_merge + bias_to_merge
                    merged_param_dict[merged_layer_list[-1].name + "_weight"] = weight_to_merge
                    merged_layer_list[-1].type = "Scale"
                    merged_layer_list[-1].bias_term = True
            elif merged_layer_list[-1].type in ["BatchNorm", "Scale"]:
                merged_param_dict[merged_layer_list[-1].name + "_weight"] = weight_linear_op * weight_to_merge
                merged_param_dict[merged_layer_list[-1].name + "_bias"] = bias_linear_op * weight_to_merge + bias_to_merge
                merged_layer_list[-1].type = "Scale"
                merged_layer_list[-1].bias_term = True
            else:
                merged_param_dict[merged_layer_list[-1].name + "_bias"] = bias_linear_op * weight_to_merge + bias_to_merge
                if len(weight_to_merge) == 1:
                    merged_param_dict[merged_layer_list[-1].name + "_weight"] = weight_linear_op * weight_to_merge
                else:
                    channel, _, _, _ = weight_linear_op.shape
                    for i in range(channel):
                        weight_linear_op[i, :, :, :] = weight_linear_op[i, :, :, :] * weight_to_merge[i]
                    merged_param_dict[merged_layer_list[-1].name + "_weight"] = weight_linear_op
                merged_layer_list[-1].bias_term = True
            merged_layer_list[-1].top = copy.deepcopy(layer.top)
        else:
            if layer.name + "_bias" in param_dict.keys():
                merged_param_dict[layer.name + "_bias"] = param_dict[layer.name + "_bias"] * 1.0
            if layer.name + "_weight" in param_dict.keys():
                merged_param_dict[layer.name + "_weight"] = param_dict[layer.name + "_weight"] * 1.0
            merged_layer_list.append(layer)
    print("merge_bn_scale_after_linear_op end!")
    return merged_layer_list, merged_param_dict

def merge_bn_scale_before_conv_ip(layer_list, param_dict, fname_log=None):
    print("merge_bn_scale_before_conv_ip start!")
    if fname_log != None:
        with open(fname_log, 'a') as f:
            f.write("merge_bn_scale_after_linear_op at time {}\n".format(datetime.datetime.now().strftime('%b_%d_%y_%H_%M_%S')))
    to_be_merged_op = ["BatchNorm", "Scale", "Bias"]

    unmerged_layer_list = copy.deepcopy(layer_list)
    unmerged_layer_list = sort_layers(unmerged_layer_list)
    assert unmerged_layer_list[0].type == "Input", "First layer should be of type Input, but got {} instead".format(unmerged_layer_list[0].type)
    dict_name_to_bot = get_dict_name_to_bottom(layer_list)

    merged_layer_list = []
    merged_param_dict = {}
    
    while len(unmerged_layer_list) > 0:
        layer = unmerged_layer_list.pop(0)
        print("get layer {}".format(layer.name))
        if layer.type in ["Convolution", "InnerProduct"] and len(dict_name_to_bot[layer.bottom]) == 1 and layer.bottom == merged_layer_list[-1].top and merged_layer_list[-1].type in to_be_merged_op:
            if fname_log != None:
                with open(fname_log, 'a') as f:
                    f.write("layer {} was merged into {}\n".format(merged_layer_list[-1].name, layer.name))
            else:
                print("layer {} was merged into {}".format(merged_layer_list[-1].name, layer.name))
            layer.bias_term = merged_layer_list[-1].name + "_bias" in merged_param_dict.keys() or layer.name + "_bias" in param_dict.keys()
            layer.bottom = copy.deepcopy(merged_layer_list[-1].bottom)

            weight_conv_ip = param_dict[layer.name + "_weight"] * 1.0
            c_o, c_i, h, w = weight_conv_ip.shape
            g = 1 if layer.type == "InnerProduct" else int(layer.group)
            bias_conv_ip = param_dict[layer.name + "_bias"] if layer.name + "_bias" in param_dict.keys() else np.zeros(c_i * g)
            weight_to_merge = merged_param_dict[merged_layer_list[-1].name + "_weight"] if merged_layer_list[-1].name + "_weight" in merged_param_dict.keys() else np.ones(c_i * g)
            bias_to_merge = merged_param_dict[merged_layer_list[-1].name + "_bias"] if merged_layer_list[-1].name + "_bias" in merged_param_dict.keys() else np.zeros(c_i * g)
            if weight_to_merge.size == 1:
                weight_to_merge = weight_to_merge * np.ones(c_i)
            if bias_to_merge.size == 1:
                bias_to_merge = bias_to_merge * np.ones(c_i)
            weight_conv_ip_group = np.split(weight_conv_ip, g)
            bias_conv_ip_group = np.split(bias_conv_ip, g)
            weight_to_merge_group = np.split(weight_to_merge, g)
            bias_to_merge_group = np.split(bias_to_merge, g)
            merged_weight_group = []
            merged_bias_group = []
            for i in range(g):
                weight_conv_mul_bias_bn = weight_conv_ip_group[i] * 1.0
                weight_conv_mul_weight_bn = weight_conv_ip_group[i] * 1.0
                for c in range(c_i):
                    weight_conv_mul_bias_bn[:, c, :, :] = weight_conv_mul_bias_bn[:, c, :, :] * bias_to_merge_group[i][c]
                    weight_conv_mul_weight_bn[:, c, :, :] = weight_conv_mul_weight_bn[:, c, :, :] * weight_to_merge_group[i][c]
                merged_weight_group.append(weight_conv_mul_weight_bn)
                merged_bias_group.append(bias_conv_ip_group[i] + np.sum(weight_conv_mul_bias_bn, axis = (1, 2, 3)))
            merged_param_dict[layer.name + "_weight"] = np.concatenate(merged_weight_group)
            merged_param_dict[layer.name + "_bias"] = np.concatenate(merged_bias_group)
            
            if merged_layer_list[-1].name + "_weight" in merged_param_dict.keys():
                merged_param_dict.pop(merged_layer_list[-1].name + "_weight")
            if merged_layer_list[-1].name + "_bias" in merged_param_dict.keys():
                merged_param_dict.pop(merged_layer_list[-1].name + "_bias")
                
            merged_layer_list[-1] = layer
        else:
            if layer.name + "_bias" in param_dict.keys():
                merged_param_dict[layer.name + "_bias"] = param_dict[layer.name + "_bias"] * 1.0
            if layer.name + "_weight" in param_dict.keys():
                merged_param_dict[layer.name + "_weight"] = param_dict[layer.name + "_weight"] * 1.0
            merged_layer_list.append(layer)
    print("merge_bn_scale_before_conv_ip end!")
    return merged_layer_list, merged_param_dict

def merge_layers(layer_list, param_dict, fname_log=None):
    merged_layer_list_step0, merged_param_dict_step0 = merge_bn_scale_after_linear_op(layer_list, param_dict, fname_log)
    merged_layer_list, merged_param_dict = merge_bn_scale_before_conv_ip(merged_layer_list_step0, merged_param_dict_step0, fname_log)
    return merged_layer_list, merged_param_dict

def add_pre_norm(layer_list, param_dict, weight, bias):
    assert isinstance(weight, np.ndarray), 'weight should be a numpy.ndarray, but got {} instead'.format(type(weight))
    assert isinstance(bias, np.ndarray), 'bias should be a numpy.ndarray, but got {} instead'.format(type(bias))
    assert len(weight.shape) == 1, 'weight should be an 1D array, but got a {}D array instead'.format(len(weight.shape))
    assert len(bias.shape) == 1, 'bias should be an 1D array, but got a {}D array instead'.format(len(bias.shape))
    assert layer_list[0].type == 'Input', '1st layer type should be "Input", but got {} instead'.format(layer_list[0].type)
    input_channel = layer_list[0].dim[1]
    input_tensor_name = layer_list[0].top
    assert weight.shape[0] == input_channel, 'length of weight should be equal to input channel, but got weight length {} and input channel {}'.format(weight.shape[0], input_channel)
    assert bias.shape[0] == input_channel, 'length of bias should be equal to input channel, but got bias length {} and input channel {}'.format(bias.shape[0], input_channel)
    # layer_name_list = [layer.name for layer in layer_list]
    # tensor_name_list = [layer.top for layer in layer_list] + [layer.bottom for layer in layer_list]
    layer_name_list = ndk.layers.get_layer_name_list(layer_list)
    tensor_name_list = ndk.layers.get_tensor_name_list(layer_list, feature_only=True)
    pre_norm_name = 'pre_norm'
    pre_norm_bottom = 'pre_norm:0'
    i = 1
    while pre_norm_name in layer_name_list or pre_norm_bottom in tensor_name_list:
        pre_norm_name = 'pre_norm_{}'.format(i)
        pre_norm_bottom = 'pre_norm_{}:0'.format(i)
        i = i + 1
    pre_norm_layer = Layer(lyr_type = 'batchnorm', name = pre_norm_name, bottom = pre_norm_bottom, top = input_tensor_name)
    layer_list.insert(1, pre_norm_layer)
    layer_list[0].top = pre_norm_bottom
    param_dict[pre_norm_name + '_weight'] = weight * 1.0
    param_dict[pre_norm_name + '_bias'] = bias * 1.0

def add_identity_between_se_and_multi_input_layer(layer_list, param_dict):
    print('add_identity_between_se_and_multi_input_layer start')
    result_layer_list = []
    result_param_dict = copy.deepcopy(param_dict)
    adjacent_layer_dict = get_adjacent_layer_dict(layer_list)
    tensor_shape = get_tensor_shape(layer_list)
    for layer in layer_list:
        print(layer)
        layer_copy = copy.deepcopy(layer)
        adjacent_layers = adjacent_layer_dict[layer.name]
        adjacent_layer_types = [l.type for l in adjacent_layers]
        need_add = False
        if layer.type == "ScaleByTensor":
            for type_ in adjacent_layer_types:
                if type_ in multi_input_layer:
                    need_add = True
                    break
        if need_add:
            c = tensor_shape[layer.top][1]
            layer_copy.top = layer.top + "_identity:0"
            identity_layer_name = layer.name + "_identity"
            identity_layer = Layer(lyr_type = 'scale', name = identity_layer_name, top = layer.top, bottom = layer_copy.top, bias_term = False)
            result_param_dict[identity_layer_name + "_weight"] = np.ones(c)
            result_layer_list.append(layer_copy)
            result_layer_list.append(identity_layer)
        else:
            result_layer_list.append(layer_copy)
    return result_layer_list, result_param_dict
    

if __name__=='__main__':
    print('Not implemented')

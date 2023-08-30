#!/usr/bin/env python3
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
import math
import sys
import os
modelpack_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(modelpack_dir)
def get_size_occupying(channel, height, width, byte = 1, opti_mode = False):
    row_size_base = int(32 / byte)
    if opti_mode:
        row_size_base = int(row_size_base / 2)
    row_size = int((width + row_size_base - 1) / row_size_base) * 32
    return channel * height * row_size

def read_tensors(layers_merge, net_configs, input_name = '', byte = 1):
    dict_to_layer_bottom = {}
    dict_to_layer_top = {}
    tensor_size = {}
    tensor_shape = {}
    input_shape = []

    if net_configs[0] == '':
        net_configs.pop(0)
    if input_name == '':
        for config in net_configs[0].split("\n"):
            config = config.strip()
            if config.find('input:') == 0:
                input_name = config.split('"')[1]
            if config.find('input_dim:') == 0:
                input_shape.append(int(config.split(':')[1]))
            if config.find('dim:') == 0:
                input_shape.append(int(config.split(':')[1]))
        if len(input_shape) == 4:
            tensor_shape[input_name] = input_shape
            tensor_size[input_name] = get_size_occupying(input_shape[-3], input_shape[-2], input_shape[-1], byte = byte)

    first_layer = layers_merge[0]
    if first_layer.type == ['input']:
        layers_merge.pop(0)
        input_name = first_layer.top[0]
        input_shape = first_layer.dim
        tensor_shape[input_name] = input_shape
        tensor_size[input_name] = get_size_occupying(input_shape[-3], input_shape[-2], input_shape[-1], byte = byte)

    for layer1 in layers_merge:
        print(layer1.name, layer1.type, layer1.bottom, layer1.top)
        for bottom in layer1.bottom:
            if bottom in dict_to_layer_bottom.keys():
                dict_to_layer_bottom[bottom].append(layer1)
            else:
                dict_to_layer_bottom[bottom] = [layer1]
        for top in layer1.top:
            if top in dict_to_layer_top.keys():
                dict_to_layer_top[top].append(layer1)
            else:
                dict_to_layer_top[top] = [layer1]
            dim_o = [0] * 4
            k_h = layer1.kernel_size if layer1.kernel_size_h == 0 else layer1.kernel_size_h
            k_w = layer1.kernel_size if layer1.kernel_size_w == 0 else layer1.kernel_size_w
            k_h_d = layer1.dilation * (k_h - 1) + 1
            k_w_d = layer1.dilation * (k_w - 1) + 1
            if 'convolution' in layer1.type:
                dim_i = tensor_shape[layer1.bottom[0]]
                dim_o[0] = dim_i[0]
                dim_o[1] = layer1.num_output
                dim_o[2] = int((dim_i[2] + layer1.pad_u + layer1.pad_d - k_h_d) / layer1.stride) + 1
                dim_o[3] = int((dim_i[3] + layer1.pad_l + layer1.pad_r - k_w_d) / layer1.stride) + 1
                tensor_shape[top] = dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif "tf_conv" in layer1.type:
                dim_i = tensor_shape[layer1.bottom[0]]
                dim_o[0] = dim_i[0]
                dim_o[1] = layer1.num_output
                if layer1.pad_u > 0:
                    dim_o[2] = int(math.ceil(dim_i[2] / layer1.stride))
                    dim_o[3] = int(math.ceil(dim_i[3] / layer1.stride))
                    h_i_need = (dim_o[2] - 1) * layer1.stride + k_h_d
                    w_i_need = (dim_o[3] - 1) * layer1.stride + k_w_d
                    layer1.pad_u = int((h_i_need - dim_i[2]) / 2)
                    layer1.pad_d = int((h_i_need - dim_i[2] + 1) / 2)
                    layer1.pad_l = int((w_i_need - dim_i[3]) / 2)
                    layer1.pad_r = int((w_i_need - dim_i[3] + 1) / 2)
                else:
                    dim_o[2] = int((dim_i[2] + k_h_d) / layer1.stride) + 1
                    dim_o[3] = int((dim_i[3] + k_w_d) / layer1.stride) + 1
                tensor_shape[top] = dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif 'pooling' in layer1.type:
                dim_i = tensor_shape[layer1.bottom[0]]
                dim_o[0] = dim_i[0]
                dim_o[1] = dim_i[1]
                if layer1.global_pooling:
                    dim_o[2] = 1
                    dim_o[3] = 1
                else:
                    if layer1.ceil == 0:
                        dim_o[2] = int((dim_i[2] + layer1.pad_u + layer1.pad_d - k_h_d) / layer1.stride) + 1
                        dim_o[3] = int((dim_i[3] + layer1.pad_l + layer1.pad_r - k_w_d) / layer1.stride) + 1
                    else:
                        dim_o[2] = int(math.ceil((dim_i[2] + layer1.pad_u + layer1.pad_d - k_h_d) / layer1.stride)) + 1
                        dim_o[3] = int(math.ceil((dim_i[3] + layer1.pad_l + layer1.pad_r - k_w_d) / layer1.stride)) + 1
                        if (dim_o[2] - 1) * layer1.stride >= dim_i[2] + layer1.pad_u:
                            dim_o[2] -= 1
                        if (dim_o[3] - 1) * layer1.stride >= dim_i[3] + layer1.pad_l:
                            dim_o[3] -= 1
                tensor_shape[top] = dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif 'tf_pool' in layer1.type:
                dim_i = tensor_shape[layer1.bottom[0]]
                dim_o[0] = dim_i[0]
                dim_o[1] = dim_i[1]
                if layer1.global_pooling:
                    dim_o[2] = 1
                    dim_o[3] = 1
                elif layer1.pad_u == 0:
                    dim_o[2] = int((dim_i[2] + k_h_d) / layer1.stride) + 1
                    dim_o[3] = int((dim_i[3] + k_w_d) / layer1.stride) + 1
                else:
                    dim_o[2] = int(math.ceil((dim_i[2] / layer1.stride)))
                    dim_o[3] = int(math.ceil((dim_i[3] / layer1.stride)))
                    h_i_need = (dim_o[2] - 1) * layer1.stride + k_h_d
                    w_i_need = (dim_o[3] - 1) * layer1.stride + k_w_d
                    layer1.pad_u = int((h_i_need - dim_i[2]) / 2)
                    layer1.pad_d = int((h_i_need - dim_i[2] + 1) / 2)
                    layer1.pad_l = int((w_i_need - dim_i[3]) / 2)
                    layer1.pad_r = int((w_i_need - dim_i[3] + 1) / 2)
                tensor_shape[top] = dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif 'slice' in layer1.type:
                index = layer1.top.index(top)
                dim_i = tensor_shape[layer1.bottom[0]]
                dim_o[0] = dim_i[0]
                if len(layer1.slice_point) == 0:
                    dim_o[1] = int(dim_i[1] / len(layer1.top))
                else:
                    dim_o[1] = dim_i[1] - layer1.slice_point[0] * (len(layer1.top) - 1) if index + 1 == len(layer1.top) else layer1.slice_point[0]
                dim_o[2] = dim_i[2]
                dim_o[3] = dim_i[3]
                tensor_shape[top] = dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif 'concat' in layer1.type:
                dim_i = tensor_shape[layer1.bottom[0]]
                dim_o = []
                for i in range(len(dim_i)):
                    dim_o.append(dim_i[i] if i != layer1.axis else sum([tensor_shape[bottom][i] for bottom in layer1.bottom]))
                tensor_shape[top] = dim_o
                dim_o = [1, 1, 1, 1] + dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif 'rms' in layer1.type:
                dim_o[0] = 1
                dim_o[1] = 1
                dim_o[2] = 1
                dim_o[3] = 1
                tensor_shape[top] = dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif 'scale' in layer1.type or 'batchnorm' in layer1.type or 'factornormalize' in layer1.type:
                dim_i = tensor_shape[layer1.bottom[0]]
                dim_o[0] = dim_i[0]
                dim_o[1] = dim_i[1]
                dim_o[2] = dim_i[2]
                dim_o[3] = dim_i[3]
                tensor_shape[top] = dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif 'innerproduct' in layer1.type:
                dim_i = tensor_shape[layer1.bottom[0]]
                dim_o[0] = dim_i[0]
                dim_o[1] = layer1.num_output
                dim_o[2] = 1
                dim_o[3] = 1
                tensor_shape[top] = dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif 'prelu' in layer1.type:
                tensor_shape[top] = tensor_shape[layer1.bottom[0]]
                tensor_size[top] = tensor_size[layer1.bottom[0]]
            elif 'permute' in layer1.type:
                dim_i = tensor_shape[layer1.bottom[0]]
                dim_o = []
                for i in range(len(dim_i)):
                    dim_o.append(dim_i[layer1.order[i]])
                tensor_shape[top] = dim_o
                dim_o = [1, 1, 1, 1] + dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif 'flatten' in layer1.type:
                dim_i = tensor_shape[layer1.bottom[0]]
                size_last_dim = 1
                dim_o = []
                for i in range(len(dim_i) - layer1.axis):
                    size_last_dim *= dim_i[i + layer1.axis]
                for i in range(layer1.axis):
                    dim_o.append(dim_i[i])
                dim_o.append(size_last_dim)
                tensor_shape[top] = dim_o
                dim_o = [1, 1, 1, 1] + dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif 'reshape' in layer1.type:
                dim_i = tensor_shape[layer1.bottom[0]]
                total_size = 1
                total_size_m1 = 1
                for i in range(len(dim_i)):
                    total_size *= dim_i[i]
                for i in range(len(layer1.dim)):
                    total_size_m1 *= layer1.dim[i] if layer1.dim[i] > 0 else (dim_i[i] if layer1.dim[i] == 0 else 1)
                dim_o = []
                for i in range(len(layer1.dim)):
                    dim_o.append(int(total_size / total_size_m1) if layer1.dim[i] < 0 else (layer1.dim[i] if layer1.dim[i] > 0 else dim_i[i]))
                tensor_shape[top] = dim_o
                dim_o = [1, 1, 1, 1] + dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif 'priorbox' in layer1.type:
                print("Warning: unsupported layer type {}, you will need to write your own C code to finish this layer".format(layer1.type))
                dim_i = tensor_shape[layer1.bottom[0]]
                dim_o = [1, 2, 4 * (2 * len(layer1.aspect_ratio) + 2) * dim_i[-1] * dim_i[-2]]
                tensor_shape[top] = dim_o
                dim_o = [1, 1, 1, 1] + dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif 'eltwise' in layer1.type:
                assert len(layer1.bottom) > 1, "An eltwise layer should have at least 2 input tensors, but got {} in layer {}".format(layer1.bottom, layer1.name)
                for i in range(len(layer1.bottom) - 1):
                    assert tensor_shape[layer1.bottom[i]] == tensor_shape[layer1.bottom[i + 1]], "An eltwise layer's all input tensors should have the same shape, but got {}'s shape is {} while {}'s shape is {} in layer {}".format(layer1.bottom[i], tensor_shape[layer1.bottom[i]], layer1.bottom[i + 1], tensor_shape[layer1.bottom[i + 1]], layer1.name)
                dim_i = tensor_shape[layer1.bottom[0]]
                dim_o = []
                for i in range(len(dim_i)):
                    dim_o.append(dim_i[i])
                tensor_shape[top] = dim_o
                dim_o = [1, 1, 1, 1] + dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif 'scalebytensor' in layer1.type:
                dim = tensor_shape[layer1.bottom[0]]
                tensor_shape[top] = [dim[0], dim[1], dim[2], dim[3]]
                tensor_size[top] = tensor_size[layer1.bottom[0]]
            elif 'input' not in layer1.type:
                dim_i = tensor_shape[layer1.bottom[0]]
                dim_o = []
                for i in range(len(dim_i)):
                    dim_o.append(dim_i[i])
                tensor_shape[top] = dim_o
                dim_o = [1, 1, 1, 1] + dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
    return tensor_size, dict_to_layer_bottom, dict_to_layer_top, tensor_shape, input_name

def check_tensor_shape(tensor_shape):
    assert type(tensor_shape) == dict, "tensor_shape should be a dict"
    tensor_shape = dict(tensor_shape)
    for key, value in tensor_shape.items():
        assert type(value) == list, "tensor's shape should be a list, but {}'s is of type {}".format(key, type(value))
        for dim in value:
            assert dim > 0, "tensor's shape should be non-negative, but {}'s shape is {}".format(key, value)
            assert dim <= 65536, "tensor's size on every dimension should be no larger than 65536, but {}'s shape is {}".format(key, value)


def read_tensors_and_input(layers_merge, byte = 1):
    dict_to_layer_bottom = {}
    dict_to_layer_top = {}
    tensor_size = {}
    tensor_shape = {}
    input_shape = []
    layers_top_used_time = {}
    first_layer = layers_merge[0]
    if 'input' not in first_layer.type:
        raise Exception("input node should be the 1st layer")
    dim = first_layer.dim
    if len(dim) != 4:
        raise Exception("input should be 4-dimension")
    input_name = first_layer.top[0]
    input_shape = first_layer.dim
    tensor_shape[input_name] = input_shape
    tensor_size[input_name] = get_size_occupying(input_shape[-3], input_shape[-2], input_shape[-1], byte = byte)
    for layer1 in layers_merge:
        if len(layer1.bottom) > 0:
            if len(layer1.bottom) > 1 or len(layer1.top) == 0 or layer1.bottom[0] != layer1.top[0]:
                for bottom in layer1.bottom:
                    if bottom in layers_top_used_time.keys():
                        layers_top_used_time[bottom] += 1
                    else:
                        layers_top_used_time[bottom] = 1
        for bottom in layer1.bottom:
            if bottom in dict_to_layer_bottom.keys():
                dict_to_layer_bottom[bottom].append(layer1)
            else:
                dict_to_layer_bottom[bottom] = [layer1]
        for top in layer1.top:
            if top in dict_to_layer_top.keys():
                dict_to_layer_top[top].append(layer1)
            else:
                dict_to_layer_top[top] = [layer1]
            dim_o = [0] * 4
            k_h = layer1.kernel_size if layer1.kernel_size_h == 0 else layer1.kernel_size_h
            k_w = layer1.kernel_size if layer1.kernel_size_w == 0 else layer1.kernel_size_w
            k_h_d = layer1.dilation * (k_h - 1) + 1
            k_w_d = layer1.dilation * (k_w - 1) + 1
            if 'convolution' in layer1.type:
                dim_i = tensor_shape[layer1.bottom[0]]
                dim_o[0] = dim_i[0]
                dim_o[1] = layer1.num_output
                dim_o[2] = int((dim_i[2] + layer1.pad_u + layer1.pad_d - k_h_d) / layer1.stride) + 1
                dim_o[3] = int((dim_i[3] + layer1.pad_l + layer1.pad_r - k_w_d) / layer1.stride) + 1
                tensor_shape[top] = dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif "tf_conv" in layer1.type:
                dim_i = tensor_shape[layer1.bottom[0]]
                dim_o[0] = dim_i[0]
                dim_o[1] = layer1.num_output
                dim_o[2] = int(math.ceil(dim_i[2] / layer1.stride))
                dim_o[3] = int(math.ceil(dim_i[3] / layer1.stride))
                tensor_shape[top] = dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif 'pooling' in layer1.type:
                dim_i = tensor_shape[layer1.bottom[0]]
                dim_o[0] = dim_i[0]
                dim_o[1] = dim_i[1]
                if layer1.global_pooling:
                    dim_o[2] = 1
                    dim_o[3] = 1
                else:
                    dim_o[2] = int((dim_i[2] + layer1.pad_u + layer1.pad_d - k_h_d) / layer1.stride) + 1
                    dim_o[3] = int((dim_i[3] + layer1.pad_l + layer1.pad_r - k_w_d) / layer1.stride) + 1
                    if layer1.ceil > 0:
                        if (dim_o[2] - 1) * layer1.stride < dim_i[2] + layer1.pad_u:
                            dim_o[2] += 1
                        if (dim_o[3] - 1) * layer1.stride < dim_i[3] + layer1.pad_l:
                            dim_o[3] += 1
                tensor_shape[top] = dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif 'tf_pool' in layer1.type:
                dim_i = tensor_shape[layer1.bottom[0]]
                dim_o[0] = dim_i[0]
                dim_o[1] = dim_i[1]
                if layer1.global_pooling:
                    dim_o[2] = 1
                    dim_o[3] = 1
                elif layer1.pad_u == 0:
                    dim_o[2] = int((dim_i[2] + k_h_d) / layer1.stride) + 1
                    dim_o[3] = int((dim_i[3] + k_w_d) / layer1.stride) + 1
                else:
                    dim_o[2] = int(math.ceil((dim_i[2] / layer1.stride)))
                    dim_o[3] = int(math.ceil((dim_i[3] / layer1.stride)))
                tensor_shape[top] = dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif 'slice' in layer1.type:
                index = layer1.top.index(top)
                dim_i = tensor_shape[layer1.bottom[0]]
                dim_o[0] = dim_i[0]
                if len(layer1.slice_point) == 0:
                    dim_o[1] = int(dim_i[1] / len(layer1.top))
                else:
                    dim_o[1] = dim_i[1] - layer1.slice_point[0] * (len(layer1.top) - 1) if index + 1 == len(layer1.top) else layer1.slice_point[0]
                dim_o[2] = dim_i[2]
                dim_o[3] = dim_i[3]
                tensor_shape[top] = dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif 'concat' in layer1.type:
                dim_i = tensor_shape[layer1.bottom[0]]
                dim_o = []
                for i in range(len(dim_i)):
                    dim_o.append(dim_i[i] if i != layer1.axis else sum([tensor_shape[bottom][i] for bottom in layer1.bottom]))
                tensor_shape[top] = dim_o
                dim_o = [1, 1, 1, 1] + dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif 'rms' in layer1.type:
                dim_o[0] = 1
                dim_o[1] = 1
                dim_o[2] = 1
                dim_o[3] = 1
                tensor_shape[top] = dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif 'scale' in layer1.type or 'batchnorm' in layer1.type or 'factornormalize' in layer1.type:
                dim_i = tensor_shape[layer1.bottom[0]]
                dim_o[0] = dim_i[0]
                dim_o[1] = dim_i[1]
                dim_o[2] = dim_i[2]
                dim_o[3] = dim_i[3]
                tensor_shape[top] = dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif 'innerproduct' in layer1.type:
                dim_i = tensor_shape[layer1.bottom[0]]
                dim_o[0] = dim_i[0]
                dim_o[1] = layer1.num_output
                dim_o[2] = 1
                dim_o[3] = 1
                tensor_shape[top] = dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif 'prelu' in layer1.type:
                tensor_shape[top] = tensor_shape[layer1.bottom[0]]
                tensor_size[top] = tensor_size[layer1.bottom[0]]
            elif 'permute' in layer1.type:
                dim_i = tensor_shape[layer1.bottom[0]]
                dim_o = []
                for i in range(len(dim_i)):
                    dim_o.append(dim_i[layer1.order[i]])
                tensor_shape[top] = dim_o
                dim_o = [1, 1, 1, 1] + dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif 'flatten' in layer1.type:
                dim_i = tensor_shape[layer1.bottom[0]]
                size_last_dim = 1
                dim_o = []
                for i in range(len(dim_i) - layer1.axis):
                    size_last_dim *= dim_i[i + layer1.axis]
                for i in range(layer1.axis):
                    dim_o.append(dim_i[i])
                dim_o.append(size_last_dim)
                tensor_shape[top] = dim_o
                dim_o = [1, 1, 1, 1] + dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif 'reshape' in layer1.type:
                dim_i = tensor_shape[layer1.bottom[0]]
                total_size = 1
                total_size_m1 = 1
                for i in range(len(dim_i)):
                    total_size *= dim_i[i]
                for i in range(len(layer1.dim)):
                    total_size_m1 *= layer1.dim[i] if layer1.dim[i] > 0 else (dim_i[i] if layer1.dim[i] == 0 else 1)
                dim_o = []
                for i in range(len(layer1.dim)):
                    dim_o.append(int(total_size / total_size_m1) if layer1.dim[i] < 0 else (layer1.dim[i] if layer1.dim[i] > 0 else dim_i[i]))
                tensor_shape[top] = dim_o
                dim_o = [1, 1, 1, 1] + dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif 'priorbox' in layer1.type:
                dim_i = tensor_shape[layer1.bottom[0]]
                dim_o = [1, 2, 4 * (2 * len(layer1.aspect_ratio) + 2) * dim_i[-1] * dim_i[-2]]
                tensor_shape[top] = dim_o
                dim_o = [1, 1, 1, 1] + dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
            elif 'input' not in layer1.type:
                dim_i = tensor_shape[layer1.bottom[0]]
                dim_o = []
                for i in range(len(dim_i)):
                    dim_o.append(dim_i[i])
                tensor_shape[top] = dim_o
                dim_o = [1, 1, 1, 1] + dim_o
                tensor_size[top] = get_size_occupying(dim_o[-3], dim_o[-2], dim_o[-1], byte = byte)
    return tensor_size, layers_top_used_time, dict_to_layer_bottom, dict_to_layer_top, tensor_shape, input_name

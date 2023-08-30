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
def get_addr_occupy(tensor_size, tensor_addr, layers_top_used_time):
    addr_occupied_raw = []
    for name in tensor_addr.keys():
        addr = tensor_addr[name]
        size = tensor_size[name]
        if addr >= 0:
            if name not in layers_top_used_time.keys() or layers_top_used_time[name] > 0:
                addr_occupied_raw.append([addr, addr + size - 1])
    
    addr_occupied_raw.sort()
    addr_occupied = []
    if len(addr_occupied_raw) > 0:
        addr_occupied.append(addr_occupied_raw.pop(0))
        for interval in addr_occupied_raw:
            if interval[0] <= addr_occupied[-1][1]:
                addr_occupied[-1][1] = max(addr_occupied[-1][1], interval[1])
            else:
                addr_occupied.append(interval)
    return addr_occupied

def get_addr_available(addr_occupied, size):
    result = -1
    if len(addr_occupied) == 0 or (addr_occupied[0][0] != 0 and size <= addr_occupied[0][0]):
        result = 0
    else:
        for i in range(len(addr_occupied) - 1):
            if addr_occupied[i + 1][0] - addr_occupied[i][1] > size:
                result = addr_occupied[i][1] + 1
                break
        if result < 0:
            result = addr_occupied[len(addr_occupied) - 1][1] + 1
    return result

def get_tensor_size_in_input_iram(one_tensor_shape, stride, pad_u, pad_l, elem_bytes):
    width_div32s = math.ceil((one_tensor_shape[3] + pad_l) / stride / 32 * elem_bytes)
    return one_tensor_shape[1] * (one_tensor_shape[2] + pad_u) * width_div32s * 32 * stride

def assign_address(layers_merge, tensor_size, dict_to_layer_bottom, 
                   dict_to_layer_top, tensor_shape, layers_top_used_time, input_name='', reserve_net_input = True, try_interal = True, elem_bytes = 1):
    tensor_addr = {}
    addr_occupied = []
    tensor_addr[input_name] = 0
    total_size_used = 0
    addr_occupied = get_addr_occupy(tensor_size, tensor_addr, layers_top_used_time)
    cnn_ops = ['tf_pool', 'pooling', 'bias', 'batchnorm', 'scale', 'tf_conv', 'convolution', 'innerproduct']
    for layer_index in range(len(layers_merge)):
        layer1 = layers_merge[layer_index]

        print("assign address for layer" + layer1.name, layer1.type)
        tops = layer1.top
        if len(tops) == 1:
            top = tops[0]
            if top == input_name:
                tensor_addr[top] = 0
            else:
                next_layers = []
                if top in dict_to_layer_bottom.keys():
                    for next_layer in dict_to_layer_bottom[top]:
                        if not next_layer.used:
                            next_layers.append(next_layer)
                if len(next_layers) == 1:
                    next_layer = next_layers[0]
                    if len(next_layer.bottom) == 1 or next_layer.type == 'scalebytensor':
                        if not(top in tensor_addr.keys()): # need addr
                            next_layer_index = layers_merge.index(next_layer)
                            output_width = tensor_shape[top][3] * elem_bytes
                            tensor_shape_per_group = [tensor_shape[layer1.bottom[0]][0], tensor_shape[layer1.bottom[0]][1] // layer1.group, tensor_shape[layer1.bottom[0]][2], tensor_shape[layer1.bottom[0]][3]]
                            if output_width <= 32 // elem_bytes and \
                               'concat' not in layer1.type and \
                               layer1.type[0] in cnn_ops and \
                               (tensor_size[top] <= 128 * 1024 or (tensor_size[top] < 256 * 1024 and tensor_shape[top][3] <= 16 // elem_bytes)) \
                               and 'bias' not in layer1.type \
                               and len(next_layer.top) == 1 \
                               and 'pooling' not in next_layer.type \
                               and 'tf_pool' not in next_layer.type \
                               and next_layer.type[0] in cnn_ops \
                               and 'eltwise' not in layer1.type \
                               and next_layer_index == layer_index + 1 \
                               and get_tensor_size_in_input_iram(tensor_shape[top], next_layer.stride, next_layer.pad_u, next_layer.pad_l, elem_bytes) < 256 * 1024 \
                               and get_tensor_size_in_input_iram(tensor_shape_per_group, layer1.stride, layer1.pad_u, layer1.pad_l, elem_bytes) < 256 * 1024 \
                               and (tensor_size[next_layer.top[0]] <= 128 * 1024 or (tensor_size[next_layer.top[0]] < 256 * 1024 and tensor_shape[next_layer.top[0]][3] <= 16 // elem_bytes)) \
                               and try_interal: # out internal mode or optimize mode
                                tensor_addr[top] = -1
                            elif 'concat' in layer1.type:
                                need_new_addr = layer1.axis > 1
                                for i in range(len(layer1.bottom) - 1):
                                    this_size = tensor_size[layer1.bottom[i]]
                                    this_addr = tensor_addr[layer1.bottom[i]]
                                    next_addr = tensor_addr[layer1.bottom[i + 1]]
                                    if this_size + this_addr != next_addr:
                                        need_new_addr = True
                                        break
                                if len(list(set(layer1.bottom))) != len(layer1.bottom):
                                    need_new_addr = True
                                if need_new_addr:
                                    tensor_addr[top] = get_addr_available(addr_occupied, tensor_size[top])
                                else:
                                    tensor_addr[top] = tensor_addr[layer1.bottom[0]]
                            else:
                                tensor_addr[top] = get_addr_available(addr_occupied, tensor_size[top])
                    else:
                        if not(top in tensor_addr.keys()): # need addr
                            index_of_one_has_addr = -1
                            for bottom in next_layer.bottom:
                                if bottom in tensor_addr.keys():
                                    index_of_one_has_addr = next_layer.bottom.index(bottom)
                                    break
                            if index_of_one_has_addr < 0: # none of the bottoms of next_layer has addr
                                total_size = sum([tensor_size[bottom] for bottom in next_layer.bottom])
                                addr_offset = get_addr_available(addr_occupied, total_size)
                                for bottom in next_layer.bottom:
                                    tensor_addr[bottom] = addr_offset
                                    addr_offset += tensor_size[bottom]
                            else:
                                print("Warning: one of the input of layer {} has been signed an address because this input has been used for another multi-input layer, you need adjust all the address manully".format(next_layer.name))
                                addr_offset = sum(tensor_size[next_layer.bottom[i]] for i in range(next_layer.bottom.index(top)))
                                tensor_addr[top] = addr_offset + tensor_addr[next_layer.bottom[0]]
                elif len(next_layers) == 0:
                    if 'concat' in layer1.type:
                        need_new_addr = layer1.axis > 1
                        for i in range(len(layer1.bottom) - 1):
                            this_size = tensor_size[layer1.bottom[i]]
                            this_addr = tensor_addr[layer1.bottom[i]]
                            next_addr = tensor_addr[layer1.bottom[i + 1]]
                            if this_size + this_addr != next_addr:
                                need_new_addr = True
                                break
                        if len(list(set(layer1.bottom))) != len(layer1.bottom):
                            need_new_addr = True
                        if need_new_addr:
                            tensor_addr[top] = get_addr_available(addr_occupied, tensor_size[top])
                        else:
                            tensor_addr[top] = tensor_addr[layer1.bottom[0]]
                    if not(top in tensor_addr.keys()): # need addr
                        tensor_addr[top] = get_addr_available(addr_occupied, tensor_size[top])
                else:
                    multi_input = False
                    next_layer = None
                    for next_layer in next_layers:
                        if len(next_layer.bottom) > 1:
                            multi_input = True
                            break
                    if multi_input:
                        if not(top in tensor_addr.keys()): # need addr
                            index_of_one_has_addr = -1
                            for bottom in next_layer.bottom:
                                if bottom in tensor_addr.keys():
                                    index_of_one_has_addr = next_layer.bottom.index(bottom)
                                    break
                            if index_of_one_has_addr < 0: # none of the bottoms of next_layer has addr
                                need_new_addr = True
                                if 'concat' in layer1.type:
                                    need_new_addr = layer1.axis > 1
                                    for i in range(len(layer1.bottom) - 1):
                                        this_size = tensor_size[layer1.bottom[i]]
                                        this_addr = tensor_addr[layer1.bottom[i]]
                                        next_addr = tensor_addr[layer1.bottom[i + 1]]
                                        if this_size + this_addr != next_addr:
                                            need_new_addr = True
                                            break
                                    if len(list(set(layer1.bottom))) != len(layer1.bottom):
                                        need_new_addr = True
                                if need_new_addr:
                                    total_size = sum([tensor_size[bottom] for bottom in next_layer.bottom])
                                    addr_offset = get_addr_available(addr_occupied, total_size)
                                    for bottom in next_layer.bottom:
                                        tensor_addr[bottom] = addr_offset
                                        addr_offset += tensor_size[bottom]
                                else:
                                    tensor_addr[top] = tensor_addr[layer1.bottom[0]]
                                    addr_offset = tensor_addr[top] - sum(tensor_size[next_layer.bottom[i]] for i in range(next_layer.bottom.index(top)))
                                    for bottom in next_layer.bottom:
                                        tensor_addr[bottom] = addr_offset
                                        addr_offset += tensor_size[bottom]
                            else:
                                print("Warning: one of the input of layer {} has been signed an address because this input has been used for another multi-input layer, you need adjust all the address manully".format(next_layer.name))
                                tensor_addr[top] = get_addr_available(addr_occupied, tensor_size[top])
                    else:
                        if not(top in tensor_addr.keys()): # need addr
                            if 'concat' in layer1.type:
                                need_new_addr = layer1.axis > 1
                                for i in range(len(layer1.bottom) - 1):
                                    this_size = tensor_size[layer1.bottom[i]]
                                    this_addr = tensor_addr[layer1.bottom[i]]
                                    next_addr = tensor_addr[layer1.bottom[i + 1]]
                                    if this_size + this_addr != next_addr:
                                        need_new_addr = True
                                        break
                                if len(list(set(layer1.bottom))) != len(layer1.bottom):
                                    need_new_addr = True
                                if need_new_addr:
                                    tensor_addr[top] = get_addr_available(addr_occupied, tensor_size[top])
                                else:
                                    tensor_addr[top] = tensor_addr[layer1.bottom[0]]
                            else:
                                tensor_addr[top] = get_addr_available(addr_occupied, tensor_size[top])
        else:
            total_size = sum([tensor_size[top] for top in layer1.top])
            if 'slice' in layer1.type and layer1.axis == 1:
                base_addr = tensor_addr[layer1.bottom[0]]
                for top in layer1.top:
                    tensor_addr[top] = base_addr
                    base_addr += tensor_size[top]
                    next_layers = []
                    if top in dict_to_layer_bottom.keys():
                        for next_layer in dict_to_layer_bottom[top]:
                            if not next_layer.used and len(next_layer.bottom) > 1:
                                next_layers.append(next_layer)
                    if len(next_layers) >= 1:
                        if len(next_layers) > 1:
                            print('Warning: this tensor is used for more than 1 branches, the addr need fixing manually')
                        next_base_addr = base_addr
                        top_in_next_index = next_layers[0].bottom.index(top)
                        for bottom_next_index in range(top_in_next_index + 1, len(next_layers[0].bottom)):
                            bottom_in_next = next_layers[0].bottom[bottom_next_index]
                            tensor_addr[bottom_in_next] = next_base_addr
                            next_base_addr += tensor_size[bottom_in_next]
                        for bottom_next_index in range(top_in_next_index, -1, -1):
                            bottom_in_next = next_layers[0].bottom[bottom_next_index]
                            next_base_addr -= tensor_size[bottom_in_next]
                            tensor_addr[bottom_in_next] = next_base_addr
            else:
                for top in tops:
                    tensor_addr[top] = get_addr_available(addr_occupied, tensor_size[top])
        addr_occupied = get_addr_occupy(tensor_size, tensor_addr, layers_top_used_time)
        if len(addr_occupied) > 0:
            total_size_used = max(total_size_used, addr_occupied[-1][-1] + 1)
        layer1.used = True
        for bottom in layer1.bottom:
            if (not reserve_net_input) or bottom != input_name:
                layers_top_used_time[bottom] -= 1
    return tensor_addr, total_size_used
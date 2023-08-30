# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:50:49 2019

@author: user
"""

import os
import sys
ndk_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ndk_dir)
import numpy
from ndk.quant_tools.numpy_net import run_layers
from ndk.layers import get_net_input_output


def running_quant(input_file, layer_list, param_dict, bitwidth):
    input_data_batch = numpy.load(input_file)
    input_data_batch = input_data_batch[0:1, :, :, :]
    output_tensor_names = []
    for layer in layer_list:
        if type(layer.top) == str:
            output_tensor_names.append(layer.top)
        else:
            output_tensor_names += layer.top
    data_dict = run_layers(input_data_batch, layer_list, output_tensor_names, param_dict, bitwidth = bitwidth, hw_aligned = True, quant = True, use_ai_framework = True)
    return data_dict

def running(input_data_batch, layer_list, param_dict):
    input_tensor_name, output_tensor_name = get_net_input_output(layer_list)
    data_dict = run_layers(input_data_batch, layer_list, output_tensor_name, param_dict, hw_aligned = False, quant=False, use_ai_framework=True)
    return data_dict[output_tensor_name[0]]

def test(output_data, golden_data):
    if len(golden_data.shape) == 2:
        batch, channel = golden_data.shape
        golden_data = golden_data.reshape(batch, channel, 1, 1)
    assert output_data.shape == golden_data.shape, "golden data should be a tensor with the same shape with output, got output shape is {} and golden shape {}".format(output_data.shape, golden_data.shape)
    N, C, H, W = output_data.shape
    error = 0
    l1_norm = 0
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    error += abs(output_data[n,c,h,w] - golden_data[n,c,h,w])
                    l1_norm += abs(golden_data[n,c,h,w])
                    if abs(output_data[n,c,h,w] - golden_data[n,c,h,w]) >= 1e-5:
                        print("at point {}, {}, {}, {}, golden: {}, output: {}, diff: {}".format(n, c, h, w, golden_data[n,c,h,w], output_data[n,c,h,w], golden_data[n,c,h,w] - output_data[n,c,h,w]))
    print("L1 norm of error: {}".format(error))
    print("L1 norm of golden: {}".format(l1_norm))
    print("average relative error: {}".format(error / l1_norm))
    with open("parsing_test_result.txt", 'w') as f:
        f.write("L1 norm of error: {}\n".format(error))
        f.write("L1 norm of golden: {}\n".format(l1_norm))
        f.write("average relative error: {}".format(error / l1_norm))
    
def running_and_test(input_file, layer_list, param_dict, golden_file):
    input_data_batch = numpy.load(input_file)
    golden_data = numpy.load(golden_file)
    output_data = running(input_data_batch, layer_list, param_dict)
    test(output_data, golden_data)
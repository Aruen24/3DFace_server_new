# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:55:58 2019

@author: user
"""

import os
import sys
ndk_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ndk_dir)

import numpy
import math

def random_quantize(layer_list, param_dict, bitwidth):
    max_ = math.pow(2.0, bitwidth - 1) - 1
    min_ = -math.pow(2.0, bitwidth - 1)
    quantized_dict = {}
    param_dict = dict(param_dict)
    for key, value in param_dict.items():
        if key.endswith("_weight"):
            quant_key_root = key[:-7]
            quant_key_param = quant_key_root + "_quant_weight"
            quant_key_frac = quant_key_root + "_frac_weight"
            quant_value_frac = bitwidth - 4
            quant_value_param = value * math.pow(2.0, quant_value_frac)
            quant_value_param = numpy.round(quant_value_param)
            quant_value_param = numpy.clip(quant_value_param, min_, max_)
            quantized_dict[quant_key_param] = quant_value_param
            quantized_dict[quant_key_frac] = quant_value_frac
        if key.endswith("_bias"):
            quant_key_root = key[:-5]
            quant_key_param = quant_key_root + "_quant_bias"
            quant_key_frac = quant_key_root + "_frac_bias"
            quant_value_frac = bitwidth - 4
            quant_value_param = value * math.pow(2.0, quant_value_frac)
            quant_value_param = numpy.round(quant_value_param)
            quant_value_param = numpy.clip(quant_value_param, min_, max_)
            quantized_dict[quant_key_param] = quant_value_param
            quantized_dict[quant_key_frac] = quant_value_frac
    for layer in layer_list:
        if type(layer.top) == str:
            quantized_dict[layer.top + "_frac"] = bitwidth - 4
        else:
            for top in layer.top:
                quantized_dict[top + "_frac"] = bitwidth - 4
    return quantized_dict

def change_nonlinear(layer_list):
    for layer in layer_list:
        if layer.type in ['Sigmoid', 'TanH']:
            print("Warn: for a faster random, quantization, layer {}'s type will be changed to ReLU".format(layer.name))
            layer.type = 'ReLU6'
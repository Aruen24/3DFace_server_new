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

import os
import sys
ndk_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ndk_dir)
from ndk.onnx_parser.onnx_parser import load_from_onnx
from running_and_test import running_and_test
from ndk.modelpack import modelpack
from random_quantize import random_quantize, change_nonlinear
from configparser import ConfigParser, NoOptionError
from ndk.optimize import merge_layers
from ndk.layers import check_layers

config_file_name = 'parser_tester_config_onnx.ini'
onnx_file = ''
input_file = ''
golden_file = ''
bitwidth = 8
packing_net_for_chip = True
testing_float_parsing = True
out_file_path = "for_testing_time_on_chip"
model_name = 'model'

def parser_tester_onnx(onnx_file, input_file, golden_file, bitwidth, packing_net_for_chip, testing_float_parsing):
    layer_list0, param_dict0 = load_from_onnx(onnx_file)
    layer_list, param_dict = merge_layers(layer_list0, param_dict0)
    check_result = check_layers(layer_list)
    if check_result and bitwidth == 16:
        print("Warning: this net is not suitable for 16bit mode")
    if testing_float_parsing:
        running_and_test(input_file, layer_list, param_dict, golden_file)
    if packing_net_for_chip:
        change_nonlinear(layer_list)
        quant_dict = random_quantize(layer_list, param_dict, bitwidth)
        quant_dict.update(param_dict)
        modelpack(bitwidth, layer_list, quant_dict, out_file_path, model_name = 'model')
    
if __name__=='__main__':
    conf = ConfigParser()
    conf.read(config_file_name, encoding="utf-8")
    onnx_file = conf.get('input_file_path', 'onnx_file')
    input_file = conf.get('input_file_path', 'input_numpy_file')
    golden_file = conf.get('input_file_path', 'golden_numpy_file')
    try:
        bitwidth = int(conf.get('quant_mode', 'bitwidth'))
    except NoOptionError:
        print("Warning: option bitwidth is not found in ini file, default value will be used")
    try:
        testing_float_parsing = conf.get('module_enable', 'testing_float_parsing')
        testing_float_parsing = testing_float_parsing.lower().find("true") >= 0
    except NoOptionError:
        print("Warning: option testing_float_parsing is not found in ini file, default value will be used")
    try:
        packing_net_for_chip = conf.get('module_enable', 'packing_net_for_chip')
        packing_net_for_chip = packing_net_for_chip.lower().find("true") >= 0
    except NoOptionError:
        print("Warning: option packing_net_for_chip is not found in ini file, default value will be used")
    
    assert bitwidth in [8, 16], "bitswidth should be 8 or 16, but got {}".format(bitwidth)
    print("config get:")
    print('onnx_file: ', onnx_file)
    print('input_numpy_file: ', input_file)
    print('golden_numpy_file: ', golden_file)
    print('bitwidth: ', bitwidth)
    print('packing_net_for_chip: ', packing_net_for_chip)
    print('testing_float_parsing: ', testing_float_parsing)
    parser_tester_onnx(onnx_file, input_file, golden_file, bitwidth, packing_net_for_chip, testing_float_parsing)

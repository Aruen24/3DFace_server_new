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
ndk_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ndk_dir)

from ndk.onnx_parser import onnx_parser

import os

def load_from_onnx(fname_onnx):
    layer_list, param_dict = onnx_parser.load_from_onnx(fname_onnx)
    return layer_list, param_dict

if __name__=='__main__':
    model_path = r'./onnx_parser/squeezenet1_0.onnx'
    layer_list, param_dict = load_from_onnx(model_path)
    for layer in layer_list:
        print(layer)

    # print('Not implemented')

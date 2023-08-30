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
from ndk.tf_interface.pb_to_K3DAITool import to_layerspb
from ndk.tf_interface.pb_read import read_graph_from_pb
from ndk.tf_interface.hdf5_read_all import read_graph_from_hdf5

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def load_from_pb(fname_pb):
    layers,param_dict = read_graph_from_pb(fname_pb)
    layer_list = to_layerspb(layers)
    return layer_list,param_dict

def load_from_hdf5(fname_hdf5):
    layers, param_dict = read_graph_from_hdf5(fname_hdf5)
    layer_list = to_layerspb(layers)
    return layer_list, param_dict

if __name__=='__main__':
    model_path = r'./tf_interface/mnist.pb'
    layer_list,param_dict = load_from_pb(model_path)
    for layer in layer_list:
        print(layer)

    # print('Not implemented')

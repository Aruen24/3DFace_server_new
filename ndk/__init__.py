#!\usr\bin\python
# -*- coding:utf-8 -*-
import os
import sys

__version__ = '1.1.3'

ndk_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ndk_dir)


try:
    import caffe
    CAFFE_FOUND=True
except ImportError:
    CAFFE_FOUND=False
try:
    import tensorflow
    TENSORFLOW_FOUND=True
except ImportError:
    TENSORFLOW_FOUND=False
try:
    import onnx
    ONNX_FOUND=True
except ImportError:
    ONNX_FOUND=False

if CAFFE_FOUND:
    import ndk.caffe_interface

if TENSORFLOW_FOUND:
    import ndk.tensorflow_interface
	
if ONNX_FOUND:
    import ndk.onnx_interface

import ndk.layers
import ndk.modelpack
import ndk.optimize
import ndk.quantize
import ndk.utils

import ndk.quant_tools

try:
    import ndk.examples
except ImportError:
    pass

try:
    import ndk.parser_tester
except ImportError:
    pass

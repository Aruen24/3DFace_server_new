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
import numpy
import copy
ndk_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ndk_dir)

from ndk.utils import AttrDisplay

no_input_layer = ['Input']
one_input_layer = ['Sigmoid', 'ReLU', 'ReLU6', 'TanH', 'InnerProduct', 'Convolution', 'Pooling', 'BatchNorm', 'Bias', 'Scale', 'Slice', 'Softmax', 'LogSoftmax', 'Permute', 'Reshape', 'Flatten', 'ShuffleChannel']
multi_input_layer = ['Concat', 'Eltwise', 'ScaleByTensor']
one_output_layer = ['Input', 'Sigmoid', 'ReLU', 'ReLU6', 'TanH', 'InnerProduct', 'Convolution', 'Pooling', 'BatchNorm', 'Bias', 'Scale', 'Concat', 'Eltwise', 'Softmax', 'LogSoftmax', 'Permute', 'Reshape', 'Flatten', 'ShuffleChannel', 'ScaleByTensor']
multi_output_layer = ['Slice']

class Layer(AttrDisplay):
    def __init__(self, lyr_type=None, **kwargs):

        self.type_dict = {'input': 'Input', 'placeholder': 'Input',
                          'sigmoid':'Sigmoid', 'sig':'Sigmoid',
                          'relu':'ReLU', 'leakyrelu':'ReLU', 'leaky_relu':'ReLU',
                          'relu6':'ReLU6',
                          'tanh':'TanH',
                          'innerproduct':'InnerProduct', 'ip':'InnerProduct', 'fc':'InnerProduct', 'inner_product':'InnerProduct',
                          'convolution':'Convolution', 'conv':'Convolution',
                          'pooling': 'Pooling', 'pool': 'Pooling',
                          'batchnorm': 'BatchNorm', 'bn': 'BatchNorm', 'batch_norm': 'BatchNorm', 'fusedbatchnorm': 'BatchNorm', 'batchnormalization': 'BatchNorm',
                          'bias': 'Bias',
                          'scale': 'Scale',
                          'scalebytensor': 'ScaleByTensor',
                          'slice': 'Slice', 'split': 'Slice',
                          'concat': 'Concat', 'concatenation':'Concat',
                          'eltwise': 'Eltwise',
                          'softmax': 'Softmax',
                          'logsoftmax': 'LogSoftmax', 'log_softmax': 'LogSoftmax',
                          # 'permute': 'Permute', 'transpose': 'Permute',
                          # 'reshape': 'Reshape',
                          # 'flatten': 'Flatten',
                          'shufflechannel': 'ShuffleChannel', 'shuffle_channel': 'ShuffleChannel', 'channel_shuffle': 'ShuffleChannel', 'channelshuffle': 'ShuffleChannel', 'shuffle': 'ShuffleChannel'
                          }
        if lyr_type==None:
            self.type = None # After layer type is initialized as None, one of the 'set_xxx_layer' functions should be called.
        else:
            assert type(lyr_type)==str, 'Layer type must be specified using a string'
            assert lyr_type.lower() in self.type_dict.keys(), 'Layer type \'{}\' is not supported.'.format(lyr_type)

            if  'Input'==self.type_dict[lyr_type.lower()]:
                self.set_input_layer(**kwargs)
            elif  'Sigmoid'==self.type_dict[lyr_type.lower()]:
                self.set_sigmoid_layer(**kwargs)
            elif 'ReLU'==self.type_dict[lyr_type.lower()]:
                self.set_relu_layer(**kwargs)
            elif 'ReLU6'==self.type_dict[lyr_type.lower()]:
                self.set_relu6_layer(**kwargs)
            elif 'TanH'==self.type_dict[lyr_type.lower()]:
                self.set_tanh_layer(**kwargs)
            elif 'InnerProduct'==self.type_dict[lyr_type.lower()]:
                self.set_innerproduct_layer(**kwargs)
            elif 'Convolution'==self.type_dict[lyr_type.lower()]:
                self.set_convolution_layer(**kwargs)
            elif 'Pooling'==self.type_dict[lyr_type.lower()]:
                self.set_pooling_layer(**kwargs)
            elif 'BatchNorm'==self.type_dict[lyr_type.lower()]:
                self.set_batchnorm_layer(**kwargs)
            elif 'Bias'==self.type_dict[lyr_type.lower()]:
                self.set_bias_layer(**kwargs)
            elif 'Scale'==self.type_dict[lyr_type.lower()]:
                self.set_scale_layer(**kwargs)
            elif 'ScaleByTensor'==self.type_dict[lyr_type.lower()]:
                self.set_scale_by_tensor_layer(**kwargs)
            elif 'Slice'==self.type_dict[lyr_type.lower()]:
                self.set_slice_layer(**kwargs)
            elif 'Concat'==self.type_dict[lyr_type.lower()]:
                self.set_concat_layer(**kwargs)
            elif 'Eltwise'==self.type_dict[lyr_type.lower()]:
                self.set_eltwise_layer(**kwargs)
            elif 'Softmax'==self.type_dict[lyr_type.lower()]:
                self.set_softmax_layer(**kwargs)
            elif 'LogSoftmax'==self.type_dict[lyr_type.lower()]:
                self.set_logsoftmax_layer(**kwargs)
            elif 'Permute'==self.type_dict[lyr_type.lower()]:
                self.set_permute_layer(**kwargs)
            elif 'Reshape'==self.type_dict[lyr_type.lower()]:
                self.set_reshape_layer(**kwargs)
            elif 'Flatten'==self.type_dict[lyr_type.lower()]:
                self.set_flatten_layer(**kwargs)
            elif 'ShuffleChannel'==self.type_dict[lyr_type.lower()]:
                self.set_shufflechannel_layer(**kwargs)
            else:
                assert False, 'Unsupported layer {}.'.format(lyr_type)

    def set_input_layer(self, name, top, dim):
        assert type(name)==str, 'Layer name must be a string, but got a {}'.format(type(name))
        if type(top)!=str:
            assert type(top)==list or type(top)==tuple, 'Output tensor(s) must be a string or a list/tuple of strings'
            for elem in top:
                assert type(elem)==str, 'Output tensor(s) must be a string or a list of string'
            if len(top)==1:
                top = top[0]
        assert type(top)==str, 'Find multiple outputs but only one is allowed, top={}'.format(top)
        assert type(dim) == tuple, 'dim should be a tuple but got a {}'.format(type(dim))
        assert len(dim)==4, 'length of dim should be 4, but got {}'.format(len(dim))

        self.type = self.type_dict['input']
        self.name = name
        # self.bottom = bottom # Input layer has no bottom
        self.top = top
        self.dim = dim # (:,C,H,W)

    def set_sigmoid_layer(self, name, bottom, top):
        assert type(name)==str, 'Layer name must be a string, but got a {}'.format(type(name))
        if type(bottom)!=str:
            assert type(bottom)==list or type(bottom)==tuple, 'Input tensor(s) must be a string or a list/tuple of strings'
            for elem in bottom:
                assert type(elem)==str, 'Input tensor(s) must be a string or a list of string'
            if len(bottom)==1:
                bottom = bottom[0]
        if type(top)!=str:
            assert type(top)==list or type(top)==tuple, 'Output tensor(s) must be a string or a list/tuple of strings'
            for elem in top:
                assert type(elem)==str, 'Output tensor(s) must be a string or a list of string'
            if len(top)==1:
                top = top[0]
        assert type(bottom)==str, 'Find multiple inputs but only one is allowed, bottom={}'.format(bottom)
        assert type(top)==str, 'Find multiple outputs but only one is allowed, top={}'.format(top)

        self.type = self.type_dict['sigmoid']
        self.name = name
        self.bottom = bottom
        self.top = top

    def set_relu_layer(self, name, bottom, top, negative_slope=0):
        assert type(name)==str, 'Layer name must be a string, but got a {}'.format(type(name))
        if type(bottom)!=str:
            assert type(bottom)==list or type(bottom)==tuple, 'Input tensor(s) must be a string or a list/tuple of strings'
            for elem in bottom:
                assert type(elem)==str, 'Input tensor(s) must be a string or a list of string'
            if len(bottom)==1:
                bottom = bottom[0]
        if type(top)!=str:
            assert type(top)==list or type(top)==tuple, 'Output tensor(s) must be a string or a list/tuple of strings'
            for elem in top:
                assert type(elem)==str, 'Output tensor(s) must be a string or a list of string'
            if len(top)==1:
                top = top[0]
        assert type(bottom)==str, 'Find multiple inputs but only one is allowed, bottom={}'.format(bottom)
        assert type(top)==str, 'Find multiple outputs but only one is allowed, top={}'.format(top)

        self.type = self.type_dict['relu']
        self.name = name
        self.bottom = bottom
        self.top = top
        self.negative_slope = negative_slope
        
    def set_relu6_layer(self, name, bottom, top):
        assert type(name)==str, 'Layer name must be a string, but got a {}'.format(type(name))
        if type(bottom)!=str:
            assert type(bottom)==list or type(bottom)==tuple, 'Input tensor(s) must be a string or a list/tuple of strings'
            for elem in bottom:
                assert type(elem)==str, 'Input tensor(s) must be a string or a list of string'
            if len(bottom)==1:
                bottom = bottom[0]
        if type(top)!=str:
            assert type(top)==list or type(top)==tuple, 'Output tensor(s) must be a string or a list/tuple of strings'
            for elem in top:
                assert type(elem)==str, 'Output tensor(s) must be a string or a list of string'
            if len(top)==1:
                top = top[0]
        assert type(bottom)==str, 'Find multiple inputs but only one is allowed, bottom={}'.format(bottom)
        assert type(top)==str, 'Find multiple outputs but only one is allowed, top={}'.format(top)

        self.type = self.type_dict['relu6']
        self.name = name
        self.bottom = bottom
        self.top = top

    def set_tanh_layer(self, name, bottom, top):
        assert type(name)==str, 'Layer name must be a string, but got a {}'.format(type(name))
        if type(bottom)!=str:
            assert type(bottom)==list or type(bottom)==tuple, 'Input tensor(s) must be a string or a list/tuple of strings'
            for elem in bottom:
                assert type(elem)==str, 'Input tensor(s) must be a string or a list of string'
            if len(bottom)==1:
                bottom = bottom[0]
        if type(top)!=str:
            assert type(top)==list or type(top)==tuple, 'Output tensor(s) must be a string or a list/tuple of strings'
            for elem in top:
                assert type(elem)==str, 'Output tensor(s) must be a string or a list of string'
            if len(top)==1:
                top = top[0]
        assert type(bottom)==str, 'Find multiple inputs but only one is allowed, bottom={}'.format(bottom)
        assert type(top)==str, 'Find multiple outputs but only one is allowed, top={}'.format(top)

        self.type = self.type_dict['tanh']
        self.name = name
        self.bottom = bottom
        self.top = top

    def set_innerproduct_layer(self, name, bottom, top, num_output, bias_term=True):
        assert type(name)==str, 'Layer name must be a string, but got a {}'.format(type(name))
        if type(bottom)!=str:
            assert type(bottom)==list or type(bottom)==tuple, 'Input tensor(s) must be a string or a list/tuple of strings'
            for elem in bottom:
                assert type(elem)==str, 'Input tensor(s) must be a string or a list of string'
            if len(bottom)==1:
                bottom = bottom[0]
        if type(top)!=str:
            assert type(top)==list or type(top)==tuple, 'Output tensor(s) must be a string or a list/tuple of strings'
            for elem in top:
                assert type(elem)==str, 'Output tensor(s) must be a string or a list of string'
            if len(top)==1:
                top = top[0]
        assert type(bottom)==str, 'Find multiple inputs but only one is allowed, bottom={}'.format(bottom)
        assert type(top)==str, 'Find multiple outputs but only one is allowed, top={}'.format(top)

        self.type = self.type_dict['inner_product']
        self.name = name
        self.bottom = bottom
        self.top = top
        self.num_output = num_output
        self.bias_term = bias_term

    def set_convolution_layer(self, name, top, bottom, num_output, kernel_size=None, kernel_size_h=None, kernel_size_w=None,
                              stride=None, stride_h=None, stride_w=None,
                              pad=None, pad_n=None, pad_s=None, pad_w=None, pad_e=None,
                              bias_term=True,
                              dilation=1, dilation_h=None, dilation_w=None,
                              group=1
                              ):
        assert type(name)==str, 'Layer name must be a string, but got a {}'.format(type(name))
        if type(bottom)!=str:
            assert type(bottom)==list or type(bottom)==tuple, 'Input tensor(s) must be a string or a list/tuple of strings'
            for elem in bottom:
                assert type(elem)==str, 'Input tensor(s) must be a string or a list of string'
            if len(bottom)==1:
                bottom = bottom[0]
        if type(top)!=str:
            assert type(top)==list or type(top)==tuple, 'Output tensor(s) must be a string or a list/tuple of strings'
            for elem in top:
                assert type(elem)==str, 'Output tensor(s) must be a string or a list of string'
            if len(top)==1:
                top = top[0]
        assert type(bottom)==str, 'Find multiple inputs but only one is allowed, bottom={}'.format(bottom)
        assert type(top)==str, 'Find multiple outputs but only one is allowed, top={}'.format(top)

        # Layer.kernel_size will always be a 2-tuple (kernel_size_h, kernel_size_w)
        if kernel_size_h != None and kernel_size_w != None and kernel_size_h>0 and kernel_size_w>0:
            assert type(kernel_size_h)==int, 'kernel_size_h is assumed to be an integer but got {}'.format(type(kernel_size_h))
            assert type(kernel_size_w)==int, 'kernel_size_w is assumed to be an integer but got {}'.format(type(kernel_size_w))
            kernel_size = (kernel_size_h, kernel_size_w,)
        else:
            assert kernel_size!=None, 'kernel_size is required'
            if type(kernel_size)==int:
                assert kernel_size > 0, 'kernel_size must be larger than zero'
                kernel_size = (kernel_size, kernel_size,)
            else:
                assert type(kernel_size)==tuple, 'kernel_size is assumed to be a tuple but got {}'.format(type(kernel_size))
                if len(kernel_size)==1:
                    assert type(kernel_size[0])==int, 'kernel_size is assumed to be a tuple of integer'
                    assert kernel_size[0] > 0, 'kernel_size must be larger than zero'
                    kernel_size = kernel_size + kernel_size
                elif len(kernel_size)==2:
                    assert type(kernel_size[0])==int, 'kernel_size is assumed to be a tuple of integer'
                    assert type(kernel_size[1])==int, 'kernel_size is assumed to be a tuple of integer'
                    assert kernel_size[0] > 0, 'kernel_size must be larger than zero.'
                    assert kernel_size[1] > 0, 'kernel_size must be larger than zero.'
                    kernel_size = kernel_size
                else:
                    assert False, 'Invalid kernel_size value {}'.format(kernel_size)

        # Layer.stride will always be a 2-tuple (stride_h, stride_w)
        if stride_h != None and stride_w != None and stride_h and stride_h > 0 and stride_w > 0:
            assert type(stride_h)==int, 'stride_h is assumed to be an integer but got {}'.format(type(stride_h))
            assert type(stride_w)==int, 'stride_w is assumed to be an integer but got {}'.format(type(stride_w))
            stride = (stride_h, stride_w,)
        else:
            assert stride!=None, 'stride is required.'
            if type(stride)==int:
                assert stride > 0, 'stride must be larger than zero'
                stride = (stride, stride,)
            else:
                assert type(stride)==tuple, 'stride is assumed to be a tuple but got {}'.format(type(stride))
                if len(stride)==1:
                    assert type(stride[0])==int, 'stride is assumed to be a tuple of integer'
                    assert stride[0] > 0, 'stride must be larger than zero'
                    stride = stride + stride
                elif len(stride)==2:
                    assert type(stride[0])==int, 'stride is assumed to be a tuple of integer'
                    assert type(stride[1])==int, 'stride is assumed to be a tuple of integer'
                    assert stride[0] > 0, 'stride must be larger than zero'
                    assert stride[1] > 0, 'stride must be larger than zero'
                    stride = stride
                else:
                    assert False, 'Invalid stride value {}'.format(stride)

        # Layer.pad will always be a 4-tuple, (pad_n, pad_s, pad_w, pad_e)
        if pad_n != None and pad_s !=None and pad_w!=None and pad_e!=None:
            assert type(pad_n)==int, 'pad_n is assumed to be an integer but got {}'.format(type(pad_n))
            assert type(pad_s)==int, 'pad_s is assumed to be an integer but got {}'.format(type(pad_s))
            assert type(pad_w)==int, 'pad_w is assumed to be an integer but got {}'.format(type(pad_w))
            assert type(pad_e)==int, 'pad_e is assumed to be an integer but got {}'.format(type(pad_e))
            assert pad_n >= 0, 'pad_n must be non-negative, but get {}'.format(pad_n)
            assert pad_w >= 0, 'pad_w must be non-negative, but get {}'.format(pad_w)
            pad = (pad_n, pad_s, pad_w, pad_e,) # upper, down, left, right
        else:
            assert pad!=None, 'pad is required.'
            if type(pad)==int:
                assert pad >= 0, 'pad must be non-negative'
                pad = (pad, pad, pad, pad,)
            else:
                assert type(pad)==tuple, 'pad is assumed to be a tuple but got {}'.format(type(pad))
                if len(pad)==1:
                    assert type(pad[0])==int, 'pad is assumed to be a tuple of integer'
                    assert pad[0] >= 0, 'pad must be non-negative'
                    pad = pad + pad + pad + pad
                elif len(pad)==2:
                    assert type(pad[0])==int, 'pad is assumed to be a tuple of integer'
                    assert type(pad[1])==int, 'pad is assumed to be a tuple of integer'
                    assert pad[0] >= 0, 'pad must be non-negative'
                    assert pad[1] >= 0, 'pad must be non-negative'
                    pad = (pad[0], pad[0], pad[1], pad[1])
                elif len(pad)==4:
                    assert type(pad[0])==int, 'pad is assumed to be a tuple of integer'
                    assert type(pad[1])==int, 'pad is assumed to be a tuple of integer'
                    assert type(pad[2])==int, 'pad is assumed to be a tuple of integer'
                    assert type(pad[3])==int, 'pad is assumed to be a tuple of integer'
                    assert pad[0] >= 0, 'pad_n must be non-negative, but get {}'.format(pad[0])
                    # assert pad[1] >= 0, 'pad_s must be non-negative, but get {}'.format(pad[1])
                    assert pad[2] >= 0, 'pad_w must be non-negative, but get {}'.format(pad[2])
                    # assert pad[3] >= 0, 'pad_e must be non-negative, but get {}'.format(pad[3])
                    pad = pad
                else: # len(stride)==2
                    assert False, 'Invalid pad value {}'.format(pad)

        # Layer.dilation will always be a 2-tuple (dilation_h, dilation_w)
        if dilation_h != None and dilation_w != None and dilation_h > 0 and dilation_w > 0:
            assert type(dilation_h)==int, 'dilation_h is assumed to be an integer but got {}'.format(type(dilation_h))
            assert type(dilation_w)==int, 'dilation_w is assumed to be an integer but got {}'.format(type(dilation_w))
            dilation = (dilation_h, dilation_w,)
        else:
            if type(dilation)==int:
                assert dilation > 0, 'dilation must be larger than zero'
                dilation = (dilation, dilation,)
            else:
                assert type(dilation)==tuple, 'dilation is assumed to be a tuple but got {}'.format(type(dilation))
                if len(dilation)==1:
                    assert type(dilation[0])==int, 'dilation is assumed to be a tuple of integer'
                    assert dilation[0] > 0, 'dilation must be larger than zero'
                    dilation = dilation + dilation
                elif len(dilation)==2:
                    assert type(dilation[0])==int, 'dilation is assumed to be a tuple of integer'
                    assert type(dilation[1])==int, 'dilation is assumed to be a tuple of integer'
                    assert dilation[0] > 0, 'dilation must be larger than zero'
                    assert dilation[1] > 0, 'dilation must be larger than zero'
                    dilation = dilation
                else:
                    assert False, 'Invalid dilation value {}'.format(dilation)

        self.type = self.type_dict['convolution']
        self.name = name
        self.bottom = bottom
        self.top = top
        self.num_output = num_output
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.bias_term = bias_term
        self.dilation = dilation
        self.group = group

    def set_pooling_layer(self, name, top, bottom, kernel_size=None, kernel_size_h=None, kernel_size_w=None,
                          stride=None, stride_h=None, stride_w=None,
                          pad=0, pad_n=0, pad_s=0, pad_w=0, pad_e=0,
                          dilation=1, dilation_h=None, dilation_w=None,
                          pool='max'):
        assert type(name)==str, 'Layer name must be a string, but got a {}'.format(type(name))
        if type(bottom)!=str:
            assert type(bottom)==list or type(bottom)==tuple, 'Input tensor(s) must be a string or a list/tuple of strings'
            for elem in bottom:
                assert type(elem)==str, 'Input tensor(s) must be a string or a list of string'
            if len(bottom)==1:
                bottom = bottom[0]
        if type(top)!=str:
            assert type(top)==list or type(top)==tuple, 'Output tensor(s) must be a string or a list/tuple of strings'
            for elem in top:
                assert type(elem)==str, 'Output tensor(s) must be a string or a list of string'
            if len(top)==1:
                top = top[0]
        assert type(bottom)==str, 'Find multiple inputs but only one is allowed, bottom={}'.format(bottom)
        assert type(top)==str, 'Find multiple outputs but only one is allowed, top={}'.format(top)

        assert type(pool)==str, 'Pooling type must be specified using a string, \'MAX\' or \'AVE\', but get {}'.format(type(pool))
        pool = pool.lower()
        assert pool in {'max', 'ave'}, 'Pooling type must be specified using a string, \'MAX\' or \'AVE\', but get {}'.format(pool)

        # Layer.kernel_size will always be a 2-tuple (kernel_size_h, kernel_size_w)
        if kernel_size_h != None and kernel_size_w != None and kernel_size_h>0 and kernel_size_w>0:
            assert type(kernel_size_h)==int, 'kernel_size_h is assumed to be an integer but got {}'.format(type(kernel_size_h))
            assert type(kernel_size_w)==int, 'kernel_size_w is assumed to be an integer but got {}'.format(type(kernel_size_w))
            kernel_size = (kernel_size_h, kernel_size_w,)
        else:
            assert kernel_size!=None, 'kernel_size is required'
            if type(kernel_size)==int:
                assert kernel_size > 0, 'kernel_size must be larger than zero'
                kernel_size = (kernel_size, kernel_size,)
            else:
                assert type(kernel_size)==tuple, 'kernel_size is assumed to be a tuple but got {}'.format(type(kernel_size))
                if len(kernel_size)==1:
                    assert type(kernel_size[0])==int, 'kernel_size is assumed to be a tuple of integer'
                    assert kernel_size[0] > 0, 'kernel_size must be larger than zero'
                    kernel_size = kernel_size + kernel_size
                elif len(kernel_size)==2:
                    assert type(kernel_size[0])==int, 'kernel_size is assumed to be a tuple of integer'
                    assert type(kernel_size[1])==int, 'kernel_size is assumed to be a tuple of integer'
                    assert kernel_size[0] > 0, 'kernel_size must be larger than zero.'
                    assert kernel_size[1] > 0, 'kernel_size must be larger than zero.'
                    kernel_size = kernel_size
                else:
                    assert False, 'Invalid kernel_size value {}'.format(kernel_size)

        # Layer.stride will always be a 2-tuple (stride_h, stride_w)
        if stride_h != None and stride_w != None and stride_h and stride_h > 0 and stride_w > 0:
            assert type(stride_h)==int, 'stride_h is assumed to be an integer but got {}'.format(type(stride_h))
            assert type(stride_w)==int, 'stride_w is assumed to be an integer but got {}'.format(type(stride_w))
            stride = (stride_h, stride_w,)
        else:
            assert stride!=None, 'stride is required.'
            if type(stride)==int:
                assert stride > 0, 'stride must be larger than zero'
                stride = (stride, stride,)
            else:
                assert type(stride)==tuple, 'stride is assumed to be a tuple but got {}'.format(type(stride))
                if len(stride)==1:
                    assert type(stride[0])==int, 'stride is assumed to be a tuple of integer'
                    assert stride[0] > 0, 'stride must be larger than zero'
                    stride = stride + stride
                elif len(stride)==2:
                    assert type(stride[0])==int, 'stride is assumed to be a tuple of integer'
                    assert type(stride[1])==int, 'stride is assumed to be a tuple of integer'
                    assert stride[0] > 0, 'stride must be larger than zero'
                    assert stride[1] > 0, 'stride must be larger than zero'
                    stride = stride
                else:
                    assert False, 'Invalid stride value {}'.format(stride)

        # Layer.pad will always be a 4-tuple, (pad_n, pad_s, pad_w, pad_e)
        if pad_n != None and pad_s !=None and pad_w!=None and pad_e!=None:
            assert type(pad_n)==int, 'pad_n is assumed to be an integer but got {}'.format(type(pad_n))
            assert type(pad_s)==int, 'pad_s is assumed to be an integer but got {}'.format(type(pad_s))
            assert type(pad_w)==int, 'pad_w is assumed to be an integer but got {}'.format(type(pad_w))
            assert type(pad_e)==int, 'pad_e is assumed to be an integer but got {}'.format(type(pad_e))
            assert pad_n >= 0, 'pad_n must be non-negative, but get {}'.format(pad_n)
            assert pad_w >= 0, 'pad_w must be non-negative, but get {}'.format(pad_w)
            pad = (pad_n, pad_s, pad_w, pad_e,) # upper, down, left, right
        else:
            assert pad!=None, 'pad is required.'
            if type(pad)==int:
                assert pad >= 0, 'pad must be non-negative'
                pad = (pad, pad, pad, pad,)
            else:
                assert type(pad)==tuple, 'pad is assumed to be a tuple but got {}'.format(type(pad))
                if len(pad)==1:
                    assert type(pad[0])==int, 'pad is assumed to be a tuple of integer'
                    assert pad[0] >= 0, 'pad must be non-negative'
                    pad = pad + pad + pad + pad
                elif len(pad)==2:
                    assert type(pad[0])==int, 'pad is assumed to be a tuple of integer'
                    assert type(pad[1])==int, 'pad is assumed to be a tuple of integer'
                    assert pad[0] >= 0, 'pad must be non-negative'
                    assert pad[1] >= 0, 'pad must be non-negative'
                    pad = (pad[0], pad[0], pad[1], pad[1])
                elif len(pad)==4:
                    assert type(pad[0])==int, 'pad is assumed to be a tuple of integer'
                    assert type(pad[1])==int, 'pad is assumed to be a tuple of integer'
                    assert type(pad[2])==int, 'pad is assumed to be a tuple of integer'
                    assert type(pad[3])==int, 'pad is assumed to be a tuple of integer'
                    assert pad[0] >= 0, 'pad_n must be non-negative, but get {}'.format(pad[0])
                    # assert pad[1] >= 0, 'pad_s must be non-negative, but get {}'.format(pad[1])
                    assert pad[2] >= 0, 'pad_w must be non-negative, but get {}'.format(pad[2])
                    # assert pad[3] >= 0, 'pad_e must be non-negative, but get {}'.format(pad[3])
                    pad = pad
                else: # len(stride)==2
                    assert False, 'Invalid pad value {}'.format(pad)

        # Layer.dilation will always be a 2-tuple (dilation_h, dilation_w)
        if dilation_h != None and dilation_w != None and dilation_h > 0 and dilation_w > 0:
            assert type(dilation_h)==int, 'dilation_h is assumed to be an integer but got {}'.format(type(dilation_h))
            assert type(dilation_w)==int, 'dilation_w is assumed to be an integer but got {}'.format(type(dilation_w))
            dilation = (dilation_h, dilation_w,)
        else:
            if type(dilation)==int:
                assert dilation > 0, 'dilation must be larger than zero'
                dilation = (dilation, dilation,)
            else:
                assert type(dilation)==tuple, 'dilation is assumed to be a tuple but got {}'.format(type(dilation))
                if len(dilation)==1:
                    assert type(dilation[0])==int, 'dilation is assumed to be a tuple of integer'
                    assert dilation[0] > 0, 'dilation must be larger than zero'
                    dilation = dilation + dilation
                elif len(dilation)==2:
                    assert type(dilation[0])==int, 'dilation is assumed to be a tuple of integer'
                    assert type(dilation[1])==int, 'dilation is assumed to be a tuple of integer'
                    assert dilation[0] > 0, 'dilation must be larger than zero'
                    assert dilation[1] > 0, 'dilation must be larger than zero'
                    dilation = dilation
                else:
                    assert False, 'Invalid dilation value {}'.format(dilation)

        self.type = self.type_dict['pooling']
        self.name = name
        self.bottom = bottom
        self.top = top
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.pool = pool
        self.dilation = dilation

    def set_batchnorm_layer(self, name, bottom, top):
        assert type(name)==str, 'Layer name must be a string, but got a {}'.format(type(name))
        if type(bottom)!=str:
            assert type(bottom)==list or type(bottom)==tuple, 'Input tensor(s) must be a string or a list/tuple of strings'
            for elem in bottom:
                assert type(elem)==str, 'Input tensor(s) must be a string or a list of string'
            if len(bottom)==1:
                bottom = bottom[0]
        if type(top)!=str:
            assert type(top)==list or type(top)==tuple, 'Output tensor(s) must be a string or a list/tuple of strings'
            for elem in top:
                assert type(elem)==str, 'Output tensor(s) must be a string or a list of string'
            if len(top)==1:
                top = top[0]
        assert type(bottom)==str, 'Find multiple inputs but only one is allowed, bottom={}'.format(bottom)
        assert type(top)==str, 'Find multiple outputs but only one is allowed, top={}'.format(top)

        self.type = self.type_dict['batchnorm']
        self.name = name
        self.bottom = bottom
        self.top = top

    def set_bias_layer(self, name, bottom, top):
        assert type(name)==str, 'Layer name must be a string, but got a {}'.format(type(name))
        if type(bottom)!=str:
            assert type(bottom)==list or type(bottom)==tuple, 'Input tensor(s) must be a string or a list/tuple of strings'
            for elem in bottom:
                assert type(elem)==str, 'Input tensor(s) must be a string or a list of string'
            if len(bottom)==1:
                bottom = bottom[0]
        if type(top)!=str:
            assert type(top)==list or type(top)==tuple, 'Output tensor(s) must be a string or a list/tuple of strings'
            for elem in top:
                assert type(elem)==str, 'Output tensor(s) must be a string or a list of string'
            if len(top)==1:
                top = top[0]
        assert type(bottom)==str, 'Find multiple inputs but only one is allowed, bottom={}'.format(bottom)
        assert type(top)==str, 'Find multiple outputs but only one is allowed, top={}'.format(top)

        self.type = self.type_dict['bias']
        self.name = name
        self.bottom = bottom
        self.top = top

    def set_scale_layer(self, name, bottom, top, bias_term=True):
        assert type(name)==str, 'Layer name must be a string, but got a {}'.format(type(name))
        if type(bottom)!=str:
            assert type(bottom)==list or type(bottom)==tuple, 'Input tensor(s) must be a string or a list/tuple of strings'
            for elem in bottom:
                assert type(elem)==str, 'Input tensor(s) must be a string or a list of string'
            if len(bottom)==1:
                bottom = bottom[0]
        if type(top)!=str:
            assert type(top)==list or type(top)==tuple, 'Output tensor(s) must be a string or a list/tuple of strings'
            for elem in top:
                assert type(elem)==str, 'Output tensor(s) must be a string or a list of string'
            if len(top)==1:
                top = top[0]
        assert type(bottom)==str, 'Find multiple inputs but only one is allowed, bottom={}'.format(bottom)
        assert type(top)==str, 'Find multiple outputs but only one is allowed, top={}'.format(top)

        self.type = self.type_dict['scale']
        self.name = name
        self.bottom = bottom
        self.top = top
        self.bias_term = bias_term
    
    def set_scale_by_tensor_layer(self, name, bottom, top):
        assert type(name)==str, 'Layer name must be a string, but got a {}'.format(type(name))
        assert type(bottom)==list or type(bottom)==tuple, 'Input tensors must be a list/tuple of strings'
        assert len(bottom) == 2, "ScaleByTensor layer must have 2 input tensors"
        if type(top)!=str:
            assert type(top)==list or type(top)==tuple, 'Output tensor(s) must be a string or a list/tuple of strings'
            for elem in top:
                assert type(elem)==str, 'Output tensor(s) must be a string or a list of string'
            if len(top)==1:
                top = top[0]
        assert type(top)==str, 'Find multiple outputs but only one is allowed, top={}'.format(top)

        self.type = self.type_dict['scalebytensor']
        self.name = name
        self.bottom = bottom
        self.top = top

    def set_slice_layer(self, name, bottom, top, axis=1, slice_point=None):
        assert type(name)==str, 'Layer name must be a string, but got a {}'.format(type(name))
        if type(bottom)!=str:
            assert type(bottom)==list or type(bottom)==tuple, 'Input tensor(s) must be a string or a list/tuple of strings'
            for elem in bottom:
                assert type(elem)==str, 'Input tensor(s) must be a string or a list of string'
            if len(bottom)==1:
                bottom = bottom[0]
        if type(top)!=str:
            assert type(top)==list or type(top)==tuple, 'Output tensor(s) must be a string or a list/tuple of strings'
            for elem in top:
                assert type(elem)==str, 'Output tensor(s) must be a string or a list of string'
            if len(top)==1:
                top = top[0]
        assert type(bottom)==str, 'Find multiple inputs but only one is allowed, bottom={}'.format(bottom)
        assert type(top)==list, 'At least 2 top tensors required.'
        assert len(top) > 1, 'At least 2 top tensors required.'

        if slice_point != None:
            assert type(slice_point)==list or type(slice_point)==tuple, 'Parameter slice_point should be none or a list/tuple with length that equals length of top minus 1'
            assert len(slice_point)==len(top)-1, 'Parameter slice_point should be none or a list with length that equals length of top minus 1'

        self.type = self.type_dict['slice']
        self.name = name
        self.bottom = bottom
        self.top = top
        self.axis = axis
        self.slice_point = slice_point

    def set_concat_layer(self, name, bottom, top, axis=1):
        assert type(name)==str, 'Layer name must be a string, but got a {}'.format(type(name))
        if type(bottom)!=str:
            assert type(bottom)==list or type(bottom)==tuple, 'Input tensor(s) must be a string or a list/tuple of strings'
            for elem in bottom:
                assert type(elem)==str, 'Input tensor(s) must be a string or a list of string'
            if len(bottom)==1:
                bottom = bottom[0]
        if type(top)!=str:
            assert type(top)==list or type(top)==tuple, 'Output tensor(s) must be a string or a list/tuple of strings'
            for elem in top:
                assert type(elem)==str, 'Output tensor(s) must be a string or a list of string'
            if len(top)==1:
                top = top[0]
        assert type(top)==str, 'Find multiple outputs but only one is allowed, top={}'.format(top)

        assert type(bottom)==list, 'At least 2 bottom tensors required.'
        assert len(bottom) > 1, 'At least 2 bottom tensors required.'
        self.type = self.type_dict['concat']
        self.name = name
        self.bottom = bottom
        self.top = top
        self.axis = axis

    def set_eltwise_layer(self, name, bottom, top, operation='sum'):
        assert type(name)==str, 'Layer name must be a string, but got a {}'.format(type(name))
        if type(bottom)!=str:
            assert type(bottom)==list or type(bottom)==tuple, 'Input tensor(s) must be a string or a list/tuple of strings'
            for elem in bottom:
                assert type(elem)==str, 'Input tensor(s) must be a string or a list of string'
            if len(bottom)==1:
                bottom = bottom[0]
        if type(top)!=str:
            assert type(top)==list or type(top)==tuple, 'Output tensor(s) must be a string or a list/tuple of strings'
            for elem in top:
                assert type(elem)==str, 'Output tensor(s) must be a string or a list of string'
            if len(top)==1:
                top = top[0]
        assert type(top)==str, 'Find multiple outputs but only one is allowed, top={}'.format(top)
        assert isinstance(operation, str), 'operation should be a string, but get a {}'.format(type(operation))
        operation = operation.lower()
        assert operation in ['sum', 'max', 'product'], 'operation should be in \{sum, max, product\}'
        assert type(bottom) == list, 'At least 2 bottom tensors required.'
        assert len(bottom) > 1, 'At least 2 bottom tensors required.'
        self.type = self.type_dict['eltwise']
        self.name = name
        self.bottom = bottom
        self.top = top
        self.operation = operation

    def set_softmax_layer(self, name, bottom, top):
        assert type(name)==str, 'Layer name must be a string, but got a {}'.format(type(name))
        if type(bottom)!=str:
            assert type(bottom)==list or type(bottom)==tuple, 'Input tensor(s) must be a string or a list/tuple of strings'
            for elem in bottom:
                assert type(elem)==str, 'Input tensor(s) must be a string or a list of string'
            if len(bottom)==1:
                bottom = bottom[0]
        if type(top)!=str:
            assert type(top)==list or type(top)==tuple, 'Output tensor(s) must be a string or a list/tuple of strings'
            for elem in top:
                assert type(elem)==str, 'Output tensor(s) must be a string or a list of string'
            if len(top)==1:
                top = top[0]
        assert type(bottom)==str, 'Find multiple inputs but only one is allowed, bottom={}'.format(bottom)
        assert type(top)==str, 'Find multiple outputs but only one is allowed, top={}'.format(top)

        self.type = self.type_dict['softmax']
        self.name = name
        self.bottom = bottom
        self.top = top

    def set_logsoftmax_layer(self, name, bottom, top):
        assert type(name)==str, 'Layer name must be a string, but got a {}'.format(type(name))
        if type(bottom)!=str:
            assert type(bottom)==list or type(bottom)==tuple, 'Input tensor(s) must be a string or a list/tuple of strings'
            for elem in bottom:
                assert type(elem)==str, 'Input tensor(s) must be a string or a list of string'
            if len(bottom)==1:
                bottom = bottom[0]
        if type(top)!=str:
            assert type(top)==list or type(top)==tuple, 'Output tensor(s) must be a string or a list/tuple of strings'
            for elem in top:
                assert type(elem)==str, 'Output tensor(s) must be a string or a list of string'
            if len(top)==1:
                top = top[0]
        assert type(bottom)==str, 'Find multiple inputs but only one is allowed, bottom={}'.format(bottom)
        assert type(top)==str, 'Find multiple outputs but only one is allowed, top={}'.format(top)

        self.type = self.type_dict['logsoftmax']
        self.name = name
        self.bottom = bottom
        self.top = top

    def set_permute_layer(self, name, bottom, top, order):
        assert type(name)==str, 'Layer name must be a string, but got a {}'.format(type(name))
        if type(bottom)!=str:
            assert type(bottom)==list or type(bottom)==tuple, 'Input tensor(s) must be a string or a list/tuple of strings'
            for elem in bottom:
                assert type(elem)==str, 'Input tensor(s) must be a string or a list of string'
            if len(bottom)==1:
                bottom = bottom[0]
        if type(top)!=str:
            assert type(top)==list or type(top)==tuple, 'Output tensor(s) must be a string or a list/tuple of strings'
            for elem in top:
                assert type(elem)==str, 'Output tensor(s) must be a string or a list of string'
            if len(top)==1:
                top = top[0]
        assert type(bottom)==str, 'Find multiple inputs but only one is allowed, bottom={}'.format(bottom)
        assert type(top)==str, 'Find multiple outputs but only one is allowed, top={}'.format(top)

        self.type = self.type_dict['permute']
        self.name = name
        self.bottom = bottom
        self.top = top
        self.order = order

    def set_reshape_layer(self, name, bottom, top, dim):
        assert type(name)==str, 'Layer name must be a string, but got a {}'.format(type(name))
        if type(bottom)!=str:
            assert type(bottom)==list or type(bottom)==tuple, 'Input tensor(s) must be a string or a list/tuple of strings'
            for elem in bottom:
                assert type(elem)==str, 'Input tensor(s) must be a string or a list of string'
            if len(bottom)==1:
                bottom = bottom[0]
        if type(top)!=str:
            assert type(top)==list or type(top)==tuple, 'Output tensor(s) must be a string or a list/tuple of strings'
            for elem in top:
                assert type(elem)==str, 'Output tensor(s) must be a string or a list of string'
            if len(top)==1:
                top = top[0]
        assert type(bottom)==str, 'Find multiple inputs but only one is allowed, bottom={}'.format(bottom)
        assert type(top)==str, 'Find multiple outputs but only one is allowed, top={}'.format(top)

        self.type = self.type_dict['reshape']
        self.name = name
        self.bottom = bottom
        self.top = top
        self.dim = dim

    def set_flatten_layer(self, name, bottom, top, axis):
        assert type(name)==str, 'Layer name must be a string, but got a {}'.format(type(name))
        if type(bottom)!=str:
            assert type(bottom)==list or type(bottom)==tuple, 'Input tensor(s) must be a string or a list/tuple of strings'
            for elem in bottom:
                assert type(elem)==str, 'Input tensor(s) must be a string or a list of string'
            if len(bottom)==1:
                bottom = bottom[0]
        if type(top)!=str:
            assert type(top)==list or type(top)==tuple, 'Output tensor(s) must be a string or a list/tuple of strings'
            for elem in top:
                assert type(elem)==str, 'Output tensor(s) must be a string or a list of string'
            if len(top)==1:
                top = top[0]
        assert type(bottom)==str, 'Find multiple inputs but only one is allowed, bottom={}'.format(bottom)
        assert type(top)==str, 'Find multiple outputs but only one is allowed, top={}'.format(top)

        self.type = self.type_dict['flatten']
        self.name = name
        self.bottom = bottom
        self.top = top
        self.axis = axis

    def set_shufflechannel_layer(self, name, bottom, top, group=1):
        assert type(name)==str, 'Layer name must be a string, but got a {}'.format(type(name))
        if type(bottom)!=str:
            assert type(bottom)==list or type(bottom)==tuple, 'Input tensor(s) must be a string or a list/tuple of strings'
            for elem in bottom:
                assert type(elem)==str, 'Input tensor(s) must be a string or a list of string'
            if len(bottom)==1:
                bottom = bottom[0]
        if type(top)!=str:
            assert type(top)==list or type(top)==tuple, 'Output tensor(s) must be a string or a list/tuple of strings'
            for elem in top:
                assert type(elem)==str, 'Output tensor(s) must be a string or a list of string'
            if len(top)==1:
                top = top[0]
        assert type(bottom)==str, 'Find multiple inputs but only one is allowed, bottom={}'.format(bottom)
        assert type(top)==str, 'Find multiple outputs but only one is allowed, top={}'.format(top)

        self.type = self.type_dict['shufflechannel']
        self.name = name
        self.bottom = bottom
        self.top = top
        self.group = group

"""
get a dict whose key is tensor name in a net and value is the Layer objects whose bottom is or contains the key 
"""
def get_dict_name_to_bottom(layer_list):
    result = {}
    for layer in layer_list:
        if layer.type!='Input':
            bottom = layer.bottom
            if type(bottom)==str:
                if bottom in result.keys():
                    result[bottom].append(layer)
                else:
                    result[bottom] = [layer]
            else:
                for bottom_name in sorted(bottom, reverse=True):
                    if bottom_name in result.keys():
                        result[bottom_name].append(layer)
                    else:
                        result[bottom_name] = [layer]
    return result            

"""
get a dict whose key is tensor name in a net and value is the Layer objects whose top is or contains the key 
"""
def get_dict_name_to_top(layer_list):
    result = {}
    for layer in layer_list:
        top = layer.top
        if type(top)==str:
            if top in result.keys():
                result[top].append(layer)
                assert False, "Tensor {} is top of layer {} and {}".format(top, result[top][0].name, result[top][1].name)
            else:
                result[top] = [layer]
        else:
            for top_name in sorted(top, reverse=True):
                if top_name in result.keys():
                    result[top_name].append(layer)
                else:
                    result[top_name] = [layer]
    return result         

'''
This function draws a tensor name list which includes features (only tops of the layers), weights and biases.
'''
def get_tensor_name_list(layer_list, feature_only=False):
    tensor_name_list = list(get_dict_name_to_top(layer_list).keys())

    if not feature_only:
        for lyr in layer_list:
            if lyr.type in ['InnerProduct', 'Convolution', 'Scale']:
                new_tensor_name = lyr.name+'_weight'
                assert new_tensor_name not in tensor_name_list, 'Tensor name collision, {} has been in the list.'
                tensor_name_list.append(new_tensor_name)
                if lyr.bias_term:
                    new_tensor_name = lyr.name + '_bias'
                    assert new_tensor_name not in tensor_name_list, 'Tensor name collision, {} has been in the list.'
                    tensor_name_list.append(new_tensor_name)
            elif lyr.type=='BatchNorm':
                new_tensor_name = lyr.name + '_weight'
                assert new_tensor_name not in tensor_name_list, 'Tensor name collision, {} has been in the list.'
                tensor_name_list.append(new_tensor_name)
                new_tensor_name = lyr.name + '_bias'
                assert new_tensor_name not in tensor_name_list, 'Tensor name collision, {} has been in the list.'
                tensor_name_list.append(new_tensor_name)
            elif lyr.type=='Bias':
                new_tensor_name = lyr.name + '_bias'
                assert new_tensor_name not in tensor_name_list, 'Tensor name collision, {} has been in the list.'
                tensor_name_list.append(new_tensor_name)
            else:
                pass

    return tensor_name_list

'''
This function draws a layer name list.
'''
def get_layer_name_list(layer_list):
    layer_name_list = []
    for lyr in layer_list:
        layer_name_list.append(lyr.name)
    return layer_name_list

"""
to check whether there is a tensor used by more 1 multi-input layer, no assertion here, only return result
"""
def check_net_multi_innput_layer(layer_list):
    result = True
    dict_name_to_bot = get_dict_name_to_bottom(layer_list)
    for bottom in dict_name_to_bot.keys():
        layer_bottom = dict_name_to_bot[bottom]
        if len(layer_bottom) < 2:
            continue
        count_multi_input_layer = 0
        for layer in layer_bottom:
            if type(layer.bottom)!=str:
                if len(layer.bottom) > 1:
                    count_multi_input_layer += 1
        if count_multi_input_layer > 1:
            print("WARNING: tensor {} is used by more than 1 multi-input layer".format(bottom))
            result = False
    return result

"""
get a list of tensor names which are only included in the tops, but never in bottoms 
"""
def get_network_output(layer_list):
    output_list = set(get_tensor_name_list(layer_list, feature_only=True))
    for layer in layer_list:
        if hasattr(layer, 'bottom'):
            bottom = layer.bottom
            if type(bottom)==str:
                if bottom in output_list:
                    output_list.remove(bottom)
            else:
                for tensor_name in bottom:
                    if tensor_name in output_list:
                        output_list.remove(tensor_name)
    return list(output_list)


"""
to get input tensor names and output tensor names of a net
"""    
def get_net_input_output(layer_list):
    input_tensor = []
    output_tensor = []
    dict_name_to_top = get_dict_name_to_top(layer_list)
    dict_name_to_bot = get_dict_name_to_bottom(layer_list)
    for top in dict_name_to_top.keys():
        if top not in dict_name_to_bot.keys():
            output_tensor.append(top)
    for bot in dict_name_to_bot:
        if bot not in dict_name_to_top.keys():
            input_tensor.append(bot)
    assert len(input_tensor)==0, "All tensors should be top of at least 1 layer, check tensor {}".format(input_tensor)
    for layer in layer_list:
        if layer.type == 'Input':
            top = layer.top
            if type(top) == str:
                input_tensor.append(top)
            else:
                for top_name in top:
                    input_tensor.append(top_name)
    return input_tensor, output_tensor

def check_bottom_top_name_conflict(layer_list):
    for layer in layer_list:
        if layer.type not in no_input_layer:
            if type(layer.bottom)==str:
                if type(layer.top)==str:
                    assert layer.bottom!=layer.top, "A layer's top name and bottom name should be different, but layer {}'s has the same top and bottom".format(layer.name)
                else:
                    assert layer.bottom not in layer.top, "Layer {}'s bottom name is one of the layer's tops".format(layer.name)
            else:
                if type(layer.top)==str:
                    assert layer.top not in layer.bottom, "Layer {}'s top is one of the layer's bottoms".format(layer.name)
                else:
                    for top in layer.top:
                        assert top not in layer.bottom, "Layer {}'s top {} is one of the layer's bottoms".format(layer.name, top)

def check_layer_name_conflict(layer_list):
    layer_name = []
    for layer in layer_list:
        assert layer.name not in layer_name, "There are more than 1 layer named {}".format(layer.name)
        layer_name.append(layer.name)

def check_bottom_and_top(layer_list):
    for layer in layer_list:
        if layer.type in one_input_layer:
            assert type(layer.bottom)==str, "{} layers should have only 1 input tensor, but get {} in layer {}".format(layer.type, layer.bottom, layer.name)
        if layer.type in one_output_layer:
            assert type(layer.top)==str, "{} layers should have only 1 output tensor, but get {} in layer {}".format(layer.type, layer.top, layer.name)
        if layer.type in multi_input_layer:
            assert type(layer.bottom)==list and len(layer.bottom)>1, "{} layers should have at least 2 input tensors, but get {} in layer".format(layer.bottom, layer.name)
        if layer.type in multi_output_layer:
            assert type(layer.top)==list and len(layer.top)>1, "{} layers should have at least 2 output tensors, but get {} in layer".format(layer.top, layer.name)

def check_layers(layer_list, param_dict=None):
    result = False
    check_bottom_and_top(layer_list)
    check_bottom_top_name_conflict(layer_list)
    check_layer_name_conflict(layer_list)
    input_tensors, output_tensors = get_net_input_output(layer_list)
    assert len(input_tensors)==1, "A net with more than 1 input tensor is not supported, but get input tensors are {}".format(input_tensors)
    assert len(output_tensors)==1, "A net with more than 1 output tensor is not supported, but get output tensors are {}".format(output_tensors)
    if type(param_dict)!=type(None):
        assert type(param_dict)==dict, "param_dict should be None or a dict"
        for k, v in param_dict.items():
            # if type(v)!=numpy.ndarray:
            #     print("Warning: item {}'s value is not of type numpy.ndarray, it will be cast to numpy.ndarray".format(k))
            #     param_dict[k] = numpy.array(v, dtype = numpy.float64)
            try:
                numpy.array(v, dtype=numpy.float64)
            except Exception:
                raise ValueError('item {}\'s value is not a number'.format(k))
    tensor_shape = get_tensor_shape(layer_list)
    for layer in layer_list:
        if layer.name in one_input_layer:
            result = result or check_by_layer_and_input_shape(tensor_shape[layer.bottom], layer, 8)
            result = result or check_by_layer_and_input_shape(tensor_shape[layer.bottom], layer, 16)
        if layer.type=='Convolution':
            if type(param_dict)!=type(None):
                assert layer.name+'_weight' in param_dict.keys(), "layer {}'s weight should be in param_dict".format(layer.name)
                shape = param_dict[layer.name+'_weight'].shape
                assert len(shape)==4, "Convolution layer's weight should be a 4D tensor, but get shape {}, check layer {}".format(shape, layer.name)
                co = shape[0]
                ci = shape[1]
                kh = shape[2]
                kw = shape[3]
                assert layer.num_output==co, "Layer {}'s output channel is {}, but weight's shape is {}".format(layer.name, layer.num_output, shape)
                assert layer.kernel_size[0]==kh, "Layer {}'s kernel height is {}, but weight's shape is {}".format(layer.name, layer.kernel_size[0], shape)
                assert layer.kernel_size[1]==kw, "Layer {}'s kernel width is {}, but weight's shape is {}".format(layer.name, layer.kernel_size[1], shape)
                if layer.bias_term:
                    assert layer.name+'_bias' in param_dict.keys(), "layer {}'s bias should be in param_dict because this layer's has bias".format(layer.name)
                    shape = param_dict[layer.name+'_bias'].shape
                    assert len(shape)==1, "Convolution layer's bias should be a vector (i.e. 1D tensor), but get shape {}, check layer {}".format(shape, layer.name)
                    assert shape[0]==layer.num_output, "Convolution layer's bias vector's length should equal to output channel, but get bias vector's length = {} and output channel = {} in layer {}".format(shape[0], layer.num_output, layer.name)
                g = layer.group
                assert g <= 65536, "Convolution layer's group should be no larger than 65536, but get {} in layer {}".format(
                    g, layer.name)
                assert ci <= 65536, "Convolution layer's input channel should be no larger than 65536, but get {} in layer {}".format(
                    ci, layer.name)
                cig = ci
                assert cig * kh * kw < 65536, "Convolution layer's number of accumulation, i.e. input_channel_per_group * kernel_height * kernel_width + 1 should be no larger than 65536, but get {} in layer {}".format(
                    cig * kh * kw + 1, layer.name)
                if cig == 1 or kh > 1 or kw > 1:
                    assert kh >= layer.stride[0], "Convolution layer's stride should be no larger than kernel size, except for the residual layer in resnet whose input channel larger than 1, but get kernel {} stride {} in height in layer {}".format(
                        kh, layer.stride[0], layer.name)
                    assert kw >= layer.stride[1], "Convolution layer's stride should be no larger than kernel size, except for the residual layer in resnet whose input channel larger than 1, but get kernel {} stride {} in width in layer {}".format(
                        kw, layer.stride[1], layer.name)

            assert layer.pad[0] < layer.kernel_size[0], "Layer {}'s upper padding ({}) must be smaller than kernel height ({}).".format(layer.name, layer.pad[0], layer.kernel_size[0])
            assert layer.pad[1] < layer.kernel_size[0], "Layer {}'s lower padding ({}) must be smaller than kernel height ({}).".format(layer.name, layer.pad[1], layer.kernel_size[0])
            assert layer.pad[2] < layer.kernel_size[1], "Layer {}'s left padding ({}) must be smaller than kernel width ({}).".format(layer.name, layer.pad[2], layer.kernel_size[1])
            assert layer.pad[3] < layer.kernel_size[1], "Layer {}'s right padding ({}) must be smaller than kernel width ({}).".format(layer.name, layer.pad[3], layer.kernel_size[1])
            assert layer.num_output<=65536, "Convolution layer's output channel should be no larger than 65536, but get {} in layer {}".format(layer.num_output, layer.name)
            assert layer.num_output%layer.group==0, "Convolution layer's output channel should be multiple of group, but get {} output channels and group {} in layer {}".format(layer.num_output, layer.group, layer.name)
            sh = layer.stride[0]
            sw = layer.stride[1]
            dh = layer.dilation[0]
            dw = layer.dilation[1]
            assert sh==sw, "Convolution layer's stride on height and width should be the same, but get stride_height = {} and stride_width = {} in layer {}".format(sh, sw, layer.name)
            assert sh<=6, "Convolution layer's stride should be no larger than 6, but get {} in layer {}".format(sh, layer.name)
            assert layer.kernel_size[0]<=16, "Convolution layer's kernel height should be no larger than 16, but get {} in layer {}".format(layer.kernel_size[0], layer.name)
            assert layer.kernel_size[1]<=16, "Convolution layer's kernel width should be no larger than 16, but get {} in layer {}".format(layer.kernel_size[1], layer.name)
            assert dh==dw, "Convolution layer's dilation on height and width should be the same, but get dilation_height = {} and dilation_width = {} in layer {}".format(dh, dw, layer.name)
            assert dh<=6, "Convolution layer's dilation should be no larger than 6, but get {} in layer {}".format(dh, layer.name)
            assert dh*(layer.kernel_size[0]-1)+1<=32, "Convolution layer's dilated kernel size, i.e. (k-1)*d+1 should be no larger than 32, but get {} in height in layer {}".format(dh*(layer.kernel_size[0]-1)+1, layer.name)
            assert dw*(layer.kernel_size[1]-1)+1<=32, "Convolution layer's dilated kernel size, i.e. (k-1)*d+1 should be no larger than 32, but get {} in width in layer {}".format(dw*(layer.kernel_size[1]-1)+1, layer.name)
        elif layer.type=="Pooling":
            kh = layer.kernel_size[0]
            kw = layer.kernel_size[1]
            sh = layer.stride[0]
            sw = layer.stride[1]
            assert kh>=sh, "Pooling layer's stride should be no larger than kernel size, but get kernel {} stride {} in height in layer {}".format(kh, sh, layer.name)
            assert kw>=sw, "Pooling layer's stride should be no larger than kernel size, but get kernel {} stride {} in width in layer {}".format(kw, sw, layer.name)
            assert sh==sw, "Pooling layer's stride on height and width should be the same, but get stride_height = {} and stride_width = {} in layer {}".format(sh, sw, layer.name)
            assert sh<=6, "Pooling layer's stride should be no larger than 6, but get {} in layer {}".format(sh, layer.name)
            assert kh<=16, "Pooling layer's kernel height should be no larger than 16, but get {} in layer {}".format(kh, layer.name)
            assert kw<=16, "Pooling layer's kernel width should be no larger than 16, but get {} in layer {}".format(kw, layer.name)
            assert layer.pad[0] < layer.kernel_size[0], "Pooling layer {}'s upper padding ({}) must be smaller than kernel height ({}).".format(layer.name, layer.pad[0], layer.kernel_size[0])
            assert layer.pad[1] < layer.kernel_size[0], "Pooling layer {}'s lower padding ({}) must be smaller than kernel height ({}).".format(layer.name, layer.pad[1], layer.kernel_size[0])
            assert layer.pad[2] < layer.kernel_size[1], "Pooling layer {}'s left padding ({}) must be smaller than kernel width ({}).".format(layer.name, layer.pad[2], layer.kernel_size[1])
            assert layer.pad[3] < layer.kernel_size[1], "Pooling layer {}'s right padding ({}) must be smaller than kernel width ({}).".format(layer.name, layer.pad[3], layer.kernel_size[1])
        elif layer.type=="InnerProduct":
            assert layer.num_output<=65536, "InnerProduct layer's output length should be no larger than 65536, but get {} in layer {}".format(layer.num_output, layer.name)
            if type(param_dict)!=type(None):
                shape = param_dict[layer.name+'_weight'].shape
                assert len(shape)==2 or len(shape)==4, "InnerProduct layer's weight should be 2D/4D tensor, but get shape {}, check layer {}".format(shape, layer.name)
                co = shape[0]
                if len(shape)==2:
                    ci = shape[1]
                else:
                    assert shape[2]<=16, "InnerProduct layer's 4D-kernel height should be no larger than 16, but get {} in layer {}".format(shape[2], layer.name)
                    assert shape[3]<=16, "InnerProduct layer's 4D-kernel width should be no larger than 16, but get {} in layer {}".format(shape[3], layer.name)
                    ci = shape[1] * shape[2] * shape[3]
                assert layer.num_output==co, "Layer {}'s output length is {}, but weight's shape is {}".format(layer.name, layer.num_output, shape)
                assert ci<=65536, "InnerProduct layer's input length should be no larger than 65536, but get {} in layer {}".format(ci, layer.name)
        elif layer.type=="Slice":
            assert layer.axis==1, "Slicing should be on channel, i.e. axis should be 1, but get {} in layer {}".format(layer.axis, layer.name)
        elif layer.type=="Concat":
            assert layer.axis==1, "Concat should be on channel, i.e. axis should be 1, but get {} in layer {}".format(layer.axis, layer.name)
            bottom = list(set(layer.bottom))
            assert len(bottom)==len(layer.bottom), "Tensors concatenated should be different, but get tensor's names {} in layer {}".format(layer.bottom, layer.name)
        elif layer.type=="Eltwise":
            assert layer.operation=='sum', "Elementwise operation should be sum, others are not supported, but get {} in layer {}".format(layer.operation, layer.name)
            bottom = list(set(layer.bottom))
            assert len(bottom)==len(layer.bottom), "Only doing elemenetwise operation on different tensors is supported, but get tensor's names {} in layer {}".format(layer.bottom, layer.name)
        elif layer.type=='Permute':
            assert False, "Permute is not supported, you might want to seperate your net"
        elif layer.type=='Reshape':
            assert False, "Reshape is not supported, you might want to seperate your net"
        elif layer.type=='Flatten':
            assert False, "Flatten is not supported, you might want to seperate your net"
        elif layer.type=='Softmax':
            print("WARNING: Softmax is not supported, it will be replaced by logsoftmax, which is supported")
        elif layer.type in ['BatchNorm', 'Scale']:
            if type(param_dict)!=type(None):
                assert layer.name+'_weight' in param_dict.keys(), "layer {}'s weight should be in param_dict".format(layer.name)
                shape = param_dict[layer.name+'_weight'].shape
                assert len(shape)==1, "BatchNorm or scale layer's weight should be a vector (i.e. 1D tensor), but get shape {}, check layer {}".format(shape, layer.name)
                co = shape[0]
                if layer.type == 'BatchNorm' or layer.bias_term:
                    assert layer.name+'_bias' in param_dict.keys(), "layer {}'s bias should be in param_dict because this layer's has bias".format(layer.name)
                    shape = param_dict[layer.name+'_bias'].shape
                    assert len(shape)==1, "BatchNorm or scale layer's bias should be a vector (i.e. 1D tensor), but get shape {}, check layer {}".format(shape, layer.name)
                    assert shape[0]==co, "BatchNorm or scale layer's bias vector's length should equal to its weight's, but get bias vector's length = {} and weight vector's length = {} in layer {}".format(shape[0], co, layer.name)
            
    return result

def check_by_layer_and_input_shape(shape_in, layer, bits_width):
    result = False
    assert bits_width in [8, 16], "bits_width needs to be 8 or 16, but get {}".format(bits_width)
    assert max(shape_in)<=65536, "tensor's size along every dimension should be no larger than 65536, but get shape {} in layer {}".format(shape_in, layer.name)
    assert min(shape_in)>=1, "tensor's size along every dimension should be positive, but get shape {} in layer {}".format(shape_in, layer.name)
    if layer.type == 'Convolution':
        assert len(shape_in)==4, "Convolution's input should be a 4D tensor, but get shape {} in layer {}".format(shape_in, layer.name)
        ci = shape_in[1]
        hi = shape_in[2]
        wi = shape_in[3]
        pn = layer.pad_n
        ps = layer.pad_s
        pe = layer.pad_e
        pw = layer.pad_w
        kh = layer.kernel_size[0]
        s = layer.stride[0]
        d = layer.dilation[0]
        g = layer.group
        cig = ci // g
        khd = (kh - 1) * d + 1
        wip = wi + pe + pw
        hip = hi + pn + ps
        wip_div = wip // ((32 // bits_width) * s)
        fi_khd_row_size = wip_div * 32 * s * cig * khd
        assert hip<=65536, "Input height with padding should be no larger than 65536, but get {} in layer {}".format(hip, layer.name)
        assert wip<=65536, "Input width with padding should be no larger than 65536, but get {} in layer {}".format(wip, layer.name)
        if bits_width == 8:
            assert fi_khd_row_size<=262144, "Input elements used for calculating one row of output should occupy no more than 262144 in iram, but they occupy {} bytes in layer {}".format(fi_khd_row_size, layer.name)
        elif fi_khd_row_size > 262144:
            result = True
            print("Input elements used for calculating one row of output should occupy no more than 262144 in iram, but they occupy {} bytes in layer {} in 16bit mode".format(fi_khd_row_size, layer.name))
    if layer.type == "Pooling":
        assert len(shape_in)==4, "Pooling's input should be a 4D tensor, but get shape {} in layer {}".format(shape_in, layer.name)
        hi = shape_in[2]
        wi = shape_in[3]
        pn = layer.pad_n
        ps = layer.pad_s
        pe = layer.pad_e
        pw = layer.pad_w
        kh = layer.kernel_size[0]
        s = layer.stride[0]        
        wip = wi + pe + pw
        hip = hi + pn + ps
        wip_div = wip // ((32 // bits_width) * s)
        fi_khd_row_size = wip_div * 32 * s * kh
        assert hip<=65536, "Input height with padding should be no larger than 65536, but get {} in layer {}".format(hip, layer.name)
        assert wip<=65536, "Input width with padding should be no larger than 65536, but get {} in layer {}".format(wip, layer.name)
        if bits_width == 8:
            assert fi_khd_row_size<=262144, "Input elements used for calculating one row of output should occupy no more than 262144 in iram, but they occupy {} bytes in layer {}".format(fi_khd_row_size, layer.name)
        elif fi_khd_row_size > 262144:
            result = True
            print("Input elements used for calculating one row of output should occupy no more than 262144 in iram, but they occupy {} bytes in layer {} in 16bit mode".format(fi_khd_row_size, layer.name))
    if layer.type in ["LogSoftmax", "Softmax"]:
        assert len(shape_in)==4, "Softmax's input and LogSoftmax's input should be a 4D tensor, but get shape {} in layer {}".format(shape_in, layer.name)
        ci = shape_in[1]
        hi = shape_in[2]
        wi = shape_in[3]
        assert ci<=4096, "Softmax's input channel and LogSoftmax's input channel should be no larger than 4096, but get {} in layer {}".format(ci, layer.name)
        assert hi==1, "Softmax's input height and LogSoftmax's input height should be 1, but get {} in layer {}".format(hi, layer.name)
        if bits_width == 8:
            assert wi<=32, "Softmax's input elements and LogSoftmax's input elements in a whole row should occupy no larger than 32bytes, but get {} in layer {}".format(wi*bits_width, layer.name)
        elif wi > 16:
            result = True
            print("Softmax's input elements and LogSoftmax's input elements in a whole row should occupy no larger than 32bytes, but get {} in layer {} in 16bit mode".format(wi*bits_width, layer.name))
    if layer.type == "InnerProduct":
        assert len(shape_in)==4, "InnerProduct's input should be a 4D tensor, but get shape {} in layer {}".format(shape_in, layer.name)
        hi = shape_in[2]
        wi = shape_in[3]
        assert hi<=16, "InnerProduct's input height should be no larger than 16, but get {} in layer {}".format(hi, layer.name)
        assert wi<=16, "InnerProduct's input width should be no larger than 16, but get {} in layer {}".format(wi, layer.name)
    if layer.type == "Eltwise":
        assert len(shape_in)==4, "Eltwise layer's inputs should be 4D tensors, but get shape {} in layer {}".format(shape_in, layer.name)
        ci = shape_in[1]
        hi = shape_in[2]
        wi = shape_in[3]
        assert ci*hi<=65536, "Eltwise's layer's input channel multiply by height should be no larger than 65536, but get {} channels and {} heights in layer {}".format(ci, hi, layer.name)
    return result

def sort_layer_by_name(layer_list, reverse=False):
    def insert_to_list_by_name(lyr_list, lyr):
        idx = 0
        for tlyr in lyr_list:
            if reverse:
                if lyr.name < tlyr.name:
                    idx +=1
                else:
                    break
            else:
                if lyr.name > tlyr.name:
                    idx +=1
                else:
                    break
        lyr_list.insert(idx, lyr)
        return lyr_list
    sorted_lyr_list = []
    for layer in layer_list:
        sorted_lyr_list = insert_to_list_by_name(sorted_lyr_list, layer)
    return sorted_lyr_list

'''
case reverse False:
    dict key: layer names; value: layers whose bottoms contains one of key layer's tops
case reverse True:
    dict key: layer names; value: layers whose tops contains one of key layer's bottoms
'''
def get_adjacent_layer_dict(layer_list, reverse=False):
    adjacent_dict = {}
    if reverse:
        dict_name_to_top = get_dict_name_to_top(layer_list)
        for layer in layer_list:
            name = layer.name
            adjacent_dict[name] = []
            if layer.type != 'Input':
                if type(layer.bottom) == str:
                    if layer.bottom in dict_name_to_top:
                        adjacent_dict[name] += dict_name_to_top[layer.bottom]
                else:
                    for bottom in layer.bottom:
                        if bottom in dict_name_to_top:
                            adjacent_dict[name] += dict_name_to_top[bottom]
    else:
        dict_name_to_bot = get_dict_name_to_bottom(layer_list)
        for layer in layer_list:
            name = layer.name
            adjacent_dict[name] = []
            if type(layer.top) == str:
                if layer.top in dict_name_to_bot:
                    adjacent_dict[name] += dict_name_to_bot[layer.top]
            else:
                for top in layer.top:
                    if top in dict_name_to_bot:
                        adjacent_dict[name] += dict_name_to_bot[top]
    return adjacent_dict

def sort_layers(layer_list):
    slayer_list = sort_layer_by_name(layer_list, reverse=False)
    adjacent_dict = get_adjacent_layer_dict(slayer_list)
    for key in adjacent_dict.keys():
        adjacent_dict[key] = sort_layer_by_name(adjacent_dict[key], reverse=True)
    dfs = []
    result = []
    visited = []
    input_layers = []
    for layer in slayer_list:
        if layer.type == 'Input':
            input_layers.append(layer)
    for input_layer in input_layers:
        dfs.append(input_layer)
        visited.append(input_layer.name)
        while len(dfs) > 0:
            current_layer = dfs[-1]
            stack_out = True
            to_push_layer = None
            adjacent_layers = list(adjacent_dict[current_layer.name])
            adjacent_layers.reverse()
            for next_layer in adjacent_layers:
                if next_layer.name not in visited:
                    visited.append(next_layer.name)
                    to_push_layer = next_layer
                    stack_out = False
                    break
            if stack_out:
                result.insert(0, current_layer)
                dfs.pop()
            else:
                dfs.append(to_push_layer)
    return result

def get_tensor_shape(layer_list):
    sorted_layer_list = sort_layers(layer_list)
    assert sorted_layer_list[0].type == 'Input', "First layer should be of type Input, but got {} instead".format(sorted_layer_list[0].type)
    tensor_shape = {}
    for layer in layer_list:
        if layer.type == "Input":
            tensor_shape[layer.top] = list(copy.deepcopy(layer.dim))
        elif layer.type in ['Sigmoid', 'ReLU', 'ReLU6', 'TanH', 'BatchNorm', 'Bias', 'Scale', 'Softmax', 'LogSoftmax', 'ShuffleChannel']:
            tensor_shape[layer.top] = copy.deepcopy(tensor_shape[layer.bottom])
        elif layer.type == 'InnerProduct':
            tensor_shape[layer.top] = [1, layer.num_output, 1, 1]
        elif layer.type in ['Convolution', 'Pooling']:
            c_i = tensor_shape[layer.bottom][1]
            h_i = tensor_shape[layer.bottom][2]
            w_i = tensor_shape[layer.bottom][3]
            k_h_d = (layer.kernel_size[0] - 1) * layer.dilation[0] + 1
            k_w_d = (layer.kernel_size[1] - 1) * layer.dilation[1] + 1
            s_h = layer.stride[0]
            s_w = layer.stride[1]
            p_n = layer.pad[0]
            p_s = layer.pad[1]
            p_w = layer.pad[2]
            p_e = layer.pad[3]
            h_o = (h_i + p_n + p_s - k_h_d) // s_h + 1
            w_o = (w_i + p_w + p_e - k_w_d) // s_w + 1
            c_o = c_i if layer.type == 'Pooling' else layer.num_output
            tensor_shape[layer.top] = [1, c_o, h_o, w_o]
        elif layer.type == 'Eltwise':
            tensor_shape[layer.top] = copy.deepcopy(tensor_shape[layer.bottom[0]])
        elif layer.type == 'Concat':
            dim_along_concat = [tensor_shape[bottom][layer.axis] for bottom in layer.bottom]
            tensor_shape[layer.top] = copy.deepcopy(tensor_shape[layer.bottom[0]])
            tensor_shape[layer.top][layer.axis] = sum(dim_along_concat)
        elif layer.type == 'Slice':
            zero_input = numpy.zeros(tensor_shape[layer.bottom])
            if layer.slice_point == None:
                try:
                    zero_output = numpy.split(zero_input, len(layer.top), layer.axis)
                    for i in range(len(layer.top)):
                        tensor_shape[layer.top[i]] = list(zero_output[i].shape)
                except ValueError as e:
                    raise Exception("Error in slice layer named {}: {}".format(layer.name, e))
            else:
                zero_output = numpy.split(zero_input, layer.slice_point, layer.axis)
                for i in range(len(layer.top)):
                    assert zero_output[i].shape[layer.axis] > 0, "output shape should be a non-zero list, but got layer {}'s input shape is {} and output tensor {}'s shape is {}, please check your slice_point".format(layer.name, tensor_shape[layer.bottom], i + i, zero_output[i]) 
                    tensor_shape[layer.top[i]] = list(zero_output[i].shape)
        elif layer.type == 'ScaleByTensor':
            tensor_shape[layer.top] = copy.deepcopy(tensor_shape[layer.bottom[0]])
    return tensor_shape

        
if __name__=='__main__':
    conv1 = Layer(lyr_type='Convolution', name='Conv1', bottom=['conv1_in'], top='conv1_out', num_output=4, kernel_size_h=0, kernel_size_w=1, kernel_size=4, stride=2, pad=2)
    fc1 = Layer()
    fc1.set_innerproduct_layer(name='fc1', bottom='fc1_in', top='fc1_out', num_output=16)
    concat1 = Layer(lyr_type='Concat', name='concat1', bottom=['fc1_out', 'conv1_in'], top=['con1_out'], axis=1)
    maxpool = Layer(lyr_type='Pooling', name='maxpool', bottom=['conv1_out'], top='conv1_out', stride_h=0, stride_w=1, kernel_size=4, stride=2, pad=(1,1,2,2), pool='max')
    print(maxpool.gatherAttrs())
    # a = layer_def(lyr_type='Sigmoid', name='Sig1', bottom='a', top='b')
    # print(conv1.name, fc1.name)

    layer_list = []
    layer_list.append(Layer(lyr_type='Input', name='layer0', top='a', dim=(1,6,5,5)))
    layer_list.append(Layer(lyr_type='Slice', name='layer1', bottom='a', top=['c', 'b', 'i']))
    layer_list.append(Layer(lyr_type='Convolution', name='layer2', bottom='b', top='d', num_output=4, kernel_size_h=0,
                            kernel_size_w=1, kernel_size=4, stride=2, pad=2))
    layer_list.append(Layer(lyr_type='ReLU', name='layer3', bottom='d', top='e'))
    layer_list.append(Layer(lyr_type='ReLU', name='layer5', bottom='f', top='g'))
    layer_list.append(Layer(lyr_type='scalebytensor', name='layer6', bottom=['g', 'e'], top='h'))
    layer_list.append(Layer(lyr_type='InnerProduct', name='layer4', bottom='c', top='f', num_output=4))
    layer_list.append(Layer(lyr_type='Concat', name='layer7', bottom=['i', 'h'], top='j'))

    layer_list = sort_layers(layer_list)

    for lyr in layer_list:
        print(lyr.name)

    print('sorted again:')
    layer_list = sort_layers(layer_list)
    
    print(check_layers(layer_list))

    for lyr in layer_list:
        print(lyr)
    print(get_adjacent_layer_dict(layer_list))
    print(get_adjacent_layer_dict(layer_list, reverse = True))
    from ndk.optimize import add_identity_between_se_and_multi_input_layer
    result_layer_list, result_param_dict = add_identity_between_se_and_multi_input_layer(layer_list, {'0': numpy.ones(2)})
    for layer in result_layer_list:
        print(layer)
    
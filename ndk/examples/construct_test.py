# -*- coding: utf-8 -*-
"""
Created on Wed May 22 09:53:55 2019

@author: qing.chuandong
"""

import os
import sys

cur_dirname = os.path.dirname(os.path.realpath(__file__))
cur_dirname = os.path.dirname(os.path.dirname(cur_dirname))
sys.path.append(cur_dirname)
from ndk.layers import Layer 


def construct_layers():
    
    layers_list = []
    
    input1 = Layer()
    input1.set_input_layer(name='input1',  top='input', dim=(100,1,28,28))
    layers_list.append(input1)
    
    bias1 = Layer()
    bias1.set_bias_layer(name='bias1',bottom='input', top='bias1')
    layers_list.append(bias1)
    
    conv1 = Layer()
    conv1.set_convolution_layer(name='conv1', bottom='bias1', top='conv1', num_output=64, kernel_size=3, stride=1,pad=2,bias_term=True)
    layers_list.append(conv1)
    
    relu1 = Layer()
    relu1.set_relu_layer(name='relu1', bottom='conv1', top='relu1')
    layers_list.append(relu1)

    pool1 = Layer()
    pool1.set_pooling_layer(name='pool1', bottom='relu1', top='pool1',  kernel_size=2, stride=2,pad=0,pool='max')
    layers_list.append(pool1)
    
    conv2 = Layer()
    conv2.set_convolution_layer(name='conv2', bottom='pool1', top='conv2', num_output=64, kernel_size=5,  stride=1,pad=2, bias_term=True)
    layers_list.append(conv2)
    
    conv3 = Layer()
    conv3.set_convolution_layer(name='conv3', bottom='pool1', top='conv3', num_output=64, kernel_size=5,  stride=1,pad=2, bias_term=True)
    layers_list.append(conv3)

    sigmoid1 = Layer()
    sigmoid1.set_sigmoid_layer(name='sigmoid1', bottom='conv2', top='sigmoid1')
    layers_list.append(sigmoid1)

    tanh1 = Layer()
    tanh1.set_tanh_layer(name='tanh1', bottom='conv3', top='tanh1')
    layers_list.append(tanh1)

    concat1 = Layer()
    concat1.set_concat_layer(name='concat1', bottom=['sigmoid1','tanh1'], top='concat1')
    layers_list.append(concat1)
    
    slice1 = Layer()
    slice1.set_slice_layer(name='slice1', bottom='concat1', top=['sliceout1','sliceout2','sliceout3'],slice_point=[32,64])
    layers_list.append(slice1)
    
    ip1 = Layer()
    ip1.set_innerproduct_layer(name='ip1', bottom='sliceout1', top='ip1', num_output=30, bias_term=True)
    layers_list.append(ip1)
 
    ip2 = Layer()
    ip2.set_innerproduct_layer(name='ip2', bottom='sliceout2', top='ip2', num_output=30, bias_term=True)
    layers_list.append(ip2)
    
    ip3 = Layer()
    ip3.set_innerproduct_layer(name='ip3', bottom='sliceout3', top='ip3', num_output=40, bias_term=True)
    layers_list.append(ip3)
    
    relu2 = Layer()
    relu2.set_relu6_layer(name='relu2', bottom='ip3', top='relu2')
    layers_list.append(relu2)
    
    batchnorm1 = Layer()
    batchnorm1.set_batchnorm_layer(name='batchnorm1', bottom='ip1', top='batchnorm1')
    layers_list.append(batchnorm1)
    
    eltwise1 = Layer()
    eltwise1.set_eltwise_layer(name='eltwise1', bottom=['batchnorm1','scale1'] , top='eltwise1',operation='sum')
    layers_list.append(eltwise1)
    
    scale1 = Layer()
    scale1.set_scale_layer(name='scale1', bottom='ip2', top='scale1')
    layers_list.append(scale1)
    
    
    concat2 = Layer()
    concat2.set_concat_layer(name='concat2', bottom=['eltwise1','relu2'], top='concat2')
    layers_list.append(concat2)    

    ip4 = Layer()
    ip4.set_innerproduct_layer(name='ip4', bottom='concat2', top='ip4', num_output=10, bias_term=True)
    layers_list.append(ip4)
    
    return layers_list
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
import sys
import os
modelpack_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(modelpack_dir)
class layer_def:
    def __init__(self, caffe_layer_str):
        configs = caffe_layer_str.split('\n')
        self.bottom = []
        self.type = []
        self.top = []
        self.middle = ''
        self.name = ''
        self.num_output = 1
        self.pad_u = 0
        self.pad_d = 0
        self.pad_l = 0
        self.pad_r = 0
        self.stride = 1
        self.group = 1
        self.pool = None # None indicate no pooling, False: maxpool, True: avgpool
        self.operation = ""
        self.ceil = 1
        self.dim = []
        self.slice_point = []
        self.used = False
        self.global_pooling = False
        self.dilation = 1
        self.bias_term = True
        self.operation = "SUM"
        self.kernel_size_h = 1
        self.kernel_size_w = 1
        self.order = [] # order of permute
        self.axis = 1
        self.aspect_ratio = []
        self.negative_slope = 0
        self.output_internal_mode = 0
        self.has_not_detected_bias = True
        for config in configs:
            config = config.replace(' ', '')
            config = config.replace('\t', '')
            config = config.split("#")[0]
            if config.find("bottom:") == 0:
                self.bottom.append(config.split('"')[1])
            if config.find("top:") == 0:
                self.top.append(config.split('"')[1])
            if config.find("name:") == 0:
                self.name = config.split('"')[1]
            if config.find("type:") == 0 and len(self.type) == 0:
                self.type.append(config.split(':')[1].lower().replace('"', '').replace("'", ""))
                if self.type == ['scale'] and self.has_not_detected_bias:
                    self.bias_term = False
            if config.find("num_output:") == 0:
                self.num_output = int(config.split(':')[1])
            if config.find("bias_term:") == 0:
                if config.split(':')[1].lower().find("true") >= 0:
                    self.bias_term = True
                    self.has_not_detected_bias = False
                elif config.split(':')[1].lower().find("false") >= 0:
                    self.bias_term = False
                    self.has_not_detected_bias = False
            if config.find("pad_u:") == 0:
                self.pad_u = int(config.split(':')[1])
            if config.find("pad_d:") == 0:
                self.pad_d = int(config.split(':')[1])
            if config.find("pad_l:") == 0:
                self.pad_l = int(config.split(':')[1])
            if config.find("pad_r:") == 0:
                self.pad_r = int(config.split(':')[1])
            if config.find("pad_n:") == 0:
                self.pad_u = int(config.split(':')[1])
            if config.find("pad_s:") == 0:
                self.pad_d = int(config.split(':')[1])
            if config.find("pad_w:") == 0:
                self.pad_l = int(config.split(':')[1])
            if config.find("pad_e:") == 0:
                self.pad_r = int(config.split(':')[1])
            if config.find("pad:") == 0:
                self.pad_u = int(config.split(':')[1])
                self.pad_d = int(config.split(':')[1])
                self.pad_l = int(config.split(':')[1])
                self.pad_r = int(config.split(':')[1])
            if config.find("pad_h:") == 0:
                self.pad_u = int(config.split(':')[1])
                self.pad_d = int(config.split(':')[1])
            if config.find("pad_w:") == 0:
                self.pad_l = int(config.split(':')[1])
                self.pad_r = int(config.split(':')[1])
            if config.find("kernel_size:") == 0:
                self.kernel_size_h = int(config.split(':')[1])
                self.kernel_size_w = int(config.split(':')[1])
            if config.find("kernel_size_h:") == 0:
                self.kernel_size_h = int(config.split(':')[1])
            if config.find("kernel_size_w:") == 0:
                self.kernel_size_w = int(config.split(':')[1])
            if config.find("kernel_h:") == 0:
                self.kernel_size_h = int(config.split(':')[1])
            if config.find("kernel_w:") == 0:
                self.kernel_size_w = int(config.split(':')[1])
            if config.find("stride:") == 0:
                self.stride = int(config.split(':')[1])
            if config.find("stride_h:") == 0:
                self.stride = int(config.split(':')[1])
            if config.find("stride_w:") == 0:
                self.stride = int(config.split(':')[1])
            if config.find("dilation:") == 0:
                self.dilation = int(config.split(':')[1])
            if config.find("dilation_h:") == 0:
                self.dilation = int(config.split(':')[1])
            if config.find("dilation_w:") == 0:
                self.dilation = int(config.split(':')[1])
            if config.find("group:") == 0:
                self.group = int(config.split(':')[1])
            if config.find("pool:") == 0:
                self.pool = config.split(':')[1].upper().find("AVE") >= 0                
            if config.find("operation:") == 0:
                self.operation = config.split(':')[1].upper()
            if config.find("dim:") == 0:
                self.dim.append(int(config.split(':')[1]))
            if config.find('slice_point:') == 0:
                self.slice_point.append(int(config.split(':')[1]))
            if config.find('global_pooling') == 0:
                self.global_pooling = config.split(':')[1].upper().find('TRUE') >= 0
            if config.find('order') == 0:
                self.order.append(int(config.split(':')[1]))
            if config.find('axis') == 0:
                self.axis = int(config.split(':')[1])
            if config.find('aspect_ratio') == 0:
                self.aspect_ratio.append(float(config.split(':')[1]))
            if config.find('negative_slope') == 0:
                self.negative_slope = float(config.split(':')[1])
            if config.find('ceil_mode') == 0:
                self.ceil = int(config.split(':')[1])
            if config.find('ceil') == 0:
                self.ceil = int(config.split(':')[1])
        if 'batchnorm' in self.type:
            self.bias_term = True
        if self.type in [['sigmoid'], ['tanh'], ['relu6'], ['relu'], ['leakyrelu'], ['prelu'], ['softmax'], ['logsoftmax']]:
            self.bias_term = False
        if self.pad_d < 0:
            print("Info: you set padding down is minus number, which means {} row(s) of input will not be used in layer {}".format(-self.pad_d, self.name))
#            self.pad_d = 0
        if self.pad_r < 0:
            print("Info: you set padding right is minus number, which means {} column(s) of input will not be used in layer {}".format(-self.pad_r, self.name))
#            self.pad_r = 0
        if self.pad_u < 0:
            raise Exception("pad on the upper side should be non negative, but got {} in layer {}".format(self.pad_u, self.name))
        if self.pad_l < 0:
            raise Exception("pad on the left side should be non negative, but got {} in layer {}".format(self.pad_l, self.name))
        assert self.num_output in range(1, 65537), "num_output should be a positive integer and no larger than 65536, but got {} in layer {}".format(self.num_output, self.name)
        assert self.group in range(1, 65537), "group should be a positive integer and no larger than 65536, but got {} in layer {}".format(self.group, self.name)
        assert self.kernel_size_h in range(1, 17), "kernel height should be a positive integer and no larger than 16, but got {} in layer {}".format(self.kernel_size_h, self.name)
        assert self.kernel_size_w in range(1, 17), "kernel width should be a positive integer and no larger than 16, but got {} in layer {}".format(self.kernel_size_w, self.name)
        assert self.stride in range(1, 7), "stride should be a positive integer and no larger than 6, but got {} in layer {}".format(self.stride, self.name)
        assert self.dilation in range(1, 7), "dilation should be a positive integer and no layer than 7, but got {} in layer {}".format(self.dilation, self.name)
        if self.dilation > 1:
            assert self.kernel_size_h > 1, "when dilation > 1, kernel height should be large than 1, but got dilation = {} and kernel_height = {} in layer {}".format(self.dilation, self.kernel_size_h, self.name)
            assert self.kernel_size_w > 1, "when dilation > 1, kernel width should be large than 1, but got dilation = {} and kernel_width = {} in layer {}".format(self.dilation, self.kernel_size_w, self.name)
        assert self.dilation * (self.kernel_size_h - 1) + 1 <= 32, "Convolution layer's dilated kernel size, i.e. (k-1)*d+1 should be no larger than 32, but got {} in height in layer {}".format(self.dilation * (self.kernel_size_h - 1) + 1, self.name)
        assert self.dilation * (self.kernel_size_w - 1) + 1 <= 32, "Convolution layer's dilated kernel size, i.e. (k-1)*d+1 should be no larger than 32, but got {} in width in layer {}".format(self.dilation * (self.kernel_size_w - 1) + 1, self.name)
        assert self.pad_u in range(16), "padding on the upper side should be a non-negative integer and no larger than 15, but got {} in layer {}".format(self.pad_u, self.name)
        assert self.pad_l in range(16), "padding on the left side should be a non-negative integer and no larger than 15, but got {} in layer {}".format(self.pad_l, self.name)
        assert self.pad_d in range(-16, 256), "padding on the down side should be a non-negative integer and no larger than 255, but got {} in layer {}".format(self.pad_d, self.name)
        assert self.pad_r in range(-16, 256), "padding on the right side should be a non-negative integer and no larger than 255, but got {} in layer {}".format(self.pad_r, self.name)

def read_layers(file_name):
    layers_read = []
    net_configs = []
    layers_top_used_time = {}
    with open(file_name) as f:
        net_conf = f.read()
        net_configs = net_conf.split('layer {')
    for i in range(len(net_configs)):
       
        layer1 = layer_def(net_configs[i])
        if layer1.type == []:
            continue
        
        if len(layer1.bottom) > 0:
            if len(layer1.bottom) > 1 or len(layer1.top) == 0 or layer1.bottom[0] != layer1.top[0]:
                for bottom in layer1.bottom:
                    if bottom in layers_top_used_time.keys():
                        layers_top_used_time[bottom] += 1
                    else:
                        layers_top_used_time[bottom] = 1
        layers_read.append(layer1)
    return layers_read, layers_top_used_time, net_configs
 
def mearge_layer(layers_read, layers_top_used_time,
                 show_major_path,
                 could_merge_layer_type = [['sigmoid'], ['tanh'], ['relu6'], ['relu'], ['leakyrelu'], ['prelu'], ['softmax'], ['logsoftmax']]):    
    layers_merge = []
    for layer1 in layers_read:
        should_merge = False
        if len(layer1.bottom) == 1 and (show_major_path or layer1.type in could_merge_layer_type): 
            bottom = layer1.bottom[0]
            if bottom in layers_top_used_time.keys() and layers_top_used_time[bottom] == 1:
                should_merge = True
            if layer1.bottom == layer1.top:
                should_merge = True
        if should_merge:
            cannot_merge = True
            for merge_layer in layers_merge:
                if len(merge_layer.top) == 1 and merge_layer.top[0] == layer1.bottom[0] and 'pooling' not in merge_layer.type and 'tf_pool' not in merge_layer.type and 'relu6' not in merge_layer.type and 'relu' not in merge_layer.type and 'leakyrelu' not in merge_layer.type and 'prelu' not in merge_layer.type and 'softmax' not in merge_layer.type and 'logsoftmax' not in merge_layer.type and 'tanh' not in merge_layer.type and 'sigmoid' not in merge_layer.type and 'input' not in merge_layer.type and 'concat' not in merge_layer.type:
                    merge_layer.middle = merge_layer.top[0]
                    merge_layer.top = layer1.top
                    merge_layer.type += layer1.type
                    merge_layer.bias_term = merge_layer.bias_term or layer1.bias_term
                    merge_layer.negative_slope = layer1.negative_slope
                    cannot_merge = False
            if cannot_merge:
                layers_merge.append(layer1)
        else:
            layers_merge.append(layer1)
        
    return layers_merge

def change_nonlinear_only(layers_read, nonlinear = [['sigmoid'], ['tanh'], ['relu6'], ['relu'], ['leakyrelu'], ['prelu']]):
    for layer in layers_read:
        if layer.type in nonlinear:
            print("Warning: a single non linear layer is found, a bias layer, whose bias value is 0, will be added before this non linear layer")
            layer.type.insert(0, 'bias')
            layer.middle = layer.bottom[0]

def merge_layer_dnn(layers_read, layers_top_used_time,
                 show_major_path,
                 could_merge_layer_type = [['sigmoid'], ['tanh'], ['relu6'], ['relu'], ['leakyrelu'], ['prelu'], ['softmax'], ['logsoftmax']]):    
    layers_merge = []
    for layer1 in layers_read:
        should_merge = False
        if len(layer1.bottom) == 1 and (show_major_path or layer1.type in could_merge_layer_type): 
            bottom = layer1.bottom[0]
            if bottom in layers_top_used_time.keys() and layers_top_used_time[bottom] == 1:
                should_merge = True
            if layer1.bottom == layer1.top:
                should_merge = True
        if should_merge:
            cannot_merge = True
            for merge_layer in layers_merge:
                if len(merge_layer.top) == 1 and merge_layer.top[0] == layer1.bottom[0] and 'pooling' not in merge_layer.type and 'tf_pool' not in merge_layer.type and 'relu6' not in merge_layer.type and 'relu' not in merge_layer.type and 'leakyrelu' not in merge_layer.type and 'prelu' not in merge_layer.type and 'softmax' not in merge_layer.type and 'logsoftmax' not in merge_layer.type and 'tanh' not in merge_layer.type and 'sigmoid' not in merge_layer.type and 'input' not in merge_layer.type and 'concat' not in merge_layer.type and (layer1.type not in [['softmax'], ['logsoftmax']] or not merge_layer.bias_term):
                    merge_layer.middle = merge_layer.top[0]
                    merge_layer.top = layer1.top
                    merge_layer.type += layer1.type
                    merge_layer.bias_term = merge_layer.bias_term or layer1.bias_term
                    merge_layer.negative_slope = layer1.negative_slope
                    cannot_merge = False
            if cannot_merge:
                layers_merge.append(layer1)
        else:
            layers_merge.append(layer1)
        
    return layers_merge
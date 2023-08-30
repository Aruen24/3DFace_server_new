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
import ndk.layers as layers
class layer_def:
    def __init__(self, caffe_layer_str):
        configs = caffe_layer_str.split('\n')
        self.bottom = []
        self.type = []
        self.top = []
        self.middle = ''
        self.name = ''
        self.num_output = 1
        self.bias_term = True
        self.pad = 0
        self.pad_h = 0
        self.pad_w = 0
        self.kernel_size = 1
        self.kernel_size_h = 0
        self.kernel_size_w = 0
        self.stride = 1
        self.stride_h = 0
        self.stride_w = 0
        self.group = 1
        self.pool = None # None indicate no pooling, False: maxpool，True: avgpool
        self.operation = ""
        self.dim = []
        self.slice_point = []
        self.used = False
        self.global_pooling = False
        self.dilation = 1
        self.dilation_h = 0
        self.dilation_w = 0
        self.operation = "sum"
        self.order = [] # order of permute
        self.axis = 1
        self.aspect_ratio = []
        self.negative_slope = 0
        self.output_internal_mode = 0
        self.eps = 1e-5
        for config in configs:            
            config = config.replace(' ', '')
            config = config.replace('{', '')
            config = config.replace('}', '')
            config = config.replace("\t",'')
            if config.find("bottom:") >= 0:
                self.bottom.append(config.split('"')[1].strip('"').split('#')[0])
            if config.find("top:") >= 0:
                self.top.append(config.split('"')[1].strip('"').split('#')[0])
            if config.find("name:") >= 0:
                if self.name=='':
                    self.name = config.split('"')[1].strip('"').split('#')[0]
            if config.find("type:") >= 0 and len(self.type) == 0:
                self.type.append(config.split('"')[1].strip('"').split('#')[0].lower())
            if config.find("num_output:") >= 0:
                self.num_output = int(config.split(':')[1].strip('"').split('#')[0])
            if config.find("bias_term:") >= 0:
                self.bias_term = config.split(':')[1].strip('"').split('#')[0].find("true") >= 0
            if config.find("pad:") >= 0:
                self.pad = int(config.split(':')[1].strip('"').split('#')[0])
            if config.find("pad_h:") >= 0:
                self.pad_h = int(config.split(':')[1].strip('"').split('#')[0])
            if config.find("pad_w:") >= 0:
                self.pad_w = int(config.split(':')[1].strip('"').split('#')[0])
            if config.find("kernel_size:") >= 0:
                self.kernel_size = int(config.split(':')[1].strip('"').split('#')[0])
            if config.find("kernel_size_h:") >= 0:
                self.kernel_size_h = int(config.split(':')[1].strip('"').split('#')[0])
            if config.find("kernel_size_w:") >= 0:        
                self.kernel_size_w = int(config.split(':')[1].strip('"').split('#')[0])
            if config.find("kernel_h:") >= 0:
                self.kernel_size_h = int(config.split(':')[1].strip('"').split('#')[0])
            if config.find("kernel_w:") >= 0:        
                self.kernel_size_w = int(config.split(':')[1].strip('"').split('#')[0])
            if config.find("dilation:") >= 0:
                self.dilation = int(config.split(':')[1].strip('"').split('#')[0])
            if config.find("dilation_h:") >= 0:
                self.dilation_h = int(config.split(':')[1].strip('"').split('#')[0])
            if config.find("dilation_w:") >= 0:        
                self.dilation_w = int(config.split(':')[1].strip('"').split('#')[0])
            if config.find("stride:") >= 0:
                self.stride = int(config.split(':')[1].strip('"').split('#')[0])
            if config.find("stride_h:") >= 0:
                self.stride_h = int(config.split(':')[1].strip('"').split('#')[0])
            if config.find("stride_w:") >= 0:
                self.stride_w = int(config.split(':')[1].strip('"').split('#')[0])
            if config.find("group:") >= 0:
                self.group = int(config.split(':')[1].strip('"').split('#')[0])
            if config.find("pool:") >= 0:
                self.pool = config.split(':')[1].strip('"').split('#')[0].find("AVE") >= 0                
            if config.find("operation:") >= 0:
                self.operation = config.split(':')[1].strip('"').split('#')[0]
            if config.find("dim:") >= 0:
                config=config.split('#')[0]
                dim_str=[]
                dim_str=config.split('dim:')
                for id in range(1,len(dim_str)):
                    self.dim.append(int(dim_str[id]))
            if config.find('slice_point:') == 0:
                self.slice_point.append(int(config.split(':')[1].strip('"').split('#')[0]))
            if config.find('global_pooling') == 0:
                self.global_pooling = config.split(':')[1].upper().strip('"').split('#')[0].find('TRUE') >= 0
            if config.find('order') == 0:
                self.order.append(int(config.split(':')[1].strip('"').split('#')[0]))
            if config.find('axis') == 0:
                self.axis = int(config.split(':')[1].strip('"').split('#')[0])
            if config.find('aspect_ratio') == 0:
                self.aspect_ratio.append(float(config.split(':')[1].strip('"').split('#')[0]))
            if config.find('negative_slope') == 0:
                self.negative_slope = float(config.split(':')[1].strip('"').split('#')[0])
            if config.find('eps') == 0:
                self.eps = float(config.split(':')[1].strip('"').split('#')[0])
        if 'batchnorm' in self.type:
            self.bias_term = True
          
def read_layers(file_name): 
    layers_read = []
    net_configs = []
    layers_top_used_time = {}
    with open(file_name) as f:
        net_conf = f.read()
        net_configs = net_conf.split('layer {')
    while '' in net_configs:
        net_configs.remove('')
    for i in range(len(net_configs)):
        layer1 = layer_def(net_configs[i])
        if len(layer1.bottom) > 0:
            if len(layer1.bottom) > 1 or len(layer1.top) == 0 or layer1.bottom[0] != layer1.top[0]:
                for bottom in layer1.bottom:
                    if bottom in layers_top_used_time.keys():
                        layers_top_used_time[bottom] += 1
                    else:
                        layers_top_used_time[bottom] = 1
        layers_read.append(layer1)
    if layers_read[0].type==[]:
        layers_read.remove(layers_read[0])
    return layers_read

def get_convolution_pad(layer,input_size,weight_size,type): #input[1,C,H,W],caffe:weight[Cout,Cin,H,W],tf:[H,W,Cin,Cout],type0:conv,type1:pooling
    pad_0=0
    pad_1=0
    pad_2=0
    pad_3=0
    kernel_size0=0
    kernel_size1=0
    stride0=0
    stride1=0
    stride=layer.stride
    stride_h=layer.stride_h
    stride_w=layer.stride_w
    kernel_size=layer.kernel_size
    kernel_size_h=layer.kernel_size_h
    kernel_size_w=layer.kernel_size_w
    pad=layer.pad
    pad_w=layer.pad_w
    pad_h=layer.pad_h
    output_size=[]
    output_size.append(input_size[0])
    output_size.append(weight_size[0])
    if pad_w!=0 and pad_h!=0:
        pad_0=pad_h
        pad_2=pad_w
    else:
        pad_0=pad
        pad_2=pad
        
    if stride_h!=0 and stride_w!=0:
        stride0=stride_h
        stride1=stride_w
    else:
        stride0=stride
        stride1=stride
        
    if kernel_size_h!=0 and kernel_size_w!=0:
        kernel_size0=kernel_size_h
        kernel_size1=kernel_size_w
    else:
        kernel_size0=kernel_size
        kernel_size1=kernel_size
        
    if type==0:
        if (input_size[2]+2*pad_0-kernel_size0)%stride0!=0:
            num_stride0=math.floor((input_size[2]+2*pad_0-kernel_size0)/stride0 )
            pad_1= pad_0- (input_size[2]+2*pad_0- num_stride0*stride0-kernel_size0  )
            output_size.append(num_stride0+1)
        else :
            pad_1=pad_0
            output_size.append(math.floor((input_size[2]+2*pad_0-kernel_size0)/stride0)+1 )
    
        if (input_size[3]+2*pad_2-kernel_size1)%stride1!=0:
            num_stride1=math.floor((input_size[3]+2*pad_2-kernel_size1)/stride1 )
            pad_3= pad_2 -(input_size[3]+2*pad_2 -num_stride1*stride1- kernel_size1 )
            output_size.append(num_stride1+1)
        else :
            pad_3=pad_2
            output_size.append(math.floor((input_size[3]+2*pad_2-kernel_size1)/stride1)+1 )
        
    else:
        if (input_size[2]+2*pad_0-kernel_size0)%stride0!=0:
            num_stride0=math.floor((input_size[2]+2*pad_0-kernel_size0)/stride0 )
            pad_1= pad_0+ (num_stride0 +1) *stride0 + kernel_size0 -input_size[2]-2*pad_0
            output_size.append(num_stride0+2)
        else :
            pad_1=pad_0
            output_size.append(math.floor((input_size[2]+2*pad_0-kernel_size0)/stride0)+1 )
        if (input_size[3]+2*pad_2-kernel_size1)%stride1!=0:
            num_stride1=math.floor((input_size[3]+2*pad_2-kernel_size1)/stride1 )
            pad_3= pad_2+ (num_stride1 +1) *stride1 + kernel_size1 -input_size[2]-2*pad_2
            output_size.append(num_stride1+2)
        else:
            pad_3=pad_2
            output_size.append(math.floor((input_size[3]+2*pad_2-kernel_size1)/stride1)+1 )
    pad_all=[pad_0,pad_1,pad_2,pad_3]
   # print(layer.name,weight_size,input_size,[kernel_size0,kernel_size1],[stride0,stride1],pad_all, output_size)
    return pad_all,output_size

def rename_multi_top_bottom(layer):
    top_dict={}
    bottom_dict={}
    for lay in layer:
        if type(lay.top)==str:
            if str(lay.top) in top_dict:
                top_dict[str(lay.top)]+=1
                lay.top=str(lay.top)+'_n'+str( top_dict[str(lay.top)]  )            
            else:
                top_dict[str(lay.top)]=0
                lay.top=str(lay.top)+'_n'+str(0)
        else:     
            length=len(lay.top)
            for i in range(length):
                if str(lay.top[i]) in top_dict:
                    top_dict[str(lay.top[i])]+=1
                    lay.top[i]=str(lay.top[i])+'_n'+str( top_dict[str(lay.top[i])]  )            
                else:
                    top_dict[str(lay.top[i])]=0
                    lay.top[i]=str(lay.top[i])+'_n'+str(0)
    for lay in layer:
        if hasattr(lay, 'bottom'):
            if type(lay.bottom)==str:
                if str(lay.bottom) in bottom_dict:
                    bottom_dict[str(lay.bottom)]+=1
                    top_index=top_dict[str(lay.bottom)]
                    if top_index>=bottom_dict[str(lay.bottom)]:
                        lay.bottom=str(lay.bottom)+'_n'+str( bottom_dict[str(lay.bottom)]  )
                    else:
                        lay.bottom=str(lay.bottom)+'_n'+str( top_dict[str(lay.bottom)]  )
                else:
                    bottom_dict[str(lay.bottom)]=0
                    lay.bottom=str(lay.bottom)+'_n'+str(0)
            else:
                length=len(lay.bottom)
                for i in range(length):
                    if str(lay.bottom[i]) in bottom_dict:
                        bottom_dict[str(lay.bottom[i])]+=1
                        lay.bottom[i]=str(lay.bottom[i])+'_n'+str( bottom_dict[str(lay.bottom[i])]  )            
                    else:
                        bottom_dict[str(lay.bottom[i])]=0
                        lay.bottom[i]=str(lay.bottom[i])+'_n'+str(0)
    return layer

def rename_multi_top_bottom1(layer):
    dict_name_to_bot = {}
    for layer in layers:
        if layer.type!='Input':
            bottom = layer.bottom
            if type(bottom)==str:
                if bottom in dict_name_to_bot.keys():
                    dict_name_to_bot[bottom].append(layer)
                else:
                    dict_name_to_bot[bottom] = [layer]
            else:
                for bottom_name in bottom:
                    if bottom_name in dict_name_to_bot.keys():
                        dict_name_to_bot[bottom_name].append(layer)
                    else:
                        dict_name_to_bot[bottom_name] = [layer]
    adjacent_dict = {}
    for layer in layers:
        name = layer.name
        adjacent_dict[name] = []
        if type(layer.top) == str:
            if layer.top in dict_name_to_bot:
                adjacent_dict[name] += dict_name_to_bot[layer.top]
        else:
            for top in layer.top:
                if top in dict_name_to_bot:
                    adjacent_dict[name] += dict_name_to_bot[top]
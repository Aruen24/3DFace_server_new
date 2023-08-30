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
ndk_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ndk_dir)

import ndk.layers
from ndk.caffe_model.load_caffemodel import read_layers
from ndk.caffe_model.load_caffemodel import get_convolution_pad
import numpy as np
import caffe.proto.caffe_pb2 as caffe_pb2
import caffe
import math

_tensor_name_dict={}
def rename_tensor(name, is_in_top):
    assert type(name)==str, 'name should be a string'
    assert type(is_in_top)==bool, 'is_in_top should be a bool type variable'
    if not name in _tensor_name_dict:
        _tensor_name_dict[name] = 0
    else:
        if is_in_top:
            _tensor_name_dict[name] += 1
    return name + ':' + str(_tensor_name_dict[name])

prelu_update_top_name={}

def load_from_caffemodel(fname_prototxt, fname_caffemodel):
    model = caffe_pb2.NetParameter()
    f = open(fname_caffemodel, 'rb')
    model.ParseFromString(f.read())
    f.close()
    net = caffe.Net(fname_prototxt, fname_caffemodel, caffe.TEST)
    layers_read=read_layers(fname_prototxt)
    #layers_read=rename_multi_top_bottom(layers_read)
    img_shape=layers_read[0].dim
    weight_dict={}
    for lay in layers_read:
        if lay.name in net.params.keys():
            if lay.type==['convolution'] or lay.type==['innerproduct']:
               # N=net.params[lay.name][0].data.shape[0]
               # C=net.params[lay.name][0].data.shape[1]
               # if len(net.params[lay.name][0].data.shape)!=4:
               #     weight_dict[str(lay.bottom[0])]= ((net.params[lay.name][0].data).reshape((N,C,1,1))  ).shape
               # else:
               weight_dict[str(lay.bottom[0])]= (net.params[lay.name][0].data).shape  #dim of init
    params_prelu={}
    for id in range(len(layers_read)):
        if layers_read[id].type==['prelu']:
            params_prelu[str(layers_read[id].name)+'_weight']=net.params[layers_read[id].name][0].data
    
    layer_list=[]
    params_dict={} 
    feature_size_out={}
    feature_size_out[str(layers_read[0].top[0])]=img_shape #featuremap 
    for i in range(len(layers_read) ):
        lay1=ndk.layers.Layer()
        if layers_read[i].type ==['convolution']:
            weight_size=weight_dict[str(layers_read[i].bottom[0])]
            bottom=layers_read[i].bottom[0]
            if bottom in prelu_update_top_name:
                bottom = prelu_update_top_name[bottom]
            input_size=feature_size_out[bottom]
            pad_all, output_size=get_convolution_pad(layers_read[i],input_size,weight_size, 0)   
            if layers_read[i].kernel_size!=0:
                layers_read[i].kernel_size_h=layers_read[i].kernel_size
                layers_read[i].kernel_size_w=layers_read[i].kernel_size
            if layers_read[i].stride!=0:
                layers_read[i].stride_h=layers_read[i].stride
                layers_read[i].stride_w=layers_read[i].stride
            lay1.set_convolution_layer(name=layers_read[i].name, top=layers_read[i].top[0], bottom=bottom, num_output=layers_read[i].num_output, 
                               kernel_size_h=layers_read[i].kernel_size_h, kernel_size_w=layers_read[i].kernel_size_w,
                               kernel_size=layers_read[i].kernel_size, stride=layers_read[i].stride,stride_h=layers_read[i].stride_h, stride_w=layers_read[i].stride_w, pad=layers_read[i].pad,pad_n=pad_all[0],
                               pad_s=pad_all[1], pad_w=pad_all[2], pad_e=pad_all[3], bias_term=layers_read[i].bias_term, dilation=layers_read[i].dilation,
                               dilation_h=layers_read[i].dilation_h, dilation_w=layers_read[i].dilation_w, group=layers_read[i].group)
            feature_size_out[str(layers_read[i].top[0])]= output_size
            layer_list.append(lay1)
        elif layers_read[i].type==['pooling']:
            global_pooling= layers_read[i].global_pooling
            bottom=layers_read[i].bottom[0]
            if bottom in prelu_update_top_name:
                bottom = prelu_update_top_name[bottom]
            input_size=feature_size_out[bottom]
            weight_size=(input_size[1], 0, 0, 0)
            if global_pooling:
                pool_chk=''
                if layers_read[i].pool:
                    pool_chk='ave'
                else:
                    pool_chk='max'
                output_size=input_size
                insert_num=0
                while 1:
                    if output_size[2] <= 16 and output_size[3] <= 16:
                        break
                    lay2= ndk.layers.Layer()
                    pool_1= 16-output_size[2] % 16
                    pool_2= 16-output_size[3] % 16
                    p1= math.floor(pool_1/2)
                    p2=pool_1-p1
                    p3= math.floor(pool_2/2)
                    p4=pool_2-p3
                    lay2.set_pooling_layer(name=layers_read[i].name+str(insert_num), top=layers_read[i].top[0], bottom=bottom,
                       kernel_size_h=16, kernel_size_w=16, stride_h=16, stride_w=16, pad=layers_read[i].pad, pad_n=p1, pad_s=p2, pad_w=p3, pad_e=p4, pool=pool_chk,                                   
                       dilation=1, dilation_h=layers_read[i].dilation_h, dilation_w=layers_read[i].dilation_w)
                    insert_num+=1
                    output_size[2]=math.ceil(output_size[2]/16 )
                    output_size[3]=math.ceil(output_size[3]/16 )
                    layer_list.append(lay2)
                layers_read[i].kernel_size_h=output_size[2]
                layers_read[i].kernel_size_w=output_size[3]
                lay1.set_pooling_layer(name=layers_read[i].name, top=layers_read[i].top[0], bottom=bottom,
                                       kernel_size_h=layers_read[i].kernel_size_h, kernel_size_w=layers_read[i].kernel_size_w, stride_h=2, stride=2,
                                       stride_w=2, pad=layers_read[i].pad, pad_n=0, pad_s=0, pad_w=0, pad_e=0, pool=pool_chk,                                   
                                       dilation=1, dilation_h=layers_read[i].dilation_h, dilation_w=layers_read[i].dilation_w)
                output_size[2]=1
                output_size[3]=1
                feature_size_out[str(layers_read[i].top[0])]=output_size
                layer_list.append(lay1)
            else:
                assert layers_read[i].kernel_size <= 16, 'kernel_size cannot larger than 16, but got {}'.format(layers_read[i].kernel_size)
                assert layers_read[i].kernel_size_h <= 16, 'kernel_size_h cannot larger than 16, but got {}'.format(layers_read[i].kernel_size_h)
                assert layers_read[i].kernel_size_w <= 16, 'kernel_size_w cannot larger than 16, but got {}'.format(layers_read[i].kernel_size_w)
                pad_all, output_size=get_convolution_pad(layers_read[i], input_size, weight_size, 1)     
                if layers_read[i].kernel_size!=0:
                    layers_read[i].kernel_size_h=layers_read[i].kernel_size
                    layers_read[i].kernel_size_w=layers_read[i].kernel_size
                if layers_read[i].stride!=0:
                    layers_read[i].stride_h=layers_read[i].stride
                    layers_read[i].stride_w=layers_read[i].stride
                pool_chk=''
                if layers_read[i].pool:
                    pool_chk='ave'
                else:
                    pool_chk='max'
                lay1.set_pooling_layer(name=layers_read[i].name, top=layers_read[i].top[0], bottom=bottom,
                                       kernel_size_h=layers_read[i].kernel_size_h, kernel_size_w=layers_read[i].kernel_size_w, stride_h=layers_read[i].stride_h, 
                                       stride_w=layers_read[i].stride_w, pad=layers_read[i].pad, pad_n=pad_all[0], pad_s=pad_all[1], pad_w=pad_all[2], pad_e=pad_all[3], pool=pool_chk,                                   
                                       dilation=1, dilation_h=layers_read[i].dilation_h, dilation_w=layers_read[i].dilation_w)
                feature_size_out[str(layers_read[i].top[0])]=output_size
                layer_list.append(lay1)
        elif layers_read[i].type==['relu']:            
            bottom=layers_read[i].bottom[0]
            if bottom in prelu_update_top_name:
                bottom = prelu_update_top_name[bottom]
            input_size=feature_size_out[bottom]
            feature_size_out[str(layers_read[i].top[0])]=input_size
            lay1.set_relu_layer(layers_read[i].name, bottom, layers_read[i].top[0], layers_read[i].negative_slope)
            layer_list.append(lay1)
        elif layers_read[i].type==['relu6']:
            bottom=layers_read[i].bottom[0]
            if bottom in prelu_update_top_name:
                bottom = prelu_update_top_name[bottom]
            input_size=feature_size_out[bottom]
            feature_size_out[str(layers_read[i].top[0])]=input_size
            lay1.set_relu6_layer(layers_read[i].name, bottom, layers_read[i].top[0])
            layer_list.append(lay1)
        elif layers_read[i].type==['concat']:
            for bottom_id in range(len(layers_read[i].bottom)):
                if layers_read[i].bottom[bottom_id] in prelu_update_top_name:
                    layers_read[i].bottom[bottom_id] = prelu_update_top_name[layers_read[i].bottom[bottom_id]]
            if layers_read[i].axis ==0:
                dim0=0
                for num in range(len(layers_read[i].bottom) ):
                    dim0 += feature_size_out[str(layers_read[i].bottom[num])][0]
                feature_size_out[str(layers_read[i].top[0])]=[dim0, feature_size_out[str(layers_read[i].bottom[num])][1],
                                                                    feature_size_out[str(layers_read[i].bottom[num])][2],
                                                                    feature_size_out[str(layers_read[i].bottom[num])][3]]
            elif layers_read[i].axis==1:
                dim1=0
                for num in range(len(layers_read[i].bottom) ):
                    dim1 += feature_size_out[str(layers_read[i].bottom[num])][1]
                feature_size_out[str(layers_read[i].top[0])]=[feature_size_out[str(layers_read[i].bottom[num])][0], dim1,
                                                              feature_size_out[str(layers_read[i].bottom[num])][2],
                                                              feature_size_out[str(layers_read[i].bottom[num])][3]]
            elif layers_read[i].axis==2:
                dim2=0
                for num in range(len(layers_read[i].bottom) ):
                    dim2 += feature_size_out[str(layers_read[i].bottom[num])][2]
                feature_size_out[str(layers_read[i].top[0])]=[feature_size_out[str(layers_read[i].bottom[num])][0],
                                                              feature_size_out[str(layers_read[i].bottom[num])][1], dim2,
                                                              feature_size_out[str(layers_read[i].bottom[num])][3]]
            else:
                dim3=0
                for num in range(len(layers_read[i].bottom) ):
                    dim3 += feature_size_out[str(layers_read[i].bottom[num])][3]
                feature_size_out[str(layers_read[i].top[0])]=[feature_size_out[str(layers_read[i].bottom[num])][0],
                                                              feature_size_out[str(layers_read[i].bottom[num])][1],
                                                              feature_size_out[str(layers_read[i].bottom[num])][2], dim3]

            lay1.set_concat_layer(layers_read[i].name, layers_read[i].bottom, layers_read[i].top[0], layers_read[i].axis)
            layer_list.append(lay1)
        elif layers_read[i].type==['slice']:
            length=len(layers_read[i].top)
            axis=layers_read[i].axis
            slice_points=layers_read[i].slice_point
            bottom=layers_read[i].bottom[0]
            if bottom in prelu_update_top_name:
                bottom = prelu_update_top_name[bottom]
            start_slice=0
            end_slice=0
            slice_collect=[]
            for j in range(length-1):
                start_slice=end_slice
                end_slice=start_slice+slice_points[j]
                slice_collect.append(end_slice - start_slice)
            if axis==0:   
                slice_collect.append( feature_size_out[bottom][0] - end_slice)
                for num in range(length):
                    feature_size_out[str(layers_read[i].top[num])]=[slice_collect[num],
                                                                  feature_size_out[bottom][1],
                                                                  feature_size_out[bottom][2],
                                                                  feature_size_out[bottom][3]]
            elif axis==1:
                slice_collect.append( feature_size_out[str(layers_read[i].bottom[0])][1] - end_slice)
                for num in range(length):
                    feature_size_out[str(layers_read[i].top[num])]=[feature_size_out[bottom][0], slice_collect[num],
                                                                    feature_size_out[bottom][2],
                                                                    feature_size_out[bottom][3]]    
            elif axis==2:
                slice_collect.append( feature_size_out[str(layers_read[i].bottom[0])][2] - end_slice)
                for num in range(length):
                    feature_size_out[str(layers_read[i].top[num])]=[feature_size_out[bottom][0],
                                                                    feature_size_out[bottom][1], slice_collect[num],
                                                                    feature_size_out[bottom][3]]  
            else:
                slice_collect.append( feature_size_out[str(layers_read[i].bottom[0])][3] - end_slice)
                for num in range(length):
                    feature_size_out[str(layers_read[i].top[num])]=[feature_size_out[bottom][0],
                                                                    feature_size_out[bottom][1],
                                                                    feature_size_out[bottom][2], slice_collect[num]]
            lay1.set_slice_layer(layers_read[i].name, bottom, layers_read[i].top, layers_read[i].axis, layers_read[i].slice_point)
            layer_list.append(lay1)
        elif layers_read[i].type==['eltwise']:
            for bottom_id in range(len(layers_read[i].bottom)):
                if layers_read[i].bottom[bottom_id] in prelu_update_top_name:
                    layers_read[i].bottom[bottom_id] = prelu_update_top_name[layers_read[i].bottom[bottom_id]]
            input_size=feature_size_out[str(layers_read[i].bottom[0])]
            feature_size_out[str(layers_read[i].top[0])]=input_size
            lay1.set_eltwise_layer(layers_read[i].name, layers_read[i].bottom, layers_read[i].top[0], layers_read[i].operation)
            layer_list.append(lay1)
        elif layers_read[i].type==['sigmoid']:
            bottom=layers_read[i].bottom[0]
            if bottom in prelu_update_top_name:
                bottom = prelu_update_top_name[bottom]
            input_size=feature_size_out[bottom]
            feature_size_out[str(layers_read[i].top[0])]=input_size
            lay1.set_sigmoid_layer(layers_read[i].name, bottom, layers_read[i].top[0])
            layer_list.append(lay1)
        elif layers_read[i].type==['tanh']:
            bottom=layers_read[i].bottom[0]
            if bottom in prelu_update_top_name:
                bottom = prelu_update_top_name[bottom]
            input_size=feature_size_out[bottom]
            feature_size_out[str(layers_read[i].top[0])]=input_size
            lay1.set_tanh_layer(layers_read[i].name, bottom, layers_read[i].top[0])
            layer_list.append(lay1)
        elif layers_read[i].type==['innerproduct']:  # output size[1,C,1,1]
            weight_size=weight_dict[str(layers_read[i].bottom[0])]
            bottom=layers_read[i].bottom[0]
            if bottom in prelu_update_top_name:
                bottom = prelu_update_top_name[bottom]
            input_size=feature_size_out[bottom]
            feature_size_out[str(layers_read[i].top[0])]=[input_size[0],weight_size[0],1,1]
            lay1.set_innerproduct_layer(layers_read[i].name, bottom, layers_read[i].top[0], layers_read[i].num_output, layers_read[i].bias_term)
            layer_list.append(lay1)
        elif layers_read[i].type==['batchnorm']:
            bottom=layers_read[i].bottom[0]
            if bottom in prelu_update_top_name:
                bottom = prelu_update_top_name[bottom]
            input_size=feature_size_out[bottom]
            feature_size_out[str(layers_read[i].top[0])]=input_size
            lay1.set_batchnorm_layer(layers_read[i].name, bottom, layers_read[i].top[0])
            lay1.eps = layers_read[i].eps
            layer_list.append(lay1)
        elif layers_read[i].type==['bias']:
            bottom=layers_read[i].bottom[0]
            if bottom in prelu_update_top_name:
                bottom = prelu_update_top_name[bottom]
            input_size=feature_size_out[bottom]
            feature_size_out[str(layers_read[i].top[0])]=input_size
            lay1.set_bias_layer(layers_read[i].name, bottom, layers_read[i].top[0])
            layer_list.append(lay1)
        elif layers_read[i].type==['scale']:
            bottom=layers_read[i].bottom[0]
            if bottom in prelu_update_top_name:
                bottom = prelu_update_top_name[bottom]
            input_size=feature_size_out[bottom]
            feature_size_out[str(layers_read[i].top[0])]=input_size
            lay1.set_scale_layer(layers_read[i].name, bottom, layers_read[i].top[0], layers_read[i].bias_term)
            layer_list.append(lay1)
        elif layers_read[i].type==['softmax']:
            bottom=layers_read[i].bottom[0]
            if bottom in prelu_update_top_name:
                bottom = prelu_update_top_name[bottom]
            input_size=feature_size_out[bottom]
            feature_size_out[str(layers_read[i].top[0])]=input_size
            lay1.set_logsoftmax_layer(layers_read[i].name, bottom, layers_read[i].top[0])
            layer_list.append(lay1)
        elif layers_read[i].type==['logsoftmax']:
            bottom=layers_read[i].bottom[0]
            if bottom in prelu_update_top_name:
                bottom = prelu_update_top_name[bottom]
            input_size=feature_size_out[bottom]
            feature_size_out[str(layers_read[i].top[0])]=input_size
            lay1.set_logsoftmax_layer(layers_read[i].name, bottom, layers_read[i].top[0])
            layer_list.append(lay1)
        elif layers_read[i].type==['permute']:
            order=layers_read[id].order
            feature_size_out[str(layers_read[i].top[0])]=order
            bottom=layers_read[i].bottom[0]
            if bottom in prelu_update_top_name:
                bottom = prelu_update_top_name[bottom]
            lay1.set_permute_layer(layers_read[i].name, bottom, layers_read[i].top[0], order)
            layer_list.append(lay1)
        elif layers_read[i].type==['reshape']: 
            dim=layers_read[i].dim            
            feature_size_out[str(layers_read[i].top[0])]=dim
            bottom=layers_read[i].bottom[0]
            if bottom in prelu_update_top_name:
                bottom = prelu_update_top_name[bottom]
            lay1.set_reshape_layer(layers_read[i].name, bottom, layers_read[i].top[0], dim)
            layer_list.append(lay1)
        elif layers_read[i].type==['flatten']: 
            bottom=layers_read[i].bottom[0]
            if bottom in prelu_update_top_name:
                bottom = prelu_update_top_name[bottom]
            input_size=feature_size_out[bottom]
            feature_size_out[str(layers_read[i].top[0])]=[1,input_size[0]*input_size[1]*input_size[2]*input_size[3], 1, 1]
            lay1.set_flatten_layer(layers_read[i].name, bottom, layers_read[i].top[0], layers_read[i].axis)
            layer_list.append(lay1)
        elif layers_read[i].type==['shufflechannel']:            
            bottom=layers_read[i].bottom[0]
            if bottom in prelu_update_top_name:
                bottom = prelu_update_top_name[bottom]
            input_size=feature_size_out[bottom]
            feature_size_out[str(layers_read[i].top[0])]=input_size
            lay1.set_shufflechannel_layer(layers_read[i].name, bottom, layers_read[i].top[0], layers_read[i].group)
            layer_list.append(lay1)
        elif layers_read[i].type==['input']:
            t=(layers_read[i].dim[0], layers_read[i].dim[1], layers_read[i].dim[2], layers_read[i].dim[3])
            lay1.set_input_layer(layers_read[i].name, layers_read[i].top[0], t)
            layer_list.append(lay1)
        elif layers_read[i].type==['prelu']:
            bottom=layers_read[i].bottom[0]
            if bottom in prelu_update_top_name:
                bottom = prelu_update_top_name[bottom]
            input_size=feature_size_out[bottom]
            top=layers_read[i].top[0]
            bottom=layers_read[i].bottom[0]
            prelu_update_top_name[top]=str(top)+str(layers_read[i].name)+'_eltwise'
            feature_size_out[str(top)+str(layers_read[i].name)+'_eltwise']=input_size
            lay2= ndk.layers.Layer()
            lay2.set_scale_layer(str(layers_read[i].name)+'_scale0', bottom, str(layers_read[i].name)+'_scale0_1', False)
            params_dict[str(layers_read[i].name)+'_scale0_weight']=np.full(input_size[1],1)
            layer_list.append(lay2)
            lay2= ndk.layers.Layer()
            lay2.set_relu_layer(str(layers_read[i].name)+'_relu0', str(layers_read[i].name)+'_scale0_1', str(layers_read[i].name)+'_relu0_1', 0)
            layer_list.append(lay2)
            lay2= ndk.layers.Layer()
            lay2.set_scale_layer(str(layers_read[i].name)+'_scale1', bottom, str(layers_read[i].name)+'_scale1_1', False)
            layer_list.append(lay2)
            params_dict[str(layers_read[i].name)+'_scale1_weight']=np.full(input_size[1],-1)
            lay2= ndk.layers.Layer()
            lay2.set_relu_layer(str(layers_read[i].name)+'_relu1', str(layers_read[i].name)+'_scale1_1', str(layers_read[i].name)+'_relu1_1', 0)
            layer_list.append(lay2)
            lay2= ndk.layers.Layer()
            lay2.set_scale_layer(str(layers_read[i].name)+'_scale2', str(layers_read[i].name)+'_relu1_1', str(layers_read[i].name)+'_scale2_1', False)
            layer_list.append(lay2)
            params_dict[str(layers_read[i].name)+'_scale2_weight']=params_prelu[str(layers_read[i].name)+'_weight']*(-1)
            bottom_list=[]
            bottom_list.append(str(layers_read[i].name)+'_relu0_1')
            bottom_list.append(str(layers_read[i].name)+'_scale2_1')
            lay1.set_eltwise_layer(str(layers_read[i].name)+'_eltwise', bottom_list, prelu_update_top_name[top])
            layer_list.append(lay1)
        elif layers_read[i].type==['dropout']:
            top=layers_read[i].top[0]
            bottom=layers_read[i].bottom[0]
            prelu_update_top_name[top]=bottom
        else :
            print('warning: unsupport type:', layers_read[i].type, 'Layer name:',layers_read[i].name)
            input_size=feature_size_out[str(layers_read[i].bottom[0])]
            feature_size_out[str(layers_read[i].top[0])]=input_size          
        
    for lay in layer_list:
        now_type=lay.type
        if lay.name in net.params.keys():
            if now_type=='Convolution':
                params_dict[str(lay.name)+'_weight']= net.params[lay.name][0].data 
                N=net.params[lay.name][0].data.shape[0]
                C=net.params[lay.name][0].data.shape[1]
                if len(net.params[lay.name][0].data.shape)!=4:
                    params_dict[str(lay.name)+'_weight']= (net.params[lay.name][0].data).reshape((N,C,1,1))
                if len(net.params[lay.name])==1:
                    pass
                else:
                    params_dict[str(lay.name)+'_bias']= net.params[lay.name][1].data
            elif now_type=='InnerProduct':
                params_dict[str(lay.name)+'_weight']= net.params[lay.name][0].data 
                N=net.params[lay.name][0].data.shape[0]
                H=feature_size_out[lay.bottom][2]
                W=feature_size_out[lay.bottom][3]
                if len(net.params[lay.name][0].data.shape)!=4:
                    params_dict[str(lay.name)+'_weight']= (net.params[lay.name][0].data).reshape((N,-1,H,W))
                if len(net.params[lay.name])==1:
                    pass
                else:
                    params_dict[str(lay.name)+'_bias']= net.params[lay.name][1].data
            elif now_type=='Scale':                
                params_dict[str(lay.name)+'_weight']= net.params[lay.name][0].data
                if len(net.params[lay.name])==1:
                    pass
                else:
                    params_dict[str(lay.name)+'_bias']= net.params[lay.name][1].data
            elif now_type=='BatchNorm':
                esp_value=net.params[lay.name][2].data
                mean=net.params[lay.name][0].data/esp_value
                var=net.params[lay.name][1].data/esp_value
                params_dict[str(lay.name)+'_weight']= 1/np.sqrt(lay.eps + var)
                params_dict[str(lay.name)+'_bias']= -(mean/np.sqrt(lay.eps + var) )
                
    for i in range(len(layer_list)):    
        if hasattr(layer_list[i], 'bottom'):
            if type(layer_list[i].bottom)==str:
                layer_list[i].bottom=rename_tensor(layer_list[i].bottom, False)     
            else:
                for d in range(len(layer_list[i].bottom)):
                    layer_list[i].bottom[d]=rename_tensor(layer_list[i].bottom[d], False)

        if type(layer_list[i].top)==str:
            layer_list[i].top=rename_tensor(layer_list[i].top, True)
        else:
            for d in range(len(layer_list[i].top)):
                layer_list[i].top[d]=rename_tensor(layer_list[i].top[d], True)    

    layer_list=ndk.layers.sort_layers(layer_list) 
    for lay_id in range(len(layer_list)):
        if layer_list[lay_id].type=='LogSoftmax':
            assert lay_id==len(layer_list)-1, 'logsoftmax or Softmax layer must be the last layer'
            assert ((layer_list[lay_id-1].type == 'Convolution' and layer_list[lay_id-1].group == 1) or layer_list[lay_id-1].type == 'InnerProduct'),'logsoftmax layer must be after Convolution or InnerProduct!'

       

    return layer_list, params_dict


if __name__=='__main__':
    print('Not implemenlayersted')

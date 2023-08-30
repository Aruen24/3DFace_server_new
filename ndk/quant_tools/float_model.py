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
#import pickle

import numpy as np
import os
import sys
ndk_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ndk_dir)
import ndk
import math
import ndk.quant_tools.quantize_weight as quantizeweight
import ndk.quant_tools.quant_func as qfun
from ndk.quant_tools import numpy_net
import time
from ndk.utils import print_log
import copy
import ndk.layers as ndk_layers
def run_float_model(input, 
                    dim, 
                    layer_list, 
                    numpy_layers, 
                    param_dict):
    feature_map_store={}
    tensor_size_list={}
    x_data=input.reshape((1, dim[1], dim[2], dim[3]))
    for id in range(len(layer_list)):
        layer= layer_list[id]   
        now_top= layer.top
        if hasattr(layer, 'bottom')==0:
            feature_map_store[str(now_top)]=numpy_layers[id].run(x_data, quant=False)
        else:
            now_bottom=layer.bottom
            if type(now_top)==str and type(now_bottom)==str:
                input_tensor=feature_map_store[str(now_bottom)]
                feature_map_store[str(now_top)] = numpy_layers[id].run(input_tensor, quant=False)
            elif type(now_top)==str:
                now_bottom= layer.bottom
                list_data_in=[]
                length=len(now_bottom)
                for i in range(length):
                    list_data_in.append(feature_map_store[str(now_bottom[i])])
                feature_map_store[str(now_top)]= numpy_layers[id].run(list_data_in, quant=False)
                tensor_size_list[str(now_top)]=feature_map_store[str(now_top)].shape
            elif type(now_bottom)==str:
                now_bottom= layer.bottom
                length=len(now_top)
                input_tensor=feature_map_store[str(now_bottom)]
                output= numpy_layers[id].run(input_tensor, quant=False)
                for i in range(length):
                    feature_map_store[str(now_top[i])]=output[i]
                    tensor_size_list[str(now_top[i])]=feature_map_store[str(now_top[i])].shape
    return feature_map_store

def get_max_and_tensor_size(data_generator, 
                            layer_list, 
                            numpy_layers, 
                            param_dict, 
                            num_step_pre):
    max_vals = {}#[0 for _ in range(0, len(layer_list))]
    tensor_size_list={}
    min_input=0
    for step in range(num_step_pre):
        feature_map_store={}
        data = next(data_generator)
        data_batch= data['input']
        tmp_min = np.min(data_batch)
        min_input = min(min_input, tmp_min)
        assert data_batch.shape[2] == layer_list[0].dim[2], "input tensor H should be {}, but get {}".format(layer_list[0].dim[2],
                                                                                                       data_batch.shape[2])
        assert data_batch.shape[3] == layer_list[0].dim[3], "input tensor W should be {}, but get {}".format(layer_list[0].dim[3],
                                                                                                           data_batch.shape[3])
        for id in range(len(layer_list)):
            layer= layer_list[id]   
            now_top= layer.top
            if hasattr(layer, 'bottom')==0:
                feature_map_store[str(now_top)]=numpy_layers[id].run(data_batch, quant=False)   
                min_value=np.min(feature_map_store[str(now_top)])
                max_value=np.max(feature_map_store[str(now_top)])
                tensor_size_list[str(now_top)]=feature_map_store[str(now_top)].shape
                if str(now_top) not in max_vals:
                    max_vals[str(now_top)]=0
                else:
                    max_vals[str(now_top)] =max(max_vals[str(now_top)], max(abs(min_value), abs(max_value)))
               # max_vals[id]= max(max_vals[id], max(abs(min_value), abs(max_value)))
            else:
                now_bottom=layer.bottom
                if type(now_top)==str and type(now_bottom)==str:
                    input_tensor=feature_map_store[str(now_bottom)]
                    feature_map_store[str(now_top)] = numpy_layers[id].run(input_tensor, quant=False)
                    tensor_size_list[str(now_top)]=feature_map_store[str(now_top)].shape
                    min_value=np.min(feature_map_store[str(now_top)])
                    max_value=np.max(feature_map_store[str(now_top)])
                    if str(now_top) not in max_vals:
                        max_vals[str(now_top)]=0
                    else:
                        max_vals[str(now_top)] =max(max_vals[str(now_top)], max(abs(min_value), abs(max_value)))
                    #max_vals[id]= max(max_vals[id], max(abs(min_value), abs(max_value)))
                elif type(now_top)==str:
                    now_bottom= layer.bottom
                    list_data_in=[]
                    length=len(now_bottom)
                    for i in range(length):
                        list_data_in.append(feature_map_store[str(now_bottom[i])])
                    feature_map_store[str(now_top)]= numpy_layers[id].run(list_data_in, quant=False)
                    tensor_size_list[str(now_top)]=feature_map_store[str(now_top)].shape
                    min_value=np.min(feature_map_store[str(now_top)])
                    max_value=np.max(feature_map_store[str(now_top)])
                    if str(now_top) not in max_vals:
                        max_vals[str(now_top)]=0
                    else:
                        max_vals[str(now_top)] =max(max_vals[str(now_top)], max(abs(min_value), abs(max_value)))
                    #max_vals[id]= max(max_vals[id], max(abs(min_value), abs(max_value)))
                elif type(now_bottom)==str:
                    now_bottom= layer.bottom
                    length=len(now_top)
                    input_tensor=feature_map_store[str(now_bottom)]
                    output= numpy_layers[id].run(input_tensor, quant=False)
                    for i in range(length):
                        feature_map_store[str(now_top[i])]=output[i]
                        tensor_size_list[str(now_top[i])]=feature_map_store[str(now_top[i])].shape
                        min_value=np.min(feature_map_store[str(now_top[i])])
                        max_value=np.max(feature_map_store[str(now_top[i])])
                        if str(now_top[i]) not in max_vals:
                            max_vals[str(now_top[i])]=0
                        else:
                            max_vals[str(now_top[i])] =max(max_vals[str(now_top[i])], max(abs(min_value), abs(max_value)))
                        #max_vals[id]= max(max_vals[id], max(abs(min_value), abs(max_value)))        
    return max_vals, tensor_size_list, min_input

def set_lower_upper_frac(layer_list, quant_params, bitwidth):
    lower_tensor_frac={}
    upper_tensor_frac={}
    for lay in layer_list:
        cur_operation =lay.name
        bias_string=str(cur_operation)+'_frac_bias'
        weight_string=str(cur_operation)+'_frac_weight'
        if lay.type in ['Convolution','BatchNorm' , 'Scale','InnerProduct','Bias']:
            cur_bottom= lay.bottom
            if bias_string in quant_params and weight_string in quant_params:
                if bitwidth==8:
                    lower_frac= int(np.max(np.array(quant_params[str(bias_string)]))- np.min(np.array(quant_params[str(weight_string)])))
                    upper_frac= 15 + int(np.min(np.array(quant_params[str(bias_string)]))- np.max(np.array(quant_params[str(weight_string)])))
                    if lower_frac > upper_frac:
                        bias= (lower_frac-upper_frac)/2
                        bias_1= lower_frac-upper_frac-bias
                        quant_params[str(weight_string)]=np.maximum(quant_params[str(weight_string)],  np.min(quant_params[str(weight_string)])+bias)
                        quant_params[str(bias_string)]=np.minimum(quant_params[str(bias_string)], np.max(quant_params[str(bias_string)])-bias_1)
                        lower_frac= int(np.max(np.array(quant_params[str(bias_string)]))- np.min(np.array(quant_params[str(weight_string)])))
                        upper_frac= 15 + int(np.min(np.array(quant_params[str(bias_string)]))- np.max(np.array(quant_params[str(weight_string)])))
                    lower_tensor_frac[str(cur_bottom)]=lower_frac
                    upper_tensor_frac[str(cur_bottom)]=upper_frac
                else:
                    lower_frac= int(np.max(np.array(quant_params[str(bias_string)]))- np.min(np.array(quant_params[str(weight_string)])))
                    upper_frac= 31 + int(np.min(np.array(quant_params[str(bias_string)]))- np.max(np.array(quant_params[str(weight_string)])))
                    if lower_frac > upper_frac:
                        bias= (lower_frac-upper_frac)/2
                        bias_1= lower_frac-upper_frac-bias
                        quant_params[str(weight_string)]=np.maximum(quant_params[str(weight_string)],  np.min(quant_params[str(weight_string)])+bias)
                        quant_params[str(bias_string)]=np.minimum(quant_params[str(bias_string)], np.max(quant_params[str(bias_string)])-bias_1)
                        lower_frac= int(np.max(np.array(quant_params[str(bias_string)]))- np.min(np.array(quant_params[str(weight_string)])))
                        upper_frac= 31 + int(np.min(np.array(quant_params[str(bias_string)]))- np.max(np.array(quant_params[str(weight_string)])))
                    lower_tensor_frac[str(cur_bottom)]=lower_frac
                    upper_tensor_frac[str(cur_bottom)]=upper_frac       
            elif bias_string in quant_params:
                if bitwidth==8:
                    lower_frac= int(np.max(np.array(quant_params[str(bias_string)])))
                    upper_frac= 15 + int(np.min(np.array(quant_params[str(bias_string)])))
                    if lower_frac > upper_frac:
                        bias= lower_frac-upper_frac
                        quant_params[str(bias_string)]=np.maximum(quant_params[str(bias_string)], np.min(quant_params[str(bias_string)])+bias)
                        lower_frac= int(np.max(np.array(quant_params[str(bias_string)])))
                        upper_frac= 15 + int(np.min(np.array(quant_params[str(bias_string)])))
                    lower_tensor_frac[str(cur_bottom)]=lower_frac
                    upper_tensor_frac[str(cur_bottom)]=upper_frac
                else:
                    lower_frac= int(np.max(np.array(quant_params[str(bias_string)])))
                    upper_frac= 31 + int(np.min(np.array(quant_params[str(bias_string)])))
                    if lower_frac > upper_frac:
                        bias= lower_frac-upper_frac
                        quant_params[str(bias_string)]=np.maximum(quant_params[str(bias_string)], np.min(quant_params[str(bias_string)])+bias)
                        lower_frac= int(np.max(np.array(quant_params[str(bias_string)])))
                        upper_frac= 31 + int(np.min(np.array(quant_params[str(bias_string)])))
                    lower_tensor_frac[str(cur_bottom)]=lower_frac
                    upper_tensor_frac[str(cur_bottom)]=upper_frac            
            else:
                pass
    return lower_tensor_frac, upper_tensor_frac


def get_frac_value_quant_string(layer_list, 
                                conv_part, 
                                usr_param_dict, 
                                feature_name_list, 
                                method_dict):
    top_group = []
    quant_string = ''
    for conv in conv_part:
        if type(layer_list[conv].top) == str:
            top_group.append(layer_list[conv].top)
        else:
            for top in layer_list[conv].top:
                top_group.append(top)
    frac_chk_same=[]
    frac_chk_value=[]
    for string in top_group:
        str_frac= str(string)+'_frac'
        if str_frac in usr_param_dict and usr_param_dict[str(str_frac)] not in frac_chk_value:
            frac_chk_same.append(str_frac)
            frac_chk_value.append(usr_param_dict[str(str_frac)])
    assert len(frac_chk_same) <= 1 , '{} value must be the same with {} value, but get {} and {}'.format(frac_chk_same[0], frac_chk_same[1], frac_chk_value[0], frac_chk_value[1])
    
    for string in top_group:
        if string in feature_name_list and string in method_dict:
            quant_string = method_dict[str(string)].upper()
            break
    return frac_chk_value, quant_string

def delivery_concat_signed(layer_list, 
                           cur_bottom, 
                           cur_top, 
                           featuremap_frac):
    all_unsigned = True
    for bottom in cur_bottom:
        if str(cur_bottom)+'_signed' in featuremap_frac and featuremap_frac[str(cur_bottom) + '_signed'] == False:
            pass
        else:
            all_unsigned=False
    if all_unsigned:
        featuremap_frac[str(cur_top) + '_signed'] = False
    else:
        pass

def update_featuremapfrac_group(layer_list, 
                                 frac_value, 
                                 bottom_group,
                                 featuremap_frac):
    for id_x in bottom_group:
        for id_g in id_x:
            if type(layer_list[id_g].top)==str:
                featuremap_frac[str(layer_list[id_g].top)+'_frac']=frac_value
            else:
                for top in layer_list[id_g].top:
                    featuremap_frac[str(top)+'_frac']=frac_value
                
def set_quant_by_string(layer_list,
                        conv_part, 
                        bitwidth, 
                        quant_string, 
                        distributions, 
                        max_vals, 
                        quant_featuremap_frac,
                        next_check,
                        next_frac,
                        QUANTIZE_NUM, 
                        INTERVAL_NUM):
    if quant_string=='KL':
        quantizeweight.normalize_distribution(distributions[conv_part[-1]])
        distributions[conv_part[-1]] = np.array(distributions[conv_part[-1]])
        thre_list= quantizeweight.exp_int_threshold(distributions[conv_part[-1]], max_vals[str(layer_list[conv_part[-1]].top)], 0.7, 1.0, INTERVAL_NUM)
        threshold_bin= quantizeweight.threshold_kl_distribution(distributions[conv_part[-1]], QUANTIZE_NUM, thre_list) 
        threshold = (threshold_bin+0.5) * max_vals[str(layer_list[conv_part[-1]].top)]/INTERVAL_NUM
        signed=True
        if str(layer_list[conv_part[0]].top) + '_signed' in quant_featuremap_frac:
            signed=quant_featuremap_frac[str(layer_list[conv_part[0]].top) + '_signed']
        conv_frac= int( math.floor(bitwidth - 1 -math.log(threshold, 2) )   )
        if signed==False:
            conv_frac= int( math.floor(bitwidth - math.log(threshold, 2)  ))
        if next_check:
            conv_frac=next_frac
        for i in conv_part:
            if type(layer_list[i].top)==str:    
                quant_featuremap_frac[str(layer_list[i].top)+'_frac']=conv_frac
            else:
                for top in layer_list[i].top:
                    quant_featuremap_frac[str(top)+'_frac']=conv_frac
    elif quant_string=='MSE':
        quantizeweight.normalize_distribution(distributions[conv_part[-1]])
        distributions[conv_part[-1]] = np.array(distributions[conv_part[-1]])
        thre_list= quantizeweight.exp_int_threshold(distributions[conv_part[-1]], max_vals[str(layer_list[conv_part[-1]].top)], 0.7, 1.0, INTERVAL_NUM)
        threshold_bin= quantizeweight.threshold_mse_distribution(distributions[conv_part[-1]], QUANTIZE_NUM, thre_list) 
        threshold = (threshold_bin+0.5) * max_vals[str(layer_list[conv_part[-1]].top)]/INTERVAL_NUM
        signed=True
        if str(layer_list[conv_part[0]].top) + '_signed' in quant_featuremap_frac:
            signed=quant_featuremap_frac[str(layer_list[conv_part[0]].top) + '_signed']
        conv_frac= int( math.floor(bitwidth - 1 -math.log(threshold, 2) )    )
        if signed==False:
            conv_frac= int( math.floor(bitwidth - math.log(threshold, 2) )    )
        if next_check:
            conv_frac=next_frac
        for i in conv_part:
            if type(layer_list[i].top)==str:    
                quant_featuremap_frac[str(layer_list[i].top)+'_frac']=conv_frac
            else:
                for top in layer_list[i].top:
                    quant_featuremap_frac[str(top)+'_frac']=conv_frac    
    else:
        if quant_string!='' and quant_string!='MINMAX':
            print('WARNING: unknown type: {}, quant as default type MINMAX'.format(quant_string) )
        signed=True
        if str(layer_list[conv_part[0]].top) + '_signed' in quant_featuremap_frac:
            signed=quant_featuremap_frac[str(layer_list[conv_part[0]].top) + '_signed']
        conv_frac= int(quantizeweight.threshold_minmax_distribution(bitwidth, max_vals[str(layer_list[conv_part[-1]].top)] ,signed) )
        if next_check:
            conv_frac=next_frac
        for i in conv_part:
            if type(layer_list[i].top)==str:    
                quant_featuremap_frac[str(layer_list[i].top)+'_frac']=conv_frac
            else:
                for top in layer_list[i].top:
                    quant_featuremap_frac[str(top)+'_frac']=conv_frac
                    
def aggressive_quant(layer_list, 
                     quant_params,
                     min_input,
                     distributions, 
                     quant_layers, 
                     max_vals,
                     bitwidth, 
                     feature_name_list, 
                     INTERVAL_NUM,
                     QUANTIZE_NUM,
                     usr_param_dict=None,
                     method_dict=None):
    quant_featuremap_frac={}
    concat_frac_enforce=[]
    if min_input >= 0 and bitwidth==8:
        quant_featuremap_frac[str(layer_list[0].top) + '_signed'] = False
    else:
        quant_featuremap_frac[str(layer_list[0].top) + '_signed'] = True
    for id in range(len(quant_layers)):
        conv_part = quant_layers[id]
        for kii in conv_part:
            print_log('quantize_model: processing layer {}/{}: name={} type={}'.format(kii+1, len(layer_list), layer_list[kii].name, layer_list[kii].type))
        frac_chk_value, quant_string=get_frac_value_quant_string(layer_list, conv_part, usr_param_dict, feature_name_list, method_dict)
        if layer_list[conv_part[0]].type in[ 'Concat','Eltwise'] :
            now_top = layer_list[conv_part[0]].top
            now_bottom = layer_list[conv_part[0]].bottom
            if layer_list[conv_part[0]].type=='Concat':
                delivery_concat_signed(layer_list, now_bottom, now_top, quant_featuremap_frac)
            else:
                pass
            bottom_group = add_concat_bottom_layer(layer_list[conv_part[0]].bottom, layer_list, quant_layers)
            bottom_group.append(conv_part)  # bottom & now
            if frac_chk_value!=[]:
                update_featuremapfrac_group(layer_list, frac_chk_value[0], bottom_group, quant_featuremap_frac)
            else:       
                aggressive_concat_model(bitwidth, quant_params, quant_featuremap_frac, distributions, INTERVAL_NUM, QUANTIZE_NUM, layer_list, 
                                        bottom_group, max_vals, quant_layers, concat_frac_enforce, quant_string)
        elif len(conv_part) == 1 and layer_list[conv_part[0]].type == 'Slice':
            now_top = layer_list[conv_part[0]].top
            now_bottom = layer_list[conv_part[0]].bottom
            if str(now_bottom)+'_signed' in quant_featuremap_frac and quant_featuremap_frac[str(now_bottom) + '_signed'] == False:
                for top in now_top:
                    quant_featuremap_frac[str(top) + '_signed'] = False
            bottom_group=[]
            bottom_group.append(quant_layers[id- 1])
            bottom_group.append(conv_part)
            if frac_chk_value!=[]:
                update_featuremapfrac_group(layer_list, frac_chk_value[0], bottom_group, quant_featuremap_frac)
            else:
                for top in now_top:
                    quant_featuremap_frac[str(top) + '_frac'] = quant_featuremap_frac[str(now_bottom) + '_frac']
        elif len(conv_part) == 1 and layer_list[conv_part[0]].type in ['ShuffleChannel' ,'Permute', 'Reshape']:
            now_top = layer_list[conv_part[0]].top
            now_bottom = layer_list[conv_part[0]].bottom
            if str(now_bottom)+'_signed' in quant_featuremap_frac and quant_featuremap_frac[str(now_bottom) + '_signed'] == False:
                quant_featuremap_frac[str(now_top) + '_signed'] = False
            bottom_group=[]
            bottom_group.append(quant_layers[id- 1])
            bottom_group.append(conv_part)
            if frac_chk_value!=[]:
                update_featuremapfrac_group(layer_list, frac_chk_value[0], bottom_group, quant_featuremap_frac)
            else:
                quant_featuremap_frac[str(now_top) + '_frac'] = quant_featuremap_frac[str(now_bottom) + '_frac']
        elif len(conv_part) == 1 and (layer_list[conv_part[0]].type in['ReLU'] or (layer_list[conv_part[0]].type=='Pooling' and layer_list[conv_part[0]].pool=='max')):
            now_top = layer_list[conv_part[0]].top
            now_bottom = layer_list[conv_part[0]].bottom
            bottom_group=[]
            bottom_group.append(quant_layers[id- 1])
            bottom_group.append(conv_part)
            if frac_chk_value!=[]:
                update_featuremapfrac_group(layer_list, frac_chk_value[0], bottom_group, quant_featuremap_frac)
            else:
                quant_featuremap_frac[str(now_top) + '_frac'] = quant_featuremap_frac[str(now_bottom) + '_frac']
        elif len(conv_part) == 1 and layer_list[conv_part[0]].type in['Sigmoid', 'TanH'] :
            now_top = layer_list[conv_part[0]].top
            if bitwidth == 8:
                if frac_chk_value!=[]:
                    assert frac_chk_value[0] == 7, '8 bit quantize Sigmoid must be 7, but get {}'.format(frac_chk_value[0])
                    quant_featuremap_frac[str(now_top)+'_frac']= frac_chk_value[0]
                else:
                    quant_featuremap_frac[str(now_top) + '_frac'] = 7
            else:
                if frac_chk_value!=[]:
                    assert frac_chk_value[0] == 15, '16 bit quantize Sigmoid must be 15, but get {}'.format(frac_chk_value[0])
                    quant_featuremap_frac[str(now_top)+'_frac']=frac_chk_value[0]
                else:
                    quant_featuremap_frac[str(now_top) + '_frac'] = 15
            if layer_list[conv_part[0]].type=='Sigmoid':
                update_sigmoid_prelayer_frac(quant_layers[id - 1], bitwidth, quant_featuremap_frac, layer_list)
            else:
                update_tanh_prelayer_frac(quant_layers[id - 1], bitwidth, quant_featuremap_frac, layer_list)
        elif len(conv_part) == 1 and layer_list[conv_part[0]].type in['LogSoftmax','Softmax','ReLU6']:
            now_top = layer_list[conv_part[0]].top
            if bitwidth == 8:
                if frac_chk_value!=[]:
                    assert frac_chk_value[0] == 4, '8 bit quantize must be 4, but get {}'.format(frac_chk_value[0])
                    quant_featuremap_frac[str(now_top)+'_frac']=frac_chk_value[0]
                else:
                    quant_featuremap_frac[str(now_top) + '_frac'] = 4
            else:
                if frac_chk_value!=[]:
                    assert frac_chk_value[0] == 12, '16 bit quantize must be 12, but get {}'.format(frac_chk_value[0])
                    quant_featuremap_frac[str(now_top)+'_frac']=frac_chk_value[0]
                else:
                    quant_featuremap_frac[str(now_top) + '_frac'] = 12
            update_softmax_prelayer_frac(quant_layers[id - 1], bitwidth, quant_featuremap_frac, layer_list)
        else:
            next_check=False
            next_frac=0
            if layer_list[conv_part[0]].type in [ 'Convolution','BatchNorm' , 'Scale','InnerProduct', 'Bias']:
                start_num=id+1
                while start_num < len(quant_layers):
                    if layer_list[quant_layers[start_num][0]].type in ['ShuffleChannel','Reshape','Permute','ReLU']:
                        start_num+=1
                    elif layer_list[quant_layers[start_num][0]].type in ['Pooling'] and layer_list[quant_layers[start_num][0]].pool=='max':
                        start_num+=1
                    elif layer_list[quant_layers[start_num][0]].type in [ 'ReLU6','Sigmoid']:
                        frac_sure=4
                        if bitwidth==16:
                            frac_sure=12
                        next_check=True
                        next_frac=frac_sure
                        break
                    elif layer_list[quant_layers[start_num][0]].type in [ 'TanH']:
                        frac_sure=5
                        if bitwidth==16:
                            frac_sure=13
                        next_check=True
                        next_frac=frac_sure
                        break
                    else:
                        break 
            set_quant_by_string(layer_list, conv_part, bitwidth, quant_string, distributions, max_vals, quant_featuremap_frac, next_check, next_frac, QUANTIZE_NUM, INTERVAL_NUM)
            for lay_id in conv_part:            
                if layer_list[lay_id].type in ['Convolution','BatchNorm' , 'Scale','InnerProduct', 'Bias']:
                    update_params_to_limit_output(bitwidth,  quant_featuremap_frac[layer_list[lay_id].top+'_frac'], quant_featuremap_frac, quant_params, layer_list[lay_id])
                    update_params_to_limit_input(bitwidth, quant_featuremap_frac[layer_list[lay_id].bottom+'_frac'], quant_featuremap_frac, quant_params, layer_list[lay_id])
    return quant_featuremap_frac

def update_input_frac(bitwidth, cur_bottom_frac, lower_tensor_frac, upper_tensor_frac, quant_params, layer_list, index):
    lay=layer_list[index]
    cur_bottom=lay.bottom
    bias_string=str(lay.name)+'_frac_bias'
    weight_string=str(lay.name)+'_frac_weight'    
    if lower_tensor_frac[str(cur_bottom)] > cur_bottom_frac:
        bias= lower_tensor_frac[str(cur_bottom)]-cur_bottom_frac
        if weight_string in quant_params:
            max_bias= np.max(quant_params[str(bias_string)])
            quant_params[str(bias_string)]= np.minimum(quant_params[str(bias_string)], max_bias-bias)     
        elif bias_string in quant_params:
            max_bias= np.max(quant_params[str(bias_string)])
            quant_params[str(bias_string)]= np.minimum(quant_params[str(bias_string)], max_bias-bias) 
    if upper_tensor_frac[str(cur_bottom)] < cur_bottom_frac:
        bias= cur_bottom_frac-upper_tensor_frac[str(cur_bottom)]
        if weight_string in quant_params:
            min_bias=np.min(np.array(quant_params[str(bias_string)]))
            quant_params[str(bias_string)]=np.maximum(min_bias+bias, quant_params[str(bias_string)])                 
        elif bias_string in quant_params:
            min_bias=np.min(np.array(quant_params[str(bias_string)]))
            quant_params[str(bias_string)]=np.maximum(min_bias+bias, quant_params[str(bias_string)])  
    else:
        pass
    
def quant_merge_layers(layer_list):
    quant_layers=[] # more
    delivery_type=['ShuffleChannel','Permute','ReLU','Reshape']
    change_type=['Convolution','Scale','BatchNorm','InnerProduct', 'Bias','ReLU6','TanH','Sigmoid']
    mid_type=['Pooling']
    mix_type=['Concat','Eltwise']
    single_type=['Slice','Input','LogSoftmax','Softmax']
    lay_merge=[]
    length=len(layer_list)
    for id in range(length):  
        lay=layer_list[id]
        cur_type=lay.type
        if cur_type in single_type:
            if lay_merge==[]:
                lay_merge.append(id)
                quant_layers.append(lay_merge)
                lay_merge=[]
            else:
                quant_layers.append(lay_merge)
                lay_merge=[]
                lay_merge.append(id)
                quant_layers.append(lay_merge)
                lay_merge=[]
        elif cur_type in delivery_type:
            lay_merge.append(id)
        elif cur_type in change_type:
            if lay_merge==[]:
                lay_merge.append(id)
            else:
                quant_layers.append(lay_merge)
                lay_merge=[]
                lay_merge.append(id)
        elif cur_type in mid_type:
            if lay_merge==[]:
                lay_merge.append(id)
            else:
                if lay.pool=='max':
                    lay_merge.append(id)
                else:
                    quant_layers.append(lay_merge)
                    lay_merge=[]
                    lay_merge.append(id)
        elif cur_type in mix_type:
            if lay_merge==[]:
                lay_merge.append(id)
            else:
                quant_layers.append(lay_merge)
                lay_merge=[]
                lay_merge.append(id)
        else:
            if lay_merge==[]:
                lay_merge.append(id)
                quant_layers.append(lay_merge)
                lay_merge=[]
            else:
                quant_layers.append(lay_merge)
                lay_merge=[]
                lay_merge.append(id)
                quant_layers.append(lay_merge)
                lay_merge=[]
    if lay_merge!=[]:
        quant_layers.append(lay_merge)
    return quant_layers
def quant_merge_layers_old(merged_layer_list):
    quant_layers=[] # more
    quant_label=[0 for x in range(0, len(merged_layer_list))]
    lay_merge=[]
    length=len(merged_layer_list)
    for i in range(length):        
        now_lay=merged_layer_list[i]
        now_type=now_lay.type
        if now_type=='Input':
            quant_label[i]=1
        elif now_type in['Convolution','Scale','BatchNorm','InnerProduct', 'Bias']:
            if i+1 < length:
                if merged_layer_list[i+1].type in ['ReLU']:
                    quant_label[i]=0
                elif merged_layer_list[i+1].type=='Pooling':
                    if merged_layer_list[i+1].pool=='ave':
                        quant_label[i]=1
                    else:
                        quant_label[i]=0
                else:
                    quant_label[i]=1
            else:
                quant_label[i]=1
        elif now_type in ['ReLU']:
            if i-1>=0:
                if merged_layer_list[i-1].type in['Convolution','Scale','BatchNorm','InnerProduct', 'Bias']:
                    if i+1 < length:
                        if merged_layer_list[i+1].type=='Pooling':
                            if merged_layer_list[i+1].pool=='ave':
                                quant_label[i]=1
                            else:
                                quant_label[i]=0
                        else:
                            quant_label[i]=1
                else:
                    quant_label[i]=1
        else:
            quant_label[i]= 1
    for i in range(length):
        if lay_merge==[]:
            lay_merge.append(i)
        else:
            if quant_label[lay_merge[-1]]== 1:
                quant_layers.append(lay_merge)
                lay_merge=[]
                lay_merge.append(i)
            else:
                lay_merge.append(i)
    if lay_merge!=[]:
        quant_layers.append(lay_merge)
   
    return quant_layers

def add_concat_bottom_layer(bottom_list, 
                            layer_list, 
                            quant_layers):
    bottom_group=[]
    for bottom in bottom_list:
        for id in range(len(layer_list)):
            if hasattr(layer_list[id], 'bottom'):
                if type(layer_list[id].top)==str:
                    if layer_list[id].top==bottom:
                        for layers in quant_layers:
                            if id in layers:
                                bottom_group.append(layers)
                else:
                    for top in layer_list[id].top:
                        if top==bottom:
                            for layers in quant_layers:
                                if id in layers:
                                    bottom_group.append(layers)
    return bottom_group
    
def get_float_distribution(layer_list, 
                           data_generator, 
                           num_step, 
                           max_vals, 
                           numpy_layers, 
                           quant_params_float,
                           INTERVAL_NUM,
                           QUANTIZE_NUM):
    distributions = []
    for i1 in range(len(layer_list)):
        distribution = [0 for _ in range(0, INTERVAL_NUM)]
        distributions.append(distribution)
    print_log('quantize_model: evaluating tensor distributions...')
    tic = time.clock()
    for id_x in range(num_step):
        feature_map_store={}
        data = next(data_generator)
        feature = data['input']
        for id in range(len(layer_list)):
            layer= layer_list[id]   
            now_top= layer.top
            if hasattr(layer, 'bottom')==0:
                feature_map_store[str(now_top)]=numpy_layers[id].run(feature, quant=False)
                quantizeweight.add_to_distribution(feature_map_store[str(now_top)], distributions[id], max_vals[str(now_top)]/INTERVAL_NUM)
            else:
                now_bottom=layer.bottom
                if type(now_top)==str and type(now_bottom)==str:
                    input_tensor=feature_map_store[str(now_bottom)]
                    feature_map_store[str(now_top)] = numpy_layers[id].run(input_tensor, quant=False)
                    quantizeweight.add_to_distribution(feature_map_store[str(now_top)], distributions[id], max_vals[str(now_top)]/INTERVAL_NUM)
                elif type(now_top)==str:
                    now_bottom= layer.bottom
                    list_data_in=[]
                    length=len(now_bottom)
                    for i in range(length):
                        list_data_in.append(feature_map_store[str(now_bottom[i])])
                    feature_map_store[str(now_top)]= numpy_layers[id].run(list_data_in, quant=False)
                    quantizeweight.add_to_distribution(feature_map_store[str(now_top)], distributions[id], max_vals[str(now_top)]/INTERVAL_NUM)
                elif type(now_bottom)==str:
                    now_bottom= layer.bottom
                    length=len(now_top)
                    input_tensor=feature_map_store[str(now_bottom)]
                    output= numpy_layers[id].run(input_tensor, quant=False)
                    for j in range(length):
                        feature_map_store[str(now_top[j])]=output[j]
                    quantizeweight.add_to_distribution(feature_map_store[str(now_top[0])], distributions[id], max_vals[str(now_top[0])]/INTERVAL_NUM)
        # print log info every 5 seconds
        if time.clock() - tic > 5:
            tic = time.clock()
            print_log('quantize_model: batch {}/{} processed'.format(id_x+1, num_step))
    print_log('quantize_model: tensor distributions are obtained.')
    return distributions
                        
def aggressive_concat_model(bitwidth,
                            quant_params,
                            featuremap_frac,
                            distributions, 
                            INTERVAL_NUM, 
                            QUANTIZE_NUM, 
                            layer_list, 
                            bottom_group,
                            max_vals, 
                            quant_layers,
                            concat_frac_enforce,
                            quant_string):
    conv_part=[]
    for part in bottom_group:
        for ki in part:
            conv_part.append(ki)
    length_group=len(bottom_group)-1 #bottom
    cover=0
    out_frac=0
    for id in range(length_group):
        if layer_list[bottom_group[id][0]].type in['ReLU6' , 'Sigmoid', 'TanH']:
            cover=1
            out_frac=featuremap_frac[str(layer_list[bottom_group[id][0]].top)+'_frac']
            concat_frac_enforce.append(layer_list[bottom_group[-1][0]].name)
            break
        elif layer_list[bottom_group[id][0]].type in['Concat' , 'Eltwise'] and layer_list[bottom_group[id][0]].name in concat_frac_enforce:
            cover=1
            out_frac=featuremap_frac[str(layer_list[bottom_group[id][0]].top)+'_frac']
            concat_frac_enforce.append(layer_list[bottom_group[-1][0]].name)
            break
        elif layer_list[bottom_group[id][0]].type in ['Slice']:
            cover=1
            out_frac=featuremap_frac[str(layer_list[bottom_group[id][0]].top[0])+'_frac']
            break
        elif layer_list[bottom_group[id][0]].type in['Concat' , 'Eltwise'] :
            cover=1
            out_frac=featuremap_frac[str(layer_list[bottom_group[id][0]].top)+'_frac']
            break
    if cover==1:
        for group in bottom_group:
            for id in group:
                if type(layer_list[id].top)==str:
                    featuremap_frac[str(layer_list[id].top)+'_frac']=out_frac
                else:
                    for top in layer_list[id].top:
                        featuremap_frac[str(top)+'_frac']=out_frac
        for lay_id in conv_part:            
            if layer_list[lay_id].type in ['Convolution','BatchNorm' , 'Scale','InnerProduct', 'Bias']:
                update_params_to_limit_output(bitwidth, featuremap_frac[layer_list[lay_id].top+'_frac'], featuremap_frac, quant_params, layer_list[lay_id])
                update_params_to_limit_input(bitwidth, featuremap_frac[layer_list[lay_id].bottom+'_frac'], featuremap_frac, quant_params, layer_list[lay_id])
        for group_id in range(len(bottom_group)-1):
            recovery_bottom=[]
            recovery_bottom.append(bottom_group[group_id][-1]) #concat's input
            update_recovery_limit_input(bitwidth, recovery_bottom, featuremap_frac, quant_params, layer_list)
        for group_id in range(len(bottom_group)-1):
            id =bottom_group[group_id][0]
            if layer_list[id].type in ['Concat','Eltwise']:
                input_group = add_concat_bottom_layer(layer_list[id].bottom, layer_list, quant_layers)
                update_featuremapfrac_group(layer_list, featuremap_frac[layer_list[id].top+'_frac'], input_group, featuremap_frac)
                for group in input_group:
                    for index in group:
                        if layer_list[index].type in ['Convolution','BatchNorm' , 'Scale','InnerProduct', 'Bias']:
                            update_params_to_limit_output(bitwidth, featuremap_frac[layer_list[index].top+'_frac'], featuremap_frac, quant_params, layer_list[index])
                            update_params_to_limit_input(bitwidth, featuremap_frac[layer_list[index].bottom+'_frac'], featuremap_frac, quant_params, layer_list[index])
            elif layer_list[id].type in ['Slice']:
                input_group = add_concat_bottom_layer(layer_list[id].bottom, layer_list, quant_layers)
                update_featuremapfrac_group(layer_list, featuremap_frac[layer_list[id].top[0]+'_frac'], input_group, featuremap_frac)
                for group in input_group:
                    for index in group:
                        if layer_list[index].type in ['Convolution','BatchNorm' , 'Scale','InnerProduct', 'Bias']:
                            update_params_to_limit_output(bitwidth, featuremap_frac[layer_list[index].top+'_frac'], featuremap_frac, quant_params, layer_list[index])
                            update_params_to_limit_input(bitwidth, featuremap_frac[layer_list[index].bottom+'_frac'], featuremap_frac, quant_params, layer_list[index])
    else:
        if quant_string=='KL':
            quantizeweight.normalize_distribution(distributions[conv_part[-1]])
            distributions[conv_part[-1]] = np.array(distributions[conv_part[-1]])
            thre_list= quantizeweight.exp_int_threshold(distributions[conv_part[-1]], max_vals[str(layer_list[conv_part[-1]].top)], 0.7, 1.0, INTERVAL_NUM)
            threshold_bin= quantizeweight.threshold_kl_distribution(distributions[conv_part[-1]], QUANTIZE_NUM, thre_list) 
            threshold = (threshold_bin+0.5) * max_vals[str(layer_list[conv_part[-1]].top)]/INTERVAL_NUM
            signed=True
            if str(layer_list[conv_part[0]].top) + '_signed' in featuremap_frac:
                signed=featuremap_frac[str(layer_list[conv_part[0]].top) + '_signed']
            conv_frac= int( math.floor(bitwidth - 1 -math.log(threshold, 2)))
            if signed==False:
                conv_frac= int( math.floor(bitwidth - math.log(threshold, 2) ))
            for i in conv_part:
                if type(layer_list[i].top)==str:    
                    featuremap_frac[str(layer_list[i].top)+'_frac']=conv_frac
                else:
                    for top in layer_list[i].top:
                        featuremap_frac[str(top)+'_frac']=conv_frac
        elif quant_string=='MSE':
            quantizeweight.normalize_distribution(distributions[conv_part[-1]])
            distributions[conv_part[-1]] = np.array(distributions[conv_part[-1]])
            thre_list= quantizeweight.exp_int_threshold(distributions[conv_part[-1]], max_vals[str(layer_list[conv_part[-1]].top)], 0.7, 1.0, INTERVAL_NUM)
            threshold_bin= quantizeweight.threshold_mse_distribution(distributions[conv_part[-1]], QUANTIZE_NUM, thre_list) 
            threshold = (threshold_bin+0.5) * max_vals[str(layer_list[conv_part[-1]].top)]/INTERVAL_NUM
            signed=True
            if str(layer_list[conv_part[0]].top) + '_signed' in featuremap_frac:
                signed=featuremap_frac[str(layer_list[conv_part[0]].top) + '_signed']
            conv_frac= int( math.floor(bitwidth - 1 -math.log(threshold, 2) )   )
            if signed==False:
                conv_frac= int( math.floor(bitwidth - math.log(threshold, 2) ) )
            for i in conv_part:
                if type(layer_list[i].top)==str:    
                    featuremap_frac[str(layer_list[i].top)+'_frac']=conv_frac
                else:
                    for top in layer_list[i].top:
                        featuremap_frac[str(top)+'_frac']=conv_frac
        else:
            if quant_string!='' and quant_string!='MINMAX':
                print('WARNING: unknown type: {}, quant as default type MINMAX'.format(quant_string) )
            signed=True
            if str(layer_list[conv_part[0]].top) + '_signed' in featuremap_frac:
                signed=featuremap_frac[str(layer_list[conv_part[0]].top) + '_signed']
            conv_frac= int( quantizeweight.threshold_minmax_distribution(bitwidth, max_vals[str(layer_list[conv_part[-1]].top)], signed))
            for i in conv_part:
                if type(layer_list[i].top)==str:    
                    featuremap_frac[str(layer_list[i].top)+'_frac']=conv_frac
                else:
                    for top in layer_list[i].top:
                        featuremap_frac[str(top)+'_frac']=conv_frac

        for lay_id in conv_part:            
            if layer_list[lay_id].type in ['Convolution','BatchNorm' , 'Scale','InnerProduct', 'Bias']:
                update_params_to_limit_output(bitwidth, featuremap_frac[layer_list[lay_id].top+'_frac'], featuremap_frac, quant_params, layer_list[lay_id])
                update_params_to_limit_input(bitwidth, featuremap_frac[layer_list[lay_id].bottom+'_frac'], featuremap_frac, quant_params, layer_list[lay_id])
                
        for group_id in range(len(bottom_group)-1):
            recovery_bottom=[]
            recovery_bottom.append(bottom_group[group_id][-1]) #concat's input
            update_recovery_limit_input(bitwidth, recovery_bottom, featuremap_frac, quant_params, layer_list)
                
            
def update_recovery_limit_input(bitwidth, recovery_bottom, featuremap_frac, quant_params, layer_list):
    top_layer= ndk_layers.get_dict_name_to_bottom(layer_list)
    for id in recovery_bottom:
        bottom_name= layer_list[id].top
        if type(bottom_name)==str:
            top_group =top_layer[bottom_name] #it's top layers
            for top_pc in top_group:
                top_pc_cp= copy.deepcopy(top_pc)
                while(1):
                    if top_pc_cp.type in ['Convolution', 'BatchNorm' , 'Scale', 'InnerProduct', 'Bias']:
                        update_params_to_limit_output(bitwidth, featuremap_frac[top_pc_cp.top+'_frac'], featuremap_frac, quant_params, top_pc_cp)
                        update_params_to_limit_input(bitwidth, featuremap_frac[top_pc_cp.bottom+'_frac'], featuremap_frac, quant_params, top_pc_cp)
                        break
                    elif top_pc_cp.type in ['ShuffleChannel','Permute','Reshape','ReLU']:
                        featuremap_frac[top_pc_cp.top+'_frac']=featuremap_frac[top_pc_cp.bottom+'_frac']
                        top_pc_cp= top_layer[top_pc_cp.top][0]
                    elif top_pc_cp.type in ['pooling'] and top_pc_cp.pool=='max':
                        featuremap_frac[top_pc_cp.top+'_frac']=featuremap_frac[top_pc_cp.bottom+'_frac']
                        top_pc_cp= top_layer[top_pc_cp.top][0]
                    else:
                        break
        else:
            for bottom in bottom_name:
                top_group =top_layer[bottom] #it's top layers
                for top_pc in top_group:
                    top_pc_cp= copy.deepcopy(top_pc)
                    while(1):
                        if top_pc_cp.type in ['Convolution', 'BatchNorm' , 'Scale', 'InnerProduct', 'Bias']:
                            update_params_to_limit_output(bitwidth, featuremap_frac[top_pc_cp.top+'_frac'], featuremap_frac, quant_params, top_pc_cp)
                            update_params_to_limit_input(bitwidth, featuremap_frac[top_pc_cp.bottom+'_frac'], featuremap_frac, quant_params, top_pc_cp)
                            break
                        elif top_pc_cp.type in ['ShuffleChannel','Permute','Reshape','ReLU']:
                            featuremap_frac[top_pc_cp.top+'_frac']=featuremap_frac[top_pc_cp.bottom+'_frac']
                            top_pc_cp= top_layer[top_pc_cp.top][0]
                        elif top_pc_cp.type in ['pooling'] and top_pc_cp.pool=='max':
                            featuremap_frac[top_pc_cp.top+'_frac']=featuremap_frac[top_pc_cp.bottom+'_frac']
                            top_pc_cp= top_layer[top_pc_cp.top][0]
                        else:
                            break
                
def update_mean_var_frac_concat(  data_generator,
                                  param_dict,
                                  quant_params,
                                  bitwidth, 
                                  featuremap_frac, 
                                  layer_list, 
                                  numpy_layers,
                                  tensor_size, 
                                  bottom_group, 
                                  tensor_mean, 
                                  tensor_var, 
                                  num_step,
                                  max_vals,
                                  INTERVAL_NUM, 
                                  QUANTIZE_NUM,
                                  gaussian_approx, 
                                  quant_layers,
                                  concat_frac_enforce,
                                  quant_string=''):
    conv_part=[]
    for part in bottom_group:
        for ki in part:
            conv_part.append(ki)
    length_group=len(bottom_group)-1 #bottom
    concat_top=layer_list[conv_part[-1]].top
    sum_tensor_num=num_step*tensor_size[str(concat_top)][0]*tensor_size[str(concat_top)][1]*tensor_size[str(concat_top)][2]*tensor_size[str(concat_top)][3]
    cover=0
    out_frac=0
    for id in range(length_group):
        if layer_list[bottom_group[id][0]].type in['ReLU6' , 'Sigmoid', 'TanH']:
            cover=1
            out_frac=featuremap_frac[str(layer_list[bottom_group[id][0]].top)+'_frac']
            concat_frac_enforce.append(layer_list[bottom_group[-1][0]].name)
            break
        elif layer_list[bottom_group[id][0]].type in['Concat' , 'Eltwise'] and layer_list[bottom_group[id][0]].name in concat_frac_enforce:
            cover=1
            out_frac=featuremap_frac[str(layer_list[bottom_group[id][0]].top)+'_frac']
            concat_frac_enforce.append(layer_list[bottom_group[-1][0]].name)
            break
        elif layer_list[bottom_group[id][0]].type in ['Slice']:
            cover=1
            out_frac=featuremap_frac[str(layer_list[bottom_group[id][0]].top[0])+'_frac']
            break
        elif layer_list[bottom_group[id][0]].type in['Concat' , 'Eltwise'] :
            cover=1
            out_frac=featuremap_frac[str(layer_list[bottom_group[id][0]].top)+'_frac']
            break
    if cover==1:
        update_featuremapfrac_group(layer_list, out_frac, bottom_group, featuremap_frac)      
        if gaussian_approx:            
            sum_all=0
            for step in range(num_step): 
                src_tensor=0
                if type(layer_list[conv_part[0]].bottom)==str:
                    src_tensor=np.random.normal(tensor_mean[str(layer_list[conv_part[0]].bottom)], math.sqrt(tensor_var[str(layer_list[conv_part[0]].bottom)]), 
                                                tensor_size[str(layer_list[conv_part[0]].bottom)])
                else:
                    src_list=[]
                    for bottom in layer_list[conv_part[0]].bottom:
                        x=np.random.normal(tensor_mean[str(bottom)], math.sqrt(tensor_var[str(bottom)]), tensor_size[str(bottom)])         
                        src_list.append(x)
                    src_tensor=src_list
                feature_map_store={}
                for id in conv_part:
                    layer=layer_list[id]           
                    cur_top= layer.top
                    cur_bottom= layer.bottom
                    if type(cur_bottom)==str:
                        input_tensor=0
                        if id==conv_part[0]:
                            input_tensor=src_tensor
                        else:
                            if str(cur_bottom) in feature_map_store:
                                input_tensor=feature_map_store[str(cur_bottom)]
                            else:
                                input_tensor=np.random.normal(tensor_mean[str(cur_bottom)], math.sqrt(tensor_var[str(cur_bottom)]), tensor_size[str(cur_bottom)])
                        output= numpy_layers[id].run(input_tensor, quant=False)
                        if type(cur_top)==str:
                            feature_map_store[str(cur_top)] =output
                        else:
                            for top_id in range(len(cur_top)):
                                feature_map_store[str(cur_top[top_id])] =output[top_id]
                    else:
                        input_list=[]
                        length_bottom=len(cur_bottom)
                        for i in range(length_bottom):
                            if str(cur_bottom[i]) in feature_map_store:
                                input_list.append(feature_map_store[str(cur_bottom[i])])
                            else:
                                x=np.random.normal(tensor_mean[str(cur_bottom[i])], math.sqrt(tensor_var[str(cur_bottom[i])]), tensor_size[str(cur_bottom[i])])
                                input_list.append(x)
                        feature_map_store[str(cur_top)] = numpy_layers[id].run(input_list, quant=False)
                signed=True
                if str(layer_list[conv_part[-1]].top) + '_signed' in featuremap_frac:
                    signed=featuremap_frac[str(layer_list[conv_part[-1]].top) + '_signed']
                q_test_tensor=qfun.quantized_value(feature_map_store[str(concat_top)], bitwidth, frac=featuremap_frac[str(concat_top) + '_frac'], floor = True, signed=signed)
                sum_all+=np.sum(q_test_tensor)
            sum_all_mean=sum_all/sum_tensor_num
            ###########################################
            sum_var_all_q=0
            for step in range(num_step): 
                src_tensor=0
                if type(layer_list[conv_part[0]].bottom)==str:
                    src_tensor=np.random.normal(tensor_mean[str(layer_list[conv_part[0]].bottom)], math.sqrt(tensor_var[str(layer_list[conv_part[0]].bottom)]), 
                                                tensor_size[str(layer_list[conv_part[0]].bottom)])
                else:
                    src_list=[]
                    for bottom in layer_list[conv_part[0]].bottom:
                        x=np.random.normal(tensor_mean[str(bottom)], math.sqrt(tensor_var[str(bottom)]), tensor_size[str(bottom)])        
                        src_list.append(x)
                    src_tensor=src_list
                feature_map_store={}
                for id in conv_part:
                    layer=layer_list[id]           
                    cur_top= layer.top
                    cur_bottom= layer.bottom
                    if type(cur_bottom)==str:
                        input_tensor=0
                        if id==conv_part[0]:
                            input_tensor=src_tensor
                        else:
                            if str(cur_bottom) in feature_map_store:
                                input_tensor=feature_map_store[str(cur_bottom)]
                            else:
                                input_tensor=np.random.normal(tensor_mean[str(cur_bottom)], math.sqrt(tensor_var[str(cur_bottom)]), tensor_size[str(cur_bottom)])
                        feature_map_store[str(cur_top)] = numpy_layers[id].run(input_tensor, quant=False)
                    else:
                        input_list=[]
                        length_bottom=len(cur_bottom)
                        for i in range(length_bottom):
                            if str(cur_bottom[i]) in feature_map_store:
                                input_list.append(feature_map_store[str(cur_bottom[i])])
                            else:
                                x=np.random.normal(tensor_mean[str(cur_bottom[i])], math.sqrt(tensor_var[str(cur_bottom[i])]), tensor_size[str(cur_bottom[i])])
                                input_list.append(x)
                        feature_map_store[str(cur_top)] = numpy_layers[id].run(input_list, quant=False)
                signed=True
                if str(layer_list[conv_part[-1]].top) + '_signed' in featuremap_frac:
                    signed=featuremap_frac[str(layer_list[conv_part[-1]].top) + '_signed']
                q_test_tensor=qfun.quantized_value(feature_map_store[str(concat_top)], bitwidth, frac=featuremap_frac[str(concat_top) + '_frac'], floor = True, signed=signed)
                sum_var_all_q+=np.sum(np.power(q_test_tensor-sum_all_mean, 2)) 
            sum_var_mean=sum_var_all_q/sum_tensor_num
           # print(sum_var_all_q, sum_var_mean)
            tensor_mean[layer_list[conv_part[-1]].top]=sum_all_mean
            tensor_var[layer_list[conv_part[-1]].top]=sum_var_mean  
        else:
            pass                 
        for group in bottom_group:
            if layer_list[group[0]].type in ['Convolution','BatchNorm' , 'Scale','InnerProduct', 'Bias']:
                update_params_to_limit_output(bitwidth, featuremap_frac[layer_list[group[0]].top+'_frac'], featuremap_frac, quant_params, layer_list[group[0]])
                update_params_to_limit_input(bitwidth, featuremap_frac[layer_list[group[0]].bottom+'_frac'], featuremap_frac, quant_params, layer_list[group[0]])
        for group_id in range(len(bottom_group)-1):
            recovery_bottom=[]
            recovery_bottom.append(bottom_group[group_id][-1]) #concat's input
            update_recovery_limit_input(bitwidth, recovery_bottom, featuremap_frac, quant_params, layer_list)
        for group_id in range(len(bottom_group)-1):
            id =bottom_group[group_id][0]
            if layer_list[id].type in ['Concat','Eltwise']:
                input_group = add_concat_bottom_layer(layer_list[id].bottom, layer_list, quant_layers)
                update_featuremapfrac_group(layer_list, featuremap_frac[layer_list[id].top+'_frac'], input_group, featuremap_frac)
                for group in input_group:
                    for index in group:
                        if layer_list[index].type in ['Convolution','BatchNorm' , 'Scale','InnerProduct', 'Bias']:
                            update_params_to_limit_output(bitwidth, featuremap_frac[layer_list[index].top+'_frac'], featuremap_frac, quant_params, layer_list[index])
                            update_params_to_limit_input(bitwidth, featuremap_frac[layer_list[index].bottom+'_frac'], featuremap_frac, quant_params, layer_list[index])
            elif layer_list[id].type in ['Slice']:
                input_group = add_concat_bottom_layer(layer_list[id].bottom, layer_list, quant_layers)
                update_featuremapfrac_group(layer_list, featuremap_frac[layer_list[id].top[0]+'_frac'], input_group, featuremap_frac)
                for group in input_group:
                    for index in group:
                        if layer_list[index].type in ['Convolution','BatchNorm' , 'Scale','InnerProduct', 'Bias']:
                            update_params_to_limit_output(bitwidth, featuremap_frac[layer_list[index].top+'_frac'], featuremap_frac, quant_params, layer_list[index])
                            update_params_to_limit_input(bitwidth, featuremap_frac[layer_list[index].bottom+'_frac'], featuremap_frac, quant_params, layer_list[index])
    else:
        next_check=False
        next_frac=0
        if gaussian_approx:
            update_normal_part_frac_dist_ga(data_generator, bitwidth, featuremap_frac, layer_list, numpy_layers, tensor_size, tensor_mean, tensor_var, 
                                              num_step,conv_part, max_vals, INTERVAL_NUM, QUANTIZE_NUM, next_check, next_frac, quant_string)
        else:
            update_normal_part_frac_noga(data_generator, param_dict, bitwidth, featuremap_frac, layer_list, numpy_layers, conv_part, max_vals, 
                                         INTERVAL_NUM, QUANTIZE_NUM, num_step, next_check, next_frac, quant_string)
        for lay_id in conv_part:            
            if layer_list[lay_id].type in ['Convolution','BatchNorm' , 'Scale','InnerProduct', 'Bias']:
                update_params_to_limit_output(bitwidth,  featuremap_frac[layer_list[lay_id].top+'_frac'], featuremap_frac, quant_params, layer_list[lay_id])
                update_params_to_limit_input(bitwidth, featuremap_frac[layer_list[lay_id].bottom+'_frac'], featuremap_frac, quant_params, layer_list[lay_id])                
        for group_id in range(len(bottom_group)):
            recovery_bottom=[]
            if group_id <len(bottom_group)-1:
                recovery_bottom.append(bottom_group[group_id][-1]) #concat's input
            update_recovery_limit_input(bitwidth, recovery_bottom, featuremap_frac, quant_params, layer_list)

def update_sigmoid_prelayer_frac(conv_part, bitwidth, featuremap_frac, layer_list):
    for id in conv_part:
        layer=layer_list[id]
        cur_top=layer.top
        if bitwidth==8:
            featuremap_frac[str(cur_top)+'_frac']=4
        else:
            featuremap_frac[str(cur_top)+'_frac']=12
            
def update_tanh_prelayer_frac(conv_part, bitwidth, featuremap_frac, layer_list):
    for id in conv_part:
        layer=layer_list[id]
        cur_top=layer.top
        if bitwidth==8:
            featuremap_frac[str(cur_top)+'_frac']=5
        else:
            featuremap_frac[str(cur_top)+'_frac']=13
            
def update_softmax_prelayer_frac(conv_part, bitwidth, featuremap_frac, layer_list):
    for id in conv_part:
        layer=layer_list[id]
        cur_top=layer.top
        if bitwidth==8:
            featuremap_frac[str(cur_top)+'_frac']=4
        else:
            featuremap_frac[str(cur_top)+'_frac']=12
          
def get_mean_var_concat(layer_list, 
                        num_step, 
                        conv_part, 
                        bitwidth, 
                        tensor_mean, 
                        tensor_var,                                                                
                        numpy_layers_quant, 
                        frac_chk_value, 
                        size_list, 
                        featuremap_frac):
    concat_var_all=0
    concat_sum_all=0
    sum_tensor_num= num_step* size_list[layer_list[conv_part[-1]].top][0]*size_list[layer_list[conv_part[-1]].top][1]*\
                              size_list[layer_list[conv_part[-1]].top][2]*size_list[layer_list[conv_part[-1]].top][3]
    for piece in range(num_step):
        feature_map_store={}
        for id in conv_part:
            cur_bottom=layer_list[id].bottom
            cur_top=layer_list[id].top
            cur_type=layer_list[id].type
            if type(cur_bottom)!=str:
                list_data_in=[]
                length_bottom=len(cur_bottom)
                for i in range(length_bottom):
                    x=np.random.normal(tensor_mean[cur_bottom[i]], math.sqrt(tensor_var[cur_bottom[i]]), size_list[cur_bottom[i]])
                    list_data_in.append(x)
                    feature_map_store[cur_top]= numpy_layers_quant[id].run(list_data_in, quant=False)
            elif cur_type in['ShuffleChannel', 'Permute', 'Reshape']:
                if str(cur_bottom)+'_signed' in featuremap_frac and featuremap_frac[str(cur_bottom) + '_signed'] == False:
                    featuremap_frac[str(cur_top) + '_signed'] = False
                    feature_map_store[cur_top]= numpy_layers_quant[id].run(feature_map_store[cur_bottom], quant=False)
            else:
                feature_map_store[cur_top]= numpy_layers_quant[id].run(feature_map_store[cur_bottom], quant=False)
        signed=True
        if str(layer_list[conv_part[-1]].top) + '_signed' in featuremap_frac:
            signed=featuremap_frac[str(layer_list[conv_part[-1]].top) + '_signed']
        q_test_tensor=qfun.quantized_value(feature_map_store[layer_list[conv_part[-1]].top], bitwidth, frac=frac_chk_value, floor = True, signed=signed)
        concat_sum_all+=np.sum(q_test_tensor)
    mean_val_q=concat_sum_all/sum_tensor_num
    for piece in range(num_step):
        feature_map_store={}
        for id in conv_part:
            cur_bottom=layer_list[id].bottom
            cur_top=layer_list[id].top
            cur_type=layer_list[id].type
            if type(cur_bottom)!=str:
                list_data_in=[]
                length_bottom=len(cur_bottom)
                for i in range(length_bottom):
                    x=np.random.normal(tensor_mean[cur_bottom[i]], math.sqrt(tensor_var[cur_bottom[i]]), size_list[cur_bottom[i]])
                    list_data_in.append(x)
                    feature_map_store[cur_top]= numpy_layers_quant[id].run(list_data_in, quant=False)
            else:
                feature_map_store[cur_top]= numpy_layers_quant[id].run(feature_map_store[cur_bottom], quant=False)
        signed=True
        if str(layer_list[conv_part[-1]].top) + '_signed' in featuremap_frac:
            signed=featuremap_frac[str(layer_list[conv_part[-1]].top) + '_signed']
        q_test_tensor=qfun.quantized_value(feature_map_store[layer_list[conv_part[-1]].top], bitwidth, frac=frac_chk_value, floor = True, signed=signed)
        concat_var_all+=np.sum(np.power(q_test_tensor-mean_val_q, 2))
    var_val_q=concat_var_all/sum_tensor_num
    return mean_val_q, var_val_q

def general_ga_dist_delivery(num_step,
                             bitwidth, 
                             cur_mean,
                             cur_var,
                             cur_numpy_layer, 
                             cur_frac, 
                             cur_signed,
                             cur_size,
                             top_size,
                             vals,
                             vals_top):    
    concat_sum_all=0
    concat_var_all=0
    sum_tensor_num= num_step* top_size[0]* top_size[1]* top_size[2]* top_size[3]
    for step in range(num_step):
        x=np.random.normal(cur_mean, math.sqrt(cur_var), cur_size)
        x0=np.minimum(vals, x)
        x1=np.maximum(-vals, x0)
        feature_map_store= cur_numpy_layer.run(x1, quant=False)
        q_test_tensor=qfun.quantized_value(feature_map_store, bitwidth, frac=cur_frac, floor = True, signed=cur_signed)
        concat_sum_all+=np.sum(q_test_tensor)
    mean_val_q=concat_sum_all/sum_tensor_num
    for step in range(num_step):
        x=np.random.normal(cur_mean, math.sqrt(cur_var), cur_size)
        x0=np.minimum(vals, x)
        x1=np.maximum(-vals, x0)
        feature_map_store= cur_numpy_layer.run(x1, quant=False)
        q_test_tensor=qfun.quantized_value(feature_map_store, bitwidth, frac=cur_frac, floor = True, signed=cur_signed)
        concat_var_all+=np.sum(np.power(q_test_tensor-mean_val_q, 2))
    var_val_q=concat_var_all/sum_tensor_num
    return mean_val_q, var_val_q

def fraknown_ga_dist_delivery(layer_list, 
                               num_step, 
                               conv_part, 
                               bitwidth, 
                               cur_mean, 
                               cur_var, 
                               numpy_layers, 
                               featuremap_frac, 
                               cur_size,
                               top_size,
                               vals,
                               vals_top):
    concat_sum_all=0
    concat_var_all=0
    sum_tensor_num= num_step* top_size[0]* top_size[1]* top_size[2]* top_size[3]
    for piece in range(num_step):
        x=np.random.normal(cur_mean, math.sqrt(cur_var), cur_size)
        x0=np.minimum(vals, x)
        x1= np.maximum(-vals, x0)
        feature_map_store={}
        q_test_tensor={}
        for id_c in conv_part:
            layer=layer_list[id_c]               
            cur_top= layer.top
            cur_bottom= layer.bottom
            input_tensor=0
            if id_c==conv_part[0]:
                input_tensor=x1
            else:
                input_tensor=feature_map_store[str(cur_bottom)]
            feature_map_store[str(cur_top)] = numpy_layers[id_c].run(input_tensor, quant=False)
            signed=True
            if str(layer_list[conv_part[-1]].top) + '_signed' in featuremap_frac:
                signed=featuremap_frac[str(cur_top) + '_signed']
            q_test_tensor[str(cur_top)]=qfun.quantized_value(feature_map_store[str(cur_top)], bitwidth, frac=featuremap_frac[str(cur_top)+'_frac'], floor = True, signed=signed)
        concat_sum_all+=np.sum(q_test_tensor[str(layer_list[conv_part[-1]].top)])
    mean_val_q=concat_sum_all/sum_tensor_num
    for piece in range(num_step):
        x=np.random.normal(cur_mean, math.sqrt(cur_var), cur_size)
        x0=np.minimum(vals, x)
        x1= np.maximum(-vals, x0)
        feature_map_store={}
        q_test_tensor={}
        for id_c in conv_part:
            layer=layer_list[id_c]             
            cur_top= layer.top
            cur_bottom= layer.bottom
            if id_c==conv_part[0]:
                input_tensor=x1
            else:
                input_tensor= feature_map_store[str(cur_bottom)]
            feature_map_store[str(cur_top)] = numpy_layers[id_c].run(input_tensor, quant=False)
            signed=True
            if str(cur_top) + '_signed' in featuremap_frac:
                signed= featuremap_frac[str(cur_top) + '_signed']
            q_test_tensor[str(cur_top)]=qfun.quantized_value(feature_map_store[str(cur_top)], bitwidth, frac=featuremap_frac[str(cur_top)+'_frac'], floor = True, signed=signed)
        concat_var_all+=np.sum(np.power(q_test_tensor[str(layer_list[conv_part[-1]].top)]-mean_val_q, 2))
    var_val_q=concat_var_all/sum_tensor_num
    return mean_val_q, var_val_q

def slice_ga_dist_delivery(cur_numpy_layer,
                           cur_size,
                           top_size,
                           cur_bottom,
                           cur_top, 
                           cur_mean,
                           cur_var, 
                           num_step,
                           vals):
    length_top=len(cur_top)
    concat_sum_all= np.zeros(length_top)
    concat_var_all= np.zeros(length_top)
    top_mean=np.zeros(length_top)
    top_var=np.zeros(length_top)
    sum_tensor_num= np.zeros(length_top)
    for id in range(length_top):
        sum_tensor_num[id]= num_step* top_size[id][0]* top_size[id][1]* top_size[id][2]* top_size[id][3]
    for step in range(num_step):
        x=np.random.normal(cur_mean, math.sqrt(cur_var), cur_size)
        x0=np.minimum(vals, x)
        x1=np.maximum(-vals,x0)
        feature_map_store= cur_numpy_layer.run(x1, quant=False)
        for id in range(length_top):
            concat_sum_all[id]+=np.sum(feature_map_store[id])
    for id in range(length_top):
        top_mean[id]=concat_sum_all[id]/sum_tensor_num[id]
    for step in range(num_step):
        x=np.random.normal(cur_mean, math.sqrt(cur_var), cur_size)
        x0=np.minimum(vals, x)
        x1=np.maximum(-vals,x0)
        feature_map_store= cur_numpy_layer.run(x1, quant=False)
        for id in range(length_top):
            concat_var_all[id]+=np.sum(np.power(feature_map_store[id]-top_mean[id], 2))
    for id in range(length_top):
        top_var[id]=concat_var_all[id]/sum_tensor_num[id]
    return top_mean, top_var   
    
def general_quant(layer_list, 
                  min_input, 
                  data_generator, 
                  quant_params,
                  params_dict, 
                  quant_layers, 
                  max_vals, 
                  bitwidth, 
                  num_step, 
                  tensor_size_list, 
                  feature_name_list, 
                  usr_param_dict, 
                  method_dict, 
                  gaussian_approx,
                  INTERVAL_NUM, 
                  QUANTIZE_NUM):
    tensor_mean={}
    tensor_var={}
    quant_featuremap_frac={}
    concat_frac_enforce=[]
    numpy_layers_quant= numpy_net.build_numpy_layers(layer_list, params_dict, quant=False, bitwidth=bitwidth)
    if min_input >= 0 and bitwidth==8:
        quant_featuremap_frac[str(layer_list[0].top) + '_signed'] = False
    else:
        quant_featuremap_frac[str(layer_list[0].top) + '_signed'] = True
    distributions = []
    for id1 in range(len(layer_list)):
        distribution = [0 for _ in range(0, INTERVAL_NUM)]
        distributions.append(distribution)
    for id in range(len(quant_layers)):
        conv_part = quant_layers[id]
        for kii in conv_part:
            print_log('quantize_model: processing layer {}/{}: name={} type={}'.format(kii+1, len(layer_list), layer_list[kii].name, layer_list[kii].type))
        frac_chk_value, quant_string= get_frac_value_quant_string(layer_list, conv_part, usr_param_dict, feature_name_list, method_dict)
        if layer_list[conv_part[0]].type in[ 'Concat','Eltwise']:
            cur_top = layer_list[conv_part[0]].top
            cur_bottom = layer_list[conv_part[0]].bottom
            if layer_list[conv_part[0]].type=='Concat':
                delivery_concat_signed(layer_list, cur_bottom, cur_top, quant_featuremap_frac)
            else:
                pass
            bottom_group = add_concat_bottom_layer(layer_list[conv_part[0]].bottom, layer_list, quant_layers)
            bottom_group.append(conv_part)  # bottom & now
            if frac_chk_value!=[]:
                update_featuremapfrac_group(layer_list, frac_chk_value[0], bottom_group, quant_featuremap_frac)
                if gaussian_approx:
                    mean_val_q, var_val_q= get_mean_var_concat(layer_list, num_step, conv_part, bitwidth, tensor_mean, tensor_var, 
                                                               numpy_layers_quant, frac_chk_value[0], tensor_size_list, quant_featuremap_frac)
                    tensor_mean[layer_list[conv_part[-1]].top]=mean_val_q
                    tensor_var[layer_list[conv_part[-1]].top]=var_val_q
                else:
                    pass
            else:
                update_mean_var_frac_concat(data_generator, params_dict, quant_params, bitwidth, quant_featuremap_frac, layer_list, numpy_layers_quant, 
                                             tensor_size_list, bottom_group, tensor_mean, tensor_var, num_step, max_vals,
                                             INTERVAL_NUM, QUANTIZE_NUM, gaussian_approx, quant_layers, concat_frac_enforce, quant_string)
        elif len(conv_part) == 1 and layer_list[conv_part[0]].type == 'Slice':
            cur_top = layer_list[conv_part[0]].top
            cur_bottom = layer_list[conv_part[0]].bottom
            if str(cur_bottom)+'_signed' in quant_featuremap_frac and quant_featuremap_frac[str(cur_bottom) + '_signed'] == False:
                for top in cur_top:
                    quant_featuremap_frac[str(top) + '_signed'] = False
            bottom_group=[]
            bottom_group.append(quant_layers[id- 1])
            bottom_group.append(conv_part)
            if frac_chk_value!=[]:
                update_featuremapfrac_group(layer_list, frac_chk_value[0], bottom_group, quant_featuremap_frac)
            else:
                for top in cur_top:
                    quant_featuremap_frac[str(top) + '_frac'] = quant_featuremap_frac[str(cur_bottom) + '_frac']
            if gaussian_approx:
                top_size=[]
                for top in cur_top:
                    top_size.append(tensor_size_list[str(top)])
                top_mean, top_var= slice_ga_dist_delivery(numpy_layers_quant[conv_part[0]], tensor_size_list[str(layer_list[conv_part[0]].bottom)], top_size, 
                                                          cur_bottom, cur_top, tensor_mean[str(cur_bottom)], tensor_var[str(cur_bottom)], num_step, max_vals[str(cur_bottom)])
                for id_x in range(len(cur_top)):
                    tensor_mean[str(cur_top[id_x])]= top_mean[id_x]
                    tensor_var[str(cur_top[id_x])]= top_var[id_x]
            else:
                pass
        elif len(conv_part) == 1 and layer_list[conv_part[0]].type in ['ShuffleChannel', 'Permute', 'Reshape']:
            cur_top = layer_list[conv_part[0]].top
            cur_bottom = layer_list[conv_part[0]].bottom
            if str(cur_bottom)+'_signed' in quant_featuremap_frac and quant_featuremap_frac[str(cur_bottom) + '_signed'] == False:
                quant_featuremap_frac[str(cur_top) + '_signed'] = False
            bottom_group=[]
            bottom_group.append(quant_layers[id- 1])
            bottom_group.append(conv_part)
            if frac_chk_value!=[]:
                update_featuremapfrac_group(layer_list, frac_chk_value[0], bottom_group, quant_featuremap_frac)
            else:
                quant_featuremap_frac[str(cur_top) + '_frac'] = quant_featuremap_frac[str(cur_bottom) + '_frac']
            if gaussian_approx:
                tensor_mean[str(cur_top)]=tensor_mean[str(cur_bottom)]
                tensor_var[str(cur_top)]=tensor_var[str(cur_bottom)]
            else:
                pass
        elif len(conv_part) == 1 and (layer_list[conv_part[0]].type in ['ReLU'] or (layer_list[conv_part[0]].type == 'Pooling' and layer_list[
            conv_part[0]].pool == 'max')):
            cur_top = layer_list[conv_part[0]].top
            cur_bottom = layer_list[conv_part[0]].bottom
            bottom_group=[]
            bottom_group.append(quant_layers[id- 1])
            bottom_group.append(conv_part)
            if frac_chk_value!=[]:
                update_featuremapfrac_group(layer_list, frac_chk_value[0], bottom_group, quant_featuremap_frac)
            else:
                quant_featuremap_frac[str(cur_top) + '_frac'] = quant_featuremap_frac[str(cur_bottom) + '_frac']
            if gaussian_approx:
                signed =True
                if str(cur_top)+'_signed' in quant_featuremap_frac:
                    signed = quant_featuremap_frac[str(cur_top)+'_signed']
                mean_val_q, var_val_q= general_ga_dist_delivery(num_step, bitwidth, tensor_mean[str(cur_bottom)], tensor_var[str(cur_bottom)], 
                                                                numpy_layers_quant[conv_part[0]], quant_featuremap_frac[str(cur_top) + '_frac'], 
                                                                signed, tensor_size_list[str(cur_bottom)], tensor_size_list[str(cur_top)],
                                                                max_vals[str(cur_bottom)], max_vals[str(cur_top)])                
                tensor_mean[str(cur_top)]=mean_val_q
                tensor_var[str(cur_top)]=var_val_q
            else:
                pass
        elif len(conv_part) == 1 and layer_list[conv_part[0]].type in ['Sigmoid','TanH']:
            cur_top = layer_list[conv_part[0]].top
            cur_bottom = layer_list[conv_part[0]].bottom
            if bitwidth == 8:
                if frac_chk_value!=[]:
                    assert frac_chk_value[0] == 7, '8 bit quantize Sigmoid must be 7, but get {}'.format(frac_chk_value[0])
                    quant_featuremap_frac[str(cur_top)+'_frac']= frac_chk_value[0]
                else:
                    quant_featuremap_frac[str(cur_top) + '_frac'] = 7
            else:
                if frac_chk_value!=[]:
                    assert frac_chk_value[0] == 15, '16 bit quantize Sigmoid must be 15, but get {}'.format(frac_chk_value[0])
                    quant_featuremap_frac[str(cur_top)+'_frac']=frac_chk_value[0]
                else:
                    quant_featuremap_frac[str(cur_top) + '_frac'] = 15
            if layer_list[conv_part[0]].type=='Sigmoid':
                update_sigmoid_prelayer_frac(quant_layers[id - 1], bitwidth, quant_featuremap_frac, layer_list)
            else:
                update_tanh_prelayer_frac(quant_layers[id - 1], bitwidth, quant_featuremap_frac, layer_list)
            if gaussian_approx:
                signed =True
                if str(cur_top)+'_signed' in quant_featuremap_frac:
                    signed = quant_featuremap_frac[str(cur_top)+'_signed']
                mean_val_q, var_val_q= general_ga_dist_delivery(num_step, bitwidth, tensor_mean[str(cur_bottom)], tensor_var[str(cur_bottom)], 
                                                                numpy_layers_quant[conv_part[0]], quant_featuremap_frac[str(cur_top) + '_frac'], 
                                                                signed, tensor_size_list[str(cur_bottom)], tensor_size_list[str(cur_top)],
                                                                max_vals[str(cur_bottom)], max_vals[str(cur_top)])                
                tensor_mean[str(cur_top)]=mean_val_q
                tensor_var[str(cur_top)]=var_val_q
            else:
                pass
        elif len(conv_part) == 1 and layer_list[conv_part[0]].type in ['ReLU6']:
            cur_top = layer_list[conv_part[0]].top
            cur_bottom = layer_list[conv_part[0]].bottom
            if bitwidth == 8:
                if frac_chk_value!=[]:
                    assert frac_chk_value[0] == 4, '8 bit ReLU6 must be 4, but get {}'.format(frac_chk_value[0])
                    quant_featuremap_frac[str(cur_top)+'_frac']=frac_chk_value[0]
                else:
                    quant_featuremap_frac[str(cur_top) + '_frac'] = 4
            else:
                if frac_chk_value!=[]:
                    assert frac_chk_value[0] == 12, '16 bit ReLU6 must be 12, but get {}'.format(frac_chk_value[0])
                    quant_featuremap_frac[str(cur_top)+'_frac']=frac_chk_value[0]
                else:
                    quant_featuremap_frac[str(cur_top) + '_frac'] = 12
            update_softmax_prelayer_frac(quant_layers[id - 1], bitwidth, quant_featuremap_frac, layer_list)
            if gaussian_approx:
                signed =True
                if str(cur_top)+'_signed' in quant_featuremap_frac:
                    signed = quant_featuremap_frac[str(cur_top)+'_signed']
                mean_val_q, var_val_q=general_ga_dist_delivery(num_step, bitwidth, tensor_mean[str(cur_bottom)], tensor_var[str(cur_bottom)], 
                                                                numpy_layers_quant[conv_part[0]], quant_featuremap_frac[str(cur_top) + '_frac'], 
                                                                signed, tensor_size_list[str(cur_bottom)], tensor_size_list[str(cur_top)],
                                                                max_vals[str(cur_bottom)], max_vals[str(cur_top)])                
                tensor_mean[str(cur_top)]=mean_val_q
                tensor_var[str(cur_top)]=var_val_q
            else:
                pass
        elif len(conv_part) == 1 and layer_list[conv_part[0]].type in ['LogSoftmax','Softmax']:
            if layer_list[conv_part[0]].type=='Softmax':
                print_log('quantize_model: WARNING: softmax will be change to logsoftmax')
            assert conv_part[0]==len(layer_list)-1,'softmax layer must be the last layer'  
            assert ((layer_list[conv_part[0]-1].type == 'Convolution' and layer_list[conv_part[0]-1].group==1) or layer_list[conv_part[0]-1].type == 'InnerProduct'),'logsoftmax layer must be after Convolution or InnerProduct!'
            cur_top = layer_list[conv_part[0]].top
            cur_bottom = layer_list[conv_part[0]].bottom
            if bitwidth == 8:
                if frac_chk_value!=[]:
                    assert frac_chk_value[0] == 4, '8 bit quantize LogSoftmax must be 4, but get {}'.format(frac_chk_value[0])
                    quant_featuremap_frac[str(cur_top)+'_frac']=frac_chk_value[0]
                else:
                    quant_featuremap_frac[str(cur_top) + '_frac'] = 4
            else:
                if frac_chk_value!=[]:
                    assert frac_chk_value[0] == 12, '16 bit quantize LogSoftmax must be 12, but get {}'.format(frac_chk_value[0])
                    quant_featuremap_frac[str(cur_top)+'_frac']=frac_chk_value[0]
                else:
                    quant_featuremap_frac[str(cur_top) + '_frac'] = 12
            update_softmax_prelayer_frac(quant_layers[id - 1], bitwidth, quant_featuremap_frac, layer_list)
            if gaussian_approx:
                signed =True
                if str(cur_top)+'_signed' in quant_featuremap_frac:
                    signed = quant_featuremap_frac[str(cur_top)+'_signed']
                mean_val_q, var_val_q=general_ga_dist_delivery(num_step, bitwidth, tensor_mean[str(cur_bottom)], tensor_var[str(cur_bottom)], 
                                                                numpy_layers_quant[conv_part[0]], quant_featuremap_frac[str(cur_top) + '_frac'], 
                                                                signed, tensor_size_list[str(cur_bottom)], tensor_size_list[str(cur_top)],
                                                                max_vals[str(cur_bottom)], max_vals[str(cur_top)])                
                tensor_mean[str(cur_top)]=mean_val_q
                tensor_var[str(cur_top)]=var_val_q
            else:
                pass
        else:
            #run back quantization before + float now 
            if frac_chk_value!=[]:
                bottom_group=[]
                bottom_group.append(conv_part)
                update_featuremapfrac_group(layer_list, frac_chk_value[0], bottom_group, quant_featuremap_frac)
                if gaussian_approx:
                    mean_val_q, var_val_q=fraknown_ga_dist_delivery(layer_list, num_step, conv_part, bitwidth, tensor_mean[str(layer_list[conv_part[0]].bottom)], 
                                                                     tensor_var[str(layer_list[conv_part[0]].bottom)], numpy_layers_quant, quant_featuremap_frac, 
                                                                     tensor_size_list[str(layer_list[conv_part[0]].bottom)], tensor_size_list[str(layer_list[conv_part[-1]].top)], 
                                                                     max_vals[str(layer_list[conv_part[0]].bottom)], max_vals[str(layer_list[conv_part[0]].top)])
                    tensor_mean[str(layer_list[conv_part[-1]].top)]=mean_val_q
                    tensor_var[str(layer_list[conv_part[-1]].top)]=var_val_q
                else:
                    pass
            else:
                next_check=False
                next_frac=0
                if layer_list[conv_part[0]].type in [ 'Convolution','BatchNorm' , 'Scale','InnerProduct', 'Bias']:
                    start_num=id+1
                    while start_num < len(quant_layers):
                        if layer_list[quant_layers[start_num][0]].type in ['ShuffleChannel','Reshape','Permute','ReLU']:
                            start_num+=1
                        elif layer_list[quant_layers[start_num][0]].type in ['Pooling'] and layer_list[quant_layers[start_num][0]].pool=='max':
                            start_num+=1
                        elif layer_list[quant_layers[start_num][0]].type in [ 'ReLU6','Sigmoid']:
                            frac_sure=4
                            if bitwidth==16:
                                frac_sure=12
                            next_check=True
                            next_frac=frac_sure
                            break
                        elif layer_list[quant_layers[start_num][0]].type in [ 'TanH']:
                            frac_sure=5
                            if bitwidth==16:
                                frac_sure=13
                            next_check=True
                            next_frac=frac_sure
                            break
                        else:
                            break 
                if gaussian_approx:
                    update_normal_part_frac_dist_ga(data_generator, bitwidth, quant_featuremap_frac, layer_list, numpy_layers_quant, tensor_size_list,
                                                    tensor_mean, tensor_var, num_step,conv_part, max_vals, INTERVAL_NUM, QUANTIZE_NUM, next_check, next_frac, quant_string)
                else:
                    update_normal_part_frac_noga(data_generator, params_dict, bitwidth, quant_featuremap_frac, layer_list, numpy_layers_quant, 
                                                 conv_part, max_vals, INTERVAL_NUM, QUANTIZE_NUM, num_step,  next_check, next_frac, quant_string)             
                for lay_id in conv_part:            
                    if layer_list[lay_id].type in ['Convolution','BatchNorm' , 'Scale','InnerProduct', 'Bias']:
                        update_params_to_limit_output(bitwidth,  quant_featuremap_frac[layer_list[lay_id].top+'_frac'], quant_featuremap_frac, quant_params, layer_list[lay_id])
                        update_params_to_limit_input(bitwidth, quant_featuremap_frac[layer_list[lay_id].bottom+'_frac'], quant_featuremap_frac, quant_params, layer_list[lay_id])
    return tensor_mean, tensor_var, quant_featuremap_frac

def update_params_to_limit_input(bitwidth, cur_bottom_frac, quant_featuremap_frac, quant_params, layer):
    bias_string= str(layer.name)+'_frac_bias'
    weight_string= str(layer.name)+'_frac_weight'
    if (bias_string in quant_params) and (weight_string in quant_params):
        if bitwidth==8:
            for c in range(quant_params[str(bias_string)].shape[0]):
                lower_frac= int(np.max(quant_params[str(bias_string)][c]- np.min(quant_params[str(weight_string)][c])) )
                upper_frac= 15 + int(np.min(quant_params[str(bias_string)][c]- np.max(quant_params[str(weight_string)][c])) )
                if lower_frac > upper_frac:
                    bias= lower_frac-upper_frac
                    quant_params[str(bias_string)][c]=np.minimum(quant_params[str(bias_string)][c],  np.max(quant_params[str(bias_string)][c])-bias)
                    lower_frac= int(np.max(quant_params[str(bias_string)][c])- np.min(quant_params[str(weight_string)][c]))
                    upper_frac= 15 + int(np.min(quant_params[str(bias_string)][c])- np.max(quant_params[str(weight_string)][c]))
                if cur_bottom_frac < lower_frac:
                    bias=lower_frac-cur_bottom_frac
                    quant_params[str(bias_string)][c]= np.minimum(quant_params[str(bias_string)][c], np.max(quant_params[str(bias_string)][c])-bias)
                if cur_bottom_frac > upper_frac:
                    bias=cur_bottom_frac- upper_frac
                    quant_params[str(bias_string)][c]= np.maximum(quant_params[str(bias_string)][c], np.min(quant_params[str(bias_string)][c])+bias)
        else:
            for c in range(quant_params[str(bias_string)].shape[0]):
                lower_frac= int(np.max(quant_params[str(bias_string)][c]- np.min(quant_params[str(weight_string)][c])) )
                upper_frac= 31 + int(np.min(quant_params[str(bias_string)][c]- np.max(quant_params[str(weight_string)][c])) )
                if lower_frac > upper_frac:
                    bias= lower_frac-upper_frac
                    quant_params[str(bias_string)][c]=np.minimum(quant_params[str(bias_string)][c],  np.max(quant_params[str(bias_string)][c])-bias)
                    lower_frac= int(np.max(quant_params[str(bias_string)][c])- np.min(quant_params[str(weight_string)][c]))
                    upper_frac= 31 + int(np.min(quant_params[str(bias_string)][c])- np.max(quant_params[str(weight_string)][c]))
                if cur_bottom_frac < lower_frac:
                    bias=lower_frac-cur_bottom_frac
                    quant_params[str(bias_string)][c]= np.minimum(quant_params[str(bias_string)][c], np.max(quant_params[str(bias_string)][c])-bias)
                if cur_bottom_frac > upper_frac:
                    bias= cur_bottom_frac- upper_frac
                    quant_params[str(bias_string)][c]= np.maximum(quant_params[str(bias_string)][c], np.min(quant_params[str(bias_string)][c])+bias)
        output_quant, output_dequant, shifts =quantizeweight.quantize_back_weight(quant_params[str(layer.name)+'_bias'], quant_params[str(bias_string)], bitwidth, 1)
        quant_params[str(layer.name)+'_quant_bias']= output_quant       
        quant_params[str(layer.name)+'_frac_bias']= shifts
    elif bias_string in quant_params:
        if bitwidth==8:
            for c in range(quant_params[str(bias_string)].shape[0]):
                lower_frac= int(np.max(quant_params[str(bias_string)][c] ))
                upper_frac= 15 + int(np.min(quant_params[str(bias_string)][c]) )
                if lower_frac > upper_frac:
                    bias= lower_frac-upper_frac
                    quant_params[str(bias_string)][c]=np.maximum(quant_params[str(bias_string)][c], np.min(quant_params[str(bias_string)][c])+bias)
                    lower_frac= int(np.max(quant_params[str(bias_string)][c]))
                    upper_frac= 15 + int(np.min(quant_params[str(bias_string)][c]))
                if cur_bottom_frac < lower_frac:
                    bias=lower_frac-cur_bottom_frac
                    quant_params[str(bias_string)][c]= np.minimum(quant_params[str(bias_string)][c], np.max(quant_params[str(bias_string)][c])-bias)
                if cur_bottom_frac > upper_frac:
                    bias=cur_bottom_frac-upper_frac
                    quant_params[str(bias_string)][c]= np.maximum(quant_params[str(bias_string)][c], np.min(quant_params[str(bias_string)][c])+bias)
        else:
            for c in range(quant_params[str(bias_string)].shape[0]):
                lower_frac= int(np.max(quant_params[str(bias_string)][c]) )
                upper_frac= 31 + int(np.min(quant_params[str(bias_string)][c]) )
                if lower_frac > upper_frac:
                    bias= lower_frac-upper_frac
                    quant_params[str(bias_string)][c]=np.maximum(quant_params[str(bias_string)][c], np.min(quant_params[str(bias_string)][c])+bias)
                    lower_frac= int(np.max(quant_params[str(bias_string)][c]))
                    upper_frac= 31 + int(np.min(quant_params[str(bias_string)][c]))
                if cur_bottom_frac < lower_frac:
                    bias=lower_frac-cur_bottom_frac
                    quant_params[str(bias_string)][c]= np.minimum(quant_params[str(bias_string)][c], np.max(quant_params[str(bias_string)][c])-bias)
                if cur_bottom_frac > upper_frac:
                    bias= cur_bottom_frac-upper_frac
                    quant_params[str(bias_string)][c]= np.maximum(quant_params[str(bias_string)][c], np.min(quant_params[str(bias_string)][c])+bias)
        output_quant, output_dequant, shifts =quantizeweight.quantize_back_weight(quant_params[str(layer.name)+'_bias'], quant_params[str(bias_string)], bitwidth, 1)
        quant_params[str(layer.name)+'_quant_bias']= output_quant       
        quant_params[str(layer.name)+'_frac_bias']= shifts


def update_params_to_limit_output(bitwidth, frac_sure, quant_featuremap_frac, quant_params, layer):
    weight_string= str(layer.name)+'_frac_weight'
    upper_frac_src=0
    lower_frac_src=0
    if weight_string in quant_params:
        if bitwidth==8:
            lower_frac_src= quant_featuremap_frac[layer.bottom+'_frac']+ int(np.max(quant_params[str(weight_string)]))-15
            upper_frac_src= quant_featuremap_frac[layer.bottom+'_frac']+ int(np.min(quant_params[str(weight_string)]))
            if lower_frac_src > upper_frac_src:
                bias= lower_frac_src -upper_frac_src
                quant_params[str(weight_string)]=np.minimum(quant_params[str(weight_string)], np.max(quant_params[str(weight_string)])-bias)
                lower_frac_src= quant_featuremap_frac[layer.bottom+'_frac']+ int(np.max(quant_params[str(weight_string)]))-15
                upper_frac_src= quant_featuremap_frac[layer.bottom+'_frac']+ int(np.min(quant_params[str(weight_string)]))
            else:
                pass            
        else:
            lower_frac_src= quant_featuremap_frac[layer.bottom+'_frac']+ int(np.max(quant_params[str(weight_string)]))-31
            upper_frac_src= quant_featuremap_frac[layer.bottom+'_frac']+ int(np.min(quant_params[str(weight_string)]))
            if lower_frac_src > upper_frac_src:
                bias= lower_frac_src -upper_frac_src
                quant_params[str(weight_string)]=np.minimum(quant_params[str(weight_string)], np.max(quant_params[str(weight_string)])-bias)
                lower_frac_src= quant_featuremap_frac[layer.bottom+'_frac']+ int(np.max(quant_params[str(weight_string)]))-31
                upper_frac_src= quant_featuremap_frac[layer.bottom+'_frac']+ int(np.min(quant_params[str(weight_string)]))
            else:
                pass
        output_quant, output_dequant, shifts =quantizeweight.quantize_back_weight(quant_params[str(layer.name)+'_weight'], quant_params[str(weight_string)], bitwidth, 1)
        quant_params[str(layer.name)+'_quant_weight']= output_quant       
        quant_params[str(layer.name)+'_frac_weight']= shifts
    else:
        pass
    
    if upper_frac_src < frac_sure:
        if weight_string in quant_params:
            bias= frac_sure- upper_frac_src
            quant_params[str(weight_string)]=np.maximum(quant_params[str(weight_string)], np.min(quant_params[str(weight_string)])+bias)
            output_quant, output_dequant, shifts =quantizeweight.quantize_back_weight(quant_params[str(layer.name)+'_weight'], quant_params[str(weight_string)], bitwidth, 1)
            quant_params[str(layer.name)+'_quant_weight']= output_quant       
            quant_params[str(layer.name)+'_frac_weight']= shifts
            if bitwidth==8:                                
                lower_frac_src= quant_featuremap_frac[layer.bottom+'_frac']+ int(np.max(quant_params[str(weight_string)]))-15
                upper_frac_src=quant_featuremap_frac[layer.bottom+'_frac']+ int(np.min(quant_params[str(weight_string)]))
            else:
                lower_frac_src= quant_featuremap_frac[layer.bottom+'_frac']+ int(np.max(quant_params[str(weight_string)]))-31
                upper_frac_src=quant_featuremap_frac[layer.bottom+'_frac']+ int(np.min(quant_params[str(weight_string)]))
        
        else:
            pass
    if frac_sure < lower_frac_src:
        if str(layer.name)+'_frac_weight' in quant_params:
            bias= lower_frac_src- frac_sure
            quant_params[str(weight_string)]=np.minimum(quant_params[str(weight_string)], np.max(quant_params[str(weight_string)])-bias)
            output_quant, output_dequant, shifts =quantizeweight.quantize_back_weight(quant_params[str(layer.name)+'_weight'], quant_params[str(weight_string)], bitwidth, 1)
            quant_params[str(layer.name)+'_quant_weight']= output_quant       
            quant_params[str(layer.name)+'_frac_weight']= shifts
            if bitwidth==8:
                lower_frac_src= quant_featuremap_frac[layer.bottom+'_frac']+ int(np.max(quant_params[str(weight_string)]))-15
                upper_frac_src=quant_featuremap_frac[layer.bottom+'_frac']+ int(np.min(quant_params[str(weight_string)]))
            else:
                lower_frac_src= quant_featuremap_frac[layer.bottom+'_frac']+ int(np.max(quant_params[str(weight_string)]))-31
                upper_frac_src=quant_featuremap_frac[layer.bottom+'_frac']+ int(np.min(quant_params[str(weight_string)]))
        else:
            pass
        
def back_check_correct_frac(layer_list, quant_layers, id):
    bottom_group=[]
    back_correct=False
    if len(quant_layers[id-1])==1:
        if layer_list[quant_layers[id-1][0]].type in ['Sigmoid','ReLU6','TanH','Pooling','Slice']:
            if layer_list[quant_layers[id-1][0]].type=='Pooling' and layer_list[quant_layers[id-1][0]].pool=='ave':
                back_correct=True
                bottom_group.append(quant_layers[id- 1])   
            else:
                back_correct=False
                bottom_group=[]
        elif layer_list[quant_layers[id-1][0]].type in ['Input']:#,'Convolution','Scale','BatchNorm','InnerProduct','Bias']:
            back_correct=True
            bottom_group.append(quant_layers[id])
            bottom_group.append(quant_layers[id- 1])   
        elif layer_list[quant_layers[id-1][0]].type in ['ShuffleChannel','Reshape','Permute','ReLU']:
            correct, group= back_check_correct_frac(layer_list, quant_layers, id-1)
            if correct==True:
                for i in group:
                    bottom_group.append(i)
        elif layer_list[quant_layers[id-1][0]].type in ['Concat','Eltwise']:
            last_group = add_concat_bottom_layer(layer_list[quant_layers[id-1][0]].bottom, layer_list, quant_layers)
            last_group.append(quant_layers[id- 1])
            length_group=len(last_group)-1 #bottom
            cover=0
            for ki in range(length_group):
                if len(last_group[ki])==1:
                    if layer_list[last_group[ki][0]].type in['Concat' ,'ShuffleChannel' ,'Permute' ,'Eltwise', 'Sigmoid', 'TanH']:
                        cover=1
                        break
                    elif layer_list[last_group[ki][0]].type in ['Slice']:
                        cover=1
                        break
            if cover==1:
                back_correct=False
            else:
                back_correct=False
        else:
            back_correct=False
    else:
        back_correct=True
        bottom_group.append(quant_layers[id- 1])    
    
    return back_correct, bottom_group

def update_normal_part_frac_noga(data_generator, 
                                 param_dict, 
                                 bitwidth, 
                                 feature_map_frac, 
                                 layer_list, 
                                 numpy_layers_quant,
                                 conv_part,
                                 max_vals,
                                 INTERVAL_NUM, 
                                 QUANTIZE_NUM, 
                                 num_step, 
                                 next_check, 
                                 next_frac, 
                                 quant_string=''):
    distributions = []
    for id1 in range(len(layer_list)):
        distribution = [0 for _ in range(0, INTERVAL_NUM)]
        distributions.append(distribution)
    length=len(conv_part)
    minmax = 0
    for step in range(num_step):
        data=next(data_generator)
        feature=data['input']    
        src_tensor=feature
        feature_map_store={}
        for id in range(0, conv_part[length-1]+ 1):
            layer=layer_list[id]
            cur_name= layer.name
            cur_type= layer.type                
            cur_top= layer.top
            if cur_type=='Input':
                if id not in conv_part:
                    numpy_layers_quant[id].set_quant_param(bitwidth=bitwidth, in_frac=feature_map_frac[str(layer_list[id].top)+'_frac'], 
                                      out_frac=feature_map_frac[str(layer_list[id].top)+'_frac'], input_signed=feature_map_frac[str(layer_list[id].top)+'_signed'])
                    feature_map_store[str(cur_top)]= numpy_layers_quant[id].run(src_tensor, quant=True)  
                else:
                    feature_map_store[cur_top]= numpy_layers_quant[id].run(src_tensor, quant=False)  
            elif cur_type in ['Convolution' ,'InnerProduct','Scale','Bias']:
                cur_bottom= layer.bottom
                input_tensor=feature_map_store[str(cur_bottom)]
                weight_tensor=param_dict[str(cur_name)+'_weight']
                bias_tensor=None
                if layer.bias_term:
                    bias_tensor= param_dict[str(cur_name)+'_bias']                  
                if id not in conv_part:
                    numpy_layers_quant[id].set_quant_param(bitwidth=bitwidth, in_frac=feature_map_frac[str(cur_bottom)+'_frac'], 
                                      out_frac=feature_map_frac[str(cur_top)+'_frac'],quant_weight=weight_tensor, quant_bias=bias_tensor)
                    feature_map_store[str(cur_top)] = numpy_layers_quant[id].run(input_tensor, quant=True)   
                else:
                    feature_map_store[str(cur_top)] = numpy_layers_quant[id].run(input_tensor, quant=False)
            elif cur_type in ['ReLU']:
                cur_bottom= layer.bottom
                input_tensor=feature_map_store[str(cur_bottom)]           
                if id not in conv_part:
                    numpy_layers_quant[id].set_quant_param(bitwidth, in_frac=feature_map_frac[str(cur_bottom)+'_frac'], 
                                      out_frac=feature_map_frac[str(cur_top)+'_frac'], quant_negative_slope=layer.negative_slope)
                    feature_map_store[str(cur_top)] = numpy_layers_quant[id].run(input_tensor, quant=True)
                else:
                    feature_map_store[str(cur_top)] = numpy_layers_quant[id].run(input_tensor, quant=False)
            elif cur_type=='BatchNorm':
                cur_bottom= layer.bottom
                input_tensor=feature_map_store[str(cur_bottom)]
                weight_tensor=param_dict[str(cur_name)+'_weight']
                bias_tensor= param_dict[str(cur_name)+'_bias']                 
                if id not in conv_part:
                    numpy_layers_quant[id].set_quant_param(bitwidth, in_frac=feature_map_frac[str(cur_bottom)+'_frac'], 
                                      out_frac=feature_map_frac[str(cur_top)+'_frac'],quant_weight=weight_tensor, quant_bias=bias_tensor )
                    feature_map_store[str(cur_top)] = numpy_layers_quant[id].run(input_tensor, quant=True)  
                else:
                    feature_map_store[str(cur_top)] = numpy_layers_quant[id].run(input_tensor, quant=False)  
            elif cur_type in ['Pooling' ,'ShuffleChannel','LogSoftmax','Softmax','ReLU6']:
                cur_bottom= layer.bottom
                input_tensor=feature_map_store[str(cur_bottom)]
                if id not in conv_part:
                    numpy_layers_quant[id].set_quant_param(bitwidth, in_frac=feature_map_frac[str(cur_bottom)+'_frac'], 
                                      out_frac=feature_map_frac[str(cur_top)+'_frac'])
                    feature_map_store[str(cur_top)] = numpy_layers_quant[id].run(input_tensor, quant=True)
                else:
                    feature_map_store[str(cur_top)] = numpy_layers_quant[id].run(input_tensor, quant=False)
            elif cur_type=='Slice': 
                cur_bottom= layer.bottom
                lens=len(cur_top)
                input_tensor=feature_map_store[str(cur_bottom)]
                output= numpy_layers_quant[id].run(input_tensor, quant=False)
                for i in range(lens):
                    feature_map_store[str(cur_top[i])]=output[i]           
            elif cur_type in ['Concat','Eltwise']: 
                cur_bottom= layer.bottom
                list_data_in=[]
                lens=len(cur_bottom)
                for i in range(lens):
                    list_data_in.append(feature_map_store[str(cur_bottom[i])])
                feature_map_store[str(cur_top)]= numpy_layers_quant[id].run(list_data_in, quant=False)
            else:
                print('unsupported type:', cur_type) 
        minmax_min=np.min(feature_map_store[str(layer_list[conv_part[length-1]].top)])
        minmax_max=np.max(feature_map_store[str(layer_list[conv_part[length-1]].top)])
        minmax=max(minmax, max(abs(minmax_min), abs(minmax_max)) )
        quantizeweight.add_to_distribution(feature_map_store[str(layer_list[conv_part[length-1]].top)], distributions[conv_part[length-1]], 
                                           max_vals[str(layer_list[conv_part[length-1]].top)]/INTERVAL_NUM)
            
    quantizeweight.normalize_distribution(distributions[conv_part[length-1]])
    distributions[conv_part[length-1]] = np.array(distributions[conv_part[length-1]])
    thre_list= quantizeweight.exp_int_threshold(distributions[conv_part[length-1]], max_vals[str(layer_list[conv_part[length-1]].top)], 0.7, 1.0, INTERVAL_NUM)
    if quant_string=='KL':
        threshold_bin= quantizeweight.threshold_kl_distribution(distributions[conv_part[length-1]], QUANTIZE_NUM, thre_list) 
        threshold = (threshold_bin+0.5) * max_vals[str(layer_list[conv_part[length-1]].top)]/INTERVAL_NUM
        signed=True
        if str(layer_list[conv_part[0]].top) + '_signed' in feature_map_frac:
            signed=feature_map_frac[str(layer_list[conv_part[0]].top) + '_signed']
        conv_frac= int( math.floor(bitwidth - 1 -math.log(threshold, 2) )   )
        if signed==False:
            conv_frac= int( math.floor(bitwidth - math.log(threshold, 2) )   )
        if  next_check:
            conv_frac= next_frac
        for i in conv_part:
            if type(layer_list[i].top)==str:    
                feature_map_frac[str(layer_list[i].top)+'_frac']=conv_frac
            else:
                for top in layer_list[i].top:
                    feature_map_frac[str(top)+'_frac']=conv_frac
    elif quant_string=='MSE':
        threshold_bin= quantizeweight.threshold_mse_distribution(distributions[conv_part[length-1]], QUANTIZE_NUM, thre_list) 
        threshold = (threshold_bin+0.5) * max_vals[str(layer_list[conv_part[length-1]].top)]/INTERVAL_NUM
        signed=True
        if str(layer_list[conv_part[0]].top) + '_signed' in feature_map_frac:
            signed=feature_map_frac[str(layer_list[conv_part[0]].top) + '_signed']
        conv_frac= int(math.floor(bitwidth - 1 -math.log(threshold, 2) )  )
        if signed==False:
            conv_frac= int(math.floor(bitwidth - math.log(threshold, 2) ))
        if  next_check:
            conv_frac= next_frac
        for i in conv_part:
            if type(layer_list[i].top)==str:    
                feature_map_frac[str(layer_list[i].top)+'_frac']=conv_frac
            else:
                for top in layer_list[i].top:
                    feature_map_frac[str(top)+'_frac']=conv_frac    
    else:
        if quant_string!='' and quant_string!='MINMAX':
            print('WARNING: unknown type: {}, quant as default type MINMAX'.format(quant_string) )
        signed=True
        if str(layer_list[conv_part[0]].top) + '_signed' in feature_map_frac:
            signed=feature_map_frac[str(layer_list[conv_part[0]].top) + '_signed']
        conv_frac= int( quantizeweight.threshold_minmax_distribution(bitwidth, minmax, signed))
        if  next_check:
            conv_frac= next_frac
        for i in conv_part:
            if type(layer_list[i].top)==str:    
                feature_map_frac[str(layer_list[i].top)+'_frac']=conv_frac
            else:
                for top in layer_list[i].top:
                    feature_map_frac[str(top)+'_frac']=conv_frac

def normal_part_dist_delivery(layer_list, 
                              data_generator, 
                              conv_part, 
                              bitwidth, 
                              numpy_layers_quant, 
                              num_step, 
                              featuremap_frac, 
                              tensor_mean, 
                              tensor_var, 
                              tensor_size_list,
                              max_vals):
    sum_all=0
    out_top=layer_list[conv_part[-1]].top
    sum_tensor_num= num_step*tensor_size_list[str(out_top)][0]*tensor_size_list[str(out_top)][1]*tensor_size_list[str(out_top)][2]*tensor_size_list[str(out_top)][3]
    
    if len(conv_part)==1 and layer_list[conv_part[0]].type=='Input':
        for step in range(num_step):
            feature_map_store={}
            data=next(data_generator)
            feature=data['input']
            cur_top=layer_list[conv_part[0]].top
            src_tensor=feature
            feature_map_store[str(cur_top)]= numpy_layers_quant[conv_part[0]].run(src_tensor, quant=False)
            signed=True
            if str(layer_list[conv_part[-1]].top) + '_signed' in featuremap_frac:
                signed=featuremap_frac[str(layer_list[conv_part[-1]].top) + '_signed']
            q_test_tensor=qfun.quantized_value(feature_map_store[str(cur_top)], bitwidth, frac=featuremap_frac[str(layer_list[conv_part[-1]].top) + '_frac'], floor = True, signed=signed)
            sum_all+=np.sum(q_test_tensor)
    else:
        for step in range(num_step): 
            feature_map_store={}
            for id in conv_part:
                layer=layer_list[id]           
                cur_top= layer.top
                cur_bottom= layer.bottom
                if type(cur_bottom)==str:
                    input_tensor=0
                    if id==conv_part[0]:
                        input_tensor=np.random.normal(tensor_mean[str(layer_list[conv_part[0]].bottom)], math.sqrt(tensor_var[str(layer_list[conv_part[0]].bottom)]),
                                        tensor_size_list[str(layer_list[conv_part[0]].bottom)])
                    else:
                        if str(cur_bottom) in feature_map_store:
                            input_tensor=feature_map_store[str(cur_bottom)]
                        else:
                            input_tensor=np.random.normal(tensor_mean[str(cur_bottom)], math.sqrt(tensor_var[str(cur_bottom)]), tensor_size_list[str(cur_bottom)])
                    feature_map_store[str(cur_top)] = numpy_layers_quant[id].run(input_tensor, quant=False)
                else:
                    input_list=[]
                    lens=len(cur_bottom)
                    for i in range(lens):
                        if str(cur_bottom[i]) in feature_map_store:
                            input_list.append(feature_map_store[str(cur_bottom[i])])
                        else:
                            x=np.random.normal(tensor_mean[str(cur_bottom[i])], math.sqrt(tensor_var[str(cur_bottom[i])]), tensor_size_list[str(cur_bottom[i])])
                            input_list.append(x)
                    feature_map_store[str(cur_top)] = numpy_layers_quant[id].run(input_list, quant=False)
            signed=True
            if str(layer_list[conv_part[-1]].top) + '_signed' in featuremap_frac:
                signed=featuremap_frac[str(layer_list[conv_part[-1]].top) + '_signed']
            q_test_tensor=qfun.quantized_value(feature_map_store[str(layer_list[conv_part[-1]].top)], bitwidth, 
                                               frac=featuremap_frac[str(layer_list[conv_part[-1]].top) + '_frac'], floor = True, signed=signed)
            sum_all+=np.sum(q_test_tensor)
    sum_all_mean=sum_all/sum_tensor_num
    sum_var_all_q=0
    if len(conv_part)==1 and layer_list[conv_part[0]].type=='Input':
        for step in range(num_step):
            feature_map_store={}
            data=next(data_generator)
            feature=data['input']
            cur_top=layer_list[conv_part[0]].top
            src_tensor=feature
            feature_map_store[str(cur_top)]= numpy_layers_quant[conv_part[0]].run(src_tensor, quant=False)
            signed=True
            if str(layer_list[conv_part[-1]].top) + '_signed' in featuremap_frac:
                signed=featuremap_frac[str(layer_list[conv_part[-1]].top) + '_signed']
            q_test_tensor=qfun.quantized_value(feature_map_store[str(cur_top)], bitwidth, frac=featuremap_frac[str(layer_list[conv_part[-1]].top) + '_frac'], floor = True, signed=signed)
            sum_var_all_q+=np.sum(np.power(q_test_tensor-sum_all_mean, 2)) 
    else:
        for step in range(num_step): 
            feature_map_store={}
            for id in conv_part:
                layer=layer_list[id]           
                cur_top= layer.top
                cur_bottom= layer.bottom
                if type(cur_bottom)==str:
                    input_tensor=0
                    if id==conv_part[0]:
                        input_tensor=np.random.normal(tensor_mean[str(layer_list[conv_part[0]].bottom)], math.sqrt(tensor_var[str(layer_list[conv_part[0]].bottom)]),
                                        tensor_size_list[str(layer_list[conv_part[0]].bottom)])
                    else:
                        if str(cur_bottom) in feature_map_store:
                            input_tensor=feature_map_store[str(cur_bottom)]
                        else:
                            input_tensor=np.random.normal(tensor_mean[str(cur_bottom)], math.sqrt(tensor_var[str(cur_bottom)]), tensor_size_list[str(cur_bottom)])
                    feature_map_store[str(cur_top)] = numpy_layers_quant[id].run(input_tensor, quant=False)
                else:
                    input_list=[]
                    lens=len(cur_bottom)
                    for i in range(lens):
                        if str(cur_bottom[i]) in feature_map_store:
                            input_list.append(feature_map_store[str(cur_bottom[i])])
                        else:
                            x=np.random.normal(tensor_mean[str(cur_bottom[i])], math.sqrt(tensor_var[str(cur_bottom[i])]), tensor_size_list[str(cur_bottom[i])])
                            input_list.append(x)
                    feature_map_store[str(cur_top)] = numpy_layers_quant[id].run(input_list, quant=False)
            signed=True
            if str(layer_list[conv_part[-1]].top) + '_signed' in featuremap_frac:
                signed=featuremap_frac[str(layer_list[conv_part[-1]].top) + '_signed']
            q_test_tensor=qfun.quantized_value(feature_map_store[str(layer_list[conv_part[-1]].top)], bitwidth, 
                                               frac=featuremap_frac[str(layer_list[conv_part[-1]].top) + '_frac'], floor = True, signed=signed)
            sum_var_all_q+=np.sum(np.power(q_test_tensor-sum_all_mean, 2)) 
    sum_var_mean=sum_var_all_q/sum_tensor_num
    return sum_all_mean, sum_var_mean

def update_normal_part_frac_dist_ga(data_generator, 
                                      bitwidth, 
                                      feature_map_frac, 
                                      layer_list, 
                                      numpy_layers_quant,
                                      tensor_size_list,
                                      tensor_mean, 
                                      tensor_var, 
                                      num_step,
                                      conv_part,
                                      max_vals,
                                      INTERVAL_NUM, 
                                      QUANTIZE_NUM, 
                                      next_check, 
                                      next_frac,
                                      quant_string=''):
    distributions = []
    for id1 in range(len(layer_list)):
        distribution = [0 for _ in range(0, INTERVAL_NUM)]
        distributions.append(distribution)
    minmax = 0
    if len(conv_part)==1 and layer_list[conv_part[0]].type=='Input':
        for step in range(num_step):
            feature_map_store={}
            data=next(data_generator)
            feature=data['input']
            layer=layer_list[conv_part[0]]          
            cur_top= layer.top
            src_tensor=feature
            feature_map_store[str(cur_top)]= numpy_layers_quant[conv_part[0]].run(src_tensor, quant=False)
            minmax_min=np.min(feature_map_store[str(layer_list[conv_part[0]].top)])
            minmax_max=np.max(feature_map_store[str(layer_list[conv_part[0]].top)])
            minmax=max(minmax, max(abs(minmax_min), abs(minmax_max)) )
            quantizeweight.add_to_distribution(feature_map_store[str(layer_list[conv_part[0]].top)], 
                                              distributions[conv_part[0]], max_vals[str(layer_list[conv_part[0]].top)]/INTERVAL_NUM)
    else:
        for step in range(num_step):
            src_tensor=0
            if type(layer_list[conv_part[0]].bottom)==str:
                src_tensor=np.random.normal(tensor_mean[str(layer_list[conv_part[0]].bottom)], math.sqrt(tensor_var[str(layer_list[conv_part[0]].bottom)]), 
                                            tensor_size_list[str(layer_list[conv_part[0]].bottom)])
            else:
                src_list=[]
                for bottom in layer_list[conv_part[0]].bottom:
                    x=np.random.normal(tensor_mean[str(bottom)], math.sqrt(tensor_var[str(bottom)]), tensor_size_list[str(bottom)])       
                    src_list.append(x)
                src_tensor=src_list
            feature_map_store={}
            for id in conv_part:
                layer=layer_list[id]           
                cur_top= layer.top
                cur_bottom= layer.bottom
                if type(cur_bottom)==str:
                    input_tensor=0
                    if id==conv_part[0]:
                        input_tensor=src_tensor
                    else:
                        if str(cur_bottom) in feature_map_store:
                            input_tensor=feature_map_store[str(cur_bottom)]
                        else:
                            input_tensor=np.random.normal(tensor_mean[str(cur_bottom)], math.sqrt(tensor_var[str(cur_bottom)]), tensor_size_list[str(cur_bottom)])
                    feature_map_store[str(cur_top)] = numpy_layers_quant[id].run(input_tensor, quant=False)
                else:
                    input_list=[]
                    length_bottom=len(cur_bottom)
                    for i in range(length_bottom):
                        if str(cur_bottom[i]) in feature_map_store:
                            input_list.append(feature_map_store[str(cur_bottom[i])])
                        else:
                            x=np.random.normal(tensor_mean[str(cur_bottom[i])], math.sqrt(tensor_var[str(cur_bottom[i])]), tensor_size_list[str(cur_bottom[i])])
                            input_list.append(x)
                    feature_map_store[str(cur_top)] = numpy_layers_quant[id].run(input_list, quant=False)
            minmax_min=np.min(feature_map_store[str(layer_list[conv_part[-1]].top)])
            minmax_max=np.max(feature_map_store[str(layer_list[conv_part[-1]].top)])
            minmax=max(minmax, max(abs(minmax_min), abs(minmax_max)) )
            quantizeweight.add_to_distribution(feature_map_store[str(layer_list[conv_part[-1]].top)], 
                                                  distributions[conv_part[-1]], max_vals[str(layer_list[conv_part[-1]].top)]/INTERVAL_NUM)
    quantizeweight.normalize_distribution(distributions[conv_part[-1]]) #for quantize frac
    distributions[conv_part[-1]]=np.array(distributions[conv_part[-1]])
    thre_list= quantizeweight.exp_int_threshold(distributions[conv_part[-1]], max_vals[str(layer_list[conv_part[-1]].top)], 0.7, 1.0, INTERVAL_NUM)
    if quant_string=='KL':
        threshold_bin= quantizeweight.threshold_kl_distribution(distributions[conv_part[-1]], QUANTIZE_NUM, thre_list) 
        threshold = (threshold_bin+0.5) * max_vals[str(layer_list[conv_part[-1]].top)]/INTERVAL_NUM
        signed=True
        if str(layer_list[conv_part[0]].top) + '_signed' in feature_map_frac:
            signed=feature_map_frac[str(layer_list[conv_part[0]].top) + '_signed']
        conv_frac=  int(math.floor(bitwidth - 1 -math.log(threshold, 2) ))
        if signed==False:
            conv_frac=  int(math.floor(bitwidth - math.log(threshold, 2) )  )
        if next_check:
            conv_frac= next_frac
        bottom_group=[]
        bottom_group.append(conv_part)
        update_featuremapfrac_group(layer_list, conv_frac, bottom_group, feature_map_frac)
    elif quant_string=='MSE':
        threshold_bin= quantizeweight.threshold_mse_distribution(distributions[conv_part[-1]], QUANTIZE_NUM, thre_list) 
        threshold = (threshold_bin+0.5) * max_vals[str(layer_list[conv_part[-1]].top)]/INTERVAL_NUM
        signed=True
        if str(layer_list[conv_part[0]].top) + '_signed' in feature_map_frac:
            signed=feature_map_frac[str(layer_list[conv_part[0]].top) + '_signed']
        conv_frac=   int(math.floor(bitwidth - 1 -math.log(threshold, 2) )  ) 
        if signed==False:
            conv_frac= int(math.floor(bitwidth - math.log(threshold, 2) )  )
        if next_check:
            conv_frac= next_frac
        bottom_group=[]
        bottom_group.append(conv_part)
        update_featuremapfrac_group(layer_list, conv_frac, bottom_group, feature_map_frac)
    else:
        if quant_string!='' and quant_string!='MINMAX':
            print('WARNING: unknown type: {}, quant as default type MINMAX'.format(quant_string) )
        signed=True
        if str(layer_list[conv_part[-1]].top) + '_signed' in feature_map_frac:
            signed=feature_map_frac[str(layer_list[conv_part[-1]].top) + '_signed']
        conv_frac=  int(quantizeweight.threshold_minmax_distribution(bitwidth, minmax, signed) )
        if next_check:
            conv_frac= next_frac
        bottom_group=[]
        bottom_group.append(conv_part)
        update_featuremapfrac_group(layer_list, conv_frac, bottom_group, feature_map_frac)
   #calculate mean:
    sum_all_mean, sum_var_mean= normal_part_dist_delivery(layer_list, data_generator, conv_part, bitwidth, numpy_layers_quant, 
                                                          num_step, feature_map_frac, tensor_mean, tensor_var, tensor_size_list, max_vals)

    tensor_mean[layer_list[conv_part[-1]].top]=sum_all_mean
    tensor_var[layer_list[conv_part[-1]].top]=sum_var_mean


if __name__=='__main__':
    print('Not implemented')    

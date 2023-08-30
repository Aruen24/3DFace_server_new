#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#clear some debugging codes. the in and out optimize mode set to be 0
#
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
modelpack_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(modelpack_dir)

import readcomputegraph as rcaffe
import read_tensor
import assignAddress
import avg_pool_multiplier
import os
import numpy
import math
import argparse
import platform
import struct
import pickle
import ctypes
#from ndk.utils import remove_dir
from ctypes import Structure, c_uint8, c_uint16, c_uint32, cdll, byref

__all__ = ['model_packer', 'write_feature_blob', 'write_weights_blob']


def remove_dir(dir):
    dir = dir.replace('\\', '/')
    if(os.path.isdir(dir)):
        for p in os.listdir(dir):
            remove_dir(os.path.join(dir,p))
        if(os.path.exists(dir)):
            os.rmdir(dir)
    else:
        if(os.path.exists(dir)):
            os.remove(dir)
            
def toHEX_complementary(x, bw):
    if x < 0:
        x = 2 ** bw + x
        padding_elem = '1'
    else:
        padding_elem = '0'

    hex_x = hex(int(x))[2:].rjust(int(bw/4), padding_elem)
    temp = str(hex_x)
    if temp.find('x') > -1:
        temp = 's'
    hex_x_bytes = [hex_x[i:i+2] for i in range(0, int(bw/4), 2)]

    return hex_x_bytes[::-1]


def write_feature_blob(blob_, f, elem_bytes):
    # Assume blob shape is (C, H, W)
    batch_, channels_, heights_, widths_ = blob_.shape

    elements_range = numpy.int32(32 / elem_bytes)
    padding_len = elements_range - numpy.remainder(widths_, elements_range)
    if padding_len == elements_range:
        padding_len = 0

    blob_to_write = []
    for j in range(0, channels_):
        if j > 0:
            f.write('\n')
            
        for i in range(0, heights_):
            for k in range(0, widths_):
                hex_elem_list = toHEX_complementary(blob_[0, j, i, k], elem_bytes*8)
                hex_elem_list[0] += '  // F({},{},{})'.format(j, i, k)
                hex_elem_list[0] += str(blob_[0, j, i, k])
                blob_to_write.extend(hex_elem_list)

            for k in range(0, padding_len):
                hex_elem_list = toHEX_complementary(0, elem_bytes * 8)
                blob_to_write.extend(hex_elem_list)
                
        f.write('\n'.join(blob_to_write))
        blob_to_write.clear()

def write_weights_blob(weights_blob_, bias_blob_, f, elem_bytes, filter_group_size = 16, start_number = 0, frac_channelwize = []):
    # Assume blob shape is (Co, Ci, H, W). last axis is width
    shapes = weights_blob_.shape
    shapes = list(shapes) + (4-len(shapes))*[1]
    weights_blob_ = numpy.reshape(weights_blob_, shapes)
    Co_, Ci_, heights_, widths_ = weights_blob_.shape
    init_Co_ = Co_

    # padding filters to be multiples of filter_group_size to match MAC array architecture
    if numpy.remainder(Co_, filter_group_size)>0:
        filter_padded_num = filter_group_size - Co_ % filter_group_size
        weights_blob_ = numpy.concatenate((weights_blob_, numpy.zeros((filter_padded_num, Ci_, heights_, widths_))))
        if len(bias_blob_) > 0:
            bias_blob_ = numpy.concatenate((bias_blob_, numpy.zeros((filter_padded_num))))
        if len(frac_channelwize) > 0:
            frac_channelwize = numpy.concatenate((frac_channelwize, numpy.zeros((filter_padded_num))))
        Co_ += filter_padded_num

    elements_range = int(16 * 2 / elem_bytes) # for 8bit fxp, 32/1; for 16 bit fxp, 32/2
    
    if len(bias_blob_)>0:
        len_bias_per_group = filter_group_size
    else:
        len_bias_per_group = 0
    
    if len(frac_channelwize)>0:
        len_frac_channelwize_per_group = filter_group_size
    else:
        len_frac_channelwize_per_group = 0
        
    padding_len = elements_range - numpy.remainder(Ci_*heights_*widths_*filter_group_size + len_bias_per_group + len_frac_channelwize_per_group, elements_range)
    if padding_len == elements_range:
        padding_len = 0

    blob_to_write = []
    written_lines = 0
    for i in range(0, numpy.int32(numpy.ceil(Co_/filter_group_size))):
        if i > 0 or start_number > 0:
            f.write('\n')
        if len(frac_channelwize) > 0:
            for n in range(0, filter_group_size):
                hex_elem_list = toHEX_complementary(frac_channelwize[i * filter_group_size + n], elem_bytes * 8)
                if i * filter_group_size + n < init_Co_:
                    hex_elem_list[0] += '  // shift_num({})'.format(i * filter_group_size + n + start_number)
                blob_to_write.extend(hex_elem_list)
        
        if len(bias_blob_) > 0:
            for n in range(0, filter_group_size):
                hex_elem_list = toHEX_complementary(bias_blob_[i * filter_group_size + n], elem_bytes * 8)
                if i * filter_group_size + n < init_Co_:
                    hex_elem_list[0] += '  // b({})'.format(i * filter_group_size + n + start_number)
                    hex_elem_list[0] += str(bias_blob_[i * filter_group_size + n])
                blob_to_write.extend(hex_elem_list)

        for j in range(0, Ci_):
            for t in range(0, heights_):
                for m in range(0, widths_):
                    for n in range(0, filter_group_size):
                        hex_elem_list = toHEX_complementary(weights_blob_[i*filter_group_size+n, j, t, m], elem_bytes * 8)
                        if i * filter_group_size + n < init_Co_:
                            hex_elem_list[0] += '  // w({},{},{},{})'.format(i * filter_group_size + n + start_number, j, t, m)
                            hex_elem_list[0] += str(weights_blob_[i*filter_group_size+n, j, t, m])
                        blob_to_write.extend(hex_elem_list)

        for iter in range(0, padding_len):
            hex_elem_list = toHEX_complementary(0, elem_bytes * 8)
            blob_to_write.extend(hex_elem_list)
        f.write('\n'.join(blob_to_write))
        written_lines = written_lines + len(blob_to_write)
        blob_to_write.clear()
        f.write(' // end of one group of 16 filters')
    
    if numpy.remainder(written_lines, 32) > 0:
        f.write('\n')
        blob_to_write = ['00'] * (32 - numpy.remainder(written_lines, 32))
        f.write('\n'.join(blob_to_write))
        written_lines = written_lines + 32 - numpy.remainder(written_lines, 32)
    return written_lines

def model_pack(elem_bytes = 1, file_name = 'WQsample.prototxt', out_file_path = 'out_sample', data_file = 'WQsample.npz', weight_suffix = '_quant_weight', bias_suffix = '_quant_bias', weight_frac_suffix = '_frac_weight', bias_frac_suffix = '_frac_bias', feature_frac_suffix = '_frac', reserve_net_input = True, try_interal = True, data_frac = 0, input_signed = False, model_name = 'model', use_machine_code = False):
    bits_width = elem_bytes * 8
    weight_path = out_file_path + '/weight_file/'
    c_file_path = out_file_path + '/c_file'
    c_file_debug_path = c_file_path + '/debug'
    for_flash_path = out_file_path + '/for_flash_in_other_running_method/'
    for_debug_path = out_file_path + '/for_debug_tool/'
    layer_to_ccode_file_name = c_file_path + '/' + 'net.c'
    layer_info_file_name = c_file_debug_path + '/' + 'layer_type_info.txt'
    machine_code_info_path = out_file_path + '/machine_code_file'
    break_info_file_name = machine_code_info_path + '/' + 'cnn_break_info.txt'
    machine_code_file_name = machine_code_info_path + '/' + 'machine_code.kl'
    tensor_addr_file_name = machine_code_info_path + '/' + 'tensor_addr.txt'
    tensor_info_file_name = machine_code_info_path + '/' + 'tensor_info.txt'
    for_debug_file_name = for_debug_path + 'net_layer_debug_info.h'
    for_debug_ini_file_name = for_debug_path + 'net_layer_debug_info.ini'
    cnn_task_to_layer_name = for_debug_path + 'cnn_task_to_layer_index.ini'
    
    no_opti = False
    
    quant_data = {}
    layers_top_used_time = {}
    layers_merge = []
    tensor_size = {}
    dict_to_layer_bottom = {}
    dict_to_layer_top = {}
    tensor_shape = {}
    input_name = ''
    cnn_one_layer_count = 0
    
    if os.path.exists(data_file):
        if data_file.find('.dic') > 0:
            quant_data_file = open(data_file, 'rb')
            quant_data_binary = pickle.load(quant_data_file, encoding='bytes')
            for key, value in quant_data_binary.items():
                if isinstance(key, str):
                    quant_data[key] = value
                else:
                    quant_data[str(key, encoding = "utf-8")] = value
        else:
            quant_data = dict(numpy.load(data_file))
    
    could_merge_layer_type = [['sigmoid'], ['tanh'], ['relu6'], ['relu'], ['leakyrelu'], ['leaky_relu'], ['prelu'], ['softmax'], ['logsoftmax']]
    show_major_path = False
    layers_read, layers_top_used_time, net_configs = rcaffe.read_layers(file_name)
    layers_merge = rcaffe.mearge_layer(layers_read, layers_top_used_time, show_major_path, could_merge_layer_type = could_merge_layer_type)
    tensor_size, dict_to_layer_bottom, dict_to_layer_top, tensor_shape, input_name = read_tensor.read_tensors(layers_merge, net_configs, byte = elem_bytes)
    read_tensor.check_tensor_shape(tensor_shape)
    rcaffe.change_nonlinear_only(layers_merge)
    tensor_adr, total_size_used = assignAddress.assign_address(layers_merge,tensor_size, dict_to_layer_bottom, dict_to_layer_top, tensor_shape, layers_top_used_time, input_name=input_name, reserve_net_input = reserve_net_input, try_interal = (elem_bytes == 1), elem_bytes = elem_bytes)
    if total_size_used % 256 != 0:
        total_size_used = total_size_used + 256 - total_size_used % 256
    tensor_signed = {}
    if input_name + '_signed' in quant_data.keys():
        input_signed = quant_data[input_name + '_signed']
    tensor_signed[input_name] = input_signed

    if not os.path.exists(out_file_path):
        os.mkdir(out_file_path)
        
    remove_dir(weight_path)
    if not os.path.exists(weight_path):
        os.mkdir(weight_path)
    
    if not os.path.exists(c_file_path):
        os.mkdir(c_file_path)
 
    if not os.path.exists(c_file_debug_path):
        os.mkdir(c_file_debug_path)
        
    if not os.path.exists(machine_code_info_path):
        os.mkdir(machine_code_info_path)
        
    if not os.path.exists(for_flash_path):
        os.mkdir(for_flash_path)
        
    if not os.path.exists(for_debug_path):
        os.mkdir(for_debug_path)
        
    with open(for_debug_file_name, 'w') as f:
        f.write('struct cnn_one_layer_debug_info{\n')
        f.write('    uint8_t  layer_index; // start from 0\n')
        f.write('    uint8_t  mem_type; // 0: in ddr, 1: in iram\n')
        f.write('    uint8_t  int_type; // 1: feature map 8bit, 0: feature map 16bit\n')
        f.write('    int8_t   output_fraction;\n')
        f.write('    uint32_t output_start_addr;\n')
        f.write('    uint32_t output_channel;\n')
        f.write('    uint32_t output_height;\n')
        f.write('    uint32_t output_valid_elem_per_row; // how many elements(8bit or 16bit) in a row of feature map\n')
        f.write('    uint32_t output_size; // how many bytes occupied by output\n')
        f.write('};\n')
        f.write('struct cnn_one_layer_debug_info debug_info[] = {\n')

    with open(for_debug_ini_file_name, 'w') as f:
        f.write('[cnn_net_reg_conf_head]\n')

    with open(cnn_task_to_layer_name, 'w') as f:
        f.write("[cnn_task_index]\n;cnn_task_index=layer_index")

    with open(layer_info_file_name, 'w') as f:
        f.write('[')

    feature_addr_offset = 0 if use_machine_code else 256
    weight_addr_offset = 0 if use_machine_code else total_size_used + feature_addr_offset
    net_out_addr = 0

    with open(layer_to_ccode_file_name, 'w') as f:
        f.write('#include "cnn_conv2d.h"\n')
        f.write('#include "cnn_pool2d.h"\n')
        f.write('#include "cnn_tensor_util.h"\n')
        f.write('/* @brief cnn_run_a_lovely_net() - We transfrom the prototxt file to C code as below, you can copy the code to complete the whole calculation of your network\n')
        f.write('/*        feature occupies {} bytes\n'.format(total_size_used))
        f.write(' * @param feature_addr_offset: the address where input and output of all layers are put\n')
        f.write(' * @param weight_addr_offset: the address where weights and biases all layers are put\n')
        f.write(' * @param layer_end: if you want to stop computation after N layers finished, please set layer_end to N, otherwise, please set layer_end to 0\n')
        f.write(' */\n')
        f.write('void cnn_run_a_lovely_net(uint32_t feature_addr_offset, uint32_t weight_addr_offset, uint16_t layer_end) {\n')
        f.write('    #define RETURN_JUDGE { if(layer_index++ == layer_end) return; }\n')
        f.write('    uint16_t layer_index = 1;\n')
        f.write('    struct cnn_conv2D_config_bean conv2d_bean;\n')
        f.write('    struct cnn_pool2D_config_bean pool2d_bean;\n')

    nonlinearity_algo = {"linear": 0, "sigmoid": 1, "relu": 2, "tanh": 3, "relu6": 6, "prelu": 7}

    CNN_BYTES_OF_CONFIG_ONE_LAYER = 256
    current_weight_addr = 0
    current_config_addr = 0
    current_cnn_engine_layer = 0
    dilation = 1
    slices = 0
    strides = [layer.stride for layer in layers_merge]
    kernels_h = [layer.kernel_size_h for layer in layers_merge]
    kernels_w = [layer.kernel_size_w for layer in layers_merge]
    groups = [layer.group for layer in layers_merge]

    class cnn_bean(Structure):
      _fields_ =[
            ('input_addr', c_uint32),
            ('weight_addr', c_uint32),
            ('output_addr', c_uint32),
            ('output_channel_per_group', c_uint16),
            ('input_height', c_uint16),
            ('input_width', c_uint16),
            ('output_height', c_uint16),
            ('output_width', c_uint16),
            ('input_channel_per_group', c_uint16),
            ('filter_size_h', c_uint8),
            ('filter_size_w', c_uint8),
            ('dilation', c_uint8),
            ('stride', c_uint8),
            ('groups', c_uint16),
            ('mac_type_8bits', c_uint8),
            ('feature_signed', c_uint8),
            ('weight_signed', c_uint8),
            ('depthwise', c_uint8),
            ('padding_left', c_uint8),
            ('padding_right', c_uint8),
            ('padding_up', c_uint8),
            ('pooling_algorithm', c_uint8),
            ('nonlinearity_algo', c_uint8),
            ('bias_en', c_uint8),
            ('output_shift_num', c_uint8),
            ('pooling_ceil_mode', c_uint8),
            ('pooling_count_l_u', c_uint16),
            ('pooling_count_r_u', c_uint16),
            ('pooling_count_l_d', c_uint16),
            ('pooling_count_r_d', c_uint16),
            ('pooling_count_c_u', c_uint16),
            ('pooling_count_c_d', c_uint16),
            ('pooling_count_l_c', c_uint16),
            ('pooling_count_r_c', c_uint16),
            ('pooling_count_c_c', c_uint16),
            ('to_do', c_uint8),
            ('bias_shift', c_uint8),
            ('huffman_en', c_uint8),
            ('most_freq', c_uint16),
            ('prelu_0', c_uint8),
            ('prelu_1', c_uint8),
            ('prelu_2', c_uint8),
            ('prelu_3', c_uint8),
            ('prelu_4', c_uint8),
            ('prelu_5', c_uint8),
            ('prelu_6', c_uint8),
            ('prelu_7', c_uint8),
            ('prelu_8', c_uint8),
            ('prelu_9', c_uint8),
            ('prelu_a', c_uint8),
            ('prelu_b', c_uint8),
            ('prelu_c', c_uint8),
            ('prelu_d', c_uint8),
            ('prelu_e', c_uint8),
            ('prelu_f', c_uint8),
            ('prelu_shift', c_uint8),
            ('F_L_mu', c_uint16),
            ('F_L_d0_mu', c_uint16),
            ('shift_channelwise', c_uint8),
            ('input_iram', c_uint8),
            ('weight_iram', c_uint8),
            ('output_iram', c_uint8),
            ('fi_dma_internal_mode', c_uint8),
            ('fo_dma_internel_mode', c_uint8),
            ('memory_map', c_uint8),
            ('softmax_enable', c_uint8),
            ('softmax_minus_enable', c_uint8),
            ('slices', c_uint8),
            ('cnn_fo_co_distance', c_uint32),
            ('cnn_hiXwi_div32', c_uint32),
            ('cnn_fi_group_distance', c_uint32),
            ('cnn_fo_group_distance', c_uint32),
            ('bias_const', c_uint8),
            ('weight_const', c_uint8),
            ('const_for_weight_bias', c_uint16),
            ('in_linklist', c_uint8),
            ('cnn_next_fi_pad_type', c_uint8),
            ('cnn_next_fi_pad_left', c_uint8),
            ('cnn_next_fi_pad_up', c_uint8),
            ('cnn_fi_optimized_mode', c_uint8),
            ('cnn_fo_optimized_mode', c_uint8),
            ('linklist_halt_en', c_uint8)
        ]
    if platform.system() == 'Windows':
        libcnn = cdll.LoadLibrary(os.path.join(modelpack_dir, 'libcnn.dll'))
    elif platform.system() =='Linux':
        libcnn = ctypes.cdll.LoadLibrary(os.path.join(modelpack_dir, "libcnn.so"))

    libcnn.cnn_get_default_bean.restype = cnn_bean

    machine_code_config = []
    break_cnn_info = []

    def verify_frac(layer, quant_data, feature_frac_suffix, output_include = True):
        if layer.bottom[0] + feature_frac_suffix not in quant_data.keys():
            return False
        frac = quant_data[layer.bottom[0] + feature_frac_suffix]
        for bottom in layer.bottom:
            if bottom + feature_frac_suffix not in quant_data.keys():
                return False
            if frac != quant_data[bottom + feature_frac_suffix]:
                return False
        if output_include:
            for top in layer.top:
                if top + feature_frac_suffix not in quant_data.keys():
                    return False
                if frac != quant_data[top + feature_frac_suffix]:
                    return False     
        return True

    before_cnn_run = True

    input_channel = 0
    input_height = 0
    input_width = 0
    output_channel = 0
    output_height = 0
    output_width = 0
    for index,layer in enumerate(layers_merge):
        assert layer.type not in [['softmax'], ['logsoftmax']], "softmax or logsoftmax should be set after an innerproduct layer or a 1-group convlution layer, but find a single one named {}".format(layer.name)

        if index != 0:
            with open(layer_to_ccode_file_name, 'a') as f:
                f.write('    RETURN_JUDGE\n\n')
                f.write('    /*\n')
                f.write('     * this layer end\n')
                f.write('     */\n')
        else:
            input_channel = tensor_shape[layer.bottom[0]][1]
            input_height = tensor_shape[layer.bottom[0]][2]
            input_width = tensor_shape[layer.bottom[0]][3]

        print("translating layer {}".format(layer.name))

        cnn_config = libcnn.cnn_get_default_bean();

        cnn_config.in_linklist = 1
        cnn_config.mac_type_8bits = 1 if elem_bytes == 1 else 0
        cnn_config.slices = 1
        cnn_config.feature_signed = 1 if tensor_signed[layer.bottom[0]] else 0
        cnn_config.softmax_enable = 1 if 'softmax' in layer.type or 'logsoftmax' in layer.type else 0
        cnn_config.softmax_minus_enable = 1 if 'softmax' in layer.type or 'logsoftmax' in layer.type else 0

        for bottom in layer.bottom:
            if bottom not in tensor_signed.keys():
                tensor_signed[bottom] = True if elem_bytes > 1 else input_signed
        for top in layer.top:
            if top not in tensor_signed.keys():
                try:
                    tensor_signed[top] = tensor_signed[layer.bottom[0]]
                except:
                    tensor_signed[top] = True if elem_bytes > 1 else input_signed

        input_frac = data_frac
        if layer.bottom[0] + feature_frac_suffix in quant_data.keys():
            input_frac = quant_data[layer.bottom[0] + feature_frac_suffix]
        else:
            print("Warning: tensor {}'s fraction is not found in the npz file, use default value {} instead".format(layer.bottom[0], input_frac))
            quant_data[layer.bottom[0] + feature_frac_suffix] = input_frac
        output_frac = bits_width - 1 if 'tanh' in layer.type or 'sigmoid' in layer.type else bits_width - 4
        if layer.top[0] + feature_frac_suffix in quant_data.keys():
            output_frac = quant_data[layer.top[0] + feature_frac_suffix]
        else:
            print("Warning: tensor {}'s fraction is not found in the npz file, use default value {} instead".format(layer.top[0], output_frac))
            quant_data[layer.top[0] + feature_frac_suffix] = output_frac
        middle_frac = bits_width - 3 if 'tanh' in layer.type else (bits_width - 4 if 'sigmoid' in layer.type or 'relu6' in layer.type or 'softmax' in layer.type or 'logsoftmax' in layer.type else output_frac)
        if layer.middle + feature_frac_suffix in quant_data.keys() and 'tanh' not in layer.type and 'sigmoid' not in layer.type:
            middle_frac = quant_data[layer.middle + feature_frac_suffix]
        neg_slope = int(round(layer.negative_slope * 64))

        if 'relu' in layer.type or 'relu6' in layer.type or 'softmax' in layer.type or 'logsoftmax' in layer.type:
            if middle_frac != output_frac:
                raise Exception('fraction of output and input of relu, relu6 or softmax layer should be the same, check layer {}'.format(layer.name))

        if 'relu6' in layer.type:
            if middle_frac > bits_width - 4:
                print('frac is too large, change relu6 to relu after layer {}'.format(layer.name))
                index_relu6 = layer.type.index('relu6')
                layer.type[index_relu6] = 'relu'
            elif middle_frac < bits_width - 4:
                raise Exception('fraction of output and input of relu6, should be 4(8bits mode) or 12(16bits mode), check layer {}'.format(layer.name))

        if 'softmax' in layer.type or 'logsoftmax' in layer.type:
            if middle_frac != bits_width - 4:
                raise Exception('fraction of output and input of softmax, should be 4(8bits mode) or 12(16bits mode), check layer {}'.format(layer.name))

        if 'sigmoid' in layer.type:
            if middle_frac != bits_width - 4:
                raise Exception('fraction of input of sigmoid, should be 4(8bits mode) or 12(16bits mode), check layer {}'.format(layer.name))
            if output_frac != bits_width - 1:
                raise Exception('fraction of output of sigmoid, should be 7(8bits mode) or 15(16bits mode), check layer {}'.format(layer.name))

        if 'tanh' in layer.type:
            if middle_frac != bits_width - 3:
                raise Exception('fraction of input of tanh, should be 5(8bits mode) or 13(16bits mode), check layer {}'.format(layer.name))
            if output_frac != bits_width - 1:
                raise Exception('fraction of output of tanh, should be 7(8bits mode) or 15(16bits mode), check layer {}'.format(layer.name))

        with open(layer_info_file_name, 'a') as f:
            if index != 0:
                f.write(', ')
            f.write('{\n')
            f.write("    name: '{}',\n".format(layer.name))
            f.write('    type: {},\n'.format(layer.type))
            f.write('    bottom: {},\n'.format(layer.bottom))
            f.write('    top: {}\n'.format(layer.top))
            f.write('}')

        if index == len(layers_merge) - 1:
            net_out_frac = output_frac
            net_out_addr = tensor_adr[layer.top[0]]
            output_channel = tensor_shape[layer.top[0]][1]
            output_height = tensor_shape[layer.top[0]][2]
            output_width = tensor_shape[layer.top[0]][3]

        with open(for_debug_file_name, 'a') as f:
            for i in range(len(layer.top)):
                top = layer.top[i]
                if index != 0 or i != 0:
                    f.write(',\n')
                total_height = 1
                for j in range(len(tensor_shape[top]) - 1):
                    total_height *= tensor_shape[top][j]
                f.write('    {')
                f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}'.format(index, 1 if tensor_adr[top] < 0 else 0, 2 - elem_bytes, int(output_frac), max(0, tensor_adr[top]), tensor_shape[top][-3], tensor_shape[top][-2], tensor_shape[top][-1], tensor_size[top]))
                f.write('}')
                cnn_one_layer_count += 1

        with open(layer_to_ccode_file_name, 'a') as f:
            if index == 0:
                f.write('    // layer index: {}, the 1st layer is of index 0\n'.format(index))
            else:
                f.write('    // layer index: {}\n'.format(index))
            f.write('    // layer name: {}\n'.format(layer.name))

        if layer.type == ['slice']:
            if not verify_frac(layer, quant_data, feature_frac_suffix):
#                raise Exception("you need to set fractions of all the inputs and outputs to the same value, check layer {}".format(layer.name))
                for top in layer.top:
                    quant_data[top + feature_frac_suffix] = input_frac
            with open(layer_to_ccode_file_name, 'a') as f:
                f.write('    // layer type: {}\n'.format(layer.type))
                f.write('    // input tensor names are {}\n'.format(layer.bottom))
                f.write('    // output tensor names are {}\n'.format(layer.top))
                f.write('    // input tensor occupy {} bytes individually\n'.format([tensor_size[bottom] for bottom in layer.bottom]))
                f.write('    // output tensor occupy {} bytes individually\n'.format([tensor_size[top] for top in layer.top]))
                if layer.axis < 2:
                    print('you can use the outputs of layer {} directly without calling any funtions'.format(layer.name))
                    f.write('    //you can use the outputs of this layer directly without calling any funtions\n')
                else:
                    f.write('    // you can also use the code below to complete this layer\n')
                    if not before_cnn_run:
                        f.write('    please flush cache you used\n')
                    f.write('    uint16_t *dim_{} = (uint16_t *)os_mem_malloc(0, 2 * {});\n'.format(index, len(tensor_shape[layer.bottom[0]])))
                    for i in range(len(tensor_shape[layer.bottom[0]])):
                        f.write('    dim_{}[{}] = {};\n'.format(index, i, tensor_shape[layer.bottom[0]][i]))
                    if len(layer.slice_point) > 0:
                        slice_point = [0] + layer.slice_point + [tensor_shape[layer.bottom[0]][layer.axis]]
                        begin = [0] * len(tensor_shape[layer.bottom[0]])
                        size = [0] * len(tensor_shape[layer.bottom[0]])
                        for i in range(len(tensor_shape[layer.bottom[0]])):
                            size[i] = tensor_shape[layer.bottom[0]][i]
                        f.write('    uint16_t *begin_{} = (uint16_t *)os_mem_malloc(0, 2 * {});\n'.format(index, len(tensor_shape[layer.bottom[0]])))
                        f.write('    uint16_t *size_{} = (uint16_t *)os_mem_malloc(0, 2 * {});\n'.format(index, len(tensor_shape[layer.bottom[0]])))
                        for i in range(len(layer.top)):
                            begin[layer.axis] = slice_point[i]
                            size[layer.axis] = slice_point[i + 1] - slice_point[i]
                            for j in range(len(tensor_shape[layer.bottom[0]])):
                                f.write('    begin_{}[{}] = {};\n'.format(index, j, begin[j]))
                                f.write('    size_{}[{}] = {};\n'.format(index, j, size[j]))
                            f.write('    cnn_tensor_slice_{}bit((int{}_t *)({} + feature_addr_offset), {}, dim_{}, begin_{}, size_{}, (int{}_t *)({} + feature_addr_offset), 1, 1);\n'\
                                    .format(bits_width, bits_width, tensor_adr[layer.bottom[0]], len(tensor_shape[layer.bottom[0]]), index, index, index, bits_width, tensor_adr[layer.top[i]]))
                            break_cnn_info.append('call function cnn_tensor_slice_{}bit((int{}_t *)({} + feature_addr_offset), {}, dim, begin, size, (int{}_t *)({} + feature_addr_offset), 1, 1) after CNN engine runs {} time(s)'.format(bits_width, bits_width, tensor_adr[layer.bottom[0]], len(tensor_shape[layer.bottom[0]]), bits_width, tensor_adr[layer.top[i]], current_cnn_engine_layer))
                        f.write('    os_mem_free(begin_{});\n'.format(index))
                        f.write('    os_mem_free(size_{});\n'.format(index))
                    else:
                        f.write('    uint32_t *output_addr_{} = os_mem_malloc(0, 4 * {});\n'.format(index, len(layer.top)))
                        for i in range(len(layer.top)):
                            f.write('    output_addr_{}[{}] = {} + feature_addr_offset;\n'.format(index, i, tensor_adr[layer.top[i]]))
                        f.write('    cnn_tensor_split_{}bit((int{}_t *)({} + feature_addr_offset), {}, dim_{}, {}, (int{}_t **)output_addr_{}, {}, 1, 1);\n'\
                                .format(bits_width, bits_width, tensor_adr[layer.bottom[0]], len(tensor_shape[layer.bottom[0]]), index, len(layer.top), bits_width, index, layer.axis))
                        f.write('    os_mem_free(output_addr_{});\n'.format(index))
                        if before_cnn_run:
                            break_cnn_info.append('call function cnn_tensor_split_{}bit((int{}_t *)({} + feature_addr_offset), {}, dim, {}, (int{}_t **)output_addr, {}, 1, 1) before CNN engine runs'.format(bits_width, bits_width, tensor_adr[layer.bottom[0]], len(layer.top), bits_width, tensor_adr[layer.top[i]], layer.axis))
                        else:
                            break_cnn_info.append('call function cnn_tensor_split_{}bit((int{}_t *)({} + feature_addr_offset), {}, dim, {}, (int{}_t **)output_addr, {}, 1, 1) after CNN engine runs {} time(s)'.format(bits_width, bits_width, tensor_adr[layer.bottom[0]], len(layer.top), bits_width, tensor_adr[layer.top[i]], layer.axis, current_cnn_engine_layer))
                    f.write('    os_mem_free(dim_{});\n'.format(index))
                    f.write('    please flush cache you used\n')
            continue
        if layer.type == ['concat']:
            if not verify_frac(layer, quant_data, feature_frac_suffix):
                raise Exception("you need to set fractions of all the inputs and outputs to the same value, check layer {}".format(layer.name))
                for top in layer.top:
                    quant_data[top + feature_frac_suffix] = input_frac
                for bottom in layer.bottom:
                    quant_data[bottom + feature_frac_suffix] = input_frac
            with open(layer_to_ccode_file_name, 'a') as f:
                f.write('    // layer type: {}\n'.format(layer.type))
                f.write('    // input tensor names are {}\n'.format(layer.bottom))
                f.write('    // output tensor names are {}\n'.format(layer.top))
                f.write('    // input tensor occupy {} bytes individually\n'.format([tensor_size[bottom] for bottom in layer.bottom]))
                f.write('    // output tensor occupy {} bytes individually\n'.format([tensor_size[top] for top in layer.top]))
                need_function = layer.axis >= 2 or tensor_adr[layer.top[0]] != tensor_adr[layer.bottom[0]]
                for i in range(len(layer.bottom) - 1):
                    this_size = tensor_size[layer.bottom[i]]
                    this_addr = tensor_adr[layer.bottom[i]]
                    next_addr = tensor_adr[layer.bottom[i + 1]]
                    if this_size + this_addr != next_addr:
                        need_function = True
                        break
                if not need_function:
                    print('you can use the outputs of layer {} directly without calling any funtions'.format(layer.name))
                    f.write('    //you can use the outputs of this layer directly without calling any funtions\n')
                else:
                    if not before_cnn_run:
                        f.write('    please flush cache you used\n')
                    f.write('    // you can also use the code below to complete this layer\n')
                    f.write('    uint16_t *common_dim_{} = (uint16_t *)os_mem_malloc(0, 2 * {});\n'.format(index, len(tensor_shape[layer.bottom[0]]) - 1))
                    commom_dim_index = 0
                    for i in range(len(tensor_shape[layer.bottom[0]])):
                        if i != layer.axis:
                            f.write('    common_dim_{}[{}] = {};\n'.format(index, commom_dim_index, tensor_shape[layer.bottom[0]][i]))
                            commom_dim_index += 1
                    f.write('    uint16_t *size_along_cat_dim_{} = (uint16_t *)os_mem_malloc(0, 2 * {});\n'.format(index, len(layer.bottom)))
                    f.write('    uint32_t *input_addr_{} = os_mem_malloc(0, 4 * {});\n'.format(index, len(layer.bottom)))
                    for i in range(len(layer.bottom)):
                        f.write('    size_along_cat_dim_{}[{}] = {};\n'.format(index, i, tensor_shape[layer.bottom[i]][layer.axis]))
                        f.write('    input_addr_{}[{}] = 0x{:x} + feature_addr_offset;\n'.format(index, i, tensor_adr[layer.bottom[i]]))
                    f.write('    cnn_tensor_concat_{}bit((int{}_t **)input_addr_{}, {}, common_dim_{}, {}, size_along_cat_dim_{}, {}, (int{}_t *)(0x{:x} + feature_addr_offset), 1, 1);\n'\
                            .format(bits_width, bits_width, index, len(tensor_shape[layer.bottom[0]]), index, len(layer.bottom), index, layer.axis, bits_width, tensor_adr[layer.top[0]]))
                    if before_cnn_run:
                        break_cnn_info.append('call function cnn_tensor_concat_{}bit((int{}_t **)input_addr, {}, common_dim, {}, size_along_cat_dim, {}, (int{}_t *)({} + feature_addr_offset), 1, 1) before CNN engine runs'.format(bits_width, bits_width, tensor_adr[layer.bottom[0]], len(layer.top), layer.axis, bits_width, tensor_adr[layer.top[0]]))
                    else:
                        break_cnn_info.append('call function cnn_tensor_concat_{}bit((int{}_t **)input_addr, {}, common_dim, {}, size_along_cat_dim, {}, (int{}_t *)({} + feature_addr_offset), 1, 1) after CNN engine runs {} time(s)'.format(bits_width, bits_width, tensor_adr[layer.bottom[0]], len(layer.top), layer.axis, bits_width, tensor_adr[layer.top[0]], current_cnn_engine_layer))
                    f.write('    os_mem_free(common_dim_{});\n'.format(index))
                    f.write('    os_mem_free(size_along_cat_dim_{});\n'.format(index))
                    f.write('    os_mem_free(input_addr_{});\n'.format(index))
                    f.write('    please flush cache you used\n')
            continue
        if layer.type == ['permute']:
            if not verify_frac(layer, quant_data, feature_frac_suffix):
                raise Exception("you need to set fractions of all the inputs and outputs to the same value, check layer {}".format(layer.name))
                for top in layer.top:
                    quant_data[top + feature_frac_suffix] = input_frac
            with open(layer_to_ccode_file_name, 'a') as f:
                f.write('    // layer type: {}\n'.format(layer.type))
                f.write('    // input tensor names are {}\n'.format(layer.bottom))
                f.write('    // output tensor names are {}\n'.format(layer.top))
                f.write('    // input tensor occupy {} bytes individually\n'.format([tensor_size[bottom] for bottom in layer.bottom]))
                f.write('    // output tensor occupy {} bytes individually\n'.format([tensor_size[top] for top in layer.top]))
                if not before_cnn_run:
                    f.write('    please flush cache you used\n')
                f.write('    uint16_t *dim_{} = (uint16_t *)os_mem_malloc(0, 2 * {});\n'.format(index, len(tensor_shape[layer.bottom[0]])))
                f.write('    uint16_t *perm_{} = (uint16_t *)os_mem_malloc(0, 2 * {});\n'.format(index, len(tensor_shape[layer.bottom[0]])))
                for i in range(len(tensor_shape[layer.bottom[0]])):
                    f.write('    dim_{}[{}] = {};\n'.format(index, i, tensor_shape[layer.bottom[0]][i]))
                    f.write('    perm_{}[{}] = {};\n'.format(index, i, layer.order[i]))
                f.write('    cnn_tensor_transpose_{}bit((int{}_t *)({} + feature_addr_offset), {}, dim_{}, perm_{}, (int{}_t *)({} + feature_addr_offset), 1, 1);\n'.format(bits_width, bits_width, tensor_adr[layer.bottom[0]], len(tensor_shape[layer.bottom[0]]), index, index, bits_width, tensor_adr[layer.top[i]]))
                if before_cnn_run:
                    break_cnn_info.append('call function cnn_tensor_transpose_{}bit((int{}_t *)({} + feature_addr_offset), {}, dim, perm, (int{}_t *)({} + feature_addr_offset), 1, 1) before CNN engine runs'.format(bits_width, bits_width, tensor_adr[layer.bottom[0]], len(tensor_shape[layer.bottom[0]]), bits_width, tensor_adr[layer.top[i]]))
                else:
                    break_cnn_info.append('call function cnn_tensor_transpose_{}bit((int{}_t *)({} + feature_addr_offset), {}, dim, perm, (int{}_t *)({} + feature_addr_offset), 1, 1) after CNN engine runs {} time(s)'.format(bits_width, bits_width, tensor_adr[layer.bottom[0]], len(tensor_shape[layer.bottom[0]]), bits_width, tensor_adr[layer.top[i]], current_cnn_engine_layer))
                f.write('    os_mem_free(dim_{});\n'.format(index))
                f.write('    os_mem_free(perm_{});\n'.format(index))
                f.write('    please flush cache you used\n')
            continue

        can_use_cnn = True
        current_weight_size = 0

        cnn_config.input_addr = tensor_adr[layer.bottom[0]] + feature_addr_offset if tensor_adr[layer.bottom[0]] >= 0 else 0
        cnn_config.output_addr = tensor_adr[layer.top[0]] + feature_addr_offset if tensor_adr[layer.top[0]] >= 0 else 0

        cnn_config.stride = strides[index]
        cnn_config.padding_left = layer.pad_l
        cnn_config.padding_right = max(layer.pad_r, 0)
        cnn_config.padding_up = layer.pad_u
        cnn_config.dilation = dilation
        cnn_config.groups = int(groups[index])

        cnn_config.input_height = tensor_shape[layer.bottom[0]][2]
        cnn_config.input_width = tensor_shape[layer.bottom[0]][3]
        cnn_config.output_height = tensor_shape[layer.top[0]][2]
        cnn_config.output_width = tensor_shape[layer.top[0]][3]

        sizeof_channel = int((tensor_shape[layer.top[0]][3] + 31) / 32) if cnn_config.mac_type_8bits != 0 else int((tensor_shape[layer.top[0]][3] + 15) / 16)
        sizeof_channel = sizeof_channel * 32 * tensor_shape[layer.top[0]][2]

        weight_file_name = weight_path + 'weight_{:04d}_layer_{}.txt'.format(index, str(layer.name).replace('/', '_'))

        cnn_config.output_shift_num = 0
        if 'convolution' in layer.type or 'tf_conv' in layer.type or 'innerproduct' in layer.type:
            cnn_config.to_do = 0
            cnn_config.bias_en = 1 if 'batchnorm' in layer.type or layer.bias_term else 0
            cnn_config.output_channel_per_group = int(tensor_shape[layer.top[0]][1] / layer.group)
            cnn_config.input_channel_per_group = int(tensor_shape[layer.bottom[0]][1] / layer.group)

            if(cnn_config.output_channel_per_group == 1 and cnn_config.input_channel_per_group == 1):
                cnn_config.depthwise = 1 
            else:
                cnn_config.depthwise = 0

            cnn_config.filter_size_h = kernels_h[index]
            cnn_config.filter_size_w = kernels_w[index]

            if 'innerproduct' in layer.type:
                assert tensor_shape[layer.bottom[0]][2] in range(1, 16), "Innerproduct layer's input height should be no larger than 16, but got {} in layer {}".format(tensor_shape[layer.bottom[0]][2], layer.name)
                assert tensor_shape[layer.bottom[0]][3] in range(1, 16), "Innerproduct layer's input width should be no larger than 16, but got {} in layer {}".format(tensor_shape[layer.bottom[0]][3], layer.name)
                cnn_config.filter_size_h = cnn_config.input_height
                cnn_config.filter_size_w = cnn_config.input_width

            if 'softmax' in layer.type or 'logsoftmax' in layer.type:
                assert cnn_config.groups == 1, "softmax or logsoftmax should be set after an innerproduct layer or a 1-group convlution layer, but find one after layer {}".format(layer.name)
                assert tensor_shape[layer.bottom[0]][2] == 1, "softmax layer input's height should be 1, but got {} in layer {}".format(tensor_shape[layer.bottom[0]][2], layer.name)
                assert tensor_shape[layer.bottom[0]][3] * elem_bytes <= 32, "softmax layer input's width should be no larger than 32(8bit mode) or 16(16bit mode), but got {} in layer {}".format(tensor_shape[layer.bottom[0]][3], layer.name)

            k_h_d = (cnn_config.filter_size_h - 1) * layer.dilation + 1

            cnn_fi_padding_wi = cnn_config.padding_left + cnn_config.padding_right + cnn_config.input_width
            cnn_fi_padding_wi_div_16s = int((cnn_fi_padding_wi - 1) / (32 * cnn_config.stride)) + 1 if cnn_config.mac_type_8bits != 0 else int((cnn_fi_padding_wi - 1) / (16 * cnn_config.stride)) + 1
            cnn_fi_padding_wi_div_16 = cnn_config.stride * cnn_fi_padding_wi_div_16s
            fi_row_size = 32 * cnn_config.input_channel_per_group * cnn_fi_padding_wi_div_16
            fi_k_row_size = fi_row_size * k_h_d
            fi_all_size = (cnn_config.input_height + cnn_config.padding_up) * fi_row_size
            assert cnn_fi_padding_wi in range(1, 65536), "Padded width should be smaller than 65536, but got {} in layer {}".format(cnn_fi_padding_wi, layer.name)
            assert fi_k_row_size in range(1, 262145), "input elements for calculating 1 row should occupy less than 256KB in iram, but got {} in layer {}".format(fi_k_row_size, layer.name)

            assert cnn_config.input_channel_per_group * cnn_config.filter_size_h * cnn_config.filter_size_w < 65536, "Convolution layer's number of accumulation, i.e. input_channel_per_group * kernel_height * kernel_width + 1 should be no larger than 65536, but get {} in layer {}".format(
                cnn_config.input_channel_per_group * cnn_config.filter_size_h * cnn_config.filter_size_w + 1, layer.name)    
            if cnn_config.input_channel_per_group == 1 or cnn_config.filter_size_h * cnn_config.filter_size_w > 1:
                assert cnn_config.stride <= cnn_config.filter_size_h, "kernel height should be larger than stride unless kernel is 1*1 and input channel per group > 1, but got stride = {}, kernel_size_h = {}".format(cnn_config.stride, cnn_config.filter_size_h)
                assert cnn_config.stride <= cnn_config.filter_size_w, "kernel width should be larger than stride unless kernel is 1*1 and input channel per group > 1, but got stride = {}, kernel_size_w = {}".format(cnn_config.stride, cnn_config.filter_size_w)

            if cnn_config.output_channel_per_group > 16:
                cnn_config.slices = int((fi_all_size - 1) / 262144) + 1
            weight = numpy.random.randint(-128, 127, (layer.num_output, cnn_config.input_channel_per_group, cnn_config.filter_size_h, cnn_config.filter_size_w))
            bias = [] if cnn_config.bias_en == 0 else numpy.zeros(layer.num_output)
            weight_frac = numpy.array([5])
            bias_frac = numpy.array([4])
            if layer.name + weight_suffix in quant_data.keys():
                weight = quant_data[layer.name + weight_suffix]
                try:
                    weight = weight.reshape((cnn_config.output_channel_per_group * cnn_config.groups, cnn_config.input_channel_per_group, cnn_config.filter_size_h, cnn_config.filter_size_w))
                except ValueError as e:
                    print(e)
                    raise Exception("weight of layer {} is npz file is not suitable, it is supposed to be {}, but get {} instead".format(layer.name, (cnn_config.output_channel_per_group * cnn_config.groups, cnn_config.input_channel_per_group, cnn_config.filter_size_h, cnn_config.filter_size_w), weight.shape))
            else:
                print("Warning: weight of layer {} is not in npz file, use random one instead".format(layer.name))
            if layer.name + bias_suffix in quant_data.keys():
                bias = quant_data[layer.name + bias_suffix]
                if cnn_config.bias_en != 0:
                    try:
                        bias = bias.reshape(cnn_config.output_channel_per_group * cnn_config.groups)
                    except ValueError as e:
                        print(e)
                        raise Exception("bias of layer {} is npz file is not suitable, it is supposed to be {}, but get {} instead".format(layer.name, (cnn_config.output_channel_per_group * cnn_config.groups), bias.shape))
            else:
                if cnn_config.bias_en != 0:
                    print("Warning: bias of layer {} is not in npz file, use random one instead".format(layer.name))
            if layer.name + weight_frac_suffix in quant_data.keys():
                weight_frac = quant_data[layer.name + weight_frac_suffix].flatten()
            else:
                print("Warning: weight frac of layer {} is not in npz file, use random one instead".format(layer.name))
            if layer.name + bias_frac_suffix in quant_data.keys():
                bias_frac = quant_data[layer.name + bias_frac_suffix].flatten()
            else:
                if cnn_config.bias_en != 0:
                    print("Warning: bias frac of layer {} is not in npz file, use random one instead".format(layer.name))
            if cnn_config.bias_en != 0:
                bias = bias * 1.0
                if len(weight_frac) == 1 and len(bias_frac) == 1:
                    cnn_config.shift_channelwise = 0
                    if input_frac + weight_frac[0] - middle_frac < 0 or int(input_frac + weight_frac[0] - middle_frac) > 31:
                        raise Exception("input_fraction + weight_fraction - output_fraction of conv layer should be in [0, 31], check layer {}".format(layer.name))
                    cnn_config.output_shift_num = int(input_frac + weight_frac[0] - middle_frac)
                    bias_shift = int(input_frac + weight_frac[0] - bias_frac[0])
                    if bias_shift < 0:
                        bias = bias * math.pow(2, bias_shift)
                        bias = numpy.floor(bias)
                        cnn_config.bias_shift = 0
                    else:
                        if bias_shift > 31:
                            raise Exception("input_fraction + weight_fraction - bias_fraction of conv layer should be in [0, 31], check layer {}".format(layer.name))
                        cnn_config.bias_shift = bias_shift
                    writing_method = 'w'
                    for group_index in range(cnn_config.groups):
                        with open(weight_file_name, writing_method) as f:
                            current_weight_size += write_weights_blob(weight[group_index * cnn_config.output_channel_per_group : (group_index + 1) * cnn_config.output_channel_per_group, :, :, :], bias[group_index * cnn_config.output_channel_per_group : (group_index + 1) * cnn_config.output_channel_per_group], f, elem_bytes, 16 if cnn_config.depthwise == 0 else 1, group_index * cnn_config.output_channel_per_group)
                        writing_method = 'a'
                else:
                    if len(weight_frac) == 1:
                        weight_frac = weight_frac.tile(layer.num_output)
                    if len(bias_frac) == 1:
                        bias_frac = bias_frac.tile(layer.num_output)
                    if len(weight_frac) != layer.num_output or len(bias_frac) != layer.num_output:
                        raise Exception("length of weight fraction and bias_fraction of conv layer should be equal with num_output when weight fraction or bias fraction differ on output channel, check layer {}".format(layer.name))
                    cnn_config.shift_channelwise = 1
                    out_shift_num = input_frac + weight_frac - middle_frac
                    bias_shift_num = input_frac + weight_frac - bias_frac
                    for i in range(len(bias_shift_num)):
                        if bias_shift_num[i] < 0:
                            bias[i] = bias[i] * math.pow(2, bias_shift_num[i])
                            bias_shift_num[i] = 0
                        if bias_shift_num[i] > 15 and bits_width == 8:
                            raise Exception("input_fraction + weight_fraction - bias_fraction of conv layer should be in [0, 15] in 8bit mode when weight fraction or bias fraction differ on output channel, check layer {}".format(layer.name))
                        if bias_shift_num[i] > 31 and bits_width == 16:
                            raise Exception("input_fraction + weight_fraction - bias_fraction of conv layer should be in [0, 31] in 16bit mode when weight fraction or bias fraction differ on output channel, check layer {}".format(layer.name))
                        if (out_shift_num[i] < 0 or out_shift_num[i] > 15) and bits_width == 8:
                            raise Exception("input_fraction + weight_fraction - output_fraction of conv layer should be in [0, 15] in 8bit mode when weight fraction or bias fraction differ on output channel, check layer {}".format(layer.name))
                        if (out_shift_num[i] < 0 or out_shift_num[i] > 31) and bits_width == 16:
                            raise Exception("input_fraction + weight_fraction - output_fraction of conv layer should be in [0, 31] in 16bit mode when weight fraction or bias fraction differ on output channel, check layer {}".format(layer.name))
                    bias = numpy.floor(bias)
                    shift_num = out_shift_num * 2 ** (elem_bytes * 4) + bias_shift_num
                    writing_method = 'w'
                    for group_index in range(cnn_config.groups):
                        with open(weight_file_name, writing_method) as f:
                            current_weight_size += write_weights_blob(weight[group_index * cnn_config.output_channel_per_group : (group_index + 1) * cnn_config.output_channel_per_group, :, :, :], bias[group_index * cnn_config.output_channel_per_group : (group_index + 1) * cnn_config.output_channel_per_group], f, elem_bytes, 16 if cnn_config.depthwise == 0 else 1, group_index * cnn_config.output_channel_per_group, shift_num[group_index * cnn_config.output_channel_per_group : (group_index + 1) * cnn_config.output_channel_per_group])
                        writing_method = 'a'
            else:
                if len(weight_frac) == 1:
                    cnn_config.shift_channelwise = 0
                    if input_frac + weight_frac[0] - middle_frac < 0 or int(input_frac + weight_frac[0] - middle_frac) > 31:
                        raise Exception("input_fraction + weight_fraction - output_fraction of conv layer should be in [0, 31], check layer {}".format(layer.name))
                    cnn_config.output_shift_num = int(input_frac + weight_frac[0] - middle_frac)
                    cnn_config.bias_shift = 0
                    writing_method = 'w'
                    for group_index in range(cnn_config.groups):
                        with open(weight_file_name, writing_method) as f:
                            current_weight_size += write_weights_blob(weight[group_index * cnn_config.output_channel_per_group : (group_index + 1) * cnn_config.output_channel_per_group, :, :, :], [], f, elem_bytes, 16 if cnn_config.depthwise == 0 else 1, group_index * cnn_config.output_channel_per_group)
                        writing_method = 'a'
                else:
                    cnn_config.shift_channelwise = 1
                    out_shift_num = input_frac + weight_frac - middle_frac
                    for i in range(len(out_shift_num)):
                        if (out_shift_num[i] < 0 or out_shift_num[i] > 15) and bits_width == 8:
                            raise Exception("input_fraction + weight_fraction - output_fraction of conv layer should be in [0, 15] in 8bit mode when weight fraction or bias fraction differ on output channel, check layer {}".format(layer.name))
                        if (out_shift_num[i] < 0 or out_shift_num[i] > 31) and bits_width == 16:
                            raise Exception("input_fraction + weight_fraction - output_fraction of conv layer should be in [0, 31] in 16bit mode when weight fraction or bias fraction differ on output channel, check layer {}".format(layer.name))
                    bias_shift_num = input_frac + weight_frac - middle_frac
                    shift_num = out_shift_num * 2 ** (elem_bytes * 4) + bias_shift_num
                    writing_method = 'w'
                    for group_index in range(cnn_config.groups):
                        with open(weight_file_name, writing_method) as f:
                            current_weight_size += write_weights_blob(weight[group_index * cnn_config.output_channel_per_group : (group_index + 1) * cnn_config.output_channel_per_group, :, :, :], [], f, elem_bytes, 16 if cnn_config.depthwise == 0 else 1, group_index * cnn_config.output_channel_per_group, shift_num[group_index * cnn_config.output_channel_per_group : (group_index + 1) * cnn_config.output_channel_per_group])
                        writing_method = 'a'

            cnn_config.weight_signed = 1
        elif 'pooling' in layer.type or 'tf_pool' in layer.type:
            cnn_config.to_do = 1
            cnn_config.shift_channelwise = 0 
            cnn_config.output_channel_per_group = int(tensor_shape[layer.top[0]][1])
            cnn_config.input_channel_per_group = int(tensor_shape[layer.bottom[0]][1])
            cnn_config.padding_left = layer.pad_l
            cnn_config.padding_right = max(layer.pad_r, 0)
            cnn_config.padding_up = layer.pad_u

            if layer.global_pooling:
                cnn_config.output_height =  1
                cnn_config.output_width = 1
                assert tensor_shape[layer.bottom[0]][2] in range(1, 16), "Global pooling layer's input height should be no larger than 16, but got {} in layer {}".format(tensor_shape[layer.bottom[0]][2], layer.name)
                assert tensor_shape[layer.bottom[0]][3] in range(1, 16), "Global pooling layer's input width should be no larger than 16, but got {} in layer {}".format(tensor_shape[layer.bottom[0]][3], layer.name)
                cnn_config.filter_size_h = cnn_config.input_height
                cnn_config.filter_size_w = cnn_config.input_width
            else:
                cnn_config.filter_size_h = kernels_h[index] if layer.kernel_size_h == 0 else layer.kernel_size_h
                cnn_config.filter_size_w = kernels_w[index] if layer.kernel_size_w == 0 else layer.kernel_size_w
            k_h_d = (cnn_config.filter_size_h - 1) * layer.dilation + 1
            k_w_d = (cnn_config.filter_size_w - 1) * layer.dilation + 1
            h_i_need = (cnn_config.output_height - 1) * cnn_config.stride + k_h_d
            p_d_real = h_i_need - cnn_config.input_height - cnn_config.padding_up

            cnn_fi_padding_wi = cnn_config.padding_left + cnn_config.padding_right + cnn_config.input_width
            cnn_fi_padding_wi_div_16s = int((cnn_fi_padding_wi - 1) / (32 * cnn_config.stride)) + 1 if cnn_config.mac_type_8bits != 0 else int((cnn_fi_padding_wi - 1) / (16 * cnn_config.stride)) + 1
            cnn_fi_padding_wi_div_16 = cnn_config.stride * cnn_fi_padding_wi_div_16s
            fi_row_size = 32 * cnn_fi_padding_wi_div_16
            fi_k_row_size = fi_row_size * k_h_d
            fi_all_size = (cnn_config.input_height + cnn_config.padding_up) * fi_row_size
            assert cnn_fi_padding_wi in range(1, 65536), "Padded width should be smaller than 65536, but got {} in layer {}".format(cnn_fi_padding_wi, layer.name)
            assert fi_k_row_size in range(1, 262145), "input elements for calculating 1 row should occupy less than 256KB in iram, but got {} in layer {}".format(fi_k_row_size, layer.name)

            if layer.pool == True:
                cnn_config.weight_signed = 0 if cnn_config.mac_type_8bits != 0 else 1 
                multiplier_bean = avg_pool_multiplier.cnn_avg_pooling_multiplier_bean()
                if layer.global_pooling:
                    if cnn_config.mac_type_8bits:
                        multiplier_bean.cnn_get_pooling_multiplier_8bits(cnn_config.filter_size_h * cnn_config.filter_size_w)
                    else:
                        multiplier_bean.cnn_get_pooling_multiplier_16bits(cnn_config.filter_size_h * cnn_config.filter_size_w)
                    cnn_config.output_shift_num = multiplier_bean.shift_num
                    cnn_config.pooling_count_l_u = multiplier_bean.multiplier
                    cnn_config.pooling_count_c_c = multiplier_bean.multiplier
                    cnn_config.pooling_count_c_d = multiplier_bean.multiplier
                    cnn_config.pooling_count_l_c = multiplier_bean.multiplier
                    cnn_config.pooling_count_l_d = multiplier_bean.multiplier
                    cnn_config.pooling_count_r_c = multiplier_bean.multiplier
                    cnn_config.pooling_count_r_d = multiplier_bean.multiplier
                    cnn_config.pooling_count_r_u = multiplier_bean.multiplier
                    cnn_config.pooling_count_c_u = multiplier_bean.multiplier
                else:
                    multiplier_bean.cnn_get_pooling_multiplier(8 if cnn_config.mac_type_8bits != 0 else 16, cnn_config.padding_left, cnn_config.padding_up, p_d_real, k_h_d, k_w_d, 0, cnn_config.input_height, cnn_config.input_width)
                    cnn_config.output_shift_num = multiplier_bean.shift_num
                    cnn_config.pooling_count_l_c = multiplier_bean.pooling_count_l_c + multiplier_bean.pooling_count_c_c * 256
                    cnn_config.pooling_count_l_d = multiplier_bean.pooling_count_l_d + multiplier_bean.pooling_count_c_d * 256
                    cnn_config.pooling_count_l_u = multiplier_bean.pooling_count_l_u + multiplier_bean.pooling_count_c_u * 256
                    cnn_config.pooling_count_r_c = multiplier_bean.pooling_count_r_c * 257
                    cnn_config.pooling_count_r_d = multiplier_bean.pooling_count_r_d * 257
                    cnn_config.pooling_count_r_u = multiplier_bean.pooling_count_r_u * 257
                    cnn_config.pooling_count_c_c = multiplier_bean.pooling_count_c_c * 257
                    cnn_config.pooling_count_c_d = multiplier_bean.pooling_count_c_d * 257
                    cnn_config.pooling_count_c_u = multiplier_bean.pooling_count_c_u * 257
                cnn_config.pooling_algorithm = 1
                cnn_config.output_shift_num = int(cnn_config.output_shift_num + input_frac - output_frac)
            else:
                assert cnn_config.feature_signed == 1, "Maxpool's input should be signed number"
                cnn_config.weight_signed = 1
                cnn_config.pooling_algorithm = 0
                if input_frac != output_frac:
                    raise Exception("you need to set input fraction and output fraction to the same value, check layer {}".format(layer.name))
        elif 'eltwise' in layer.type:
            if not verify_frac(layer, quant_data, feature_frac_suffix, False):
                raise Exception("you need to set fractions of all the inputs to the same value, check layer {}".format(layer.name))
            if layer.operation == "SUM":
                out_shift_num = int(input_frac - middle_frac)
                if out_shift_num < 0 or out_shift_num > 31:
                    raise Exception("input_fraction - output_fraction of add layer should be in [0, 31], check layer {}".format(layer.name))
                cnn_config.output_shift_num = out_shift_num
                if len(layer.bottom) == 2:
                    cnn_config.input_addr = min([tensor_adr[layer.bottom[0]], tensor_adr[layer.bottom[1]]]) + feature_addr_offset
                    cnn_config.input_channel_per_group = 2;
                    cnn_config.input_width = tensor_shape[layer.top[0]][3]
                    cnn_config.input_height = tensor_shape[layer.top[0]][1] * tensor_shape[layer.top[0]][2]
                    cnn_config.bias_en = False
                    cnn_config.output_addr = tensor_adr[layer.top[0]] + feature_addr_offset if tensor_adr[layer.top[0]] > 0 else 0
                    cnn_config.output_channel_per_group = 1;
                    cnn_config.output_height = cnn_config.input_height
                    cnn_config.output_width = cnn_config.input_width
                    cnn_config.filter_size_h = 1
                    cnn_config.filter_size_w = 1
                    cnn_config.feature_signed = 1
                    cnn_config.weight_const = 1
                    cnn_config.const_for_weight_bias = 1 if cnn_config.mac_type_8bits == 0 else 257
                    cnn_config.weight_signed = 1
                    cnn_config.cnn_hiXwi_div32 = max([tensor_adr[layer.bottom[0]], tensor_adr[layer.bottom[1]]]) - min([tensor_adr[layer.bottom[0]], tensor_adr[layer.bottom[1]]])
                    cnn_config.cnn_hiXwi_div32 = int(cnn_config.cnn_hiXwi_div32 / 32)
                else:
                    addrs = [tensor_adr[bottom] for bottom in layer.bottom]
                    addrs.sort()
                    for i in range(len(addrs) - 2):
                        if addrs[i] + addrs[i + 2] != addrs[i + 1] * 2:
                            can_use_cnn = False
                    if can_use_cnn:
                        cnn_config.input_addr = addrs[0] + feature_addr_offset
                        cnn_config.input_channel_per_group = 3;
                        cnn_config.input_width = tensor_shape[layer.top[0]][3]
                        cnn_config.input_height = tensor_shape[layer.top[0]][1] * tensor_shape[layer.top[0]][2]
                        cnn_config.bias_en = False
                        cnn_config.output_addr = tensor_adr[layer.top[0]] + feature_addr_offset if tensor_adr[layer.top[0]] > 0 else 0
                        cnn_config.output_channel_per_group = 1;
                        cnn_config.output_height = cnn_config.input_height
                        cnn_config.output_width = cnn_config.input_width
                        cnn_config.filter_size_h = 1
                        cnn_config.filter_size_w = 1
                        cnn_config.feature_signed = 1
                        cnn_config.weight_const = 1
                        cnn_config.const_for_weight_bias = 1
                        cnn_config.weight_signed = 1
                        cnn_config.cnn_hiXwi_div32 = addrs[1] - addrs[0]
            else:
                can_use_cnn = False
            assert not ('softmax' in layer.type or 'logsoftmax' in layer.type), "softmax or logsoftmax should be set after an innerproduct layer or a 1-group convlution layer, but find one after layer {}".format(layer.name)
        elif 'batchnorm' in layer.type or 'scale' in layer.type:
            assert not ('softmax' in layer.type or 'logsoftmax' in layer.type), "softmax or logsoftmax should be set after an innerproduct layer or a 1-group convlution layer, but find one after layer {}".format(layer.name)
            cnn_config.to_do = 0
            cnn_config.bias_en = 1 if 'batchnorm' in layer.type or layer.bias_term else 0 
            cnn_config.output_channel_per_group = 1
            cnn_config.input_channel_per_group = 1
            cnn_config.groups = tensor_shape[layer.bottom[0]][1]
            cnn_config.depthwise = 1
            cnn_config.filter_size_h = 2
            cnn_config.filter_size_w = 1
            weight = numpy.random.randint(-128, 127, (layer.num_output, cnn_config.input_channel_per_group, cnn_config.filter_size_h, cnn_config.filter_size_w))
            bias = [] if cnn_config.bias_en == 0 else numpy.zeros(layer.num_output)
            weight_frac = numpy.array([5])
            bias_frac = numpy.array([4])
            if layer.name + weight_suffix in quant_data.keys():
                weight = quant_data[layer.name + weight_suffix]
                weight = weight.reshape(cnn_config.groups, 1, 1, 1)
            if layer.name + bias_suffix in quant_data.keys():
                bias = quant_data[layer.name + bias_suffix]
            if layer.name + weight_frac_suffix in quant_data.keys():
                weight_frac = quant_data[layer.name + weight_frac_suffix].flatten()
            if layer.name + bias_frac_suffix in quant_data.keys():
                bias_frac = quant_data[layer.name + bias_frac_suffix].flatten()
            if cnn_config.bias_en != 0:
                bias = bias * 1.0
                if len(weight_frac) == 1 and len(bias_frac) == 1:
                    cnn_config.shift_channelwise = 0
                    if input_frac + weight_frac[0] - middle_frac < 0 or int(input_frac + weight_frac[0] - middle_frac) > 31:
                        raise Exception("input_fraction + weight_fraction - output_fraction of scale layer should be in [0, 31], check layer {}".format(layer.name))
                    cnn_config.output_shift_num = int(input_frac + weight_frac[0] - middle_frac)
                    bias_shift = int(input_frac + weight_frac[0] - bias_frac[0])
                    if bias_shift < 0:
                        bias = bias * math.pow(2, bias_shift)
                        bias = numpy.floor(bias)
                        cnn_config.bias_shift = 0
                    else:
                        if bias_shift > 31:
                            raise Exception("input_fraction + weight_fraction - bias_fraction of scale layer should be in [0, 31], check layer {}".format(layer.name))
                        cnn_config.bias_shift = bias_shift
                    writing_method = 'w'
                    for group_index in range(cnn_config.groups):
                        with open(weight_file_name, writing_method) as f:
                            current_weight_size += write_weights_blob(weight[group_index * cnn_config.output_channel_per_group : (group_index + 1) * cnn_config.output_channel_per_group, :, :, :], bias[group_index * cnn_config.output_channel_per_group : (group_index + 1) * cnn_config.output_channel_per_group], f, elem_bytes, 1, group_index * cnn_config.output_channel_per_group)
                        writing_method = 'a'
                else:
                    if len(weight_frac) == 1:
                        weight_frac = weight_frac.tile(layer.num_output)
                    if len(bias_frac) == 1:
                        bias_frac = bias_frac.tile(layer.num_output)
                    cnn_config.shift_channelwise = 1
                    out_shift_num = input_frac + weight_frac - middle_frac
                    bias_shift_num = input_frac + weight_frac - bias_frac
                    for i in range(len(bias_shift_num)):
                        if bias_shift_num[i] < 0:
                            bias[i] = bias[i] * math.pow(2, bias_shift_num[i])
                            bias_shift_num[i] = 0
                        if bias_shift_num[i] > 15 and bits_width == 8:
                            raise Exception("input_fraction + weight_fraction - bias_fraction of scale layer should be in [0, 15] in 8bit mode when weight fraction or bias fraction differ on output channel, check layer {}".format(layer.name))
                        if bias_shift_num[i] > 31 and bits_width == 16:
                            raise Exception("input_fraction + weight_fraction - bias_fraction of scale layer should be in [0, 31] in 16bit mode when weight fraction or bias fraction differ on output channel, check layer {}".format(layer.name))
                        if (out_shift_num[i] < 0 or out_shift_num[i] > 15) and bits_width == 8:
                            raise Exception("input_fraction + weight_fraction - output_fraction of scale layer should be in [0, 15] in 8bit mode when weight fraction or bias fraction differ on output channel, check layer {}".format(layer.name))
                        if (out_shift_num[i] < 0 or out_shift_num[i] > 31) and bits_width == 16:
                            raise Exception("input_fraction + weight_fraction - output_fraction of scale layer should be in [0, 31] in 16bit mode when weight fraction or bias fraction differ on output channel, check layer {}".format(layer.name))
                    bias = numpy.floor(bias)
                    shift_num = out_shift_num * 2 ** (elem_bytes * 4) + bias_shift_num
                    writing_method = 'w'
                    for group_index in range(cnn_config.groups):
                        with open(weight_file_name, writing_method) as f:
                            current_weight_size += write_weights_blob(weight[group_index * cnn_config.output_channel_per_group : (group_index + 1) * cnn_config.output_channel_per_group, :, :, :], bias[group_index * cnn_config.output_channel_per_group : (group_index + 1) * cnn_config.output_channel_per_group], f, elem_bytes, 1, group_index * cnn_config.output_channel_per_group, shift_num[group_index * cnn_config.output_channel_per_group : (group_index + 1) * cnn_config.output_channel_per_group])
                        writing_method = 'a'
            else:
                if len(weight_frac) == 1:
                    cnn_config.shift_channelwise = 0
                    if input_frac + weight_frac[0] - middle_frac < 0 or int(input_frac + weight_frac[0] - middle_frac) > 31:
                        raise Exception("input_fraction + weight_fraction - output_fraction of scale layer should be in [0, 31], check layer {}".format(layer.name))
                    cnn_config.output_shift_num = int(input_frac + weight_frac[0] - middle_frac)
                    cnn_config.bias_shift = 0
                    writing_method = 'w'
                    for group_index in range(cnn_config.groups):
                        with open(weight_file_name, writing_method) as f:
                            current_weight_size += write_weights_blob(weight[group_index * cnn_config.output_channel_per_group : (group_index + 1) * cnn_config.output_channel_per_group, :, :, :], [], f, elem_bytes, 1, group_index * cnn_config.output_channel_per_group)
                        writing_method = 'a'
                else:
                    cnn_config.shift_channelwise = 1
                    out_shift_num = input_frac + weight_frac - middle_frac
                    for i in range(len(out_shift_num)):
                        if (out_shift_num[i] < 0 or out_shift_num[i] > 15) and bits_width == 8:
                            raise Exception("input_fraction + weight_fraction - output_fraction of scale layer should be in [0, 15] in 8bit mode when weight fraction or bias fraction differ on output channel, check layer {}".format(layer.name))
                        if (out_shift_num[i] < 0 or out_shift_num[i] > 31) and bits_width == 16:
                            raise Exception("input_fraction + weight_fraction - output_fraction of scale layer should be in [0, 31] in 16bit mode when weight fraction or bias fraction differ on output channel, check layer {}".format(layer.name))
                    bias_shift_num = input_frac + weight_frac - middle_frac
                    shift_num = out_shift_num * 2 ** (elem_bytes * 4) + bias_shift_num
                    writing_method = 'w'
                    for group_index in range(cnn_config.groups):
                        with open(weight_file_name, writing_method) as f:
                            current_weight_size += write_weights_blob(weight[group_index * cnn_config.output_channel_per_group : (group_index + 1) * cnn_config.output_channel_per_group, :, :, :], [], f, elem_bytes, 1, group_index * cnn_config.output_channel_per_group, shift_num[group_index * cnn_config.output_channel_per_group : (group_index + 1) * cnn_config.output_channel_per_group])
                        writing_method = 'a'

            cnn_config.weight_signed = 1 
        elif 'shuffle' in layer.type or 'shufflechannel' in layer.type:
            cnn_config.feature_signed = 1
            if not verify_frac(layer, quant_data, feature_frac_suffix):
                raise Exception("you need to set fractions of all the inputs and outputs to the same value, check layer {}".format(layer.name))
            cnn_config.depthwise = 1;
            cnn_config.input_channel_per_group = 1;
            cnn_config.output_channel_per_group = 1;
            cnn_config.to_do = 0;
            cnn_config.groups = int(tensor_shape[layer.top[0]][1] / int(layer.group));
            cnn_config.pooling_algorithm = 0;
            cnn_config.filter_size_h = 2;
            cnn_config.filter_size_w = 1;
            cnn_config.cnn_fo_group_distance = sizeof_channel * layer.group;
            weight = numpy.ones([cnn_config.groups, 1, 1, 1])
            writing_method = 'w'
            with open(weight_file_name, writing_method) as f:
                current_weight_size += write_weights_blob(weight, [], f, elem_bytes, filter_group_size = 1)
                writing_method = 'a'
        elif 'scalebytensor' in layer.type:
            cnn_config.to_do = 0
            cnn_config.bias_en = 0
            cnn_config.output_channel_per_group = 1
            cnn_config.input_channel_per_group = 1
            cnn_config.groups = tensor_shape[layer.bottom[0]][1]
            cnn_config.depthwise = 1
            cnn_config.filter_size_h = 2
            cnn_config.filter_size_w = 1
            cnn_config.shift_channelwise = 0
            cnn_config.output_shift_num = int(input_frac + quant_data[layer.bottom[1] + feature_frac_suffix] - middle_frac)
            cnn_config.weight_addr = tensor_adr[layer.bottom[1]] + feature_addr_offset + 1
            cnn_config.weight_signed = 1
        elif 'bias' in layer.type:
            assert not ('softmax' in layer.type or 'logsoftmax' in layer.type), "softmax or logsoftmax should be set after an innerproduct layer or a 1-group convlution layer, but find one after layer {}".format(layer.name)
            cnn_config.to_do = 0
            cnn_config.bias_en = 1
            cnn_config.output_channel_per_group = 1
            cnn_config.input_channel_per_group = 1
            cnn_config.groups = tensor_shape[layer.bottom[0]][1]
            cnn_config.depthwise = 1
            cnn_config.filter_size_h = 2
            cnn_config.filter_size_w = 1
            if middle_frac > input_frac:
                raise Exception('fraction of output cannot be larger than that of input in bias layer, check layer {}'.format(layer.name))
            weight = numpy.ones([tensor_shape[layer.bottom[0]][1], 1, 1, 1])
            bias = numpy.zeros(tensor_shape[layer.bottom[0]][1])
            weight_frac = numpy.array([0])
            bias_frac = numpy.array([input_frac])
            if layer.name + weight_suffix in quant_data.keys():
                bias = quant_data[layer.name + weight_suffix]
                try:
                    len(bias)
                except(TypeError):
                    bias = numpy.array([bias])
            if layer.name + weight_frac_suffix in quant_data.keys():
                bias_frac = quant_data[layer.name + weight_frac_suffix].flatten()
            bias = bias * 1.0
            if len(bias_frac) == 1:
                cnn_config.shift_channelwise = 0
                if input_frac + weight_frac[0] - middle_frac < 0 or int(input_frac + weight_frac[0] - middle_frac) > 31:
                    raise Exception("input_fraction - output_fraction of bias layer should be in [0, 31], check layer {}".format(layer.name))
                cnn_config.output_shift_num = int(input_frac + weight_frac[0] - middle_frac)
                bias_shift = int(input_frac + weight_frac[0] - bias_frac[0])
                if bias_shift < 0:
                    bias = bias * math.pow(2, bias_shift)
                    bias = numpy.floor(bias)
                    cnn_config.bias_shift = 0
                else:
                    if bias_shift > 31:
                        raise Exception("input_fraction - bias_fraction of bias layer should be in [0, 31], check layer {}".format(layer.name))
                    cnn_config.bias_shift = bias_shift
                writing_method = 'w'
                for group_index in range(cnn_config.groups):
                    with open(weight_file_name, writing_method) as f:
                        current_weight_size += write_weights_blob(weight[group_index * cnn_config.output_channel_per_group : (group_index + 1) * cnn_config.output_channel_per_group, :, :, :], bias[group_index * cnn_config.output_channel_per_group : (group_index + 1) * cnn_config.output_channel_per_group], f, elem_bytes, 1, group_index * cnn_config.output_channel_per_group)
                    writing_method = 'a'
            else:
                weight_frac = weight_frac.tile(layer.num_output)
                cnn_config.shift_channelwise = 1
                out_shift_num = input_frac + weight_frac - middle_frac
                bias_shift_num = input_frac + weight_frac - bias_frac
                for i in range(len(bias_shift_num)):
                    if bias_shift_num[i] < 0:
                        bias[i] = bias[i] * math.pow(2, bias_shift_num[i])
                        bias_shift_num[i] = 0
                    if bias_shift_num[i] > 15 and bits_width == 8:
                        raise Exception("input_fraction - bias_fraction of bias layer should be in [0, 15] in 8bit mode when weight fraction or bias fraction differ on output channel, check layer {}".format(layer.name))
                    if bias_shift_num[i] > 31 and bits_width == 16:
                        raise Exception("input_fraction - bias_fraction of bias layer should be in [0, 31] in 16bit mode when weight fraction or bias fraction differ on output channel, check layer {}".format(layer.name))
                    if (out_shift_num[i] < 0 or out_shift_num[i] > 15) and bits_width == 8:
                        raise Exception("input_fraction - output_fraction of bias layer should be in [0, 15] in 8bit mode when weight fraction or bias fraction differ on output channel, check layer {}".format(layer.name))
                    if (out_shift_num[i] < 0 or out_shift_num[i] > 31) and bits_width == 16:
                        raise Exception("input_fraction - output_fraction of bias layer should be in [0, 31] in 16bit mode when weight fraction or bias fraction differ on output channel, check layer {}".format(layer.name))
                bias = numpy.floor(bias)
                shift_num = out_shift_num * 2 ** (elem_bytes * 4) + bias_shift_num
                writing_method = 'w'
                for group_index in range(cnn_config.groups):
                    with open(weight_file_name, writing_method) as f:
                        current_weight_size += write_weights_blob(weight[group_index * cnn_config.output_channel_per_group : (group_index + 1) * cnn_config.output_channel_per_group, :, :, :], bias[group_index * cnn_config.output_channel_per_group : (group_index + 1) * cnn_config.output_channel_per_group], f, elem_bytes, 1, group_index * cnn_config.output_channel_per_group, shift_num[group_index * cnn_config.output_channel_per_group : (group_index + 1) * cnn_config.output_channel_per_group])
                    writing_method = 'a'
            cnn_config.weight_signed = 1 
        else:
            can_use_cnn = False

        cnn_config.memory_map = 9
        cnn_config.input_iram = 0
        cnn_config.output_iram = 0
        if (cnn_config.input_width + layer.pad_l) * elem_bytes > 16 or tensor_size[layer.top[0]] >= 262144:
            cnn_config.memory_map = 4
        nextLayer = None
        if layer.top[0] in dict_to_layer_bottom.keys():
            nextLayer = dict_to_layer_bottom[layer.top[0]][0]
            if 'innerproduct' in nextLayer.type:
                nextLayer.kernel_size_h = tensor_shape[nextLayer.bottom[0]][2]
                nextLayer.kernel_size_w = tensor_shape[nextLayer.bottom[0]][3]

        cnn_config.fi_dma_internal_mode = 0
        cnn_config.fo_dma_internel_mode = 0

        if tensor_adr[layer.bottom[0]] == -1:
            prevLayer = dict_to_layer_top[layer.bottom[0]][0]
            prev_group = tensor_shape[layer.bottom[0]][1] if 'tf_conv' not in prevLayer.type and 'convolution' not in prevLayer.type and 'innerproduct' not in prevLayer.type and 'eltwise' not in prevLayer.type else prevLayer.group
            this_group = tensor_shape[layer.bottom[0]][1] if 'tf_conv' not in layer.type and 'convolution' not in layer.type and 'innerproduct' not in layer.type and 'eltwise' not in layer.type else layer.group
            if layer.stride > 1 or (layer.group == 1 and layer.kernel_size_h * layer.kernel_size_w > 1) or prevLayer.output_internal_mode != 0 or no_opti:
                cnn_config.fi_dma_internal_mode = 1
            elif prev_group != 1 and this_group != 1:
                cnn_config.fi_dma_internal_mode = 1
            elif prev_group == 1 and this_group == 1:
                cnn_config.fi_dma_internal_mode = 1
            else:
                cnn_config.cnn_fi_optimized_mode = 1
                cnn_config.input_iram =1
                if cnn_config.depthwise == 1 or 'pooling' in layer.type or 'tf_pool' in layer.type:
                    cnn_config.memory_map = 5
                elif layer.kernel_size_h * layer.kernel_size_w == 1:
                    cnn_config.memory_map = 6
        else:
            cnn_config.cnn_fi_optimized_mode = 0 

        if tensor_adr[layer.top[0]] == -1:
            next_group = tensor_shape[layer.bottom[0]][1] if 'tf_conv' not in nextLayer.type and 'convolution' not in nextLayer.type and 'innerproduct' not in nextLayer.type and 'eltwise' not in nextLayer.type else nextLayer.group
            this_group = tensor_shape[layer.bottom[0]][1] if 'tf_conv' not in layer.type and 'convolution' not in layer.type and 'innerproduct' not in layer.type and 'eltwise' not in layer.type else layer.group
            if nextLayer.stride > 1 or cnn_config.output_width + nextLayer.pad_l + max(nextLayer.pad_r, 0) > (16 if cnn_config.mac_type_8bits != 0 else 8) or (layer.group == 1 and layer.kernel_size_h * layer.kernel_size_w > 1) or cnn_config.fi_dma_internal_mode != 0 or no_opti:
                cnn_config.fo_dma_internel_mode = 1
            elif cnn_config.groups > 1 and nextLayer.pad_l + max(nextLayer.pad_r, 0) + nextLayer.pad_u > 0:
                cnn_config.fo_dma_internel_mode = 1
            elif next_group != 1 and this_group != 1:
                cnn_config.fo_dma_internel_mode = 1
            elif next_group == 1 and this_group == 1:
                cnn_config.fo_dma_internel_mode = 1
            elif layer.kernel_size_h * layer.kernel_size_w > 1 and nextLayer.kernel_size_h * nextLayer.kernel_size_w > 1:
                cnn_config.fo_dma_internel_mode = 1
            elif nextLayer.group == 1 and nextLayer.kernel_size_h * nextLayer.kernel_size_w > 1:
                cnn_config.fo_dma_internel_mode = 1
            elif tensor_shape[layer.bottom[0]][3] + layer.pad_l > (16 if cnn_config.mac_type_8bits != 0 else 8):
                cnn_config.fo_dma_internel_mode = 1
            else:
                cnn_config.cnn_next_fi_pad_left = nextLayer.pad_l
                cnn_config.cnn_next_fi_pad_up = nextLayer.pad_u
                cnn_config.cnn_next_fi_pad_type = 1 if ('pooling' in nextLayer.type or 'tf_pool' in nextLayer.type) and (not nextLayer.pool) else 0
                cnn_config.cnn_fo_optimized_mode = 1
                cnn_config.output_iram = 1
                if cnn_config.depthwise == 1 or 'pooling' in layer.type or 'tf_pool' in layer.type:
                    cnn_config.memory_map = 5
                elif layer.kernel_size_h * layer.kernel_size_w == 1:
                    cnn_config.memory_map = 6
            layer.output_internal_mode = cnn_config.fo_dma_internel_mode
        else:
            cnn_config.cnn_fo_optimized_mode = 0        

        if 'scalebytensor' not in layer.type:
            cnn_config.weight_addr = current_weight_addr + weight_addr_offset

        if layer.type[-1] in nonlinearity_algo.keys():
            cnn_config.nonlinearity_algo = nonlinearity_algo[layer.type[-1]]
            if cnn_config.nonlinearity_algo == 2:
                if neg_slope != 0:
                    cnn_config.nonlinearity_algo = 7
                    cnn_config.prelu_0 = neg_slope
                    cnn_config.prelu_1 = neg_slope
                    cnn_config.prelu_2 = neg_slope
                    cnn_config.prelu_3 = neg_slope
                    cnn_config.prelu_4 = neg_slope
                    cnn_config.prelu_5 = neg_slope
                    cnn_config.prelu_6 = neg_slope
                    cnn_config.prelu_7 = neg_slope
                    cnn_config.prelu_8 = neg_slope
                    cnn_config.prelu_9 = neg_slope
                    cnn_config.prelu_a = neg_slope
                    cnn_config.prelu_b = neg_slope
                    cnn_config.prelu_c = neg_slope
                    cnn_config.prelu_d = neg_slope
                    cnn_config.prelu_e = neg_slope
                    cnn_config.prelu_f = neg_slope
        else: 
            cnn_config.nonlinearity_algo = 0

        if index == len(layers_merge)-1:
            next_config_addr = 0xffffffff 
        else:
            next_config_addr = (current_config_addr + CNN_BYTES_OF_CONFIG_ONE_LAYER * cnn_config.slices)
        with open(layer_to_ccode_file_name, 'a') as f:
            f.write('    // layer type: {}\n'.format(layer.type))
            f.write('    // this layer is completed by using CNN engine from Round {} to Round {}\n'.format(slices + 1, slices + cnn_config.slices))
            f.write('    // input tensor names are {}\n'.format(layer.bottom))
            f.write('    // output tensor names are {}\n'.format(layer.top))
            f.write('    // input tensor occupy {} bytes individually\n'.format([tensor_size[bottom] for bottom in layer.bottom]))
            f.write('    // output tensor occupy {} bytes individually\n'.format([tensor_size[top] for top in layer.top]))
            f.write('    // weight occupies {} bytes\n'.format(current_weight_size))
            f.write('    // you can also use the code below to complete this layer\n')
            if 'convolution' in layer.type or 'tf_conv' in layer.type or 'batchnorm' in layer.type or 'scale' in layer.type or 'innerproduct' or 'scalebytensor' in layer.type:
                leaky_param = 0
                non_linear_name = 'none'
                if 'sigmoid' in layer.type:
                    non_linear_name = 'sigmoid'
                if 'tanh' in layer.type:
                    non_linear_name = 'tanh'
                if 'relu' in layer.type:
                    if neg_slope == 0:
                        non_linear_name = 'relu'
                    else:
                        non_linear_name = 'leaky_relu'
                        leaky_param = neg_slope
                if 'relu6' in layer.type:
                    non_linear_name = 'relu6'
                if 'prelu' in layer.type:
                    if cnn_config.groups == 1 and cnn_config.output_channel_per_group <= 16:
                        non_linear_name = 'prelu'
                f.write('    conv2d_bean.in_channel = {}; // how many channels of input\n'.format(cnn_config.input_channel_per_group * cnn_config.groups))
                f.write('    conv2d_bean.in_height = {}; // height of input\n'.format(cnn_config.input_height))
                f.write('    conv2d_bean.in_width = {}; // output of input\n'.format(cnn_config.input_width))
                f.write('    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes\n')
                f.write('    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes\n')
                f.write('    conv2d_bean.out_channel = {}; // how many channels of output, if you want to run a depthwise conv, you do not need to set it\n'.format(cnn_config.output_channel_per_group * cnn_config.groups))
                f.write('    conv2d_bean.group = {}; // how many groups, if you want to run a depthwise conv, you do not need to set it\n'.format(cnn_config.groups))
                f.write('    conv2d_bean.kernel_size_h = {}; // size of the convolving kernel\n'.format(cnn_config.filter_size_h))
                f.write('    conv2d_bean.kernel_size_w = {}; // size of the convolving kernel\n'.format(cnn_config.filter_size_w))
                f.write('    conv2d_bean.stride = {};\n'.format(cnn_config.stride))
                f.write('    conv2d_bean.dilation = {};\n'.format(cnn_config.dilation))
                f.write('    conv2d_bean.bias_en = {}; // whether add bias when calculating conv\n'.format(cnn_config.bias_en))
                f.write('    conv2d_bean.softmax = {}; // whether calculate softmax after calculating conv and activation function\n'.format(cnn_config.softmax_enable))
                f.write('    conv2d_bean.mac_8bit = {}; // non_zero: 8bit mode; zero: 16bit mode\n'.format(cnn_config.mac_type_8bits))
                f.write('    conv2d_bean.pad_u = {}; // zero-pad added to up    sides of the input\n'.format(cnn_config.padding_up))
                f.write('    conv2d_bean.pad_d = {}; // zero-pad added to down  sides of the input\n'.format(1 if 'batchnorm' in layer.type or 'scale' in layer.type else max(layer.pad_d, 0)))
                f.write('    conv2d_bean.pad_l = {}; // zero-pad added to left  sides of the input\n'.format(cnn_config.padding_left))
                f.write('    conv2d_bean.pad_r = {}; // zero-pad added to right sides of the input\n'.format(cnn_config.padding_right))
                f.write('    conv2d_bean.input_signed = {}; // whether input is signed\n'.format(cnn_config.feature_signed))
                f.write('    conv2d_bean.weight_bias_signed = {}; // whether weight and bias are signed\n'.format(cnn_config.weight_signed))
                f.write('    conv2d_bean.filter_lsb_channelwise = {}; // whether filter lsb differ from channels\n'.format(cnn_config.shift_channelwise))
                f.write('    conv2d_bean.acc_out_shift = {}; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0\n'.format(cnn_config.output_shift_num))
                f.write('    conv2d_bean.bias_shift = {}; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0\n'.format(cnn_config.bias_shift))
                if non_linear_name == 'leaky_relu':
                    f.write('    conv2d_bean.leaky_param = {}; // the multiplier of leaky relu, the LSB is 2^(-6)\n'.format(leaky_param))
                else:
                    f.write('    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)\n')
                f.write('    conv2d_bean.input_iram = {}; // nonzero - read input from iram, 0 - read input from ddr\n'.format(cnn_config.fi_dma_internal_mode))
                f.write('    conv2d_bean.output_iram = {}; // nonzero - put output into iram, 0 - put output into ddr\n'.format(cnn_config.fo_dma_internel_mode))
                f.write('    conv2d_bean.in_sep_mode = {}; // whether read input from iram as separable conv mode\n'.format(cnn_config.cnn_fi_optimized_mode))
                f.write('    conv2d_bean.out_sep_mode = {}; // whether put output into iram as separable conv mode\n'.format(cnn_config.cnn_fo_optimized_mode))
                f.write('    conv2d_bean.next_padding_left = {}; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0\n'.format(cnn_config.cnn_next_fi_pad_left))
                f.write('    conv2d_bean.next_padding_up = {}; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0\n'.format(cnn_config.cnn_next_fi_pad_up))
                f.write('    conv2d_bean.next_padding_type = {}; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0\n'.format(cnn_config.cnn_next_fi_pad_type))
                f.write('    conv2d_bean.nonlinearty = {}; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"\n'.format(non_linear_name))
                if tensor_adr[layer.bottom[0]] >= 0:
                    f.write('    cnn_conv2D(0x{:x} + feature_addr_offset, '.format(tensor_adr[layer.bottom[0]]))
                else:
                    f.write('    cnn_conv2D(0, ')
                if tensor_adr[layer.top[0]] >= 0:
                    f.write('0x{:x} + feature_addr_offset, '.format(tensor_adr[layer.top[0]]))
                else:
                    f.write('0, ')
                if 'scalebytensor' in layer.type:
                    f.write('0x{:x} + feature_addr_offset, &conv2d_bean);\n'.format(tensor_adr[layer.bottom[1]]))
                else:
                    f.write('0x{:x} + weight_addr_offset, &conv2d_bean);\n'.format(current_weight_addr))
            elif 'pooling' in layer.type:
                f.write('    pool2d_bean.in_channel = {}; // how many channels of input\n'.format(cnn_config.input_channel_per_group))
                f.write('    pool2d_bean.in_height = {}; // height of input\n'.format(cnn_config.input_height))
                f.write('    pool2d_bean.in_width = {}; // output of input\n'.format(cnn_config.input_width))
                f.write('    pool2d_bean.out_height = {}; // do not need to be setted, this is used to save the height of output when the calculation finishes\n'.format(0))
                f.write('    pool2d_bean.out_width = {}; // do not need to be setted, this is used to save the height of output when the calculation finishes\n'.format(0))
                f.write('    pool2d_bean.kernel_size_h = {}; // size of the convolving kernel\n'.format(cnn_config.filter_size_h))
                f.write('    pool2d_bean.kernel_size_w = {}; // size of the convolving kernel\n'.format(cnn_config.filter_size_w))
                f.write('    pool2d_bean.stride = {};\n'.format(cnn_config.stride))
                f.write('    pool2d_bean.pad_u = {}; // zero-pad added to up    sides of the input\n'.format(layer.pad_u))
                f.write('    pool2d_bean.pad_d = {}; // zero-pad added to down  sides of the input\n'.format(max(layer.pad_d, 0)))
                f.write('    pool2d_bean.pad_l = {}; // zero-pad added to left  sides of the input\n'.format(layer.pad_l))
                f.write('    pool2d_bean.pad_r = {}; // zero-pad added to right sides of the input\n'.format(max(layer.pad_r, 0)))
                f.write('    pool2d_bean.input_signed = {}; // whether input is signed\n'.format(cnn_config.feature_signed))
                f.write('    pool2d_bean.mac_8bit = {}; // non_zero: 8bit mode; zero: 16bit mode\n'.format(cnn_config.mac_type_8bits))
                f.write('    pool2d_bean.count_include_pad = 0; // when non-0, include the zero-pad in the averaging calculation\n')
                if 'tf_pool' in layer.type:
                    f.write('    pool2d_bean.ceil_mode = 0; // when non-0, use ceil instead of floor to compute the output shape.\n')
                else:
                    f.write('    pool2d_bean.ceil_mode = 1; // when non-0, use ceil instead of floor to compute the output shape.\n'.format(layer.ceil))
                f.write('    pool2d_bean.avg = {}; // non-zero: avg pooling, zero: max pooling\n'.format(cnn_config.pooling_algorithm))
                f.write('    pool2d_bean.out_shift = {}; // the fraction of output minus the fraction of input, you can set this to a positive number for a higher precision, only valid in avgpooling\n'.format(int(output_frac - input_frac)))
                f.write('    pool2d_bean.input_iram = {}; // nonzero - read input from iram, 0 - read input from ddr\n'.format(cnn_config.fi_dma_internal_mode))
                f.write('    pool2d_bean.output_iram = {}; // nonzero - put output into iram, 0 - put output into ddr\n'.format(cnn_config.fo_dma_internel_mode))
                f.write('    pool2d_bean.in_sep_mode = {}; // whether read input from iram as separable conv mode\n'.format(cnn_config.cnn_fi_optimized_mode))
                f.write('    pool2d_bean.out_sep_mode = {}; // whether put output into iram as separable conv mode\n'.format(cnn_config.cnn_fo_optimized_mode))
                if tensor_adr[layer.bottom[0]] >= 0:
                    f.write('    cnn_pool2D(0x{:x} + feature_addr_offset, '.format(tensor_adr[layer.bottom[0]]))
                else:
                    f.write('    cnn_pool2D(0, ')
                if tensor_adr[layer.top[0]] >= 0:
                    f.write('0x{:x} + feature_addr_offset, '.format(tensor_adr[layer.top[0]]))
                else:
                    f.write('0, ')
                f.write('&pool2d_bean);\n'.format(cnn_config.input_addr, cnn_config.output_addr))
            elif 'eltwise' in layer.type:
                if layer.operation == "SUM":
                    if len(layer.bottom) == 2:
                        f.write('    uint16_t dim_{}[3];\n'.format(index))
                        f.write('    dim_{}[0] = {};\n'.format(index, tensor_shape[layer.top[0]][1]))
                        f.write('    dim_{}[1] = {};\n'.format(index, tensor_shape[layer.top[0]][2]))
                        f.write('    dim_{}[2] = {};\n'.format(index, tensor_shape[layer.top[0]][3]))
                        f.write('    cnn_tensor_add(0x{:x} + feature_addr_offset, 0x{:x} + feature_addr_offset, 0x{:x} + feature_addr_offset, 3, dim_{}, 1, {}, {}, 1, 1, 1);\n'.format(tensor_adr[layer.bottom[0]], tensor_adr[layer.bottom[1]], tensor_adr[layer.top[0]], index, cnn_config.mac_type_8bits, cnn_config.output_shift_num))
                    else:
                        f.write('    uint16_t dim_{}[3];\n'.format(index))
                        f.write('    dim_{}[0] = {};\n'.format(index, tensor_shape[layer.top[0]][1]))
                        f.write('    dim_{}[1] = {};\n'.format(index, tensor_shape[layer.top[0]][2]))
                        f.write('    dim_{}[2] = {};\n'.format(index, tensor_shape[layer.top[0]][3]))
                        f.write('    uint32_t *input_addr_{} = (uint32_t *)os_mem_malloc(0, {} * 4);\n'.format(index, len(layer.bottom)))
                        for bottom_index, bottom_name in layer.bottom:
                            f.write('    input_addr_{}[{}] = 0x{:x} + feature_addr_offset;\n'.format(index, bottom_index, tensor_adr[bottom_name]))
                        if cnn_config.mac_type_8bits != 0:
                            f.write('    cnn_tensor_add_n_8bit((int8_t **)input_addr_{}, 3, dim_{}, {}, {}, (int8_t *)(0x{:x} + feature_addr_offset), 1, 1);\n'.format(index, index, len(layer.bottom), cnn_config.output_shift_num, tensor_adr[layer.top[0]]))
                        else:
                            f.write('    cnn_tensor_add_n_16bit((int16_t **)input_addr_{}, 3, dim_{}, {}, {}, (int16_t *)(0x{:x} + feature_addr_offset), 1, 1);\n'.format(index, index, len(layer.bottom), cnn_config.output_shift_num, tensor_adr[layer.top[0]]))
                        f.write('    os_mem_free(input_addr_{});\n'.index)
                else:
                    f.write('    // please write you code to complete this layer below\n')
            elif 'shuffle' in layer.type or 'shufflechannel' in layer.type:
                if not verify_frac(layer, quant_data, feature_frac_suffix):
                    print("warn: you need to set fractions of all the inputs and outputs to the same value, in layer {}".format(layer.name))
                    for top in layer.top:
                        quant_data[top + feature_frac_suffix] = input_frac
                f.write('    cnn_channel_shuffle(0x{:x} + feature_addr_offset, {}, {}, {}, {}, 0x{:x} + feature_addr_offset, 1, 1, {});\n'.format(tensor_adr[layer.bottom[0]], tensor_shape[layer.top[0]][1], tensor_shape[layer.top[0]][2], tensor_shape[layer.top[0]][3], layer.group, tensor_adr[layer.top[0]], cnn_config.mac_type_8bits))
            elif 'bias' in layer.type:
                non_linear_name = 'none'
                if 'relu' in layer.type:
                    non_linear_name = 'relu'
                if 'relu6' in layer.type:
                    non_linear_name = 'relu6'
                if tensor_adr[layer.bottom[0]] >= 0:
                    f.write('    cnn_bias(0x{:x} + feature_addr_offset, '.format(tensor_adr[layer.bottom[0]]))
                else:
                    f.write('0, ')
                if tensor_adr[layer.top[0]] >= 0:
                    f.write('0x{:x} + feature_addr_offset, '.format(tensor_adr[layer.top[0]]))
                else:
                    f.write('0, ')
                f.write('0x{:x} + weight_addr_offset, {}, {}, {}, {}, {}, {}, {}, {});\n'.format(current_weight_addr, tensor_shape[layer.top[0]][1], tensor_shape[layer.top[0]][2], tensor_shape[layer.top[0]][3], cnn_config.mac_type_8bits, cnn_config.feature_signed, cnn_config.weight_signed, cnn_config.output_shift_num, non_linear_name))
            else:
                print("layer {} is of type {}, can't be completed by our API".format(index + 1, layer.type))
                f.write('    please write you code to complete this layer below and delete this line\n')
                f.write('    /*\n')
                f.write('     * this layer end\n')
                f.write('     */\n')
                continue

        if can_use_cnn:
            if 'shuffle' in layer.type or 'shufflechannel' in layer.type:
                if index == len(layers_merge) - 1:
                    next_config_addr = (current_config_addr + CNN_BYTES_OF_CONFIG_ONE_LAYER * cnn_config.slices)
                    for index_group in range(layer.group):
                        if index_group == layer.group - 1:
                            next_config_addr = 0xffffffff
                        machine_code_config_tmp = (c_uint32 * 640) ()
                        libcnn.cnn_one_layer_slices_to_config_array(byref(machine_code_config_tmp), next_config_addr, byref(cnn_config))
                        machine_code_config += machine_code_config_tmp[0: 64* cnn_config.slices]
                        current_config_addr += CNN_BYTES_OF_CONFIG_ONE_LAYER * cnn_config.slices
                        next_config_addr += CNN_BYTES_OF_CONFIG_ONE_LAYER * cnn_config.slices
                        cnn_config.input_addr += sizeof_channel * cnn_config.groups
                        cnn_config.output_addr += sizeof_channel
                        with open(cnn_task_to_layer_name, 'a') as f:
                            f.write("\n{}={}".format(slices,index))
                        slices += 1
                else:
                    for index_group in range(layer.group):
                        machine_code_config_tmp = (c_uint32 * 640) ()
                        libcnn.cnn_one_layer_slices_to_config_array(byref(machine_code_config_tmp), next_config_addr, byref(cnn_config))
                        machine_code_config += machine_code_config_tmp[0: 64* cnn_config.slices]
                        current_config_addr += CNN_BYTES_OF_CONFIG_ONE_LAYER * cnn_config.slices
                        next_config_addr += CNN_BYTES_OF_CONFIG_ONE_LAYER * cnn_config.slices
                        cnn_config.input_addr += sizeof_channel * cnn_config.groups
                        cnn_config.output_addr += sizeof_channel
                        with open(cnn_task_to_layer_name, 'a') as f:
                            f.write("\n{}={}".format(slices,index))
                        slices += 1
            else:
                if cnn_config.slices > 1:
                    print("input large")
                len_machine_code_config_tmp = int(cnn_config.slices) * 64
                machine_code_config_tmp = (c_uint32 * len_machine_code_config_tmp) ()
                libcnn.cnn_one_layer_slices_to_config_array(byref(machine_code_config_tmp), next_config_addr, byref(cnn_config))
                for slice_index in range(cnn_config.slices - 1):
                    machine_code_config_tmp[64 * slice_index] = current_config_addr + CNN_BYTES_OF_CONFIG_ONE_LAYER * (slice_index + 1)
                machine_code_config_tmp[64 * (cnn_config.slices - 1)] = next_config_addr
                machine_code_config += machine_code_config_tmp[0: 64 * cnn_config.slices]
                current_config_addr += CNN_BYTES_OF_CONFIG_ONE_LAYER * cnn_config.slices
                for slice_index in range(cnn_config.slices):
                    with open(cnn_task_to_layer_name, 'a') as f:
                        f.write("\n{}={}".format(slices + slice_index, index))
                slices = slices + cnn_config.slices
            before_cnn_run = False
        else:
            print("[WARN] unsupport layer occurs, you need to split your net if you want to use netpack")
            if before_cnn_run:
                break_cnn_info.append("call your own function to complete layer {} before CNN engine runs".format(layer.name))
            else:
                break_cnn_info.append("call your own function to complete layer {} after CNN engine runs {} time(s)".format(layer.name, current_cnn_engine_layer))
        
        current_weight_addr += current_weight_size
        if current_weight_addr % 256 != 0:
            current_weight_addr += (256 - current_weight_addr % 256)

        for top in layer.top:
            tensor_signed[top] = True

    with open(layer_to_ccode_file_name, 'a') as f:
        f.write('    /*\n')
        f.write('     * this layer end\n')
        f.write('     */\n')
        f.write("    // total_weight_size: 0x{:x}\n".format(current_weight_addr))
        f.write('}\n')
    with open(layer_info_file_name, 'a') as f:
        f.write(']')

    with open(for_debug_file_name, 'a') as f:
        f.write('\n};\n')
        f.write('uint32_t input_size = {};\n'.format(tensor_size[input_name]))
        f.write('uint32_t total_weight_size = {};\n'.format(current_weight_addr))
        f.write('uint32_t features_total_size = {};\n'.format(total_size_used))

    with open(for_debug_ini_file_name, 'a') as f:
        f.write('cnn_one_layer_count = {}\n'.format(cnn_one_layer_count))
        f.write('input_size = {}\n'.format(tensor_size[input_name]))
        f.write('total_weight_size = {}\n'.format(current_weight_addr))
        f.write('features_total_size = {}\n'.format(total_size_used))

    if machine_code_config[-64] != 0xffffffff:
        print("info: this net ends with an operation without multiplying or add")
        machine_code_config[-64] = 0xffffffff
    with open(machine_code_file_name,'w') as f:
        for machine_code in machine_code_config:
            f.write("{:08x}\n".format(machine_code))
    if len(break_cnn_info) > 0:           
        with open(break_info_file_name, 'w') as f:
            f.write('\n'.join(break_cnn_info))

    with open(tensor_addr_file_name, 'w') as f:
        f.write("{:<60}".format('tensor name'))
        f.write("{:<20}".format('tensor addr'))
        f.write("{:<30}".format('memory occupied (byte)'))
        for key, value in tensor_adr.items():
            f.write('\n')
            f.write("{:<60}".format(key))
            if value < 0:
                f.write("{:<20}".format("in iram"))
            else:
                f.write("0x{:<18x}".format(value + feature_addr_offset))
            f.write("{:<30}".format(tensor_size[key]))

    with open(tensor_info_file_name, 'w') as f:
        f.write("{:<60}".format('tensor name'))
        f.write("{:<20}".format('tensor addr'))
        f.write("{:<30}".format('memory occupied (byte)'))
        f.write("{:<20}".format('channel'))
        f.write("{:<20}".format('height'))
        f.write("{:<20}".format('width'))
        f.write("{:<5}".format('frac'))
        for key, value in tensor_adr.items():
            f.write('\n')
            f.write("{:<60}".format(key))
            if value < 0:
                f.write("{:<20}".format("in iram"))
            else:
                f.write("0x{:<18x}".format(value + feature_addr_offset))
            f.write("{:<30}".format(tensor_size[key]))
            f.write("{:<20}".format(tensor_shape[key][1]))
            f.write("{:<20}".format(tensor_shape[key][2]))
            f.write("{:<20}".format(tensor_shape[key][3]))
            f.write("{:<5}".format(quant_data[key + feature_frac_suffix]))

    for current_path, subfolders, filenames in os.walk(weight_path):
        for filename in filenames:
            file_lines = 0
            with open(os.path.join(weight_path, filename), 'r') as f:
                file_lines = len(f.read().rstrip().split('\n'))
            if file_lines % 256 != 0:
                with open(os.path.join(weight_path, filename), 'a') as f:
                    while file_lines % 256 != 0:
                        f.write('\n00')
                        file_lines += 1

    with open(for_flash_path + 'w.bin', 'wb') as f:
        print("generate weight bin for flash...\n")

    for current_path, subfolders, filenames in os.walk(weight_path):
        filenames.sort()
        for filename in filenames:
            contents = []
            with open(os.path.join(current_path, filename), 'r') as f:
                contents = f.read().rstrip().split('\n')
            with open(for_flash_path + 'w.bin', 'ab') as f:
                for content in contents:
                    num = int(content[0:2], 16)
                    str_num = struct.pack("B", num)
                    f.write(str_num)
    with open(for_flash_path + 'code.bin', 'wb') as f:
         for machine_code in machine_code_config:
            to_write_machine_code = "{:08x}\n".format(machine_code)
            num = int(to_write_machine_code[6:8], 16)
            str_num = struct.pack("B", num)
            f.write(str_num)
            num = int(to_write_machine_code[4:6], 16)
            str_num = struct.pack("B", num)
            f.write(str_num)
            num = int(to_write_machine_code[2:4], 16)
            str_num = struct.pack("B", num)
            f.write(str_num)
            num = int(to_write_machine_code[0:2], 16)
            str_num = struct.pack("B", num)
            f.write(str_num)

    model_size_configs = [0] * 64
    total_model_size = 256 + total_size_used + current_weight_addr + CNN_BYTES_OF_CONFIG_ONE_LAYER * slices
    model_size_configs[0] = total_model_size
    model_size_configs[1] = net_out_addr
    model_size_configs[2] = total_size_used
    model_size_configs[3] = current_weight_addr
    model_size_configs[4] = 256 + total_size_used + current_weight_addr
    model_size_configs[5] = CNN_BYTES_OF_CONFIG_ONE_LAYER * slices
    model_size_configs[6] = input_channel
    model_size_configs[7] = input_height
    model_size_configs[8] = input_width
    model_size_configs[9] = output_channel
    model_size_configs[10] = output_height
    model_size_configs[11] = output_width
    model_size_configs[12] = elem_bytes
    model_size_configs[13] = int(net_out_frac)
    with open(out_file_path + '/' + model_name + '.bin', 'wb') as f:
         for model_size_config in model_size_configs:
            to_write = "{:08x}\n".format(model_size_config)
            num = int(to_write[6:8], 16)
            str_num = struct.pack("B", num)
            f.write(str_num)
            num = int(to_write[4:6], 16)
            str_num = struct.pack("B", num)
            f.write(str_num)
            num = int(to_write[2:4], 16)
            str_num = struct.pack("B", num)
            f.write(str_num)
            num = int(to_write[0:2], 16)
            str_num = struct.pack("B", num)
            f.write(str_num)

    for current_path, subfolders, filenames in os.walk(weight_path):
        filenames.sort()
        for filename in filenames:
            contents = []
            with open(os.path.join(current_path, filename), 'r') as f:
                contents = f.read().rstrip().split('\n')
            with open(out_file_path + '/' + model_name + '.bin', 'ab') as f:
                for content in contents:
                    num = int(content[0:2], 16)
                    str_num = struct.pack("B", num)
                    f.write(str_num)
    with open(out_file_path + '/' + model_name + '.bin', 'ab') as f:
         for machine_code in machine_code_config:
            to_write_machine_code = "{:08x}\n".format(machine_code)
            num = int(to_write_machine_code[6:8], 16)
            str_num = struct.pack("B", num)
            f.write(str_num)
            num = int(to_write_machine_code[4:6], 16)
            str_num = struct.pack("B", num)
            f.write(str_num)
            num = int(to_write_machine_code[2:4], 16)
            str_num = struct.pack("B", num)
            f.write(str_num)
            num = int(to_write_machine_code[0:2], 16)
            str_num = struct.pack("B", num)
            f.write(str_num)

    print("outputs of all layers occupy {} bytes in total".format(total_size_used))
    print("weights of all layers occupy {} bytes in total".format(current_weight_addr))
    print("machine code of whole net occupy {} bytes in total".format(CNN_BYTES_OF_CONFIG_ONE_LAYER * slices))

    print("model pack completed")

"""
this function is used to pack a quantized model
bitwidth: quantized bits or the net, should be 8 or 16
prototxt_name: the prototxt file name
out_file_path: the path name where the packed net and relevant information are put
data_file: the file saving quantized data, including weight, bias and fraction of all tensors
input_signed: whether your input picture of the net is signed number, only valid if this information is not in data_file
model_name: the name of your model, this is for load your model from the flash in the chip
use_machine_code: set True when you want to use run net with machine code, set False when you want to run net with net_pack
"""
def model_packer(bitwidth = 8, prototxt_name = "WQ_sample.prototxt", out_file_path = "out_sample", data_file = "WQ_sample.npz", input_signed = False, model_name = "WQ_sample", use_machine_code = False):
    assert bitwidth in [8, 16], "bitwidth should be 8 or 16"
    assert type(prototxt_name) == str, "prototxt_name should be a string"
    assert type(out_file_path) == str, "prototxt_name should be a string"
    assert type(data_file) == str, "prototxt_name should be a string"
    assert type(input_signed) == bool, "input_signed should be a bool variable"
    assert type(model_name) == str, "prototxt_name should be a string"
    model_pack(bitwidth // 8, prototxt_name, out_file_path, data_file, weight_suffix = '_quant_weight', bias_suffix = '_quant_bias', weight_frac_suffix = '_frac_weight', bias_frac_suffix = '_frac_bias', feature_frac_suffix = '_frac', reserve_net_input = True, try_interal = True, data_frac = bitwidth - 4, input_signed=input_signed, model_name=model_name, use_machine_code = use_machine_code)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_type',
        type=str,
        default='int8',
        help='The type of features and weights, should be int8 or int16. Default: int8'
    )
    parser.add_argument(
        '--prototxt_name',
        type=str,
        default='mobilenetSSD_025_256half.prototxt',
        help='The name of prototxt file'
    )
    parser.add_argument(
        '--out_file_path',
        type=str,
        default='mobilenetSSD_025_256half',
        help='The path of folder where the output files. Default: out'
    )
    parser.add_argument(
        '--data_numpy_file',
        type=str,
        default='ShuffleNet_V2_inference.npz',
        help='The name of npz file, where weights, bias, and fraction of all the tensors are saved'
    )
    parser.add_argument(
        '--weight_suffix',
        type=str,
        default='_quanti_weights_0',
        help="The suffix of the weight tensors name in npz file, i.e. a weight tensor's name in the npz file is layer_name + weight_suffix. Default: _quanti_weights_0"
    )
    parser.add_argument(
        '--bias_suffix',
        type=str,
        default='_quanti_weights_1',
        help="The suffix of the bias tensors name in npz file, i.e. a bias's name in the npz file is layer_name + bias_suffix. Default: _quanti_weights_1"
    )
    parser.add_argument(
        '--weight_frac_suffix',
        type=str,
        default='_fl_params_0',
        help="The suffix of the fraction of weight tensors name in npz file, i.e. the name of fraction of a weight tensor in the npz file is layer_name + weight_frac_suffix. Default: _fl_params_0"
    )
    parser.add_argument(
        '--bias_frac_suffix',
        type=str,
        default='_fl_params_1',
        help="The suffix of the fraction of bias tensors name in npz file, i.e. the name of fraction of a bias in the npz file is layer_name + bias_frac_suffix. Default: _fl_params_1"
    )
    parser.add_argument(
        '--feature_frac_suffix',
        type=str,
        default='_fl_out',
        help="The suffix of the fraction of feature tensors name in npz file, i.e. the name of fraction of a feature tensor in the npz file is layer_name + feature_frac_suffix. default : _fl_out"
    )
    parser.add_argument(
        '--override_input_image',
        type=str,
        default='false',
        help="Whether the input image in ddr can be override by the output of a layer in the net. Default: false"
    )
    parser.add_argument(
        '--debug_mode',
        type=str,
        default='false',
        help="Whether the converted C code is for debugging, if true, all the outputs of layers in the net will be put into ddr, Default: false"
    )
    parser.add_argument(
        '--frac_input_image',
        type=int,
        default=0,
        help="The fraction of input image. Default: 0"
    )
    parser.add_argument(
        '--input_signed',
        type=str,
        default='false',
        help="Whether the input is signed number. Default: false"
    )
    FLAGS, unparsed = parser.parse_known_args()
    elem_bytes = 1 
    if FLAGS.data_type == 'int8':
        elem_bytes = 1
    elif FLAGS.data_type == 'int16':
        elem_bytes = 2
    else:
        raise Exception("data_type error, should be int8 or int16")
    file_name = FLAGS.prototxt_name
    out_file_path = FLAGS.out_file_path
    data_file = FLAGS.data_numpy_file
    weight_suffix = FLAGS.weight_suffix
    bias_suffix = FLAGS.bias_suffix
    weight_frac_suffix = FLAGS.weight_frac_suffix
    bias_frac_suffix = FLAGS.bias_frac_suffix
    feature_frac_suffix = FLAGS.feature_frac_suffix
    reserve_net_input = True
    if FLAGS.override_input_image.lower() == "true":
        reserve_net_input = False
    elif FLAGS.override_input_image.lower() == "false":
        reserve_net_input = True
    else:
        raise Exception("override_input_image error, should be true or false")
    try_interal = True
    if FLAGS.debug_mode.lower() == "true":
        try_interal = False
    elif FLAGS.debug_mode.lower() == "false":
        try_interal = True
    else:
        raise Exception("debug_mode error, should be true or false")
    data_frac = FLAGS.frac_input_image
    input_signed = False
    if FLAGS.input_signed.lower() == "false":
        input_signed = False
    elif FLAGS.input_signed.lower() == "true":
        input_signed = True
    else:
        raise Exception("input_signed error, should be true or false")
    if elem_bytes > 1:
        input_signed = True
    model_pack(elem_bytes, file_name, out_file_path, data_file, weight_suffix, bias_suffix, weight_frac_suffix, bias_frac_suffix, feature_frac_suffix, reserve_net_input, try_interal, data_frac, input_signed, 'lenet')

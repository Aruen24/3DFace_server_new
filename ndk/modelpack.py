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
import zipfile
import numpy
import ndk.layers as ndk_layers
from ndk.layers import sort_layers
from ndk.layers import no_input_layer
from ndk.layers import one_input_layer
from ndk.layers import multi_input_layer
from ndk.layers import one_output_layer
from ndk.model_pack.model_packer import model_packer

default_npy_dir = 'it_includes_temp_quantized_param'

def save_to_npz(param_dict, fname_npz):
    assert isinstance(param_dict, dict), 'param_dict must be dict type, but get {}'.format(type(param_dict))
    assert isinstance(fname_npz, str), 'fname_npz must be str type, but get {}'.format(type(fname_npz))
    if not fname_npz.endswith('.npz'):
        fname_npz += '.npz'
        print('\033[0;31m Warning: save_to_npz: will save param_dict to file {}\033[0m'.format(fname_npz))
    numpy.savez(fname_npz, **param_dict)
    return fname_npz

def load_from_npz(fname_npz):
    assert isinstance(fname_npz, str), 'fname_npz must be str type, but get {}'.format(type(fname_npz))
    if not fname_npz.endswith('.npz'):
        fname_npz += '.npz'
        print('\033[0;31m Warning: load_from_npz: will load param_dict from file {}\033[0m'.format(fname_npz))
    param_dict = dict(numpy.load(fname_npz))
    for key,val in param_dict.items():
        if len(val.shape)==0:
            if val.dtype==int or val.dtype==numpy.int32 or val.dtype==numpy.int64:
                param_dict[key] = int(val)
            elif val.dtype==bool:
                param_dict[key] = bool(val)
            else:
                param_dict[key] = float(val)
    return param_dict

def zip_npy(npy_dir, fname_npz):
    z = zipfile.ZipFile(fname_npz, 'w', zipfile.ZIP_STORED)
    for current_path, subfolders, filenames in os.walk(npy_dir):
        fpath = current_path.replace(npy_dir, '', 1)
        for filename in filenames:
            z.write(os.path.join(current_path, filename),  os.path.join(fpath, filename))
    z.close()

def save_param_dict(param_dict, folder_name):
    assert type(param_dict)==dict, "param_dict should be type of dict"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    for key, value in param_dict.items():
        paths = key.split('/')
        file_name = folder_name
        for path in paths[:-1]:
            file_name = file_name + '/' + path
            if not os.path.exists(file_name):
                os.mkdir(file_name)
        file_name = file_name + '/' + paths[-1]
        numpy.save(file_name, value)
        
def save_to_prototxt(layer_list, fname):
    assert isinstance(layer_list, list), 'layer_list must be dict type, but get {}'.format(type(layer_list))
    assert isinstance(fname, str), 'fname must be str type, but get {}'.format(type(fname))
    if not fname.endswith('.prototxt'):
        fname += '.prototxt'
        print('\033[0;31m Warning: save_to_prototxt: will save layer_list to file {}\033[0m'.format(fname))

    # assert layer_list[0].type=="Input", "First layer should be input."
    with open(fname, 'w') as f:
        # f.write('input: "{}"\n'.format(layer_list[0].top))
        # f.write('input_dim: {}\n'.format(layer_list[0].dim[0]))
        # f.write('input_dim: {}\n'.format(layer_list[0].dim[1]))
        # f.write('input_dim: {}\n'.format(layer_list[0].dim[2]))
        # f.write('input_dim: {}\n'.format(layer_list[0].dim[3]))
        for index in range(0, len(layer_list)):
            layer = layer_list[index]
            f.write('layer {\n')
            f.write('  name : "{}"\n'.format(layer.name))
            f.write('  type : "{}"\n'.format(layer.type))
            if layer.type in one_input_layer:
                f.write('  bottom: "{}"\n'.format(layer.bottom))
            elif layer.type in multi_input_layer:
                for bottom in layer.bottom:
                    f.write('  bottom: "{}"\n'.format(bottom))
            if layer.type in no_input_layer:
                f.write('  top: "{}"\n'.format(layer.top))
                f.write('  input_param {\n')
                f.write('    shape {\n')
                for dim in layer.dim:
                    f.write('      dim: {}\n'.format(dim))
                f.write('    }\n')
                f.write('  }\n')
            elif layer.type in one_output_layer:
                f.write('  top: "{}"\n'.format(layer.top))
            else:
                for top in layer.top:
                    f.write('  top: "{}"\n'.format(top))
            if layer.type == 'ReLU':
                if abs(layer.negative_slope) > 1.0 / 128.0:
                    f.write('  relu_param {\n')
                    f.write('    negative_slope: {}\n'.format(layer.negative_slope))
                    f.write('  }\n')
            elif layer.type == 'InnerProduct':
                f.write('  inner_product_param {\n')
                f.write('    num_output: {}\n'.format(layer.num_output))
                f.write('    bias_term: {}\n'.format(layer.bias_term))
                f.write('  }\n')
            elif layer.type == 'Convolution':
                f.write('  convolution_param {\n')
                f.write('    num_output: {}\n'.format(layer.num_output))
                f.write('    kernel_size_h: {}\n'.format(layer.kernel_size[0]))
                f.write('    kernel_size_w: {}\n'.format(layer.kernel_size[1]))
                f.write('    stride_h: {}\n'.format(layer.stride[0]))
                f.write('    stride_w: {}\n'.format(layer.stride[1]))
                f.write('    pad_n: {}\n'.format(layer.pad[0]))
                f.write('    pad_s: {}\n'.format(layer.pad[1]))
                f.write('    pad_w: {}\n'.format(layer.pad[2]))
                f.write('    pad_e: {}\n'.format(layer.pad[3]))
                f.write('    bias_term: {}\n'.format(layer.bias_term))
                f.write('    dilation_h: {}\n'.format(layer.dilation[0]))
                f.write('    dilation_w: {}\n'.format(layer.dilation[1]))
                f.write('    group: {}\n'.format(layer.group))
                f.write('  }\n')
            elif layer.type == 'Pooling':
                f.write('  pooling_param {\n')
                f.write('    kernel_size_h: {}\n'.format(layer.kernel_size[0]))
                f.write('    kernel_size_w: {}\n'.format(layer.kernel_size[1]))
                f.write('    stride_h: {}\n'.format(layer.stride[0]))
                f.write('    stride_w: {}\n'.format(layer.stride[1]))
                f.write('    pad_n: {}\n'.format(layer.pad[0]))
                f.write('    pad_s: {}\n'.format(layer.pad[1]))
                f.write('    pad_w: {}\n'.format(layer.pad[2]))
                f.write('    pad_e: {}\n'.format(layer.pad[3]))
                f.write('    dilation_h: {}\n'.format(layer.dilation[0]))
                f.write('    dilation_w: {}\n'.format(layer.dilation[1]))
                f.write('    pool: {}\n'.format(layer.pool.upper()))
                f.write('  }\n')
            elif layer.type == 'Scale':
                f.write('  scale_param {\n')
                f.write('    bias_term: {}\n'.format(layer.bias_term))
                f.write('  }\n')
            elif layer.type == 'Slice':
                f.write('  slice_param {\n')
                f.write('    axis: {}\n'.format(layer.axis))
                if layer.slice_point != None:
                    for slice_point in layer.slice_point:
                        f.write('    slice_point: {}\n'.format(slice_point))
                f.write('  }\n')
            elif layer.type == 'Concat':
                f.write('  concat_param {\n')
                f.write('    axis: {}\n'.format(layer.axis))
                f.write('  }\n')
            elif layer.type == 'Eltwise':
                f.write('  eltwise_param {\n')
                f.write('    operation: {}\n'.format(layer.operation.upper()))
                f.write('  }\n')
            elif layer.type == 'ShuffleChannel':
                f.write('  shuffle_channel_param {\n')
                f.write('    group: {}\n'.format(layer.group))
                f.write('  }\n')

            f.write('}\n')

def load_from_prototxt(fname_prototxt):
    def warning_msg_set_default(layer_name, attr, default_value):
        print('\033[0;31m Warning: load_from_prototxt: Layer {}: {} is set to default value:{}\033[0m'.format(layer_name, attr, default_value))

    if not fname_prototxt.endswith('.prototxt'):
        fname_prototxt += '.prototxt'
        print('\033[0;31m Warning: load_from_prototxt: will load layer_list from file {}\033[0m'.format(fname_prototxt))
        
    layer_list = []
    with open(fname_prototxt) as f:
        list_layer_str = f.read().split('layer {')
        assert len(list_layer_str)>1, 'no layer definition found in {}'.format(fname_prototxt)
        list_layer_str = list_layer_str[1:]

        for layer_str in list_layer_str:
            list_config_str = layer_str.split('\n')
            for idx,config_str in enumerate(list_config_str):
                list_config_str[idx] = config_str.strip().replace(' ', '')
            config_dict = {}
            for config_str in list_config_str:
                if config_str.find(':') and len(config_str.split(':'))>=2:
                    key = config_str.split(':',1)[0].lower()  # keys: lower case
                    val = config_str.split(':',1)[1].split('#',1)[0].strip('"').strip('\'') # vals: could be upper case
                    if key not in config_dict:
                        config_dict[key] = []
                    config_dict[key].append(val)
                else:
                    pass # useless rows
            if 'type' in config_dict:
                type = config_dict['type'][0].lower()
                name = config_dict['name'][0]
                top = config_dict['top']
                if type == 'input':
                    dim = config_dict['dim']
                    for i in range(len(dim)):
                        dim[i] = int(dim[i])
                    dim = tuple(dim)
                    layer_list.append(ndk_layers.Layer(lyr_type=type, name=name, top=top, dim=dim))
                elif type in ['sigmoid', 'relu6', 'tanh', 'batchnorm', 'bias', 'softmax', 'logsoftmax']:
                    bottom = config_dict['bottom']
                    layer_list.append(ndk_layers.Layer(lyr_type=type, name=name, bottom=bottom, top=top))
                elif type == 'scale':
                    bottom = config_dict['bottom']
                    if 'bias_term' in config_dict:
                        if config_dict['bias_term'][0].lower() == 'false':
                            bias_term = False
                        else:
                            bias_term = True
                    else:
                        bias_term = True
                        warning_msg_set_default(name, 'bias_term', bias_term)
                    layer_list.append(ndk_layers.Layer(lyr_type=type, name=name, bottom=bottom, top=top, bias_term=bias_term))
                elif type == 'relu':
                    bottom = config_dict['bottom']
                    if 'negative_slope' in config_dict:
                        negative_slope = float(config_dict['negative_slope'][0])
                    else:
                        negative_slope = 0
                        warning_msg_set_default(name, 'negative_slope', negative_slope)
                    layer_list.append(ndk_layers.Layer(lyr_type=type, name=name, bottom=bottom, top=top, negative_slope=negative_slope))
                elif type == 'innerproduct':
                    bottom = config_dict['bottom']
                    num_output = int(config_dict['num_output'][0])
                    if 'bias_term' in config_dict:
                        if config_dict['bias_term'][0].lower() == 'false':
                            bias_term = False
                        else:
                            bias_term = True
                    else:
                        bias_term = True
                        warning_msg_set_default(name, 'bias_term', bias_term)
                    layer_list.append(ndk_layers.Layer(lyr_type=type, name=name, bottom=bottom, top=top, num_output=num_output, bias_term=bias_term))
                elif type == 'convolution':
                    bottom = config_dict['bottom']
                    num_output = int(config_dict['num_output'][0])
                    if 'kernel_size' in config_dict:
                        kernel_size = int(config_dict['kernel_size'][0])
                        kernel_size_h = None
                        kernel_size_w = None
                    elif 'kernel_h' in config_dict and 'kernel_w' in config_dict:
                        kernel_size = None
                        kernel_size_h = int(config_dict['kernel_h'][0])
                        kernel_size_w = int(config_dict['kernel_w'][0])
                    elif 'kernel_size_h' in config_dict and 'kernel_size_w' in config_dict:
                        kernel_size = None
                        kernel_size_h = int(config_dict['kernel_size_h'][0])
                        kernel_size_w = int(config_dict['kernel_size_w'][0])
                    else:
                        raise Exception('kernel size must be defined in Convolution layer.')

                    if 'stride' in config_dict:
                        stride = int(config_dict['stride'][0])
                        stride_h = None
                        stride_w = None
                    elif 'stride_h' in config_dict and 'stride_w' in config_dict:
                        stride = None
                        stride_h = int(config_dict['stride_h'][0])
                        stride_w = int(config_dict['stride_w'][0])
                    else:
                        stride = 1 # default=1
                        stride_h = None
                        stride_w = None
                        warning_msg_set_default(name, 'stride', stride)

                    if 'pad' in config_dict:
                        pad = int(config_dict['pad'][0])
                        pad_n = None
                        pad_s = None
                        pad_w = None
                        pad_e = None
                    elif 'pad_h' in config_dict and 'pad_w' in config_dict:
                        pad = None
                        pad_n = int(config_dict['pad_h'][0])
                        pad_s = int(config_dict['pad_h'][0])
                        pad_w = int(config_dict['pad_w'][0])
                        pad_e = int(config_dict['pad_w'][0])
                    elif 'pad_n' in config_dict and 'pad_s' in config_dict and 'pad_w' in config_dict and 'pad_e' in config_dict:
                        pad = None
                        pad_n = int(config_dict['pad_n'][0])
                        pad_s = int(config_dict['pad_s'][0])
                        pad_w = int(config_dict['pad_w'][0])
                        pad_e = int(config_dict['pad_e'][0])
                    else:
                        pad = 0 # default=0
                        pad_n = None
                        pad_s = None
                        pad_w = None
                        pad_e = None
                        warning_msg_set_default(name, 'pad', pad)

                    if 'dilation' in config_dict:
                        dilation = int(config_dict['dilation'][0])
                        dilation_h = None
                        dilation_w = None
                    elif 'dilation_h' in config_dict and 'dilation_w' in config_dict:
                        dilation = None
                        dilation_h = int(config_dict['dilation_h'][0])
                        dilation_w = int(config_dict['dilation_w'][0])
                    else:
                        dilation = 1 # default=1
                        dilation_h = None
                        dilation_w = None
                        warning_msg_set_default(name, 'dilation', dilation)

                    if 'bias_term' in config_dict:
                        if config_dict['bias_term'][0].lower() == 'false':
                            bias_term = False
                        else:
                            bias_term = True
                    else:
                        bias_term = True
                        warning_msg_set_default(name, 'bias_term', bias_term)

                    if 'group' in config_dict:
                        group = int(config_dict['group'][0])
                    else:
                        group = 1
                        warning_msg_set_default(name, 'group', group)

                    layer_list.append(
                        ndk_layers.Layer(lyr_type=type, name=name, top=top, bottom=bottom, num_output=num_output,
                                         kernel_size = kernel_size, kernel_size_h = kernel_size_h, kernel_size_w = kernel_size_w,
                                         stride = stride, stride_h = stride_h, stride_w = stride_w,
                                         pad = pad, pad_n = pad_n, pad_s = pad_s, pad_w = pad_w, pad_e = pad_e,
                                         bias_term=bias_term,
                                         dilation = dilation, dilation_h = dilation_h, dilation_w = dilation_w,
                                         group = group)
                    )
                elif type == 'pooling':
                    bottom = config_dict['bottom']
                    pool = config_dict['pool'][0]
                    if 'kernel_size' in config_dict:
                        kernel_size = int(config_dict['kernel_size'][0])
                        kernel_size_h = None
                        kernel_size_w = None
                    elif 'kernel_h' in config_dict and 'kernel_w' in config_dict:
                        kernel_size = None
                        kernel_size_h = int(config_dict['kernel_h'][0])
                        kernel_size_w = int(config_dict['kernel_w'][0])
                    elif 'kernel_size_h' in config_dict and 'kernel_size_w' in config_dict:
                        kernel_size = None
                        kernel_size_h = int(config_dict['kernel_size_h'][0])
                        kernel_size_w = int(config_dict['kernel_size_w'][0])
                    else:
                        raise Exception('kernel size must be defined in Convolution layer.')

                    if 'stride' in config_dict:
                        stride = int(config_dict['stride'][0])
                        stride_h = None
                        stride_w = None
                    elif 'stride_h' in config_dict and 'stride_w' in config_dict:
                        stride = None
                        stride_h = int(config_dict['stride_h'][0])
                        stride_w = int(config_dict['stride_w'][0])
                    else:
                        stride = 1 # default=1
                        stride_h = None
                        stride_w = None
                        warning_msg_set_default(name, 'stride', stride)

                    if 'pad' in config_dict:
                        pad = int(config_dict['pad'][0])
                        pad_n = None
                        pad_s = None
                        pad_w = None
                        pad_e = None
                    elif 'pad_h' in config_dict and 'pad_w' in config_dict:
                        pad = None
                        pad_n = int(config_dict['pad_h'][0])
                        pad_s = int(config_dict['pad_h'][0])
                        pad_w = int(config_dict['pad_w'][0])
                        pad_e = int(config_dict['pad_w'][0])
                    elif 'pad_n' in config_dict and 'pad_s' in config_dict and 'pad_w' in config_dict and 'pad_e' in config_dict:
                        pad = None
                        pad_n = int(config_dict['pad_n'][0])
                        pad_s = int(config_dict['pad_s'][0])
                        pad_w = int(config_dict['pad_w'][0])
                        pad_e = int(config_dict['pad_e'][0])
                    else:
                        pad = 0 # default=0
                        pad_n = None
                        pad_s = None
                        pad_w = None
                        pad_e = None
                        warning_msg_set_default(name, 'pad', pad)

                    if 'dilation' in config_dict:
                        dilation = int(config_dict['dilation'][0])
                        dilation_h = None
                        dilation_w = None
                    elif 'dilation_h' in config_dict and 'dilation_w' in config_dict:
                        dilation = None
                        dilation_h = int(config_dict['dilation_h'][0])
                        dilation_w = int(config_dict['dilation_w'][0])
                    else:
                        dilation = 1 # default=1
                        dilation_h = None
                        dilation_w = None
                        warning_msg_set_default(name, 'dilation', dilation)

                    layer_list.append(
                        ndk_layers.Layer(lyr_type=type, name=name, top=top, bottom=bottom,
                                         kernel_size=kernel_size, kernel_size_h=kernel_size_h, kernel_size_w=kernel_size_w,
                                         stride=stride, stride_h=stride_h, stride_w=stride_w,
                                         pad=pad, pad_n=pad_n, pad_s=pad_s, pad_w=pad_w, pad_e=pad_e,
                                         dilation=dilation, dilation_h=dilation_h, dilation_w=dilation_w,
                                         pool=pool)
                    )
                elif type == 'slice':
                    bottom = config_dict['bottom']
                    if 'axis' in config_dict:
                        axis = int(config_dict['axis'][0])
                    elif 'slice_dim' in config_dict:
                        axis = int(config_dict['slice_dim'][0])
                    else:
                        axis = 1
                        warning_msg_set_default(name, 'axis', axis)
                    if 'slice_point' in config_dict:
                        slice_point = config_dict['slice_point']
                        for i in range(len(slice_point)):
                            slice_point[i] = int(slice_point[i])
                    else:
                        slice_point = None
                        warning_msg_set_default(name, 'slice_point', slice_point)
                    layer_list.append(ndk_layers.Layer(lyr_type=type, name=name, bottom=bottom, top=top, axis=axis, slice_point=slice_point))
                elif type == 'concat':
                    bottom = config_dict['bottom']
                    if 'axis' in config_dict:
                        axis = int(config_dict['axis'][0])
                    elif 'concat_dim ' in config_dict:
                        axis = int(config_dict['concat_dim '][0])
                    else:
                        axis = 1
                        warning_msg_set_default(name, 'axis', axis)
                    layer_list.append(ndk_layers.Layer(lyr_type=type, name=name, bottom=bottom, top=top, axis=axis))
                elif type == 'eltwise':
                    bottom = config_dict['bottom']
                    if 'operation' in config_dict:
                        operation = config_dict['operation'][0]
                    else:
                        operation = 'sum'
                        warning_msg_set_default(name, 'operation', operation)
                    layer_list.append(ndk_layers.Layer(lyr_type=type, name=name, bottom=bottom, top=top, operation=operation))
                elif type == 'shufflechannel':
                    bottom = config_dict['bottom']
                    if 'group' in config_dict:
                        group = int(config_dict['group'][0])
                    else:
                        group = 1
                        warning_msg_set_default(name, 'group', group)
                    layer_list.append(ndk_layers.Layer(lyr_type=type, name=name, bottom=bottom, top=top, group=group))
                elif type == 'scalebytensor':
                    bottom = config_dict['bottom']
                    layer_list.append(ndk_layers.Layer(lyr_type=type, name=name, bottom=bottom, top=top))
                else:
                    raise Exception('unsupported layer type: {}'.format(type))
            else:
                pass # useless dict
    return layer_list
# def save_to_npz(param_dict, fname_npz):
#     assert type(fname_npz) == str, "Your npz file's name should be a string"
#     if os.path.exists(default_npy_dir):
#         # os.removedirs(default_npy_dir) # it fails when the directory is not empty.
#         ndk.utils.remove_dir(default_npy_dir)
#     save_param_dict(param_dict, default_npy_dir)
#     zip_npy(default_npy_dir, fname_npz)
#     # os.removedirs(default_npy_dir) # it fails when the directory is not empty.
#     ndk.utils.remove_dir(default_npy_dir)

def save_to_file(layer_list, fname_prototxt, param_dict=None, fname_npz=None):
    if layer_list != None:
        assert fname_prototxt != None, "You need to specify a prototxt file name if you set layer_list not none"
        save_to_prototxt(layer_list=layer_list, fname=fname_prototxt)
    # save_to_prototxt(fname_prototxt, layer_list) # it seems to be useless, CK 2019-5-27
    if param_dict != None:
        assert fname_npz != None, "You need to specify an npz file name if you set param_dict not none"
        save_to_npz(param_dict, fname_npz)

def load_from_file(fname_prototxt, fname_npz):
    layer_list = load_from_prototxt(fname_prototxt)
    param_dict = load_from_npz(fname_npz)
    return layer_list, param_dict

def modelpack_from_file(bitwidth, fname_prototxt, fname_npz, out_file_path, model_name = 'model', use_machine_code = True):
    if not fname_prototxt.endswith('.prototxt'):
        fname_prototxt += '.prototxt'
        print('\033[0;31m Warning: modelpack_from_file: will load layer_list from file {}\033[0m'.format(fname_prototxt))
    if not fname_npz.endswith('.npz'):
        fname_npz += '.npz'
        print('\033[0;31m Warning: modelpack_from_file: will load param from file {}\033[0m'.format(fname_npz))
    model_packer(bitwidth, fname_prototxt, out_file_path, fname_npz, True, model_name, use_machine_code = use_machine_code)
    
def modelpack(bitwidth, layer_list, param_dict, out_file_path, model_name = 'model', use_machine_code = False):
    if not os.path.exists(out_file_path):
        os.mkdir(out_file_path)
    layer_list = sort_layers(layer_list)
    fname_prototxt = os.path.join(out_file_path, model_name + ".prototxt")
    fname_npz = os.path.join(out_file_path, model_name + '.npz')

    # check quantized weight and bias
    max_q = 2 ** (bitwidth - 1) - 1
    min_q = -2 ** (bitwidth - 1)
    for layer in layer_list:
        if layer.name+'_quant_weight' in param_dict:
            param_dict[layer.name + '_quant_weight'] = numpy.clip(param_dict[layer.name+'_quant_weight'], min_q, max_q).astype(int)
        if layer.name+'_quant_bias' in param_dict:
            param_dict[layer.name + '_quant_bias'] = numpy.clip(param_dict[layer.name+'_quant_bias'], min_q, max_q).astype(int)
    save_to_file(layer_list, fname_prototxt, param_dict, fname_npz)
    modelpack_from_file(bitwidth, fname_prototxt, fname_npz, out_file_path, model_name, use_machine_code = use_machine_code)

if __name__=='__main__':
    #modelpack_from_file(8, 'C:/Users/admin/Documents/WeChat Files/wxid_ra437qko9fx811/FileStorage/File/2020-11/quant.prototxt', 'examples/shufflenetv2_05x/shufflenet_v2_0.5x.npz', 'out_shufflenetv2_05x')
    #modelpack_from_file(16, './examples/spc_2d_220714_no_ir_quant_v2.prototxt', './examples/spc_2d_220714_no_ir_quant_v2.npz', 'out_spc_2d_220714_no_ir')
    #modelpack_from_file(16, './examples/spc_2d_220713_no_ir_quant_v2.prototxt', './examples/spc_2d_220713_no_ir_quant_v2.npz', 'out_spc_2d_220713_no_ir')
    #modelpack_from_file(16, './examples/spc_2d_220715_feathernet_quant_v2.prototxt', './examples/spc_2d_220715_feathernet_quant_v2.npz', 'out_spc_2d_220715_feathernet')
    #modelpack_from_file(16, './examples/spc_2d_220715_feathernet_add_quant_v2.prototxt', './examples/spc_2d_220715_feathernet_add_quant_v2.npz', 'out_spc_2d_220715_feathernet_add')
    modelpack_from_file(16, './examples/spc_2d_220715_feathernet_add_epoch150_quant_v2.prototxt', './examples/spc_2d_220715_feathernet_add_epoch150_quant_v2.npz', 'out_spc_2d_220715_feathernet_add_epoch150')
    
    
    
#    layer_list, param_dict = load_from_file('D:/AI_code/K3DAITool/ndk/dilake/model.prototxt', 'D:/AI_code/K3DAITool/ndk/dilake/model.npz')
#    import cv2
#    import numpy as np
#    input_frac = param_dict['input_frac']
#    #exit()
#    img=cv2.imread('D:/AI_code/K3DAITool/ndk/dilake/0.jpg',0)
#    img = cv2.resize(img, (512,512))
#    #print(param_dict['layer186_signed'])
#    #exit()
#    #img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)/256.0
#    #img = (img-0.5)/0.5
#    res=img.flatten()
#    img = np.expand_dims(img,0)
#    input_data_batch= np.expand_dims(img,0)
#    from ndk.quant_tools.numpy_net import run_layers
#    from ndk.quant_tools.numpy_net import build_numpy_layers
#    numpy_layers= build_numpy_layers(layer_list, param_dict, quant=True, bitwidth=8)
#    data_dict = run_layers(input_data_batch,
#                layer_list,
#                target_feature_tensor_list=[layer.top for layer in layer_list if type(layer.top) == str],
#                param_dict=None,
#                quant=True,
#                bitwidth=8,
#                hw_aligned=True,
#                numpy_layers=numpy_layers,
#                log_on=False,
#                use_ai_framework=False,
#                quant_weight_only=False,
#                quant_feature_only=False
#                )
#    data_dict_int = {}
#    for out_name in data_dict.keys():
#        data_dict_int[out_name] = data_dict[out_name] * 2 ** param_dict[out_name + '_frac']
#    np.savez('data_dict.npz', **data_dict)
#    np.savez('data_dict_int.npz', **data_dict_int)
#    modelpack(8, layer_list[0:22], param_dict, 'model')
#    from ndk.layers import get_tensor_shape
#    tensor_shape = get_tensor_shape(layer_list)
#    modelpack_from_file(8, 'examples/shufflenetv2_05x/shufflenet_v2_0.5x.prototxt', 'examples/shufflenetv2_05x/shufflenet_v2_0.5x.npz', 'out_shufflenetv2_05x')

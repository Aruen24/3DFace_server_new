import os
import sys
ndk_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ndk_dir)
from ndk.layers import Layer
from ndk.layers import check_layers
from ndk.tf_interface.pb_read import read_graph_from_pb
from ndk.tf_interface.hdf5_read_all import read_graph_from_hdf5

# from layers import Layer
import tensorflow as tf
import os
import numpy as np
import math
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
conv = ['Conv2D','DepthwiseConv2dNative','DepthwiseConv2D']
pool = ['AvgPool','MaxPool','AveragePooling2D','GlobalAveragePooling2D','GlobalMaxPool2D','MaxPool2D','MaxPooling2D']
relu = ['Relu','Relu6','ReLU6','LeakyRelu','LeakyReLU','Tanh','Sigmoid','ReLU','relu','sigmoid','tanh']
scale_index = ['RealDiv','Sub','Mul']

def to_layerspb(layers):

    result = []
    for layer in layers:
        layer_type = layer[1]
        result_temp = Layer()
        if layer_type == 'Split'or layer_type == 'SplitV':
            name = layer[2]
            bottom = layer[0]
            top = layer[-2]
            axis = layer[4]
            slice_point = layer[5]
            result_temp.set_slice_layer(name,bottom,top,axis,slice_point)
            result.append(result_temp)
            # print(result_temp)
            continue
        elif layer_type == 'Slice' and layer[-1] != 1:
            name = layer[2]
            bottom = layer[0]
            top = layer[-2]
            axis = layer[4]
            slice_point = layer[5]
            result_temp.set_slice_layer(name, bottom, top, axis, slice_point)
            result.append(result_temp)
            # print(result_temp)
            continue
        elif layer_type ==  'Placeholder' or layer_type == 'InputLayer':
            name = layer[2]
            top = layer[3]
            output = layer[4]
            dim = (output[0],output[3],output[1],output[2])
            result_temp.set_input_layer(name,top,dim)
            result.append(result_temp)
            # print(result_temp)
            continue
        elif layer_type in conv:
            # print('############3')
            # print(layer)
            name = layer[2]
            top = layer[-2]
            bottom = layer[0]
            num_output = layer[4][3]
            kernel_size = None
            kernel_size_h = layer[4][0]
            kernel_size_w = layer[4][1]
            stride = None
            stride_h = layer[6][1]
            stride_w = layer[6][2]

            ###for pad
            padding = layer[7]
            if padding == 'SAME' or padding == 'same':
                in_height = layer[3][1]
                in_width = layer[3][2]
                out_shape = layer[-1][1]
                # stride = layer[6][1]
                stride_h = layer[6][1]
                stride_w = layer[6][2]
                dilation_h = layer[5][1]
                dilation_w = layer[5][2]
                filter_height = (layer[4][0] - 1) * dilation_h + 1
                filter_width = (layer[4][1] - 1) * dilation_w + 1
                new_height = layer[-1][1]
                new_width = layer[-1][2]
                pad_needed_height = (new_height - 1) * stride_h + filter_height - in_height
                pad_needed_width = (new_width - 1) * stride_w + filter_width - in_width
                pad_n = math.floor(pad_needed_height / 2)
                pad_s = int(pad_needed_height - pad_n)
                pad_w = math.floor(pad_needed_width / 2)
                pad_e = int(pad_needed_width - pad_w)

                if pad_needed_height <= 0:
                    pad_n = pad_s = 0
                if pad_needed_width <= 0:
                    pad_w = pad_e = 0
                h0 = math.ceil(in_height/stride_h)
                need = pad_n + pad_s + in_height
                temp = (h0-1)*stride_h + filter_height
                if temp == need:
                    pass
                elif need>temp:
                    n = need-temp
                    pad_s = pad_s - n
                # w0 = math.floor((pad_w + pad_e + in_width - filter_width) / stride_w) + 1
                w0 = math.ceil(in_width/stride_w)
                need = pad_w + pad_e + in_width

                temp = (w0 - 1) * stride_w + filter_width
                if temp == need:
                    pass
                elif need > temp:
                    n = need - temp
                    pad_e = pad_e - n


            elif padding == 'VALID' or padding == 'valid':
                pad_n = int(0)
                pad_s = int(0)
                pad_w = int(0)
                pad_e = int(0)
            else:
                pad_n = int(padding[0])
                pad_s = int(padding[1])
                pad_w = int(padding[2])
                pad_e = int(padding[3])
            padding = (pad_n,pad_s,pad_w,pad_e)
            bias_term = layer[9][0]
            dilation = None
            dilation_h = layer[5][1]
            dilation_w = layer[5][2]
            group = 1
            if layer_type == 'DepthwiseConv2dNative' or layer_type == 'DepthwiseConv2D':
                num_output = layer[4][2]*layer[4][3]
                group = layer[4][2]
            result_temp.set_convolution_layer(name ,top,bottom,num_output, kernel_size, kernel_size_h, kernel_size_w,stride, stride_h, stride_w,padding, pad_n, pad_s, pad_w, pad_e,bias_term,dilation, dilation_h, dilation_w,group)
            result.append(result_temp)
            continue
            # print(result_temp)
        elif layer_type in relu:
            if layer_type == 'Relu' or layer_type == 'LeakyRelu' or layer_type ==  'LeakyReLU' or layer_type == 'ReLU' or layer_type == 'relu':
                name = layer[2]
                top = layer[-2]
                bottom = layer[0]
                negative_slop = 0
                if layer_type == 'LeakyRelu' or layer_type ==  'LeakyReLU':
                    negative_slop = layer[-3]
                result_temp.set_relu_layer(name,bottom,top,negative_slop)
                result.append(result_temp)
            elif layer_type == 'Relu6' or layer_type == 'ReLU6':
                name = layer[2]
                top = layer[-2]
                bottom = layer[0]
                result_temp.set_relu6_layer(name,bottom,top)
                result.append(result_temp)
            elif layer_type == 'Sigmoid' or layer_type == 'sigmoid':
                name = layer[2]
                top = layer[-2]
                bottom = layer[0]
                result_temp.set_sigmoid_layer(name, bottom, top)
                result.append(result_temp)
            elif layer_type == 'Tanh' or layer_type == 'tanh':
                name = layer[2]
                top = layer[-2]
                bottom = layer[0]
                result_temp.set_tanh_layer(name, bottom, top)
                result.append(result_temp)
                continue
            # print(result_temp)
        elif layer_type in pool:
            """
            注意：如果原始的type是Max或者Mean，在result里面该层加入了一个指示'Mean''Max'
            所以该层pad的计算，索引一般都是用负数
            """
            name = layer[2]
            top = layer[-2]
            bottom = layer[0]
            kernel_size = None
            kernel_size_h = layer[-5][1]
            kernel_size_w = layer[-5][2]
            stride = None
            stride_h = layer[-4][1]
            stride_w = layer[-4][2]
            dilation = 1
            dilation_h = None
            dilation_w = None

            ###for pad
            padding = layer[-3]
            if padding == 'SAME' or padding == 'same':
                in_height = layer[-6][1]
                in_width = layer[-6][2]
                stride_h = layer[-4][1]
                stride_w = layer[-4][2]
                filter_height = layer[-5][1]
                filter_width = layer[-5][2]
                new_height = layer[-1][1]
                new_width = layer[-1][2]
                pad_needed_height = (new_height - 1) * stride_h + filter_height - in_height
                pad_needed_width = (new_width - 1) * stride_w + filter_width - in_width
                pad_n = math.floor(pad_needed_height/2)
                pad_s = int(pad_needed_height - pad_n)
                pad_w =  math.floor(pad_needed_width/2)
                pad_e = int(pad_needed_width - pad_w)
                if pad_needed_height <= 0:
                    pad_n = pad_s = 0
                if pad_needed_width <= 0:
                    pad_w = pad_e = 0

                h0 = math.ceil(in_height / stride_h)
                need = pad_n + pad_s + in_height
                temp = (h0 - 1) * stride_h + filter_height
                if temp == need:
                    pass
                elif need > temp:
                    n = need - temp
                    pad_s = pad_s - n
                # w0 = math.floor((pad_w + pad_e + in_width - filter_width) / stride_w) + 1
                w0 = math.ceil(in_width / stride_w)
                need = pad_w + pad_e + in_width
                temp = (w0 - 1) * stride_w + filter_width
                if temp == need:
                    pass
                elif need > temp:
                    n = need - temp
                    pad_e = pad_e - n

            elif padding == 'VALID' or padding == 'valid':
                # pad = padding
                pad_n = int(0)
                pad_s = int(0)
                pad_w = int(0)
                pad_e = int(0)
            else:
                pad_n = int(padding[0])
                pad_s = int(padding[1])
                pad_w = int(padding[2])
                pad_e = int(padding[3])
            padding = (pad_n, pad_s, pad_w, pad_e)
            if layer_type == 'AvgPool' or layer_type == 'AveragePooling2D' or layer_type == 'GlobalAveragePooling2D':
                pool_type = 'AVE'#####注意是avg还是max
            else:
                pool_type = 'MAX'

            result_temp.set_pooling_layer(name,top, bottom, kernel_size, kernel_size_h, kernel_size_w,  stride, stride_h, stride_w, padding, pad_n, pad_s, pad_w, pad_e, dilation, dilation_h, dilation_w,pool_type)
            result.append(result_temp)
            continue
            # print(result_temp)
        elif layer_type == 'Softmax' or layer_type == 'softmax':
            pass
            # name = layer[2]
            # top = layer[-2]
            # bottom = layer[0]
            # result_temp.set_logsoftmax_layer(name, bottom, top)
            # # result_temp.set_softmax_layer(name,bottom,top)
            # result.append(result_temp)
            # continue
            # print(result_temp)
        elif layer_type == 'Add':
            name = layer[2]
            top = layer[-2]
            bottom = layer[0]
            operation = 'sum'
            result_temp.set_eltwise_layer(name, bottom, top, operation)
            result.append(result_temp)
            # print(result_temp)
            continue
        elif layer_type == 'ConcatV2' or layer_type == 'Concatenate':
            name = layer[2]
            top = layer[-2]
            bottom = layer[0]
            axis = layer[-3]
            result_temp.set_concat_layer(name, bottom, top,axis)
            result.append(result_temp)
            continue
            # print(result_temp)
        elif layer_type == 'FusedBatchNorm' or layer_type == 'BatchNormalizationV1':
            name = layer[2]
            top = layer[-2]
            bottom = layer[0]
            result_temp.set_batchnorm_layer(name, bottom, top)
            result.append(result_temp)
            continue
            # print(result_temp)
        elif layer_type == 'shufflechannel':
            name = layer[2]
            bottom = layer[0]
            top = layer[-2]
            group = layer[-3]
            result_temp.set_shufflechannel_layer(name,bottom,top,group)
            result.append(result_temp)
            continue
        elif layer_type in scale_index and layer[-1] != 1:
            name = layer[2]
            bottom = layer[0]
            top = layer[-2]
            bias_term = layer[-3][0]
            result_temp.set_scale_layer(name,bottom,top,bias_term)
            result.append(result_temp)
            continue
        elif layer_type == 'MatMul_conv' or layer_type == 'Dense':
            # pass
            name = layer[2]
            bottom = layer[0]
            top = layer[-2]
            num_output = layer[-1][-1]
            bias_term = layer[-3][0]
            result_temp.set_innerproduct_layer(name, bottom, top, num_output,bias_term)
            result.append(result_temp)
            continue
        elif layer_type == 'bias' and layer[-1] != 1:
            name = layer[2]
            bottom = layer[0]
            top = layer[-2]
            result_temp.set_bias_layer(name,bottom,top)
            result.append(result_temp)
            continue
        elif layer_type == 'LogSoftmax':
            name = layer[2]
            bottom = layer[0]
            top = layer[-2]
            result_temp.set_logsoftmax_layer(name, bottom, top)
            result.append(result_temp)
            continue
        elif layer_type == 'Reshape' or layer[-1] == 1:
            pass
        else:
            raise Exception(
                "This program can't deal with the layer of {} ,please check the code of pb_to_K3DAITool.py".format(layer_type))
    print('Calculating..............')

    return result

if __name__ == '__main__':
    # model_path_1 =  r'C:/Users/75831/Desktop/2019.6.26/ndk/examples/tf_parser_test_1/densenet121.pb'
    # model_path_1 = r'lenet_model.pb'
    model_path_1 =  'C:/Users/75831/Desktop/pb_2019/tf_nn_layers_keras/keras_hdf5_net1.hdf5'
    # layers,layer_w_b = read_graph_from_hdf5(model_path_1)
    layers,layer_w_b = read_graph_from_hdf5(model_path_1)
    #打印分层结果
    print(len(layers))
    print('layer_data:')
    for layer in layers:
        print(layer)
    print('##############################################################')
    result = to_layerspb(layers)
    #打印layer类的结果
    for layer in result:
        print(layer)

    print('Key in layer_w_b:')
    # 打印存储w/b的字典的键
    for key in layer_w_b:
        print(key)
        # if len(layer_w_b[key].shape) == 1:
        print(layer_w_b[key].shape)
    check_layers(result, layer_w_b)




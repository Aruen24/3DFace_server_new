from tensorflow.python.keras.models import load_model
import numpy as np
import math
import os
import time
import warnings
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


#conv_index卷积操作，matmul虽然是全连接层，但是在硬件上还是按照卷积处理
conv_index = ['Dense','DepthwiseConv2D','Conv2D','SeparableConv2D']
#relu_index:激活函数，暂时没有支持softmax以及logSogtmax，后面处理了再修改
relu_index = ['Sigmoid', 'Tanh', 'ReLU','Relu6','LeakyReLU','LeakyRelu','Softmax','LogSoftmax']
#pool_index:池化操作
pool_index = ['AveragePooling2D','GlobalAveragePooling2D','GlobalMaxPool2D','MaxPool2D','MaxPooling2D']

single_layer = ['Concatenate','InputLayer','Activation','BatchNormalizationV1','ReLU','Add','AveragePooling2D','GlobalAveragePooling2D','GlobalMaxPool2D','MaxPool2D','MaxPooling2D','Lambda','Flatten','LeakyReLU','PReLU']
#all_index：所有该程序能够处理的节点列表
all_index = ['GlobalAveragePooling2D','Concatenate','InputLayer','Activation','BatchNormalizationV1','ReLU','Add','MaxPooling2D','LeakyReLU','PReLU']

layer_w_b = {}
N = 0
H = 1
W = 2
C = 3


def name_shape(model,layers):
    name_shape = {}
    input = model.input
    input_shape = input.get_shape().as_list()
    if input_shape[0] == None:
        input_shape[0] = 1
    input_im = np.ones(input_shape)
    name = input.name
    name_shape[name] = input_shape
    for layer in layers[1:]:
        inputs = layer.input
        if isinstance(inputs, list):
            for input in inputs:
                name = input.name
                shape = input.get_shape().as_list()
                temp = 0
                for i in range(len(shape)):
                    n = shape[i]
                    if n == None:
                        shape[i] = 1
                        temp += 1
                if temp >= 2:
                    shape = list(tf.Session().run(input, feed_dict={model.input:input_im}).shape)
                name_shape[name] = shape
        elif not isinstance(inputs, list):
            name = inputs.name
            shape = inputs.get_shape().as_list()
            temp = 0
            for i in range(len(shape)):
                n = shape[i]
                if n == None:
                    shape[i] = 1
                    temp += 1
            if temp >= 2:
                shape = list(tf.Session().run(input, feed_dict={model.input: input_im}).shape)
            name_shape[name] = shape



        outputs = layer.output
        if isinstance(outputs, list):
            for output in outputs:
                name = output.name
                shape = output.get_shape().as_list()
                temp = 0
                for i in range(len(shape)):
                    n = shape[i]
                    if n == None:
                        shape[i] = 1
                        temp += 1
                if temp >= 2:
                    shape = list(tf.Session().run(output, feed_dict={model.input: input_im}).shape)
                name_shape[name] = shape
        elif not isinstance(outputs, list):
            name = outputs.name
            shape = outputs.get_shape().as_list()
            temp = 0
            for i in range(len(shape)):
                n = shape[i]
                if n == None:
                    shape[i] = 1
                    temp += 1
            if temp >= 2:
                shape = list(tf.Session().run(outputs, feed_dict={model.input: input_im}).shape)
            name_shape[name] = shape

    return name_shape

    # name =
def find_my_input(model):
    result = {}
    layers = model.layers
    for layer in layers[1:]:
        inputs = layer.input
        if isinstance(inputs, list):
            for input in inputs:
                name = input.name
                result.setdefault(name,[]).append(layer)
        elif not isinstance(inputs, list):
            name = inputs.name
            result.setdefault(name,[]).append(layer)
    output = layers[len(layers)-1].output
    if output == []:
        name = 'NULL'
    else:
        name = layers[len(layers)-1].output[0].name
    layer = []
    result[name] = layer
    return result
def find_my_output(model,find_op_by_input):
    result = {}
    layers = model.layers
    for layer in layers:
        outputs = layer.output
        if isinstance(outputs, list):
            for output in outputs:
                name = output.name
                result.setdefault(name,[]).append(layer)
        elif not isinstance(outputs, list):
            name = outputs.name
            result.setdefault(name,[]).append(layer)
    for key in result:
        if not key in find_op_by_input:
            name = key
            layer = []
            find_op_by_input[name] = layer
    return result,find_op_by_input
def padding_data(layer,padding,find_op_by_output,name_shape_dic):
    layer_type = layer.__class__.__name__
    config = layer.get_config()
    ksize = []
    if layer_type in pool_index:
        ksize = config['pool_size']
    else:
        ksize = config['kernel_size']
    strides = config['strides']
    input_name = layer.input.name
    input_shape = []
    input_shape_ori = name_shape_dic[input_name]
    output = layer.output
    output_name = output.name
    output_shape = name_shape_dic[output_name]


    input1 = input_name
    input1_from_layer = find_op_by_output[input1]
    layers_type = [layer.__class__.__name__  for layer in input1_from_layer]
    # print(layers_type)
    if 'ZeroPadding2D' in layers_type:
        if padding == 'same':
            raise Exception(
                'The "padding" type of convolution which  operation after "pad" should be "VALID" while your type is "SAME"')
        else:
            input_shape_1 = name_shape_dic[input1_from_layer[0].input.name]
            if not len(input_shape_1) == []:
                input_shape = input_shape_1
                input_name = input1_from_layer[0].input.name
            pad_data = input1_from_layer[0].get_config()['padding']
            # if (pad_data[0] == pad_data[1]) and (pad_data[0][0] == pad_data[1][1]):
            # if pad_data[0] == pad_data[1]:
            pad_data1 = [pad_data[0][0],pad_data[0][1],pad_data[1][0],pad_data[1][1]]
            # if (pad_data[0] == pad_data[1]) and (pad_data[0][0] == 0) and (pad_data[0][1] == 1):
            #     pad_data1 = 'same'


            res1_1 = math.floor((input_shape_ori[H] - input_shape_1[H])/2)
            res1_2 = math.ceil((input_shape_ori[H] - input_shape_1[H])/2)
            res2_1 = math.floor((input_shape_ori[W] - input_shape_1[W])/2)
            res2_2 = math.ceil((input_shape_ori[W] - input_shape_1[W])/2)
            pad = [res1_1, res1_2, res2_1, res2_2]

            cons1 = list(pad_data[0])
            cons2 = list(pad_data[1])

            if ([res1_1, res1_2] == cons1) and ([res2_1, res2_2] == cons2):
                padding = pad_data1
            else:
                raise Exception('The operation of "pad" cannot be merged with the operation of convolution,'
                                'because the transformed "SAME" operation is different from the "pad" operation in data processing.')

            # if not padding == 'same':
            input_h = input_shape_1[H]
            input_w = input_shape_1[W]
            output_h = output_shape[H]
            output_w = output_shape[W]
            if layer_type in conv_index:
                dilation = layer.get_config()['dilation_rate']
                dilation_h = dilation[0]
                dilation_w = dilation[1]
                ksize_h = (ksize[0] - 1) * dilation_h + 1
                ksize_w = (ksize[1] - 1) * dilation_w + 1
            else:
                ksize_h = ksize[0]
                ksize_w = ksize[1]
            strids_h = strides[0]
            strids_w = strides[1]
            pad_n = padding[0]
            pad_s = padding[1]
            pad_w = padding[2]
            pad_e = padding[3]
            h0 = math.floor((pad_n + pad_s + input_h - ksize_h) / strids_h) + 1

            need = pad_n + pad_s + input_h
            temp = (h0 - 1) * strids_h + ksize_h

            if temp == need:
                pass
            elif need > temp:
                n = need - temp
                pad_s = pad_s - n
            w0 = math.floor((pad_w + pad_e + input_w - ksize_w) / strids_w) + 1
            need = pad_w + pad_e + input_w
            temp = (w0 - 1) * strids_w + ksize_w

            if temp == need:
                pass
            elif need > temp:
                n = need - temp
                pad_e = pad_e - n
            padding = [pad_n, pad_s, pad_w, pad_e]
            if layer_type in pool_index:
                warnings.warn(
                    "Because there is a 'Pad' node(we don't support it),which at the top of AvgPool/MaxPool node,so our program has to merge the two nodes(Pad&pool).The result of this merge operation will cause errors in the results.")
    else:
        return padding,name_shape_dic[input_name]
    return padding, input_shape




def conv_defeat(layers,layer,name_shape_dic,find_op_by_input,find_op_by_output):


    result = []
    result_temp = []
    layer_type = layer.__class__.__name__
    input = layer.input
    input_name = layer.input.name
    input_shape = name_shape_dic[input_name]
    output = layer.output
    output_name = output.name
    output_shape = name_shape_dic[output_name]
    layer_name = layer.name
    weight = layer.get_weights()
    config = layer.get_config()
    dilations = config['dilation_rate']
    strides = config['strides']
    padding = config['padding']
    activation = config['activation']
    bias = config['use_bias']
    beta = []
    if bias == True:
        weight, beta = layer.get_weights()
    result_temp.append([input_name])
    result_temp.append(layer_type)
    result_temp.append(layer_name)
    ##############################################
    if layer == layers[0]:
        layer_name_all = []
    else:
        input1_from_layer = find_op_by_output[input_name]
        layer_name_all = [layer.__class__.__name__  for layer in input1_from_layer]
        if layer_name_all[0] == 'ZeroPadding2D':
            input_name = input1_from_layer[0].input.name
            result_temp[0] = input_name
            padding,input_shape = padding_data(layer,padding,find_op_by_output,name_shape_dic)
    w_name = layer_name + '_weight'
    layer_w_b[w_name] = weight
    if bias == True:
        b_name = layer_name+'_bias'
        layer_w_b[b_name] = beta
    ksize = list(np.array(layer_w_b[w_name]).shape)
    if len(ksize) == 4:
        pass
    else:
        ksize = ksize[1:]
        layer_w_b[w_name] = np.array(layer_w_b[w_name]).reshape(ksize)


    strides = list(strides)
    dilations = list(dilations)
    strides = [1,strides[0],strides[1],1]
    dilations = [1,dilations[0],dilations[1],1]




    result_temp.append(input_shape)
    result_temp.append(ksize)
    result_temp.append(dilations)
    result_temp.append(strides)
    result_temp.append(padding)
    result_temp.append('w')
    result_temp.append([bias, 'beta'])
    result_temp.append(output_name)
    result_temp.append(output_shape)
    result.append(result_temp)

    if layer_type == 'DepthwiseConv2D':
        key = layer_name+'_weight'
        layer_w_b[key] = np.array(layer_w_b[key])
        shape = layer_w_b[key].shape
        shape = list(shape)
        if shape[-1] != 1:
            shape[-2] = shape[-2]*shape[-1]
            shape[-1] = 1
            layer_w_b[key] = layer_w_b[key].reshape(shape)
        layer_w_b[key] = layer_w_b[key].transpose((0, 1, 3, 2))


    if activation == 'linear':
        pass
    else:
        result_temp = []
        result[0][-2] = result[0][-2]+'_temp'
        layer1_name = layer_name+'_'+activation
        layer1_type = activation
        input1_name = result[0][-2]
        input1_shape = result[0][-1]
        output1_name = layer.output.name
        output1_shape = input1_shape
        result_temp = []
        result_temp.append([input1_name])
        result_temp.append(layer1_type)
        result_temp.append(layer1_name)
        result_temp.append(input1_shape)
        result_temp.append(output1_name)
        result_temp.append(output1_shape)
        result.append(result_temp)


    return result
def separable_conv_data(layer,name_shape_dic,find_op_by_input,find_op_by_output):
    result = []
    result_temp = []
    layer_type = layer.__class__.__name__
    input = layer.input
    input_name = layer.input.name
    input_shape = name_shape_dic[input_name]
    output = layer.output
    output_name = output.name
    output_shape = name_shape_dic[output_name]
    layer_name = layer.name
    # weight = layer.get_weights()
    config = layer.get_config()
    dilations = config['dilation_rate']
    strides = config['strides']
    padding = config['padding']
    activation = config['activation']
    weigth1 = weight2 = []
    bias1 = False
    beat1 = []
    bias2 = config['use_bias']
    beta2 = []
    if bias2 == True:
        weight1,weight2,beta2 = layer.get_weights
    else:
        weight1,weight2 = layer.get_weights()
    result_temp.append([input_name])
    result_temp.append('DepthwiseConv2D')
    result_temp.append(layer_name+'_depth')

    ##############################################
    #depthwiseconv2d
    input1_from_layer = find_op_by_output[input_name]
    layer_name_all = [layer.__class__.__name__ for layer in input1_from_layer]
    if layer_name_all[0] == 'ZeroPadding2D':
        input_name = input1_from_layer[0].input.name
        result_temp[0] = input_name
        padding, input_shape = padding_data(layer, padding, find_op_by_output, name_shape_dic)
    w_name = layer_name + '_depth'+'_weight'
    layer_w_b[w_name] = weight1
    ksize = list(np.array(layer_w_b[w_name]).shape)
    if len(ksize) == 4:
        pass
    else:
        ksize = ksize[1:]
        layer_w_b[w_name] = np.array(layer_w_b[w_name]).reshape(ksize)

    strides = list(strides)
    dilations = list(dilations)
    strides = [1, strides[0], strides[1], 1]
    dilations = [1, dilations[0], dilations[1], 1]
    output1_shape = []
    if config['padding'] == 'same':
        output_h = math.ceil(input_shape[H]/strides[1])
        output_w = math.ceil(input_shape[W]/strides[2])
    else:
        output_h = math.floor((input_shape[H]-ksize[1])/strides[1]) +1
        output_w = math.floor((input_shape[W]-ksize[2])/strides[2]) +1
    if C == 3:
        output1_shape = [1,output_h,output_w,weight1.shape[-1]*weight1.shape[-2]]
    else:
        output1_shape = [1,weight1.shape[-1]*weight1.shape[-2],output_h,output_w]


    result_temp.append(input_shape)
    result_temp.append(ksize)
    result_temp.append(dilations)
    result_temp.append(strides)
    result_temp.append(padding)
    result_temp.append('w')
    result_temp.append([bias1, 'beta'])
    result_temp.append(output_name+'_depth')
    result_temp.append(output1_shape)
    result.append(result_temp)

    key = layer_name + '_depth'+'_weight'
    layer_w_b[key] = np.array(layer_w_b[key])
    shape = layer_w_b[key].shape
    shape = list(shape)
    if shape[-1] != 1:
        shape[-2] = shape[-2] * shape[-1]
        shape[-1] = 1
        layer_w_b[key] = layer_w_b[key].reshape(shape)
    layer_w_b[key] = layer_w_b[key].transpose((0, 1, 3, 2))




    w_name = layer_name + '_weight'
    layer_w_b[w_name] = weight2
    if bias2 == True:
        b_name = layer_name + '_bias'
        layer_w_b[b_name] = beta2
    result_temp =[]
    result_temp.append([output_name+'_depth'])
    result_temp.append('Conv2D')
    result_temp.append(layer_name)
    result_temp.append(output1_shape)
    result_temp.append(list(weight2.shape))
    result_temp.append([1,1,1,1])
    result_temp.append([1,1,1,1])
    result_temp.append(padding)
    result_temp.append('w')
    result_temp.append([bias2, 'beta'])
    result_temp.append(output_name)
    result_temp.append(output_shape)
    result.append(result_temp)


    if activation == 'linear':
        pass
    else:
        result_temp = []
        result[1][-2] = result[0][-2]+'_temp'
        layer1_name = layer_name+'_'+activation
        layer1_type = activation
        input1_name = result[1][-2]
        input1_shape = result[1][-2]
        output1_name = layer.output.name
        output1_shape = input1_shape
        result_temp = []
        result_temp.append([input1_name])
        result_temp.append(layer1_type)
        result_temp.append(layer1_name)
        result_temp.append(input1_shape)
        result_temp.append(output1_name)
        result_temp.append(output1_shape)
        result.append(result_temp)
    return result
def w_data(w):
    shape = list(w.shape)
    if C == 3:
        index = shape[-1]
    else:
        index = shape[0]
    result = np.ones([index])
    w = tf.transpose(w,(2,0,1))
    w = tf.Session().run(w)
    tf.Session().close()
    i = 0
    while i < index:
        array = w[i]
        temp = array[0][0]
        array_shape = list(array.shape)
        temp_array = np.ones(array_shape)*temp
        if (w[i] == temp_array).all:
            result[i] = temp
        else:
            raise Exception("all elements in the parament of 'w' which is the weight of depthwiseconv2d should be same, but your 'w' is different ")
        i = i+1
    result.resize(index)
    result = np.array(result)
    # print(result)
    return result
def prelu_data(layer,name_shape_dic):
    layer_type = layer.__class__.__name__
    result_temp = []
    result = []
    #第一部分
    layer_name = layer.name
    input_name = layer.input.name
    input_shape = name_shape_dic[input_name]
    output_name = layer.output.name
    output_shape = name_shape_dic[output_name]
    config = layer.get_config()
    result_temp.append([input_name])
    result_temp.append('Relu')
    result_temp.append(layer_name+'_relu1')
    result_temp.append(input_shape)
    result_temp.append(output_name+'_relu1')
    result_temp.append(output_shape)
    result.append(result_temp)

    #第二部分
    result_temp = []
    result_temp.append([input_name])
    result_temp.append('Mul')
    result_temp.append(layer_type+'/neg_1')
    result_temp.append(input_shape)
    weight = np.ones(input_shape[C])*-1
    layer_w_b[layer_type+'/neg_1'+'_weight'] = weight
    bias = False
    beta = []
    result_temp.append([bias,beta])
    out_neg_name = output_name+'/neg1'
    result_temp.append(out_neg_name)
    result_temp.append(output_shape)
    result.append(result_temp)


    #relu
    result_temp = []
    result_temp.append([out_neg_name])
    result_temp.append('Relu')
    result_temp.append(layer_name+'/relu2')
    result_temp.append(input_shape)
    result_temp.append(output_name + '/relu2')
    result_temp.append(output_shape)
    result.append(result_temp)

    #neg
    result_temp = []
    result_temp.append([output_name + '/relu2'])
    result_temp.append('Mul')
    result_temp.append(layer_type + '/neg_2')
    result_temp.append(input_shape)
    weight = layer.get_weights()
    weight = w_data(weight[0])
    layer_w_b[layer_type + '/neg_2' + '_weight'] = weight*-1
    bias = False
    beta = []
    result_temp.append([bias, beta])
    out_neg_name = output_name + '/neg2'
    result_temp.append(out_neg_name)
    result_temp.append(output_shape)
    result.append(result_temp)

    #第三部分
    input_name = [output_name+'_relu1',out_neg_name]
    result_temp = []
    result_temp.append(input_name)
    result_temp.append('Add')
    result_temp.append(layer_name+'_add')
    result_temp.append(output_name)
    result_temp.append(output_shape)
    result.append(result_temp)
    return result

def matmul_data(layer,name_shape_dic,find_op_by_input,find_op_by_output):
    result = []
    result_temp = []
    input = layer.input
    layer_type = layer.__class__.__name__
    layer_name = layer.name
    input_name = layer.input.name
    input_shape = name_shape_dic[input_name]
    output = layer.output
    output_name = output.name
    output_shape = name_shape_dic[output_name]
    config = layer.get_config()
    activation = config['activation']
    bias = config['use_bias']
    weight = []
    beta = []
    if bias == True:
        weight,beta = layer.get_weights()
    else:
        weight = layer.get_weights()[0]
    units = config['units']
    input1_from_layer = find_op_by_output[input_name][0]
    if input1_from_layer.__class__.__name__ == 'Flatten':
        input_name = input1_from_layer.input.name
        input_shape = name_shape_dic[input1_from_layer.input.name]
    input1 = input1_from_layer.input
    input1_shape = input1.shape
    if len(input_shape) == 2:
        input_shape.insert(0,1)
        input_shape.insert(0,1)
    if len(input1_shape) == 2:
        ksize = list(weight.shape)
        ksize = [1,1,ksize[0],ksize[1]]
    else:
        ksize = [input_shape[1],input_shape[2],input_shape[3],units]
    #########33
    w_name = layer_name+'_weight'
    w = weight.reshape(ksize)
    layer_w_b[w_name] = w
    if bias == True:
        b_name = layer_name+'_bias'
        layer_w_b[b_name] = beta
    strids = [1,1,1,1]
    dilations = [1,1,1,1]
    padding = 'valid'
    output_shape.insert(0,1)
    output_shape.insert(0,1)
    result_temp.append([input_name])
    result_temp.append(layer_type)
    result_temp.append(layer_name)
    result_temp.append(input_shape)
    result_temp.append(ksize)
    result_temp.append(dilations)
    result_temp.append(strids)
    result_temp.append(padding)
    result_temp.append('w')
    result_temp.append([bias,'beat'])
    result_temp.append(output_name)
    result_temp.append(output_shape)
    result.append(result_temp)

    if activation == 'linear':
        pass
    elif activation == 'softmax':
        result[0][-2] = result[0][-2]+'_temp'
        layer1_name = layer_name+'_softmax'
        layer1_type = activation
        input1_name = result[0][-2]
        input1_shape = result[0][-2]
        output1_name = layer.output.name
        output1_shape = input1_shape
        result_temp = []
        result_temp.append([input1_name])
        result_temp.append(layer1_type)
        result_temp.append(layer1_name)
        result_temp.append(input1_shape)
        result_temp.append(output1_name)
        result_temp.append(output1_shape)
        result.append(result_temp)
    elif activation == 'relu':
        result[0][-2] = result[0][-2] + '_temp'
        layer1_name = layer_name + '_relu'
        layer1_type = activation
        input1_name = result[0][-2]
        input1_shape = result[0][-2]
        output1_name = layer.output.name
        output1_shape = input1_shape
        result_temp = []
        result_temp.append([input1_name])
        result_temp.append(layer1_type)
        result_temp.append(layer1_name)
        result_temp.append(input1_shape)
        result_temp.append(output1_name)
        result_temp.append(output1_shape)
        result.append(result_temp)
    return result



def operation_data(model,name_shape_dic,find_op_by_input,find_op_by_output):
    result = []
    layers = model.layers
    for layer in layers:
        result_temp = []
        layer_type = layer.__class__.__name__
        if layer_type in single_layer:
            if layer_type == 'InputLayer':
                input_name = []
                layer_name = layer.name
                output_name = layer.output.name
                output_shape = name_shape_dic[output_name]
                result_temp.append(input_name)
                result_temp.append(layer_type)
                result_temp.append(layer_name)
                result_temp.append(output_name)
                result_temp.append(output_shape)
                result.append(result_temp)
            elif layer_type == 'PReLU':
                result_temp = prelu_data(layer,name_shape_dic)
                for res in result_temp:
                    result.append(res)
            elif layer_type == 'Add' or layer_type == 'Lambda':
                layer_name = layer.name
                input_name = [v.name for v in layer.input]
                input_shape = [name_shape_dic[name] for name in input_name]
                output_name = layer.output.name
                output_shape = name_shape_dic[output_name]

                result_temp.append(input_name)
                if layer_type == 'Lambda':
                    layer_type == 'Sub'
                result_temp.append('Add')
                result_temp.append(layer_name)
                for shape in input_shape:
                    result_temp.append(shape)
                result_temp.append(output_name)
                result_temp.append(output_shape)
                result.append(result_temp)

            elif layer_type == 'Concatenate':
                layer_name = layer.name
                input_name = [v.name for v in layer.input]
                input_shape = [name_shape_dic[name] for name in input_name]
                output_name = layer.output.name
                output_shape = name_shape_dic[output_name]
                config = layer.get_config()
                axis = config['axis']
                a = [0, 1, 2, 3]
                b = [0, 1]
                if axis < 0 and len(input_shape[0]) == 4:
                    axis = a[axis]
                elif axis < 0 and len(input_shape[0]) == 2:
                    axis = b[axis]
                if C == 3 and len(input_shape[0]) == 4:
                    if axis == 3:
                        axis_true = 1
                    elif axis == 1:
                        axis_true = 2
                    elif axis == 2:
                        axis_true = 3
                    else:
                        axis_true = axis
                else:
                    axis_true = axis

                result_temp.append(input_name)
                result_temp.append(layer_type)
                result_temp.append(layer_name)
                for shape in input_shape:
                    result_temp.append(shape)
                if len(output_shape) == 2:
                    output_shape.insert(0, 1)
                    output_shape.insert(0, 1)
                result_temp.append(axis_true)
                result_temp.append(output_name)
                result_temp.append(output_shape)
                result.append(result_temp)



            elif layer_type == 'GlobalAveragePooling2D' or layer_type == 'GlobalMaxPool2D':
                layer_name = layer.name
                input_name = layer.input.name
                input_shape = name_shape_dic[input_name]
                output_name = layer.output.name
                output_shape = name_shape_dic[output_name]

                ksize = [1,input_shape[H],input_shape[W],1]
                strides = [1,1,1,1]
                padding = 'valid'
                output_shape.insert(0,1)
                output_shape.insert(0,1)

                result_temp.append([input_name])
                result_temp.append(layer_type)
                result_temp.append(layer_name)
                result_temp.append(input_shape)
                result_temp.append(ksize)
                result_temp.append(strides)
                result_temp.append(padding)
                result_temp.append(output_name)
                result_temp.append(output_shape)
                result.append(result_temp)
            elif layer_type == 'MaxPool2D' or layer_type ==  'AveragePooling2D' or layer_type == 'MaxPooling2D':
                layer_name = layer.name
                input_name = layer.input.name
                input_shape = name_shape_dic[input_name]
                output_name = layer.output.name
                output_shape = name_shape_dic[output_name]
                config = layer.get_config()


                strides = config['strides']
                padding = config['padding']

                ##############################################
                input1_from_layer = find_op_by_output[input_name]
                layer_name_all = [layer.__class__.__name__ for layer in input1_from_layer]
                if layer_name_all[0] == 'ZeroPadding2D':
                    input_name = input1_from_layer[0].input.name

                    padding, input_shape = padding_data(layer, padding, find_op_by_output, name_shape_dic)
                ksize = config['pool_size']
                ksize = list(ksize)
                strides = list(strides)

                ksize = [1, ksize[0], ksize[1], 1]
                strides = [1, strides[0], strides[1], 1]


                result_temp.append([input_name])
                result_temp.append(layer_type)
                result_temp.append(layer_name)
                result_temp.append(input_shape)
                result_temp.append(ksize)
                result_temp.append(strides)
                result_temp.append(padding)
                result_temp.append(output_name)
                result_temp.append(output_shape)
                result.append(result_temp)

            elif layer_type in relu_index and not layer_type == 'LeakyReLU':
                layer_name = layer.name
                input_name = layer.input.name
                input_shape = name_shape_dic[input_name]
                output_name = layer.output.name
                output_shape = name_shape_dic[output_name]
                config = layer.get_config()
                if config['max_value'] != None:
                    layer_type = 'ReLU6'
                result_temp.append([input_name])
                result_temp.append(layer_type)
                result_temp.append(layer_name)
                result_temp.append(input_shape)
                result_temp.append(output_name)
                result_temp.append(output_shape)
                result.append(result_temp)
            elif layer_type in relu_index and layer_type == 'LeakyReLU':
                layer_name = layer.name
                input_name = layer.input.name
                input_shape = name_shape_dic[input_name]
                output_name = layer.output.name
                output_shape = name_shape_dic[output_name]
                config = layer.get_config()
                alpha = config['alpha']
                result_temp.append([input_name])
                result_temp.append(layer_type)
                result_temp.append(layer_name)
                result_temp.append(input_shape)
                result_temp.append(alpha)
                result_temp.append(output_name)
                result_temp.append(output_shape)
                result.append(result_temp)
            elif layer_type == 'Activation':
                layer_type = layer.get_config()['activation']
                layer_name = layer.name
                input_name = layer.input.name
                input_shape = name_shape_dic[input_name]
                output_name = layer.output.name
                output_shape = name_shape_dic[output_name]

                result_temp.append([input_name])
                result_temp.append(layer_type)
                result_temp.append(layer_name)
                result_temp.append(input_shape)
                result_temp.append(output_name)
                result_temp.append(output_shape)
                result.append(result_temp)
            elif layer_type == 'BatchNormalizationV1':
                config = layer.get_config()
                layer_name = layer.name
                input_name = layer.input.name
                input_shape = name_shape_dic[input_name]
                output_name = layer.output.name
                output_shape = name_shape_dic[output_name]

                bias = True
                momentum = config['momentum']
                epsilon = config['epsilon']
                center = layer.center
                scale = layer.scale
                if center is False and scale == True:
                    gamma,mean,variance = layer.get_weights()
                    beta = 0
                elif scale is False and center == True:
                    beta, mean, variance = layer.get_weights()
                    gamma = 1
                else:
                    gamma,beta,mean,variance = layer.get_weights()
                beta = beta - gamma * mean / np.sqrt(variance + epsilon)
                w = gamma / np.sqrt(variance + epsilon)
                ##权重和偏置保存到字典中
                w_name = layer_name + '_weight'
                b_name = layer_name + '_bias'
                layer_w_b[w_name] = w
                layer_w_b[b_name] = beta
                ##########################
                result_temp.append([input_name])
                result_temp.append(layer_type)
                result_temp.append(layer_name)
                result_temp.append(input_shape)
                result_temp.append('w')
                result_temp.append([bias, 'beta'])
                result_temp.append(output_name)
                result_temp.append(output_shape)
                result.append(result_temp)






        elif layer_type in conv_index:
            if layer_type == 'Conv2D' or layer_type ==  'DepthwiseConv2D':
                result_temp = conv_defeat(layers,layer,name_shape_dic,find_op_by_input,find_op_by_output)
                if len(result_temp) == 2:
                    for res in result_temp:
                        result.append(res)
                else:
                    result.append(result_temp[0])
            elif layer_type == 'Dense':
                result_temp = matmul_data(layer,name_shape_dic,find_op_by_input,find_op_by_output)
                if len(result_temp) == 2:
                    for res in result_temp:
                        result.append(res)
                else:
                    result.append(result_temp[0])
            elif layer_type == 'SeparableConv2D':
                result_temp = separable_conv_data(layer,name_shape_dic,find_op_by_input,find_op_by_output)
                for res in result_temp:
                    result.append(res)
    input_index = False
    for res in result:
        type = res[1]
        if 'InputLayer' == type:
            input_index = True
    if input_index is False:
        result_temp = []
        input_name = []
        layer_name = 'InputLayer'
        output_name = model.input.name
        output_shape = name_shape_dic[output_name]
        result_temp.append(input_name)
        result_temp.append('InputLayer')
        result_temp.append(layer_name)
        result_temp.append(output_name)
        result_temp.append([1,28,28,1])
        result.insert(0,result_temp)

    return result
def pool_padding(result):
    for i in range(len(result)):
        layer = result[i]
        type = layer[1]
        name = layer[2]
        if type in pool_index:
            padding = layer[-3]
            if padding == 'valid' and layer[-4][1] != 1 and layer[-4][1] != 1:
                input_h = layer[-6][H]
                input_w = layer[-6][W]
                output_h = layer[-1][H]
                output_w = layer[-1][W]
                ksize_h = layer[-5][1]
                ksize_w = layer[-5][2]
                strids_h = layer[-4][1]
                strids_w = layer[-4][2]
                pad_n = 0
                pad_s = 0
                pad_w = 0
                pad_e = 0
                temp = input_h
                need = (output_h-1)*strids_h + ksize_h
                if need < temp:
                    pad_s = pad_s-(temp - need)
                temp = input_w
                need = (output_w - 1) * strids_w + ksize_w
                if need < temp:
                    pad_e = pad_e-(temp - need)
                padding = [pad_n,pad_s,pad_w,pad_e]
                result[i][-3] = padding
        if type in conv_index:
            padding = layer[-5]

            if padding == 'valid' and layer[-6][1] != 1 and layer[-6][2] != 1:
                input_h = layer[3][H]
                input_w = layer[3][W]
                output_h = layer[-1][H]
                output_w = layer[-1][W]
                dilation_h = layer[5][1]
                dilation_w = layer[5][2]
                ksize_h = (layer[4][0]-1)*dilation_h+1
                ksize_w = (layer[4][1]-1)*dilation_w+1
                strids_h = layer[-6][1]
                strids_w = layer[-6][2]
                pad_n = 0
                pad_s = 0
                pad_w = 0
                pad_e = 0
                temp = input_h
                need = (output_h - 1) * strids_h + ksize_h
                if need < temp:
                    pad_s = pad_s - (temp - need)
                temp = input_w
                need = (output_w - 1) * strids_w + ksize_w
                if need < temp:
                    pad_e = pad_e - (temp - need)
                padding = [pad_n,pad_s,pad_w,pad_e]
                result[i][-5] = padding
    return result
def global_maxpooling_valid(layers,index):
    """
    处理全局池化不满足硬件条件的情况
    :param result:
    :param index:
    :return:
    """
    result = layers.copy()
    i = index
    j = 0
    input_shape = result[i][3]
    layer_name = result[i][2]
    output_name = result[i][-2]
    result[i][2] = result[i][2] + '_step_' + str(j)
    if result[i][4][1]<=16 and result[i][4][1]<=16:
        return result,j
    result[i][4][1] = 6
    result[i][4][2] = 6
    result[i][5] = result[i][4]

    output_h = math.ceil(input_shape[H]/6)
    output_w = math.ceil(input_shape[W]/6)

    result[i][-2] = result[i][-2] + '_step_' + str(j)


    stride_h = result[i][5][1]
    stride_w = result[i][5][2]

    ksize_h = result[i][4][1]
    ksize_w = result[i][4][2]

    pad_needed_height = (output_h - 1) * stride_h + ksize_h
    pad_needed_width = (output_w - 1) * stride_w + ksize_w
    pad_n = 0
    pad_s = pad_needed_height-input_shape[H]
    pad_w = 0
    pad_e = pad_needed_width-input_shape[W]
    padding = [pad_n, pad_s, pad_w, pad_e]
    result[i][-3] = padding
    pool = result[i]
    if C == 3:
        outputs = [1, output_h, output_w, pool[-1][C]]
    else:
        outputs = [1, pool[-1][C], output_h, output_w]
    result[i][-1] = outputs
    j = j + 1
    while (output_h > 1 and output_w > 1):
        # print(11111111111111111)
        if (pool[-1][1] > 16 and pool[-1][2] > 16):
            inputs_name = [pool[-2]]
            type = pool[1]
            input_shape = pool[-1]
            ksize_h = 16
            ksize_w = 16
            stride_h = ksize_h
            stride_w = ksize_w
            output_h = math.ceil(input_shape[H] / 6)
            output_w = math.ceil(input_shape[W] / 6)

            pad_needed_height = (output_h - 1) * stride_h + ksize_h
            pad_needed_width = (output_w - 1) * stride_w + ksize_w
            pad_n = 0
            pad_s = pad_needed_height - input_shape[H]
            pad_w = 0
            pad_e = pad_needed_width - input_shape[W]

            padding = [pad_n,pad_s,pad_w,pad_e]
            if C == 3:
                out = [1,output_h, output_w,pool[-1][C]]
            else:
                out = [1,pool[-1][C],output_h,output_w]

            pool = [inputs_name, type, layer_name + '_step_' + str(j), input_shape, [1, ksize_h, ksize_w, 1],
                    [1, ksize_h, ksize_w, 1], padding, output_name + '_step_' + str(j),
                    out]
            j = j + 1
            result.insert(i + 1, pool)
            i = i+1
        else:
            inputs_name = [pool[-2]]
            type = pool[1]
            input_shape = pool[-1]
            ksize_h = pool[-1][1]
            ksize_w = pool[-1][2]
            output_h = 1
            output_w = 1
            if C == 3:
                out = [1,output_h, output_w,pool[-1][C]]
            else:
                out = [1,pool[-1][C],output_h,output_w]
            pool = [inputs_name, type, layer_name + '_step_' + str(j), input_shape, [1, ksize_h, ksize_w, 1],
                    [1, 1, 1, 1], 'VALID', output_name, out]
            result.insert(i + 1, pool)

    result[i][-1] = outputs

    return result,j
def global_avgpooling_valid(layers,index):
    """
    处理全局池化不满足硬件条件的情况
    :param result:
    :param index:
    :return:
    """
    result = layers.copy()
    i = index
    j = 0
    input_shape = result[i][3]
    layer_name = result[i][2]
    output_name = result[i][-2]
    result[i][2] = result[i][2] + '_step_' + str(j)
    if result[i][4][1]<=16 and result[i][4][1]<=16:
        return result,j
    warnings.warn(
        "Because the kernal_size of this avgpool is large than 16, we don't support of it.So we have to divide this layer into two or more layers(kernal_size<16), The result of this divided operation will cause errors in the results.")
    result[i][4][1] = 6
    result[i][4][2] = 6
    result[i][5] = result[i][4]

    output_h = math.ceil(input_shape[H]/6)
    output_w = math.ceil(input_shape[W]/6)

    result[i][-2] = result[i][-2] + '_step_' + str(j)


    stride_h = result[i][5][1]
    stride_w = result[i][5][2]

    ksize_h = result[i][4][1]
    ksize_w = result[i][4][2]

    pad_needed_height = (output_h - 1) * stride_h + ksize_h
    pad_needed_width = (output_w - 1) * stride_w + ksize_w
    pad_n = 0
    pad_s = pad_needed_height-input_shape[H]
    pad_w = pad_needed_width-input_shape[W]
    pad_e = 0
    padding = [pad_n, pad_s, pad_w, pad_e]
    result[i][-3] = padding
    pool = result[i]
    if C == 3:
        outputs = [1, output_h, output_w, pool[-1][C]]
    else:
        outputs = [1, pool[-1][C], output_h, output_w]
    result[i][-1] = outputs
    j = j + 1
    while (output_h > 1 and output_w > 1):
        # print(11111111111111111)
        if (pool[-1][1] > 16 and pool[-1][2] > 16):
            inputs_name = [pool[-2]]
            type = pool[1]
            input_shape = pool[-1]
            ksize_h = 16
            ksize_w = 16
            stride_h = ksize_h
            stride_w = ksize_w
            output_h = math.ceil(input_shape[H] / 6)
            output_w = math.ceil(input_shape[W] / 6)

            pad_needed_height = (output_h - 1) * stride_h + ksize_h
            pad_needed_width = (output_w - 1) * stride_w + ksize_w
            pad_n = 0
            pad_s = pad_needed_height - input_shape[H]
            pad_w = pad_needed_width - input_shape[W]
            pad_e = 0

            padding = [pad_n,pad_s,pad_w,pad_e]
            if C == 3:
                out = [1,output_h, output_w,pool[-1][C]]
            else:
                out = [1,pool[-1][C],output_h,output_w]

            pool = [inputs_name, type, layer_name + '_step_' + str(j), input_shape, [1, ksize_h, ksize_w, 1],
                    [1, ksize_h, ksize_w, 1], padding, output_name + '_step_' + str(j),
                    out]
            j = j + 1
            result.insert(i + 1, pool)
            i = i+1
        else:
            inputs_name = [pool[-2]]
            type = pool[1]
            input_shape = pool[-1]
            ksize_h = pool[-1][1]
            ksize_w = pool[-1][2]
            output_h = 1
            output_w = 1
            if C == 3:
                out = [1,output_h, output_w,pool[-1][C]]
            else:
                out = [1,pool[-1][C],output_h,output_w]
            pool = [inputs_name, type, layer_name + '_step_' + str(j), input_shape, [1, ksize_h, ksize_w, 1],
                    [1, 1, 1, 1], 'VALID', output_name, out]
            result.insert(i + 1, pool)

    result[i][-1] = outputs

    return result,j
def load_graph_hdf5(model_path):
    model = load_model(model_path)
    return model
def read_graph_from_hdf5(tf_model_path):
    model = load_graph_hdf5(tf_model_path)
    layers = model.layers
    name_shape_dic = name_shape(model, layers)
    for layer in layers:
        layer_type = layer.__class__.__name__
        if layer_type == 'Conv2D':
            if layer.data_format == 'channels_last':
                func_1()
                break
            else:
                func_2()
                break
    operation_type = [layer.__class__.__name__ for layer in layers]

    operation_name = [layer.name for layer in layers]

    find_op_by_input = find_my_input(model)  # 节点与节点输入组成的字典
    find_op_by_output, find_op_by_input = find_my_output(model, find_op_by_input)

    result = operation_data(model, name_shape_dic, find_op_by_input, find_op_by_output)
    layers_type = [v[1] for v in result]
    if 'Cast' in layers_type:
        warnings.warn(
            "There is a cast operation in your net,we don't support of it, this operation will cause errors in the results")
    if 'softmax' in layers_type:
        warnings.warn(
            "There is a softmax operation in your net,we don't support of it,used logsoftmax instead.")

    result = pool_padding(result)  # 处理pool层stride=2时候的pad四个方向上的选择（主要是右边和下边）
    length = len(result)
    i = 0
    while i < length:
        n = 0
        if result[i][1] in pool_index and result[i][-1][1] == 1 and result[i][-1][2] == 1:
            if result[i][-3] == 'valid':
                if result[i][1] == 'GlobalMaxPool2D' or result[i][1] == 'MaxPool2D'or result[i][0] == 'MaxPooling2D':
                    result, n = global_maxpooling_valid(result, i)
                elif result[i][1] == 'AveragePooling2D' or result[i][1] == 'GlobalAveragePooling2D':
                    result, n = global_avgpooling_valid(result, i)
                i = i + n + 1
                length += n
            else:
                i = i + 1
        else:
            i = i + 1
    find_identity_name = find_my_identity(model)


    # 处理identity
    i = 0
    for layer in result:
        name = layer[-2]
        if isinstance(name, str):
            index = True
            while index:
                if name in find_identity_name:
                    name = find_identity_name[name][0]
                    index = True
                else:
                    index = False

        else:
            j = 0
            index = True
            for name_n in name:
                while index:
                    if name_n in find_identity_name:
                        name_n = find_identity_name[name_n][0]
                        name[j] = name_n
                        index = True
                    else:
                        index = False
                        j += 1
                        break
        result[i][-2] = name
        i += 1
    length = len(result)
    i = 0
    while i < length:
        n = 0
        if result[i][1] in pool_index and result[i][-1][1] == 1 and result[i][-1][2] == 1:
            if result[i][-3] == 'valid':
                if result[i][1] == 'GlobalMaxPool2D':
                    result, n = global_maxpooling_valid(result, i)
                elif result[i][1] == 'GlobalAveragePooling2D':
                    result, n = global_avgpooling_valid(result, i)
                i = i + n + 1
                length += n
            else:
                i = i + 1
        else:
            i = i + 1

    for key in layer_w_b:
        layer_w_b[key] = np.array(layer_w_b[key])
        if len(layer_w_b[key].shape) == 4:
            layer_w_b[key] = layer_w_b[key].transpose((C,W,N,H))
    return result,layer_w_b




def func_1():
    """
    由于pb_read.py是被调用文件，存在变量的作用域问题，所以func_1和func_2函数是处理该问题的，没有特殊的作用
    :return:
    """
    global N
    global H
    global W
    global C
    N = 0
    H = 1
    W = 2
    C = 3
def func_2():
    global N
    global H
    global W
    global C
    N = 0
    H = 2
    W = 3
    C = 1
def find_my_identity(model):
    identity_in_out = {}
    layers = model.layers
    ops = []
    for layer in layers:
        if layer.__class__.__name__ == 'Reshape' or layer.__class__.__name__ == 'Dropout':
            ops.append(layer)
    for op in ops:
        name_in = []
        input = op.input
        if isinstance(input,list):
            name_in = input[0].name
        else:
            name_in=input.name
        name_out = []
        output = op.output
        if isinstance(output, list):
            name_out = output[0].name
        else:
            name_out=output.name

        # name_in = [v.name for v in op.input][0]
        # name_out = [v.name for v in op.output][0]
        identity_in_out.setdefault(name_in, []).append(name_out)
    name_dic = identity_in_out.copy()
    for key in identity_in_out:
        name = identity_in_out[key]
        if name[0] in identity_in_out.keys():
            name_dic[key] = name_dic[name[0]]
            name_dic.pop(name[0])
    # for key,name in identity_in_out.items():
    #     if name in identity_in_out.keys():
    #         identity_in_out[key] = identity_in_out[name]
    #         name_dic.pop(name)
    # ops = []
    #
    # for layer in layers:
    #     if layer.__class__.__name__ == 'Lambda':
    #         ops.append(layer)
    # for op in ops:
    #     name_in = []
    #     input = op.input
    #     if isinstance(input,list):
    #         name_in = input[0].name
    #     else:
    #         name_in=input.name
    #     name_out = []
    #     output = op.output
    #     if isinstance(output, list):
    #         name_out = output[0].name
    #     else:
    #         name_out=output.name
    #
    #     # name_in = [v.name for v in op.input][0]
    #     # name_out = [v.name for v in op.output][0]
    #     name_dic.setdefault(name_in, []).append(name_out)



    return name_dic

if __name__ == '__main__':
    model_path = 'C:/Users/75831/Desktop/keras_lenet/weights.best.hdf5'
    model = load_graph_hdf5(model_path)
    layers = model.layers
    name_shape_dic = name_shape(model, layers)
    for layer in layers:
        layer_type = layer.__class__.__name__
        if layer_type == 'Conv2D':
            if layer.data_format == 'channels_last':
                func_1()
                break
            else:
                func_2()
                break
    operation_type = [layer.__class__.__name__  for layer in layers]

    operation_name = [layer.name for layer in layers]


    find_op_by_input = find_my_input(model)  # 节点与节点输入组成的字典
    find_op_by_output, find_op_by_input = find_my_output(model,find_op_by_input)



    result = operation_data(model,name_shape_dic,find_op_by_input,find_op_by_output)
    layers_type = [v[1] for v in result]
    if 'Cast' in layers_type:
        warnings.warn(
            "There is a cast operation in your net,we don't support of it, this operation will cause errors in the results")
    if 'softmax' in layers_type:
        warnings.warn(
            "There is a softmax operation in your net,we don't support of it,used logsoftmax instead.")




    result = pool_padding(result)  # 处理pool层stride=2时候的pad四个方向上的选择（主要是右边和下边）
    length = len(result)
    i = 0
    while i < length:
        n = 0
        if result[i][1] in pool_index and result[i][-1][1] == 1 and result[i][-1][2] == 1:
            if result[i][-3] == 'valid':
                if result[i][1] == 'GlobalMaxPool2D'or result[i][1] == 'MaxPool2D'or result[i][1]=='MaxPooling2D':
                    result, n = global_maxpooling_valid(result, i)
                elif result[i][1] == 'AveragePooling2D'or result[i][1]=='GlobalAveragePooling2D':
                    result, n = global_avgpooling_valid(result, i)
                i = i + n + 1
                length += n
            else:
                i = i + 1
        else:
            i = i + 1

    find_identity_name = find_my_identity(model)


    # 处理identity
    i = 0
    for layer in result:
        name = layer[-2]
        if isinstance(name, str):
            index = True
            while index:
                if name in find_identity_name:
                    name = find_identity_name[name][0]
                    index = True
                else:
                    index = False

        else:
            j = 0
            index = True
            for name_n in name:
                while index:
                    if name_n in find_identity_name:
                        name_n = find_identity_name[name_n][0]
                        name[j] = name_n
                        index = True
                    else:
                        index = False
                        j += 1
                        break
        result[i][-2] = name
        i += 1
    length = len(result)
    i = 0
    while i < length:
        n = 0
        if result[i][1] in pool_index and result[i][-1][1] == 1 and result[i][-1][2] == 1:
            if result[i][-3] == 'valid':
                if result[i][1] == 'GlobalMaxPool2D':
                    result, n = global_maxpooling_valid(result, i)
                elif result[i][1] == 'GlobalAveragePooling2D':
                    result, n = global_avgpooling_valid(result, i)
                i = i + n + 1
                length += n
            else:
                i = i + 1
        else:
            i = i + 1



    # for key in layer_w_b:
    #     layer_w_b[key] = np.array(layer_w_b[key])
    #     if len(layer_w_b[key].shape) == 4:
    #         layer_w_b[key] = layer_w_b[key].transpose((C,W,N,H))
    for layer in result:
        print(layer)
    print(layer_w_b['conv2d_1_weight'])





# for layer in layers:
#     result_temp = []
#     type = layer.__class__.__name__
#     if type in single_layer:
#         result_temp = single_op_layer(layer)







# print(model.__dict__)
# print(model.layers[0].output.name)
# result = []
# result_temp = []
# for layer in layers:
#     print(layer.__class__.__name__)
#     print(len(layer.input_shape))
#     print(layer.input)









# with tf.Session(graph=graph) as sess:
#     ops = graph.get_operations()
#     print(ops)
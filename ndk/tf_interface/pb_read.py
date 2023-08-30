import tensorflow as tf
import numpy as np
import math
import os
import warnings
import time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

name_shape1 = ['Sigmoid', 'Tanh', 'Relu','Relu6','LeakyRelu','Softmax','LogSoftmax','MaxPool','AvgPool','Mean','Max','BiasAdd','ConcatV2','MatMul','DepthwiseConv2dNative','Conv2D','Reshape']
#reshape_delet：后面处理的时候需要用的列表，是针对某个函数设置的
reshape_delet = ['Sigmoid', 'Tanh', 'Relu','Relu6','LeakyRelu','Softmax','LogSoftmax','ConcatV2']
#conv_index卷积操作，matmul虽然是全连接层，但是在硬件上还是按照卷积处理
conv_index = ['MatMul','DepthwiseConv2dNative','Conv2D']
#relu_index:激活函数，暂时没有支持softmax以及logSogtmax，后面处理了再修改
relu_index = ['Sigmoid', 'Tanh', 'Relu','Relu6','LeakyRelu','Softmax','LogSoftmax']
#pool_index:池化操作
pool_index = ['MaxPool','AvgPool','Mean','Max']
#add_index：对张量的操作合并等
add_index = ['Add','ConcatV2','Mul','Neg', 'Maximum']
#slice_index:切片节点，还有split虽然也属于该类，但是处理方式不同，被放到了single_layer里面
slice_index = ['Slice','StridedSlice']
#scale_index:张量与常数的乘除加法
scale_index = ['RealDiv','Sub','Mul','Rsqrt','bias']
#single_layer:所有会被按单层处理的节点
single_layer = ['Sigmoid', 'Tanh', 'Relu','Relu6','Mul','LeakyRelu','Softmax','LeakyRelu','MaxPool','AvgPool','Mean','Max','Add','ConcatV2','Mul','Neg', 'Maximum', 'Placeholder','Split','Reshape','Slice','StridedSlice','LogSoftmax', 'FusedBatchNorm','bias','Squeeze','Pack','Shape','RealDiv','Sub','Fill','Rsqrt','RandomStandardNormal','SplitV']
#all_index：所有该程序能够处理的节点列表
all_index = ['Sigmoid', 'Tanh', 'Relu','Relu6','Mul','LeakyRelu','Softmax','LeakyRelu','MaxPool','AvgPool','Mean','Max','Add','ConcatV2','Mul','Neg', 'Placeholder','Split','Reshape','Slice','StridedSlice','MatMul','DepthwiseConv2dNative',
             'Conv2D','Const','Identity','Squeeze','NoOp', 'BiasAdd', 'FusedBatchNorm', 'Transpose','Pack','Shape','Pad','Maximum','LogSoftmax','Cast','Pack','RealDiv','Sub','Fill','Rsqrt','RandomStandardNormal','SplitV','SpaceToBatchND','BatchToSpaceND','Squeeze']
sess_list = ['RealDiv','Sub','Mul','Rsqrt','bias','Add','Shape','Reshape']

layer_w_b = {}
N = 0
H = 1
W = 2
C = 3
# def name_get_shape(ops):
#     """
#     给网络输入填充全1数据，以获取每一个张量的维数
#     :param ops:
#     :param graph:
#     :return:
#     """
#     # name_list = list()
#     # for op in graph.get_operations():
#     #     # print出tensor的name和值
#     #     name_list.append(op.name)
#
#     opo = None
#     for op in ops:
#         if op.type == 'Placeholder':
#             opo = op
#             break
#     if opo == None:
#         raise Exception("This graph has none of 'Placeholderr' node")
#     # opo = graph.get_operation_by_name('input')
#     op_out_shape = opo.outputs[0].get_shape().as_list()
#     if op_out_shape[0] == None:
#         op_out_shape[0] = 1
#     input_im = np.ones(op_out_shape)
#     name_shape_dic = {}
#     name_shape_dic[opo.outputs[0].name] = list(input_im.shape)
#     for op in ops:
#         inputs = [v for v in op.inputs]
#         for name in inputs:
#             input = name.eval({opo.outputs[0]: input_im})
#             # input = tf.Session().run(name, feed_dict={opo.outputs[0]: input_im})
#             # print(tf.Session().run(ops[5].outputs[0].name))
#             key = name.name
#             value = list(input.shape)
#             name_shape_dic[key] = value
#             tf.Session().close()
#     for op in ops:
#         outputs = [v for v in op.outputs]
#         if op.type in name_shape1 and not outputs[0].name in name_shape_dic:
#             output = outputs[0].eval({opo.outputs[0]: input_im})
#             # output = tf.Session().run(outputs[0], feed_dict={opo.outputs[0]: input_im})
#             key = op.outputs[0].name
#             value = list(output.shape)
#             name_shape_dic[key] = value
#             tf.Session().close()
#     return name_shape_dic
# def const_input_output_op(ops):
#     const_dic = {}
#     input_name_op = {}
#     output_name_op = {}
#     for op in ops:
#         type = op.type
#         inputs_name = [v.name for v in op.inputs]
#         outputs_name = [v.name for v in op.outputs]
#         for name in inputs_name:
#             input_name_op.setdefault(name,[]).append(op)
#         for name in outputs_name:
#             output_name_op.setdefault(name,[]).append(op)
#         if type == 'Const' or type == 'Identity':
#             name = outputs_name[0]
#             value = tf.Session().run(name)
#             const_dic[name] = value
#             tf.Session().close()
#     output = ops[len(ops) - 1].outputs
#     if output == []:
#         name = 'NULL'
#     else:
#         name = ops[len(ops) - 1].outputs[0].name
#     op = []
#     input_name_op[name] = op
#     return const_dic,input_name_op,output_name_op
def name_shape(ops):
    opo = None
    for op in ops:
        if op.type == 'Placeholder':
            opo = op
            break
    if opo == None:
        raise Exception("This graph has none of 'Placeholderr' node")
    op_out_shape = opo.outputs[0].get_shape().as_list()
    if op_out_shape[0] == None:
        op_out_shape[0] = 1
    input_im = np.ones(op_out_shape)
    name_shape_dic = {}
    for op in ops:
        txt = [v.get_shape().as_list() for v in op.inputs]
        names = [v.name for v in op.inputs]
        for i in range(len(txt)):
            shape = txt[i]
            temp = 0
            for j in range(len(shape)):
                n = shape[j]
                if n == None:
                    shape[j] = 1
                    temp+=1
            if temp >= 2:
                shape = list(tf.Session().run(names[i], feed_dict={opo.outputs[0]: input_im}).shape)
            name_shape_dic[names[i]] = shape
    for op in ops:
        outputs = [v for v in op.outputs]
        if op.type in name_shape1 and not outputs[0] in name_shape_dic:
            txt = [v.get_shape().as_list() for v in op.outputs]
            names = [v.name for v in op.outputs]
            for i in range(len(txt)):
                shape = txt[i]
                temp = 0
                for j in range(len(shape)):
                    n = shape[j]
                    if n == None:
                        shape[j] = 1
                        temp += 1
                if temp >= 2:
                    shape = list(tf.Session().run(names[i], feed_dict={opo.outputs[0]: input_im}).shape)
                name_shape_dic[names[i]] = shape
    return name_shape_dic
def deal_with_op(ops):
    for op in ops:
        if not op.type in all_index:
            raise Exception("Dont't support this operation:",op.type)

def get_const_value(ops):
    constant_values = {}
    constant_ops = [op for op in ops if op.type=='Const']
    name_list = list()
    value_list = list()
    for constant_op in constant_ops:
        # b = constant_op.outputs[0].eval()
        # name = [v.name for v in constant_op.outputs][0]
        name_list.append(constant_op.outputs[0])
        # constant_values[name] = b
        # tf.Session().close()
        # print('Saved constant values_1:{}'.format(name))
    value_list = tf.Session().run(name_list)
    tf.Session().close()
    print('Saving constant values..............')
    for i in range(len(name_list)):
        constant_values[name_list[i].name] = value_list[i]
        # print('Saved constant values_1:{}'.format(name_list[i].name))
    constant_ops = [op for op in ops if op.type == 'Identity']
    for constant_op in constant_ops:
        input_name = [v.name for v in constant_op.inputs][0]
        if input_name in constant_values:
            name = [v for v in constant_op.outputs][0]
            name_1 = [v for v in constant_op.inputs][0]
            # value = tf.Session().run(constant_op.outputs)
            # value = name.eval()
            constant_values[name.name] = constant_values[name_1.name]
            # tf.Session().close()
            # print('Saved constant values_2:{}'.format(name.name))
        else:
            continue
    for op in ops:
        if op.type == 'Shape':
            name = op.outputs[0].name
            name_list.append(op.outputs[0])
            shape = op.inputs[0].get_shape().as_list()
            constant_values[name] = shape
    #处理假常数问题例如shape->stidedslice->pack
    ops_not_placeholder = [op for op in ops if op.type != 'Identity' and op.type != 'Const' and op.type != 'Placeholder']
    for op in ops_not_placeholder:
        if op.inputs[0].name in constant_values:
            if op.type == 'StridedSlice':
                constant_values[op.outputs[0].name] = None
    for op in ops_not_placeholder:
        if op.inputs[0].name in constant_values:
            if op.type == 'Pack':
                constant_values[op.outputs[0].name] = []
                print(op.outputs[0].name)
    return constant_values
def get_const_value_1(ops):
    """
    处理所有输入输出，看是不是常量，是常量就保存到字典中
    :param ops:
    :return:
    """
    name_tensor = list()
    const_dic = {}
    name = []
    constant_ops = [op for op in ops if not op.type == 'Const']
    constant_ops = [op for op in constant_ops if not op.type == 'Identity' and op.type in sess_list]
    for op in constant_ops:
        inputs_name = [v for v in op.inputs]
        outputs_name = [v for v in op.outputs]
        name = name + inputs_name + outputs_name
    name = list(set(name))
    for op_name in name:
        try:
            value = op_name.eval()
            if type(value) == np.ndarray:
                const_dic[op_name.name] = value
            # print('Saved constant values_3:{}'.format(op_name.name))
        except:
            continue



    return const_dic


def find_my_input(ops,operation):
    result = {}
    for op in ops:
        inputs_name = [v.name for v in op.inputs]
        for name in inputs_name:
            result.setdefault(name, []).append(op)
    # print(len(ops))
    # print(len(operation))
    # print(ops[len(operation)-1].outputs)
    output = ops[len(operation) - 1].outputs
    if output == []:
        name = 'NULL'
    else:
        name = ops[len(operation) - 1].outputs[0].name
    op = []
    result[name] = op
    return result
def find_my_output(ops,find_op_by_input):
    result = {}
    for op in ops:
        inputs_name = [v.name for v in op.outputs]
        for name in inputs_name:
            result.setdefault(name, []).append(op)
    for key in result:
        if not key in find_op_by_input:
            name = key
            op = []
            find_op_by_input[name] = op
    return result,find_op_by_input
def find_my_identity(ops):
    identity_in_out = {}
    ops = [op for op in ops if op.type == 'Identity' or op.type == 'Cast']
    for op in ops:
        name_in = [v.name for v in op.inputs][0]
        name_out = [v.name for v in op.outputs][0]
        identity_in_out.setdefault(name_in, []).append(name_out)
    return identity_in_out

def reshape_bool(op,find_op_by_output, find_op_by_input):
    """
    该函数是判断Reshape节点是否需要讨论，reshape_index = False表示该节点无用（在网络首部或底部）
    :param op:
    :param find_op_byoutput:
    :param find_op_by_input:
    :return:
    """
    inputs_name = op.inputs[0].name
    outputs_name = op.outputs[0].name
    reshape_index = True#False表示改Reshape无用
    if inputs_name in find_op_by_output:
        in_op = find_op_by_output[inputs_name][0]
        if in_op.type == 'Placeholder':
            reshape_index = False
    if find_op_by_input[outputs_name] == []:
        reshape_index = False
    return reshape_index
def reshape_creat(result_temp):
    reshape_add = [[result_temp[-2]],'Reshape',result_temp[2],result_temp[-1],result_temp[-2],[result_temp[-1][2],result_temp[-1][3]]]
    # print(reshape_add)################################################################################################
    return reshape_add
def reshape_conv(result,reshape_index,reshape_list):
    length = len(result)
    i = 0
    if reshape_index == length - 1:
        return result
    rp_out = result[reshape_index][-2]
    i = reshape_index+1
    length = len(result)
    while(i < length):
        if rp_out == result[i][0][0] and result[i][1] ==  'MatMul_conv':
            reshape = result[reshape_index]
            conv = result[i]
            filter_last = conv[-1][-1]
            input = reshape[3]
            conv[3] = input
            conv[4] = [input[1], input[2], input[3], filter_last]
            conv[0] = reshape[0]

            filter = conv[4]
            w_name = conv[2] + '_weight'
            w = layer_w_b[w_name]
            w = w.reshape(filter)
            layer_w_b[w_name] = w
            ################
            result[i] = conv
            reshape_list.append(i)
        if rp_out == result[i][0][0] and result[i][1] != 'MatMul_conv':
            reshape = result[reshape_index]
            layer_no_mat = result[i]
            input_name = reshape[0]
            layer_no_mat[0] = input_name
            result[i] = layer_no_mat
            reshape_list.append(i)

        i = i+1
    return result,reshape_list


def next_exam(op,find_op_by_input):
    """
    判断该输出是否被多个节点当做输入，如果是，则表示需要在该节点处分层
    :param op:
    :param find_op_by_input:
    :return:
    """
    name = op.outputs[0].name
    temp = 0
    op_next = find_op_by_input[name]
    if  not len(op_next) == 1:
        return False
    else:
        return True

def input_trans(inputs,ops_dic_bias):
    """
    输入是节点的输入名称，输出是张量名称
    :param inputs:
    :param ops_dic_bias:
    :return:
    """
    input_tensor = []
    for name in inputs:
        if not name in ops_dic_bias:
            input_tensor.append(name)
    return input_tensor
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
def padding_data(op,padding,find_op_by_output,name_shape_dic,ops_dic_bias):
    # inputs_name = [v.name for v in op.inputs]
    # outputs_name = [v.name for v in op.outputs]
    # txt = [v.get_shape().as_list() for v in op.inputs]
    # inputs_shape = txt[0]
    strides = op.get_attr('strides')
    padding = bytes.decode(op.get_attr('padding'))
    inputs_name = [v.name for v in op.inputs]
    txt = []

    for name in inputs_name:
        txt.append(name_shape_dic[name])
    if op.type in pool_index:
        filter = op.get_attr('ksize')
    else:
        filter = txt[1]
    inputs_shape = name_shape_dic[inputs_name[0]]
    outputs_name = op.outputs[0].name
    outputs_shape = name_shape_dic[outputs_name]

    input1 = inputs_name[0]
    input1_form_ops = find_op_by_output[input1]
    op_name = [v.type for v in input1_form_ops]
    inputs_shape_1 = []
    layer_name = op.name
    if 'Pad' in op_name:
        if padding == 'SAME':
            raise Exception(
                'The "padding" type of convolution which  operation after "pad" should be "VALID" while your type is "SAME"')
        else:
            inputs_name[0] = [v.name for v in input1_form_ops[0].inputs][0]
            inputs_shape_1 = name_shape_dic[inputs_name[0]]
            if not len(inputs_shape_1) == []:
                inputs_shape = inputs_shape_1
            pad_data = input1_form_ops[0].inputs[1].name
            pad_data = ops_dic_bias[pad_data]
            if not ((pad_data[N] == pad_data[C]).all() and (pad_data[N] == [0, 0]).all()):
                raise Exception('The operation of "pad" cannot be merged with the operation of convolution,'
                                'because the transformed "SAME" operation is different from the "pad" operation in data processing.')
            if (pad_data[H] == pad_data[W]).all() and (pad_data[N][0] == pad_data[N][1]):
                pad_data1 = [pad_data[H][0], pad_data[H][1], pad_data[W][0], pad_data[W][0]]
            if ((pad_data[H] == pad_data[W]).all()) and (pad_data[H][0] == 0) and (pad_data[H][1] == 1):
                pad_data1 = 'SAME'


            res1_1 = math.floor((txt[0][H] - input1_form_ops[0].inputs[0].get_shape().as_list()[H]) / 2)
            res1_2 = math.ceil((txt[0][H] - input1_form_ops[0].inputs[0].get_shape().as_list()[H]) / 2)
            res2_1 = math.floor((txt[0][W] - input1_form_ops[0].inputs[0].get_shape().as_list()[W]) / 2)
            res2_2 = math.ceil((txt[0][W] - input1_form_ops[0].inputs[0].get_shape().as_list()[W]) / 2)
            pad = [res1_1, res1_2, res2_1, res2_2]
            # print(inputs_shape)
            # print(pad)
            cons1 =ops_dic_bias[input1_form_ops[0].inputs[1].name][H]
            cons2 = ops_dic_bias[input1_form_ops[0].inputs[1].name][W]
            # print(cons1, cons2)
            if ([res1_1, res1_2] == cons1).all() and ([res2_1, res2_2] == cons2).all():
                padding = pad_data1
            else:
                raise Exception('The operation of "pad" cannot be merged with the operation of convolution,'
                                'because the transformed "SAME" operation is different from the "pad" operation in data processing.')

            if not padding == 'SAME':

                input_h = inputs_shape[H]
                input_w = inputs_shape[W]
                output_h = outputs_shape[H]
                output_w = outputs_shape[W]
                if op.type in conv_index:
                    dilations = op.get_attr('dilations')
                    dilation_h = dilations[1]
                    dilation_w = dilations[2]
                    ksize_h = (filter[0] - 1) * dilation_h + 1
                    ksize_w = (filter[1] - 1) * dilation_w + 1
                else:
                    ksize_h = filter[1]
                    ksize_w = filter[2]
                strids_h = strides[1]
                strids_w = strides[2]
                pad_n = padding[0]
                pad_s = padding[1]
                pad_w = padding[2]
                pad_e = padding[3]
                h0 = math.floor((pad_n+pad_s+input_h-ksize_h)/strids_h) +1


                need = pad_n+pad_s+input_h
                temp = (h0 - 1)*strids_h + ksize_h

                if temp == need:
                    pass
                elif need > temp:
                    n = need-temp
                    pad_s = pad_s - n
                w0 = math.floor((pad_w + pad_e + input_w - ksize_w) / strids_w) + 1
                need = pad_w + pad_e + input_w
                temp = (w0 - 1) * strids_w + ksize_w
                # if layer_name ==  'conv2d_1/Conv2D':
                #     print(w0)
                #     print(strids_w)
                #     print(ksize_w)
                #     print(padding)
                #     print(need)
                #     print(temp)
                if temp == need:
                    pass
                elif  need > temp:
                    n = need-temp
                    pad_e = pad_e - n
                padding = [pad_n,pad_s,pad_w,pad_e]
            if op.type in pool_index:
                warnings.warn("Because there is a 'Pad' node(we don't support it),which at the top of AvgPool/MaxPool node,so our program has to merge the two nodes(Pad&pool).The result of this merge operation will cause errors in the results.")
    return padding,inputs_shape
def layer_index(ops,ops_dic_bias,find_op_by_output,find_op_by_input):
    result = []
    op_use = []
    #处理单独层的情况
    for op in ops:
        result_temp = []
        op_type = op.type
        if op_type in single_layer:
            if not op_type == 'Add' and not op_type == 'Neg' and not op_type == 'Reshape' and not op.type=='Shape'and not op.type=='Pack'  and not op.type == 'Squeeze' and not op.type ==  'FusedBatchNorm' and not op_type in slice_index:
                result_temp.append(op_type)
                result_temp.append(op)
                result.append(result_temp)
            elif op_type == 'Squeeze':
                inputs_name = [v.name for v in op.inputs][0]
                outputs_name = [v.name for v in op.outputs][0]
                op_next_list = find_op_by_input[outputs_name]
                for op_1 in op_next_list:
                    op_1_inputs = [v.name for v in op_1.inputs]
                    if outputs_name in op_1_inputs and op_1.type == 'Reshape':
                        op_next = op_1
                        result_temp.append(op_type)
                        result_temp.append(op)
                        result_temp.append(op_next)
                        result.append(result_temp)

                if len(op_next_list) == 1 and not op_next_list[0].type == 'Reshape':
                    result_temp.append(op_type)
                    result_temp.append(op)
                    result.append(result_temp)
            elif op_type == 'Add':
                #处理的是两个张量相加的Add节点，如果有一个张量加上一个常量，并且不能被合并到卷积层，该节点会在本函数最后处理
                temp = 0
                inputs_name = [v.name for v in op.inputs]
                for add_input in inputs_name:
                    if not add_input in ops_dic_bias:
                        temp += 1
                if temp >= 2:
                    result_temp.append(op_type)
                    result_temp.append(op)
                    result.append(result_temp)
            # elif op_type == 'Neg':
            #     inputs_name = [v.name for v in op.inputs][0]
            #     input_to_op = find_op_by_input[inputs_name]
            #     num = len(input_to_op)
            #     if inputs_name[0] in ops_dic_bias:
            #         pass
            #     elif num == 1:
            #         pass
            #     else:
            #         result_temp.append(op_type)
            #         result_temp.append(op)
            #         result.append(result_temp)
            elif op_type == 'Pack':
                inputs_name = [v.name for v in op.inputs]
                txt = [v.get_shape().as_list() for v in op.inputs]
                try:
                    output_name = [v.name for v in op.outputs][0]
                    op_next = find_op_by_input[output_name]  # 处理shape后面有多个slice情况
                    op_next_1 = find_op_by_input[output_name][0]
                    op_next_type = op_next_1.type
                    next1_inputs = [v.name for v in op_next_1.inputs]
                    next1_outputs = [v.name for v in op_next_1.outputs]
                    op_next_2 = find_op_by_input[next1_outputs[0]][0]
                    op_next2_type = op_next_2.type
                    if len(txt[0]) == 4 and op_next_type == 'Transpose' and op_next2_type == 'Reshape':
                        result_temp.append(op_type)
                        result_temp.append(op)
                        result_temp.append(op_next_1)
                        result_temp.append(op_next_2)
                        result.append(result_temp)
                    elif len(txt[0]) == 4 and not op_next_type == 'Transpose':
                        result_temp.append(op_type)
                        result_temp.append(op)
                        result.append(result_temp)
                except:
                    pass
            elif op_type == 'Shape':
                #暂时只遇到了两种shape节点，一种是可以合并为shufflechannal的，一种是mobilenet_v1.pb出现在网络最后shape->reshape
                try:
                    #可以合并的shape
                    output_name = [v.name for v in op.outputs][0]
                    op_next = find_op_by_input[output_name]  # 处理shape后面有多个slice情况
                    op_next_1 = find_op_by_input[output_name][0]
                    op_next_type = op_next_1.type
                    next1_inputs = [v.name for v in op_next_1.inputs]
                    next1_outputs = [v.name for v in op_next_1.outputs]
                    op_next_2 = find_op_by_input[next1_outputs[0]][0]
                    op_next2_type = op_next_2.type
                    if op_next_type == 'StridedSlice' and op_next2_type == 'Pack':
                        result_temp.append(op.type)
                        result_temp.append(op)
                        result_temp.append(op_next)
                        result_temp.append(op_next_2)
                        result.append(result_temp)
                except:
                    pass

            elif op_type == 'Reshape':#处理reshape->Transpose->reshape
                inputs_sh = op.inputs[0].get_shape().as_list()
                outputs_sh = op.outputs[0].get_shape().as_list()
                output_name = [v.name for v in op.outputs][0]
                reshape_index = reshape_bool(op,find_op_by_output,find_op_by_input)#判断reshape节点是否可以被舍弃
                if reshape_index == False:
                    #如果出现了reshape是处理placeholder的或者是网络最后一个节点是reshape，则跳过
                    continue
                op_next_1 = find_op_by_input[output_name][0]
                op_next_type = op_next_1.type
                next1_inputs = [v.name for v in op_next_1.inputs]
                next1_outputs = [v.name for v in op_next_1.outputs]
                op_next_2 = find_op_by_input[next1_outputs[0]][0]
                op_next2_type = op_next_2.type
                if op_next_type == 'Transpose' and op_next2_type == 'Reshape':
                    result_temp.append(op.type)
                    result_temp.append(op)
                    result_temp.append(op_next_1)
                    result_temp.append(op_next_2)
                    result.append(result_temp)
                elif op_next_type == 'MatMul':
                    result_temp.append(op.type)
                    result_temp.append(op)
                    result.append(result_temp)
                elif len(inputs_sh) == len(outputs_sh):
                    #例如将[3,4]改变成了[2,6]，必须报错
                    for i in range(len(inputs_sh)):
                        if inputs_sh[i] == outputs_sh[i]:
                            pass
                        else:
                            raise Exception("Reshape op has change the inputs_shape: {} into {}.".format(inputs_sh,outputs_sh))
                    continue
                else:
                    continue
            elif op.type in slice_index:
                try:
                    outputs_name = [v.name for v in op.outputs][0]
                    next1 = find_op_by_input[outputs_name][0]
                    next1_out = [v.name for v in next1.outputs][0]
                    next2 = find_op_by_input[next1_out][0]
                    if next1.type == 'Pack' and next2.type == 'Reshape':
                        continue
                    else:
                        result_temp.append(op.type)
                        result_temp.append(op)
                        result.append(result_temp)
                except:
                    result_temp.append(op.type)
                    result_temp.append(op)
                    result.append(result_temp)
            continue

        # 'SpaceToBatchND'以及'BatchToSpaceND'是由于空洞卷积（tf.keras/tf.layers）固化网络文件时产生的.其他情况下都不会支持该操作。

        elif op_type == 'SpaceToBatchND':
            output_name = [v.name for v in op.outputs][0]
            if output_name in find_op_by_input and find_op_by_input[output_name] != []:
                op_next_1 = find_op_by_input[output_name][0]
                if not op_next_1.type == 'Conv2D' and not op_next_1.type == 'DepthwiseConv2dNative':
                    raise Exception(
                        "We don't support this operation ('SpaceToBatchND'),opration name is:{}".format(op.name))
        elif op_type == 'BatchToSpaceND':
            input_name = [v.name for v in op.inputs][0]
            if input_name in find_op_by_output and find_op_by_output[input_name] != []:
                op_next_1 = find_op_by_output[input_name][0]
                if not op_next_1.type == 'Conv2D' and not op_next_1.type == 'DepthwiseConv2dNative':
                    raise Exception(
                        "We don't support this operation ('SpaceToBatchND'),opration name is:{}".format(op.name))

    for  op in ops:
        result_temp = []
        op_type = op.type
        if op_type in conv_index:
            result_temp.append(op.type)
            result_temp.append(op)
            output_name = [v.name for v in op.outputs][0]
            # print(output_name)################################################################################################################
            # next_index = next_exam(op,find_op_by_input)
            if output_name in find_op_by_input and find_op_by_input[output_name] != []:
                op_next_1 = find_op_by_input[output_name][0]
                if op_next_1.type == 'BatchToSpaceND':
                    next_output = [v.name for v in op_next_1.outputs][0]
                    op_next_1 = find_op_by_input[next_output][0]
                    next1_inputs = [v.name for v in op_next_1.inputs]
                    next1_outputs = [v.name for v in op_next_1.outputs]
                    op_next_type = op_next_1.type
                else:
                    next1_inputs = [v.name for v in op_next_1.inputs]
                    next1_outputs = [v.name for v in op_next_1.outputs]
                    op_next_type = op_next_1.type
            else:
                result.append(result_temp)
                continue

            #检查Add
            if op_next_type == 'Add' or op_next_type == 'BiasAdd':
                if op_next_type == 'Add':
                    temp = 0
                    for add_input in next1_inputs:
                        if not add_input in ops_dic_bias:
                            temp += 1
                    if temp < 2:
                        result_temp.append(op_next_1)
                    else:
                        result.append(result_temp)
                        continue
                elif op_next_type == 'BiasAdd':
                    result_temp.append(op_next_1)
                if not next_exam(op_next_1,find_op_by_input):#如果在conv->baisadd后面有多个分支，则没有必要进行下一个节点的寻找，包括neg也一样
                    result.append(result_temp)
                    continue
                op_next_2 = find_op_by_input[next1_outputs[0]][0]
                op_next2_type = op_next_2.type
                if op_next2_type ==  'FusedBatchNorm':
                    # print('******************')###################################################################
                    result_temp.append(op_next_2)
                    result.append(result_temp)
                    continue
                elif op_next2_type == 'Neg':
                    result_temp.append(op_next_2)
                    result.append(result_temp)
                    continue
                else:
                    result.append(result_temp)
                    continue
            elif op_next_type == 'FusedBatchNorm':
                # print('**********')####################################################################################
                result_temp.append(op_next_1)
                result.append(result_temp)
                continue
            elif op_next_type in single_layer or op_next_type in conv_index:
                result.append(result_temp)
                continue
            else:
                raise Exception("This {} can't deal,while the name of the layer is {}".format(op.type,op.name))
        continue
    op_use = result[0]
    for layer in result[1:]:
        op_use = op_use + layer
    for op in ops:
        result_temp = []
        temp = 0
        if op.type == 'Add':
            inputs_name = [v.name for v in op.inputs]
            for add_input in inputs_name:
                if not add_input in ops_dic_bias:
                    temp += 1
            if temp >= 1 and not op in op_use:
                result_temp.append('bias')
                result_temp.append(op)
                result.append(result_temp)
            elif temp == 0:
                result_temp.append('bias')
                result_temp.append(op)
                result.append(result_temp)
        if op.type == 'FusedBatchNorm':
            if not op in op_use:
                result_temp.append(op.type)
                result_temp.append(op)
                result.append(result_temp)
        #因为部分网络有reshape->softmax或者softmax->reshape这样的节点，所以reshape不能全部被舍弃。
        #网络最后一个节点是reshape，这种reshape直接忽略就可以了，不然会在程序中报错

        if op.type == 'Reshape' and op != ops[len(ops)-1]:
            if not op in op_use:
                result_temp.append(op.type)
                result_temp.append(op)
                result.append(result_temp)
    for op in ops:
        result_temp = []
        if op.type == 'Neg':
            if not op.inputs[0].name in ops_dic_bias and not op in op_use:
                result_temp.append(op.type)
                result_temp.append(op)
                result.append(result_temp)

    return result
def layer_batchnorm_out(layer_op):
    layers = layer_op.copy()
    result = layer_op.copy()
    for i in range(len(layers)):
        layer = layers[i]
        if layer[0] in conv_index:
            op = layer[-1]
            if op.type == 'FusedBatchNorm':
                temp = [op.type,op]
                layer.pop()
                result[i] = layer
                result.append(temp)
    return result


def conv_defeat(layer,find_op_by_output,find_op_by_input,ops_dic_bias,name_shape_dic):
    """
    处理卷积
    :param layer:
    :param find_op_by_output:
    :param find_op_by_input:
    :param ops_dic_bias:
    :param name_shape_dic:
    :return:
    """
    result = []
    op = layer[1]
    ops = layer[1:]
    op_num = len(ops)#该层节点数
    # print(op_num)#################################################################################################
    layer_name = op.name
    layer_type = layer[0]
    inputs_name = [v.name for v in op.inputs]#存储该层所有的输入
    inputs_tensor = input_trans(inputs_name, ops_dic_bias)#存储该层所有的tensor输入
    txt = []
    for name in inputs_name:
        txt.append(name_shape_dic[name])
    inputs_shape = name_shape_dic[inputs_name[0]]  # 默认输入第一个是张量，如果不是，需要在后面对应的层中修改即可
    outputs_name = [v.name for v in op.outputs]
    outputs_shape = name_shape_dic[outputs_name[0]]#一般的节点输出只有一个，故选择列表中第一个当作该层输出
    ######################
    result.append(inputs_tensor)  # 输入张量
    result.append(layer_type)  # 层的类型
    result.append(layer_name)  # 层的名称
    #######################
    w = 0
    bias = False
    beta = 0
    n = 0
    while (n<op_num):
        op = ops[n]
        layer_type = op.type
        if layer_type == 'Conv2D' or layer_type == 'DepthwiseConv2dNative':
            ###################################################
            # 处理conv前面有Reshape（Mnist.pb）或者有‘Identity
            input1 = inputs_name[0]
            input1_form_ops = find_op_by_output[input1]
            op_name = [v.type for v in input1_form_ops]
            if op_name[0] == 'Reshape' or op_name[0] == 'Pad' or op_name[0] == 'SpaceToBatchND':
                inputs_name[0] = input1_form_ops[0].inputs[0].name
                result[0] = [inputs_name[0]]############针对输入name的修改
            ############################################
            filter = txt[1]
            dilations = op.get_attr('dilations')
            strides = op.get_attr('strides')
            padding = bytes.decode(op.get_attr('padding'))
            padding, inputs_shape = padding_data(op, padding, find_op_by_output,name_shape_dic,ops_dic_bias)
            w = ops_dic_bias[inputs_name[1]]  # 这里也是假设卷积里面的输入都是按照[tensor,const]形式保存的，不然需要先判断谁是const
            # 保存权重到字典中
            w_name = layer_name + '_weight'
            layer_w_b[w_name] = w
            outputs_name = outputs_name[0]
            #########################
            if op_name[0] == 'SpaceToBatchND':
                inputs_shape = name_shape_dic[inputs_name[0]]
                dilations = ops_dic_bias[input1_form_ops[0].inputs[1].name]
                # print(dilations)
                dilations = [1,int(dilations[0]),int(dilations[1]),1]
            result.append(inputs_shape)
            result.append(filter)
            result.append(dilations)
            result.append(strides)
            result.append(padding)
        elif layer_type == 'Add' or layer_type == 'BiasAdd':
            add_inputs = [v.name for v in op.inputs]
            bias = True
            beta = ops_dic_bias[add_inputs[1]]  # 这里假设的是Add输入形式是[tensor,const]
            ##添加本层偏置
            b_name = layer_name + '_bias'
            layer_w_b[b_name] = beta
            outputs_name = [v.name for v in op.outputs][0]  # batch_normal输出有四个
            outputs_shape = name_shape_dic[outputs_name]
        elif layer_type == 'FusedBatchNorm':
            bias = True
            bn_inputs = [v.name for v in op.inputs]
            gamma = ops_dic_bias[bn_inputs[1]]
            beta = ops_dic_bias[bn_inputs[2]]
            mean = ops_dic_bias[bn_inputs[3]]
            variance = ops_dic_bias[bn_inputs[4]]
            epsilon = op.get_attr('epsilon')
            beta = beta - gamma * mean / np.sqrt(variance + epsilon)
            if result[1] == 'Conv2D':
                w = w * gamma / np.sqrt(variance + epsilon)
            else:
                w = tf.squeeze(w, [C])
                w = w * gamma / np.sqrt(variance + epsilon)
                w = tf.expand_dims(w, [C])
                w = tf.Session().run(w)
                tf.Session().close()
            ##权重和偏置保存到字典中
            layer_name = result[2]
            w_name = layer_name + '_weight'
            layer_w_b[w_name] = w
            b_name = layer_name + '_bias'
            layer_w_b[b_name] = beta
            ##########################
            outputs_name = [v.name for v in op.outputs][0] # batch_normal输出有四个
            outputs_shape = name_shape_dic[outputs_name]
        elif layer_type == 'Neg':
            w = w*-1
            beta = beta*-1
            outputs_name = [v.name for v in op.outputs][0]  # batch_normal输出有四个
            outputs_shape = name_shape_dic[outputs_name]
        n = n+1
    op = layer[1]
    if op.type == 'DepthwiseConv2dNative':
        key = layer_name+'_weight'
        layer_w_b[key] = np.array(layer_w_b[key])
        shape = layer_w_b[key].shape
        shape = list(shape)
        if shape[-1] != 1:
            shape[-2] = shape[-2]*shape[-1]
            shape[-1] = 1
            layer_w_b[key] = layer_w_b[key].reshape(shape)
        layer_w_b[key] = layer_w_b[key].transpose((0, 1, 3, 2))
    if outputs_shape[H] == math.ceil(inputs_shape[H]/strides[1]) and op_name[0] == 'SpaceToBatchND':
        result[-1] = 'SAME'



    result.append('w')
    result.append([bias,'beta'])
    result.append(outputs_name)
    result.append(outputs_shape)
    return result
def matmul_data(layer,ops_dic_bias,find_op_by_output,find_op_by_input,name_shape_dic):
    """
    处理matmul
    :param layer:
    :param ops_dic_bias:
    :param find_op_by_output:
    :param find_op_by_input:
    :param name_shape_dic:
    :return:
    """
    result = []
    op = layer[1]
    ops = layer[1:]
    op_num = len(ops)  # 该层节点数
    layer_name = op.name
    layer_name = layer_name.replace('MatMul', 'dense')
    layer_type = layer[0]
    inputs_name = [v.name for v in op.inputs]  # 存储该层所有的输入
    txt = []
    for name in inputs_name:
        txt.append(name_shape_dic[name])
    inputs_shape = name_shape_dic[inputs_name[0]]  # 默认输入第一个是张量，如果不是，需要在后面对应的层中修改即可
    outputs_name = [v.name for v in op.outputs][0]
    outputs_shape = name_shape_dic[outputs_name]  # 一般的节点输出只有一个，故选择列表中第一个当作该层输出
    ###################################################
    inputs_tensor = input_trans(inputs_name, ops_dic_bias)  # 存储该层所有的tensor输入
    result.append(inputs_tensor)  # 输入张量
    result.append(layer_type)  # 层的类型
    result.append(layer_name)  # 层的名称
    ######################
    input1 = inputs_name[0]
    input1_form_ops = find_op_by_output[input1]
    op_name = [v.type for v in input1_form_ops]
    # if op_name[0] == 'Reshape' :
    #     inputs_name[0] = input1_form_ops[0].inputs[0].name
    #     result[0] = [inputs_name[0]]  ############针对输入name的修改
    # ######################
    w = 0
    bias = False
    beta = 0
    n = 0
    while (n < op_num):
        op = ops[n]
        layer_type = op.type
        if layer_type == 'MatMul':
            filter = []
            # print(filter)
            result[1] = 'MatMul_conv'
            result.append(inputs_shape)
            result.append(filter)
            result.append([1,1,1,1])
            result.append([1,1,1,1])
            result.append('VALID')
            #处理权重
            w_shape = filter
            w = ops_dic_bias[inputs_name[1]]
            ##权重和偏置保存到字典中
            w_name = layer_name + '_weight'
            layer_w_b[w_name] = w
            #########################
        elif layer_type == 'Add' or layer_type == 'BiasAdd':
            add_inputs = [v.name for v in op.inputs]
            bias = True
            beta = ops_dic_bias[add_inputs[1]]  # 这里假设的是Add输入形式是[tensor,const]
            ##添加本层偏置
            b_name = layer_name + '_bias'
            layer_w_b[b_name] = beta
            outputs_name = [v.name for v in op.outputs][0]  # batch_normal输出有四个
            outputs_shape = name_shape_dic[outputs_name]
        elif layer_type == 'FusedBatchNorm':
            bias = True
            bn_inputs = [v.name for v in op.inputs]
            gamma = ops_dic_bias[bn_inputs[1]]
            beta = ops_dic_bias[bn_inputs[2]]
            mean = ops_dic_bias[bn_inputs[3]]
            variance = ops_dic_bias[bn_inputs[4]]
            epsilon = op.get_attr('epsilon')
            beta = beta - gamma * mean / np.sqrt(variance + epsilon)
            if result[1] == 'Conv2D':
                w = w * gamma / np.sqrt(variance + epsilon)
            else:
                w = tf.squeeze(w, [C])
                w = w * gamma / np.sqrt(variance + epsilon)
                w = tf.expand_dims(w, [C])
                w = tf.Session().run(w)
                tf.Session().close()
            ##权重和偏置保存到字典中
            layer_name = result[2]
            w_name = layer_name + '_weight'
            layer_w_b[w_name] = w
            b_name = layer_name + '_bias'
            layer_w_b[b_name] = beta
            ##########################
            outputs_name = [v.name for v in op.outputs][0]  # batch_normal输出有四个
            outputs_shape = name_shape_dic[outputs_name]
        elif layer_type == 'Neg':
            w = w * -1
            beta = beta * -1
            outputs_name = [v.name for v in op.outputs][0]  # batch_normal输出有四个
            outputs_shape = name_shape_dic[outputs_name]

        n = n + 1
    result.append('w')
    result.append([bias, 'beta'])
    result.append(outputs_name)
    result.append(outputs_shape)
    return result
def layer_connect(result,operaion_name):
    """
    处理未被排序的result，并且寻找二维输入和输出，进行“改造”
    :param result:
    :param operaion_name:
    :return:
    """
    layer_result = []
    for name in operaion_name:
        for layer in result:
            if layer[2] == name:
                layer_result.append(layer)
            else:
                continue
    for i in range(len(layer_result)):
        layer = layer_result[i]
        layer_type = layer[1]
        output_shape = layer[-1]
        if layer_type in relu_index and not layer_type == 'ConcatV2':
            input_shape = layer[3]
            output_shape = layer[-1]
            if len(input_shape)==2:
                input_shape.insert(0,1)
                input_shape.insert(0,1)
                layer_result[i][3] = input_shape
            if len(output_shape) == 2:
                output_shape.insert(0,1)
                output_shape.insert(0,1)
                layer_result[i][-1] = output_shape
        if layer_type == 'ConcatV2':
            if len(output_shape) == 2:
                output_shape.insert(0,1)
                output_shape.insert(0,1)
                layer_result[i][-1] = output_shape
    return layer_result

def operation_data(layer,ops_dic_bias,find_op_by_input,find_op_by_output,name_shape_dic):
    """
    处理layer_index处理好的分层列表，返回详细的每一层参数
    :param layer:
    :param ops_dic_bias:
    :param find_op_by_input:
    :param find_op_by_output:
    :param name_shape_dic:
    :return:
    """
    result = []
    layer_type = layer[0]
    op = layer[1]
    if layer_type in single_layer:
        inputs_name = [v.name for v in op.inputs]
        inputs_tensor = input_trans(inputs_name, ops_dic_bias)
        txt = [v.get_shape().as_list() for v in op.inputs]
        inputs_shape = [v.get_shape().as_list() for v in op.inputs]
        if layer_type == 'Placeholder':
            inputs_shape = [v.get_shape().as_list() for v in op.inputs]
        else:
            i = 0
            for name in inputs_name:
                if not name in ops_dic_bias:
                    inputs_shape = txt[i]
                i = i+1
            # inputs_shape = [v.get_shape().as_list() for v in op.inputs][0]  # 默认输入第一个是张量，如果不是，需要在后面对应的层中修改即可
            if len(inputs_shape) == 0:
                pass
            else:
                if inputs_shape[0] == None:
                    inputs_shape[0] = 1

        outputs_name = [v.name for v in op.outputs][0]
        outputs_shape = [v.get_shape().as_list() for v in op.outputs][0]#一般的节点输出只有一个，故选择列表中第一个当作该层输出
        if not outputs_shape == [] and outputs_shape[N] == None:
            outputs_shape[N] = 1
        layer_name = op.name
        #当节点前面是Identity节点
        # if len(inputs_tensor) >=1 and inputs_tensor[0] in find_op_by_output:
        #     next_op = find_op_by_output[inputs_tensor[0]][0]
        #     if next_op.type == 'Identity':
        #         inputs_tensor[0] = next_op.inputs[0].name
        ######################
        result.append(inputs_tensor)  # 输入张量
        result.append(layer_type)  # 层的类型
        result.append(layer_name)  # 层的名称
        #######################



        if layer_type == 'Placeholder':
            if outputs_name in find_op_by_input:
                next_op = find_op_by_input[outputs_name][0]
                next_type = next_op.type
                if next_type == 'Reshape':
                    outputs_shape = next_op.outputs[0].get_shape().as_list()
                    if outputs_shape[N] == None:
                        outputs_shape[N] = 1
            if len(outputs_shape) == 2:
                outputs_shape.insert(0,1)
                outputs_shape.insert(0,1)
            result.append(outputs_name)
            result.append(outputs_shape)
            return result
        elif layer_type == 'Sub':
            # inputs_shape = [v.get_shape().as_list() for v in op.inputs][0]
            inputs_name = [v.name for v in op.inputs]
            inputs_shape = name_shape_dic[inputs_name[0]]
            outputs_name = op.outputs[0].name
            outputs_shape = name_shape_dic[outputs_name]
            index = False#如果是TRUE则需要说明是const-tensor,需要分两层
            if len(inputs_name) == 2:
                if inputs_name[0] in ops_dic_bias and not inputs_tensor == []:#判据inputs_tensor == []表示是两个常数相加
                    const = ops_dic_bias[inputs_name[0]]
                    inputs_shape = name_shape_dic[inputs_name[1]]
                    index = True
                elif inputs_name [1] in ops_dic_bias or inputs_tensor == []:
                    const = ops_dic_bias[inputs_name[1]]
                    index = False
                else:
                    raise Exception("This sub operation's input in two subtraction, sub operation name is :{}".format(op.name))
            else:
                const = ops_dic_bias[op.inputs[0].name]
            ##权重和偏置保存到字典中
            layer_name = layer[1].name
            if layer_type == 'Sub':
                b_name = layer_name + '_bias'
                layer_w_b[b_name] = const*-1
                temp = [True,'b']
            if index == True:
                result = []
                result_temp = []
                result_temp.append([inputs_name[1]])
                result_temp.append('Mul')
                layer_name = op.name
                layer_w_b[layer_name+'_weight'] = -1
                result_temp.append(op.name)
                result_temp.append(inputs_shape)
                result_temp.append([False,[]])
                result_temp.append(outputs_name+'_mul')
                result_temp.append(outputs_shape)
                result.append(result_temp)
                result_temp = []
                result_temp.append([outputs_name+'_mul'])
                result_temp.append('bias')
                result_temp.append(op.name)
                result_temp.append(inputs_shape)
                result_temp.append([True,'b'])
                b_name = layer_name + '_bias'
                layer_w_b[b_name] = const * 1
                result_temp.append(outputs_name)
                result_temp.append(outputs_shape)
                result.append(result_temp)
                result.append(1)
                result.append(1)
                # print(result)
                return result
            else:
                result.append(inputs_shape)
                result.append(temp)
                result.append(outputs_name)
                result.append(outputs_shape)
                return result
        elif layer_type == 'RealDiv':
            # inputs_shape = [v.get_shape().as_list() for v in op.inputs][0]
            inputs_name = [v.name for v in op.inputs]
            inputs_shape = name_shape_dic[inputs_name[0]]
            outputs_name = op.outputs[0].name
            outputs_shape = name_shape_dic[outputs_name]
            index = False
            if len(inputs_name) == 2:
                if inputs_name[0] in ops_dic_bias and not inputs_tensor == []:#判据inputs_tensor == []表示是两个常数相加
                    const = ops_dic_bias[inputs_name[0]]
                    inputs_shape = name_shape_dic[inputs_name[1]]
                    index = True
                elif inputs_name [1] in ops_dic_bias or inputs_tensor == []:
                    const = ops_dic_bias[inputs_name[1]]
                    index = False
            else:
                const = ops_dic_bias[op.inputs[0].name]
            ##权重和偏置保存到字典中
            layer_name = layer[1].name
            if layer_type == 'RealDiv':
                w_name = layer_name + '_weight'
                layer_w_b[w_name] = 1/const
                temp = [False,[]]
            if index == True:
                raise Exception("Don't support this RealDiv(const/tensor),this operation name is:{}".format(op.name))
            else:
                result.append(inputs_shape)
                result.append(temp)
                result.append(outputs_name)
                result.append(outputs_shape)
                return result
        elif layer_type == 'Rsqrt':
            # inputs_shape = [v.get_shape().as_list() for v in op.inputs][0]
            inputs_name = [v.name for v in op.inputs]
            inputs_shape = name_shape_dic[inputs_name[0]]
            outputs_name = op.outputs[0].name
            outputs_shape = name_shape_dic[outputs_name]
            if not inputs_tensor == []:
                raise Exception("Don't support this RealDiv(sqrt(tensor)),this operation name is: {}".format(op.name))
            if len(inputs_name) == 2:
                for name in inputs_name:
                    if name in ops_dic_bias:
                        const = ops_dic_bias[name]
            else:
                if op.inputs[0].name in ops_dic_bias:
                    const = ops_dic_bias[op.inputs[0].name]
            ##权重和偏置保存到字典中
            layer_name = layer[1].name
            if layer_type == 'Rsqrt':
                b_name = layer_name + 'weight'
                layer_w_b[b_name] = const
                temp = [False,[]]
            result.append(inputs_shape)
            result.append(temp)
            result.append(outputs_name)
            result.append(outputs_shape)
            return result
        elif layer_type == 'Shape':
            # inputs_shape = [v.get_shape().as_list() for v in op.inputs][0]
            inputs_name = op.inputs[0].name
            inputs_shape = name_shape_dic[inputs_name]
            op_last = layer[-1]
            outputs_name = [v.name for v in op_last.outputs][0]
            outputs_shape = name_shape_dic[outputs_name]
            # outputs_shape = op_last.outputs[0].get_shape().as_list()
            result.append(inputs_shape)
            result.append(outputs_name)
            result.append(outputs_shape)
            result.append(1)
            return result
        elif layer_type == 'Fill' or layer_type =='RandomStandardNormal':
            inputs_name = op.inputs[0].name
            inputs_shape = name_shape_dic[inputs_name]
            op_last = layer[-1]
            outputs_name = [v.name for v in op_last.outputs][0]
            outputs_shape = name_shape_dic[outputs_name]
            # outputs_shape = op_last.outputs[0].get_shape().as_list()
            result.append(inputs_shape)
            result.append(outputs_name)
            result.append(outputs_shape)
            return result
        elif layer_type == 'Pack':
            if len(layer) == 4:
                #['Pack',pack_op,transpose_op,reshape_op]
                result_temp = []
                result = []
                inputs_name = [v.name for v in op.inputs]
                outputs_name = op.outputs[0].name
                outputs_shape = name_shape_dic[outputs_name]
                result_temp.append(inputs_name)
                result_temp.append('ConcatV2')
                result_temp.append(op.name)
                for name in inputs_name:
                    result_temp.append(name_shape_dic[name])
                input_sh = name_shape_dic[inputs_name[0]]
                for i in range(len(input_sh)):
                    if input_sh[i] == outputs_shape[i]:
                        pass
                    else:
                        axis = i
                        if axis == C and C == 3:
                            axis = 1
                        elif axis == C and C == 1:
                            pass
                        else:
                            raise Exception("We can't support this axis,because C = {},but axis = {}".format(C,axis))
                        break
                group = outputs_shape[C]
                result_temp.append(axis)
                result_temp.append(outputs_name)
                result_temp.append(outputs_shape)
                result.append(result_temp)
                result_temp = []
                op = layer[2]
                inputs_name = [v.name for v in op.inputs]
                outputs_name = op.outputs[0].name
                outputs_shape = name_shape_dic[outputs_name]
                inputs_tensor = input_trans(inputs_name, ops_dic_bias)
                result_temp.append(inputs_tensor)
                result_temp.append('shufflechannel')
                result_temp.append(op.name)
                for name in inputs_name:
                    result_temp.append(name_shape_dic[name])
                input_sh = name_shape_dic[inputs_name[0]]
                op_output = layer[3]
                outputs_name = op_output.outputs[0].name
                outputs_shape = name_shape_dic[outputs_name]
                # for i in range(len(input_sh)):
                #     if input_sh[i] == outputs_shape[i]:
                #         pass
                #     else:
                #         axis = i
                #         if axis == C and C == 3:
                #             axis = 1
                #         elif axis == C and C == 1:
                #             pass
                #         else:
                #             raise Exception("We can't support this axis,because C = ", C, ",but axis = ", axis)
                #         break
                result_temp.append(group)
                result_temp.append(outputs_name)
                result_temp.append(outputs_shape)
                result.append(result_temp)
                result.append(1)
                result.append(1)
            else:
                #['pack',pack_op]这种情况暂时没有发现，如果后面出现的话可能要处理，首先是去layer_index里面修改
                inputs_name = [v.name for v in op.inputs]
                result[0] = inputs_name
                for name in inputs_name:
                    result.append(name_shape_dic[name])
                outputs_name = [v.name for v in op.outputs][0]
                outputs_shape = name_shape_dic[outputs_name]
                result.append(outputs_name)
                result.append(outputs_shape)
            return result
        elif layer_type ==  'Maximum':
            result.append(outputs_name)
            result.append(outputs_shape)
            return result
        elif layer_type == 'bias':
            inputs = [v.name for v in op.inputs]
            for name in inputs:
                if name in ops_dic_bias:
                    bias = ops_dic_bias[name]
            layer_name = layer[1].name
            bias_name = layer_name + '_bias'
            layer_w_b[bias_name] = bias

            outputs_name = op.outputs[0].name
            outputs_shape = name_shape_dic[outputs_name]
            result.append(inputs_shape)
            result.append([True,[]])
            result.append(outputs_name)
            result.append(outputs_shape)
            return result
        elif layer_type == 'Add':
            outputs_name = op.outputs[0].name
            outputs_shape = name_shape_dic[outputs_name]
            result.append(outputs_name)
            result.append(outputs_shape)
            return result
        elif layer_type in slice_index:
            if layer_type == 'Slice':
                inputs_name = [v.name for v in op.inputs][1:]
                for i in range(len(inputs_name)):
                    if (tf.Session().run(inputs_name[i])[0] == 0):
                        begin = ops_dic_bias[inputs_name[i]]
                    else:
                        end = ops_dic_bias[inputs_name[i]]

                axis_point = [begin[C],begin[C]+end[C]]

                inputs_name = op.inputs[0].name
                inputs_shape = name_shape_dic[inputs_name]
                outputs_name = op.outputs[0].name
                outputs_shape = name_shape_dic[outputs_name]
                axis_index = True#True存在切点，FALSE不存在切点
                for i in range(len(inputs_shape)):
                    if inputs_shape[i] == outputs_shape[i]:
                        axis_index = False
                        pass
                    else:
                        axis_index = True
                        axis = i
                        if axis == C and C == 3:
                            axis = 1
                        elif axis == C and C == 1:
                            pass
                        else:
                            raise Exception("We can't support this axis,because C = {} but axis ={}.While this operation name is:{} ".format(C,axis,layer_name))
                        break
                if axis_index == False:
                    raise Exception("This slice point:(op_name = {}),slice none of the tensor:{}".format(layer_name,inputs_tensor))
                result.append(inputs_shape)
                result.append(axis)
                result.append(axis_point)
                result.append(outputs_name)
                result.append(outputs_shape)
                return result
            else:
                inputs_name = [v.name for v in op.inputs]
                # inputs_shape =  [v.get_shape().as_list() for v in op.inputs]
                result[0] = inputs_name
                inputs_name = op.inputs[0].name
                inputs_shape = name_shape_dic[inputs_name]
                outputs_name = op.outputs[0].name
                outputs_shape = name_shape_dic[outputs_name]
                result.append(inputs_shape)
                result.append(outputs_name)
                result.append(outputs_shape)
                return result
        elif layer_type in relu_index:
            inputs_name = op.inputs[0].name
            inputs_shape = name_shape_dic[inputs_name]
            outputs_name = op.outputs[0].name
            outputs_shape = name_shape_dic[outputs_name]
            result.append(inputs_shape)
            if layer_type == 'LeakyRelu':
                alpha = op.get_attr('alpha')
                result.append(alpha)
            result.append(outputs_name)
            result.append(outputs_shape)
            return result
        elif layer_type == 'Reshape' and len(layer) == 4:
            #layer = ['Reshape',reshape_op,transpose_op,reshape_op]
            outputs_name_1 = op.outputs[0].name
            outputs_shape_1 = name_shape_dic[outputs_name_1]
            axis = int(outputs_shape_1[C])
            result[1] = 'shufflechannel'
            op_output = layer[3]
            inputs_name = op.inputs[0].name
            inputs_shape = name_shape_dic[inputs_name]
            outputs_name = op_output.outputs[0].name
            outputs_shape = name_shape_dic[outputs_name]

            # for i in range(len(inputs_shape)):
            #     if inputs_shape[i] == outputs_shape[i]:
            #         pass
            #     else:
            #         axis = i
            #         if axis == C and C == 3:
            #             axis = 1
            #         elif axis == C and C == 1:
            #             pass
            #         else:
            #             raise Exception("We can't support this axis,because C = ", C, ",but axis = ", axis)
            #         break
            result.append(inputs_shape)
            result.append(axis)
            result.append(outputs_name)
            result.append(outputs_shape)
            return result
        elif layer_type == 'Squeeze':
            if len(layer) == 3:
                result[1] = 'Reshape'
                result.append(inputs_shape)
                outputs_name = layer[2].outputs[0].name
                outputs_shape = name_shape_dic[outputs_name]
                result.append(outputs_name)
                result.append(outputs_shape)
            else:
                result[1] = 'Reshape'
                result.append(inputs_shape)
                result.append(outputs_name)
                result.append(outputs_shape)
            return result
        elif layer_type == 'Reshape' and len(layer) == 2:
            # layer = ['Reshape',reshape_op]
            #这一层是matmul上面reshape
            i = 0
            for name1 in inputs_name:
                if  not name1 in ops_dic_bias:
                    inputs_name = name1
                    inputs_shape = name_shape_dic[inputs_name]
                i = i+1
            outputs_name = op.outputs[0].name
            outputs_shape = name_shape_dic[outputs_name]
            if len(inputs_shape)==4 and len(outputs_shape) == 2:
                outputs_shape = [1,inputs_shape[1]*inputs_shape[2]*inputs_shape[3]]
            result[0] = [inputs_name]
            result.append(inputs_shape)
            result.append(outputs_name)
            result.append(outputs_shape)
            return result
        elif layer_type == 'Split':
            inputs_shape = [v.get_shape().as_list() for v in op.inputs]
            axis_name = ''
            i = 0
            while i < len(inputs_name):
                name = inputs_name[i]
                if name in ops_dic_bias:
                    axis_name = name
                else:
                    inputs_shape = name_shape_dic[name]#找到tensor对应的shape
                i = i+1
            axis = ops_dic_bias[axis_name]
            if C == 3:
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
            if len(inputs_shape) == 2:
                axis_true = axis
            outputs_name = [v.name for v in op.outputs]
            slice_point = []#在那个地方分割
            slice_num = len(outputs_name)  # 例如输出三个张量，则slice_num = 3
            if C == 3 and len(inputs_shape) == 2:
                num = inputs_shape[1]/slice_num
            elif C == 1 and len(inputs_shape) == 2:
                num = inputs_shape[0]/slice_num
            else:
                num = inputs_shape[C] / slice_num  # 使用输入最后一个维度，除以slice_num就可以得到每一层的大小，例如15/3 = 5
            i = 1
            while i < slice_num:
                slice_point.append(int(num * i))
                i = i + 1
            outputs_name = [v.name for v in op.outputs]
            outputs_shape = name_shape_dic[outputs_name[0]]
            result.append(inputs_shape)
            result.append(axis_true)
            result.append(slice_point)
            result.append(outputs_name)
            result.append(outputs_shape)
            # print(result)############################################################
            return result
        elif layer_type == 'SplitV':
            inputs_shape = [v.get_shape().as_list() for v in op.inputs]
            axis_name = ''
            i = 0
            while i < len(inputs_name):
                name = inputs_name[i]
                if name in ops_dic_bias:
                    axis_name = name
                else:
                    inputs_shape = name_shape_dic[name]#找到tensor对应的shape
                    # if inputs_shape[N] == None:
                    #     inputs_shape[N] = 1
                i = i+1
            axis = ops_dic_bias[axis_name]
            if C == 3:
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
            outputs_name = [v.name for v in op.outputs]
            slice_point = []#在那个地方分割
            split_num = ops_dic_bias[inputs_name[1]]
            # print('split',split_num)
            num = 0
            for n in split_num[:-1]:
                num = num+n
                slice_point.append(num)
            # slice_num = len(outputs_name)  # 例如输出三个张量，则slice_num = 3
            # if C == 3 and len(inputs_shape) == 2:
            #     num = inputs_shape[1]/slice_num
            # elif C == 1 and len(inputs_shape) == 2:
            #     num = inputs_shape[0]/slice_num
            # else:
            #     num = inputs_shape[C] / slice_num  # 使用输入最后一个维度，除以slice_num就可以得到每一层的大小，例如15/3 = 5
            # i = 1
            # while i < slice_num:
            #     slice_point.append(int(num * i))
            #     i = i + 1
            outputs_name = [v.name for v in op.outputs]
            outputs_shape = name_shape_dic[outputs_name[0]]
            result.append(inputs_shape)
            result.append(axis_true)
            result.append(slice_point)
            result.append(outputs_name)
            result.append(outputs_shape)
            # print(result)############################################################
            return result
        elif layer_type == 'Mul':
            txt = [v.get_shape().as_list() for v in op.inputs]
            op = layer[1]
            inputs_name_all = [v.name for v in op.inputs]
            if not inputs_name_all[0] in ops_dic_bias and not inputs_name_all[1] in ops_dic_bias:
                raise Exception("This Mul operation's input is two tensor, this operation name is {}".format(op.name))
            neg_index = False
            inputs_shape = []
            i = 0
            for ip in inputs_name:  # 如果输入有neg，input找到上一级
                op = find_op_by_output[ip][0]
                if op.type == 'Neg':
                    input = [v.name for v in op.inputs][0]
                    if input in ops_dic_bias:
                        neg_index = True  # 如果是真，则表示是对常数的neg操作，在后面导入dic_e_b的时候需要添负号
                    inputs_name[i] = input
                i = i + 1

            # 获取卷积信息，输入的图片以及filter参数
            i = 0
            for name in inputs_name:
                if name in ops_dic_bias :#首先找到mul输入的常数项以哪一个
                    filter_name = name
                    w = ops_dic_bias[filter_name]
                    w_shape = list(w.shape)
                    if len(w.shape) == 3:
                        # w_shape.insert(0, 1)
                        # w = w.reshape(w_shape)
                        w = w_data(w)

                    ##权重和偏置保存到字典中
                    layer_name = layer[1].name
                    w_name = layer_name + '_weight'
                    if neg_index:
                        layer_w_b[w_name] = w*-1
                    else:
                        layer_w_b[w_name] = w
                    ######################
                else:
                    inputs_name = name
                    inputs_shape = name_shape_dic[inputs_name]
                i = i + 1
            outputs_name = layer[1].outputs[0].name
            outputs_shape = name_shape_dic[outputs_name]
            # result[0] = [inputs_name]
            result.append(inputs_shape)
            result.append([False, []])
            result.append(outputs_name)
            result.append(outputs_shape)
            return result
        elif layer_type == 'ConcatV2':
            inputs_name = [v.name for v in op.inputs]
            txt = []
            for name in inputs_name:
                txt.append(name_shape_dic[name])
            inputs_shape = txt[0]
            axis = ops_dic_bias[inputs_name[-1]]
            inputs_shape1 = txt[0:-1]
            for inputs in inputs_shape1:
                if len(inputs) == 2:
                    inputs.insert(0,1)
                    inputs.insert(0,1)
                result.append(inputs)
            txt = []
            for name in inputs_name:
                txt.append(name_shape_dic[name])
            a = [0,1,2,3]
            b = [0,1]
            if axis<0 and len(op.inputs[0].get_shape().as_list()) == 4:
                axis = a[axis]
            elif axis<0 and len(op.inputs[0].get_shape().as_list()) == 2:
                axis = b[axis]
            if C == 3 and len(op.inputs[0].get_shape().as_list()) == 4:
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
            result.append(axis_true)
            result.append(outputs_name)
            outputs_name = op.outputs[0].name
            outputs_shape = name_shape_dic[outputs_name]
            if len(outputs_shape) == 2:
                outputs_shape.insert(0,1)
                outputs_shape.insert(0,1)
            result.append(outputs_shape)
            return result
        elif layer_type =='FusedBatchNorm':
            #在layer_index函数里面，是将不能合并的neg节点当做fusedbatchnorm处理
            bias = True
            gamma = ops_dic_bias[inputs_name[1]]
            beta = ops_dic_bias[inputs_name[2]]
            mean = ops_dic_bias[inputs_name[3]]
            variance = ops_dic_bias[inputs_name[4]]
            epsilon = op.get_attr('epsilon')
            beta = beta - gamma * mean / np.sqrt(variance + epsilon)
            w = gamma / np.sqrt(variance + epsilon)
            ##权重和偏置保存到字典中
            w_name = layer_name + '_weight'
            b_name = layer_name + '_bias'
            layer_w_b[w_name] = w
            layer_w_b[b_name] = beta
            ##########################
            result.append('w')
            result.append([bias, 'beta'])
            outputs_name = op.outputs[0].name
            outputs_shape = name_shape_dic[outputs_name]
            result.append(outputs_name)#batch_normal有四个输出


            result.append(outputs_shape)
            return result
        elif layer_type == 'Mean'or layer_type == 'Max':
            inputs_name = op.inputs[0].name
            inputs_shape = name_shape_dic[inputs_name]
            outputs_name = op.outputs[0].name
            outputs_shape = name_shape_dic[outputs_name]
            ksize = [1, inputs_shape[1], inputs_shape[2], 1]
            strides = [1, 1, 1, 1]
            padding = 'VALID'
            outputs_shape.insert(0, 1)
            outputs_shape.insert(0, 1)
            if layer_type == 'Max':
                result[1] = 'MaxPool'
                result.append('Max')
            else:
                result[1] = 'AvgPool'
                result.append('Mean')
            result.append(inputs_shape)
            result.append(ksize)
            result.append(strides)
            result.append(padding)
            result.append(outputs_name)
            result.append(outputs_shape)
            return result
        elif layer_type == 'AvgPool' or layer_type == 'MaxPool':
            input1 = inputs_name[0]
            input1_form_ops = find_op_by_output[input1]
            op_name = [v.type for v in input1_form_ops]
            if op_name[0] == 'Reshape' or op_name[0] == 'Pad':
                inputs_name[0] = input1_form_ops[0].inputs[0].name
                result[0] = [inputs_name[0]]  ############针对输入name的修改
            ksize = op.get_attr('ksize')
            strides = op.get_attr('strides')
            padding = bytes.decode(op.get_attr('padding'))
            padding,inputs_shape = padding_data(op,padding,find_op_by_output,name_shape_dic,ops_dic_bias)
            result.append(inputs_shape)
            result.append(ksize)
            result.append(strides)
            result.append(padding)
            outputs_name = op.outputs[0].name
            outputs_shape = name_shape_dic[outputs_name]
            result.append(outputs_name)
            result.append(outputs_shape)
            return result
        elif layer_type == 'Neg':
            inputs_name = op.inputs[0].name
            inputs_shape = name_shape_dic[inputs_name]
            result[1] = 'Mul'
            w = -1
            ##权重和偏置保存到字典中
            w_name = layer_name + '_weight'
            layer_w_b[w_name] = w
            bias = False
            ##########################
            result.append(inputs_shape)
            result.append([bias, []])
            outputs_name = op.outputs[0].name
            outputs_shape = name_shape_dic[outputs_name]
            result.append(outputs_name)  # batch_normal有四个输出
            result.append(outputs_shape)
            return result


    elif layer_type in conv_index:
        if layer_type == 'Conv2D' or layer_type == 'DepthwiseConv2dNative':
            result = conv_defeat(layer,find_op_by_output,find_op_by_input,ops_dic_bias,name_shape_dic)
            return result
        elif layer_type == 'MatMul':
            result = matmul_data(layer,ops_dic_bias,find_op_by_output,find_op_by_input,name_shape_dic)
            return result
def load_graph_tf(model_path):
    """
    对待处理的神经网络文件，会区分是目录还是单个的.pb文件
    :param model_path:
    :return:
    """
    if os.path.isdir(model_path):
        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with tf.Session(graph=graph) as sess:
            tf.saved_model.loader.load(
                sess, [tf.saved_model.tag_constants.SERVING], model_path)
        return graph
    elif os.path.isfile(model_path):
        with open(model_path, 'rb') as f:
            serialized = f.read()
        tf.reset_default_graph()
        gdef = tf.GraphDef()
        gdef.ParseFromString(serialized)
        with tf.Graph().as_default() as g:
            tf.import_graph_def(gdef, name='')
            graph = g
    else:
        raise Exception('Wrong model_path')
    return graph
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

def global_pooling_same(result,index):
    """
    处理全局池化不满足硬件条件的情况
    :param result:
    :param index:
    :return:
    """
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

    result[i][-1][H] = math.ceil((input_shape[H])/6)
    result[i][-1][W] = math.ceil((input_shape[W])/6)
    result[i][-2] = result[i][-2] + '_step_' + str(j)


    output_h = result[i][-1][H]
    output_w = result[i][-1][W]

    stride_h = result[i][5][1]
    stride_w = result[i][5][2]

    ksize_h = result[i][4][1]
    ksize_w = result[i][4][2]

    pad_needed_height = (output_h - 1) * stride_h + ksize_h
    pad_needed_width = (output_w - 1) * stride_w + ksize_w
    pad_n = int(pad_needed_height-input_shape[1])
    pad_s = int(0)
    pad_w = int(pad_needed_width-input_shape[2])
    pad_e = int(0)
    padding = [pad_n, pad_s, pad_w, pad_e]
    result[i][-3] = padding

    pool = result[i]
    j = j + 1
    while (pool[-1][1] > 1 and pool[-1][2] > 1):
        # print(11111111111111111)
        if (pool[-1][1] > 16 and pool[-1][2] > 16):
            inputs_name = [pool[-2]]
            type = pool[1]
            input_shape = pool[-1]
            ksize_h = 16
            ksize_w = 16
            stride_h = ksize_h
            stride_w = ksize_w
            output_h = math.ceil(input_shape[H] / stride_h)
            output_w = math.floor(input_shape[W]/stride_w)

            pad_needed_height = (output_h - 1) * stride_h + ksize_h
            pad_needed_width = (output_w - 1) * stride_w + ksize_w
            pad_n = int(pad_needed_height)
            pad_s = int(0)
            pad_w = int(pad_needed_width)
            pad_e = int(0)


            need = pad_n + pad_s + input_shape[H]
            temp = (output_h - 1) * stride_h + ksize_h
            if temp == need:
                pass
            elif need > temp:
                n = need - temp
                pad_s = pad_s - n

            need = pad_w + pad_e + input_shape[W]
            temp = (output_w - 1) * stride_w + ksize_w
            if temp == need:
                pass
            elif need > temp:
                n = need - temp
                pad_e = pad_e - n
            padding = [pad_n, pad_s, pad_w, pad_e]

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
            pool = [inputs_name, type, layer_name + '_step_' + str(j), input_shape, [1, ksize_h, ksize_w, 1],
                    [1, 1, 1, 1], 'VALID', output_name, [1, output_h, output_w, pool[-1][-1]]]
            result.insert(i + 1, pool)

    return result,j
def slice_delet(result):
    """
    前面对于单个的slice节点已经处理了，现在是将slice合并
    :param result:
    :return:
    """

    for i in range(len(result)):
        name = result[i][1]
        index = result[i][-1]
        output_dic = {}
        if name == 'Slice' and not index == 1:
            slice = result[i]
            output = [slice[-2]]
            axis = slice[-4]
            slice_point = slice[-3]
            k = i+1
            n = 1
            output_dic[result[i][-3][0]] = result[i][-2]
            while(k<len(result)):
                layer = result[k]
                if name == layer[1] and slice[0] == layer[0]:
                    if not axis == layer[-4]:
                        raise Exception("slice's axis erro")
                    else:
                        slice_point = slice_point+layer[-3]
                        output.append(layer[-2])
                        output_dic[result[k][-3][0]] = result[k][-2]
                        result[k].append(1)
                        n = n+1
                k = k+1
            slice_point = set(slice_point)
            slice_point = list(slice_point)
            if not len(slice_point) == n+1:
                raise Exception("slice's axis_point erro")

            else:
                output_name = list()
                dic_index = sorted(output_dic)
                for key in dic_index:
                    output_name.append(output_dic[key])

                slice_point.pop()
                del slice_point[0]
                slice[-3] = slice_point
                slice[-2] = output_name
                result[i] = slice
    # for j in range(len(result)-1):
    #     if result[j][-1] == 1:
    #         del result[j]
    #         j = j-1
    return result
def div_sub_mul_delect(result):
    """
    对于RealDiv,Mul,Sub都已经解析完了，该函数是对合并能合并的scale层
    :param result:
    :return:
    """
    for i in range(len(result)):
        name = result[i][1]#类型
        index = result[i][-1]
        if name in scale_index and not index == 1:
            type_list = []
            const_list = []
            scale = result[i]
            output = scale[-2]
            layer_name = scale[2]
            bias = False
            weight = False
            if name == 'Sub'or name == 'bias':
                const = layer_w_b[layer_name+'_bias']
                bias = True
            else:
                const = layer_w_b[layer_name+'_weight']
                weight = True
            type_list.append(name)
            const_list.append(const)
            k = i + 1
            while (k < len(result)):
                layer = result[k]
                name = layer[1]
                if layer[1] in scale_index and output == layer[0][0]:
                    layer_name = layer[2]
                    if name == 'Sub' or name == 'bias':
                        const = layer_w_b[layer_name + '_bias']
                        bias = True
                    else:
                        const = layer_w_b[layer_name + '_weight']
                        weight = True
                    type_list.append(name)
                    const_list.append(const)
                    output = layer[-2]
                    result[k].append(1)
                    k = k+1
                else:
                    break
            w = 1.0
            b = 0
            for j in range(len(type_list)):
                type = type_list[j]
                if (not type == 'Sub' and not type == 'bias') and not j == len(type_list):
                    while (j < len(type_list)):
                        if type_list[j] == 'Sub' or type_list[j] == 'bias':
                            j = j+1
                            pass
                        else:
                            w = w * const_list[j]
                            j = j + 1
                    break
            for j in range(len(type_list)):
                type = type_list[j]
                if (type == 'Sub'or type == 'bias')and not j == len(type_list):
                    while(j < len(type_list)):
                        if type_list[j] == 'Sub' or type_list[j] == 'bias':
                            b = b+const_list[j]
                            j = j+1
                        else:
                            b = b*const_list[j]
                            j = j+1
                    break
                elif (type == 'Sub'or type == 'bias')and  j == len(type_list)-1:
                    b = b-const_list[-1]
            if weight == False:
                pass
            else:
                layer_name = scale[2]+'_weight'
                w = np.array(w, dtype = np.float64)
                layer_w_b[layer_name] = w
                if layer_w_b[layer_name].shape == ():
                    shape = scale[3][C]
                    layer_w_b[layer_name] = np.ones(shape)*w
            if bias == False:
                pass
            else:
                layer_name = scale[2] + '_bias'
                b = np.array(b, dtype=np.float64)
                layer_w_b[layer_name] = b
                if layer_w_b[layer_name].shape == ():
                    shape = scale[3][C]
                    layer_w_b[layer_name] = np.ones(shape)*b
            if bias:
                scale[-3] = [True,'b']
            scale[-2] = output
            result[i] = scale
    #去掉不用的值
    for i in range(len(result)):
        name = result[i][1]#类型
        index = result[i][-1]
        if name in scale_index and  index == 1:
            scale = result[i]
            layer_name = scale[2]
            if layer_name+'_bias' in layer_w_b:
                layer_w_b.pop(layer_name+'_bias')
            if layer_name+'_weight' in layer_w_b:
                layer_w_b.pop(layer_name+'_weight')
    # print(b)
    # print(layer_w_b['p_re_lu/Neg_1'+'_weight'])

    return result
def const_delet(result,ops_dic_bias):
    for i in range(len(result)):
        layer = result[i]
        if isinstance(layer[-2],str):
            if layer[-2] in ops_dic_bias:
                result[i].append(1)
    res = []
    for i in range(len(result)):
        layer = result[i]
        if not layer[-1] == 1:
            res.append(layer)
    #去掉不用的值
    for i in range(len(result)):
        name = result[i][1]#类型
        index = result[i][-1]
        if name in scale_index and  index == 1:
            scale = result[i]
            layer_name = scale[2]
            if layer_name+'_bias' in layer_w_b:
                layer_w_b.pop(layer_name+'_bias')
            if layer_name+'_weight' in layer_w_b:
                layer_w_b.pop(layer_name+'_weight')
    return res
def index_delet(result):
    res = []
    for i in range(len(result)):
        layer = result[i]
        if not layer[-1] == 1:
            res.append(layer)
    return res
def conv_scale(result):
    """
    合并conv和mul节点（bias,div）
    :param result:
    :return:
    """
    for i in range(len(result)):
        if result[i][1] in conv_index or result[i][1] == 'MatMul_conv':
            output = result[i][-2]
            temp = 0
            k = i+1
            while(k<len(result)):
                layer = result[k]
                if output in layer[0]:
                    temp+=1
                k = k+1
            if temp >1:
                continue
            conv = result[i]
            j = i+1
            bias_index = True
            conv_bias = False
            conv_weight = False
            if j < len(result):
                if result[j][1] in scale_index:
                    scale = result[j]
                    conv_layer_name = conv[2]
                    scale_layer_name = scale[2]
                    scale_bias = False
                    scale_weight = False
                    if scale_layer_name+'_bias' in layer_w_b:
                        scale_bias = True
                        bias_s = layer_w_b[scale_layer_name+'_bias']
                    if scale_layer_name+'_weight' in layer_w_b:
                        scale_weight = True
                        weight_s = layer_w_b[scale_layer_name + '_weight']
                    if conv_layer_name+'_bias' in layer_w_b:
                        conv_bias = True
                        bias_c = layer_w_b[conv_layer_name+'_bias']
                    if conv_layer_name+'_weight' in layer_w_b:
                        conv_weight = True
                        weight_c = layer_w_b[conv_layer_name + '_weight']
                    if scale_weight == True:
                        layer_w_b[conv_layer_name+'_weight'] = weight_c*weight_s
                        if conv_bias == True:
                            if scale_bias == True:
                                bias = weight_s*bias_c + bias_s
                                layer_w_b[conv_layer_name + '_bias'] = bias
                                weight = weight_c*weight_s
                            elif scale_bias == False:
                                weight = weight_c*weight_s
                                bias = weight_s*bias_c
                                layer_w_b[conv_layer_name + '_bias'] = bias
                        if conv_bias == False:
                            if scale_bias == True:
                                bias = bias_s
                                layer_w_b[conv_layer_name + '_bias'] = bias
                                weight = weight_c*weight_s
                            elif scale_bias == False:
                                weight = weight_c*weight_s
                                bias_index = False
                    if scale_weight == False:
                        return result
                    if bias_index == False:
                        conv[-2] = scale[-2]
                        scale.append(1)
                    else:
                        conv[-2] = scale[-2]
                        conv[-3][0] = True
                        scale.append(1)
                    result[i] = conv
                    result[j] = scale
    return result
def pool_padding(result):
    for i in range(len(result)):
        layer = result[i]
        type = layer[1]
        name = layer[2]
        if type == 'AvgPool' or type == 'MaxPool':
            padding = layer[-3]
            if padding == 'VALID' and layer[-4][1] != 1 and layer[-4][1] != 1:
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
            if padding == 'VALID' and layer[-6][1] != 1 and layer[-6][1] != 1:
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
def read_graph_from_pb(tf_model_path):
    """
    pb_read.py文件的核心函数，整个pb文件的解析是靠该文件“指导”完成的
    :param tf_model_path:
    :return:
    """
    print('Loading the graph..............')
    graph = load_graph_tf(tf_model_path)
    with tf.Session(graph = graph) as sess:
        ops = graph.get_operations()
        ops_num = len(ops)
        start = time.time()
        name_shape_dic = name_shape(ops)#获取name:shape字典，方便后面出现[None,None,None,n]的情况
        deal_with_op(ops)#查看有没有不能处理的节点
        for op in ops:
            if op.type=='Conv2D':
                # print(op.get_attr("data_format"))
                if op.get_attr("data_format") == b'NCHW':
                    func_2()
                    break
                else:
                    func_1()
                    break
        operation = [op.type for op in ops]
        operation_name = [op.name for op in ops]
        operation_type = [op.type for op in ops]
        if 'Cast' in operation_type:
            warnings.warn(
                "There is a cast operation in your net,we don't support of it, this operation will cause errors in the results")
        if 'Softmax' in operation_type:
            warnings.warn(
                "There is a softmax operation in your net,we don't support of it,used logsoftmax instead.")

        for i in range(len(operation_name)):
            if operation_type[i] == 'MatMul':
                operation_name[i] = operation_name[i].replace('MatMul','dense')
        dict1 = get_const_value(ops)  # 常量的名称和值组成的字典
        dict2 = get_const_value_1(ops)
        ops_dic_bias = dict(dict1, **dict2)
        # 解析网络并根据constant和Identity获取节点以及值，构成字典
        find_op_by_input = find_my_input(ops, operation)  # 节点与节点输入组成的字典
        find_op_by_output,find_op_by_input = find_my_output(ops, find_op_by_input)  # 节点和节点输出组成的字典
        find_identity_name = find_my_identity(ops)
        """
        ops_dic_bias:存储const和name
        find_op_by_input:存储op->(name)
        find_op_by_output:存储(name)->op
        """
        batch_norm_in = False
        ops_data = [op for op in ops if not op.type=='Identity' and not op.type == 'Squeeze' and  not op.type == 'Const'or not op.type == 'NoOp']
        layers_op = layer_index(ops_data,ops_dic_bias,find_op_by_output,find_op_by_input)
        if batch_norm_in == False:
            layers_op = layer_batchnorm_out(layers_op)
        result = []
        reshape_crt = False
        for layer in layers_op:
            if layer[0] == 'Reshape'and len(layer) == 2:
                reshape_crt = True
        for layer in layers_op:
            # print(layer[1])
            result_temp_1 = operation_data(layer,ops_dic_bias,find_op_by_input,find_op_by_output,name_shape_dic)
            if result_temp_1 == None:
                continue
            else:
                result_temp = result_temp_1
            if result_temp[3] == 'Max' or result_temp[3] == 'Mean' :
                reshape_add = reshape_creat(result_temp)
                result.append(result_temp)
                if reshape_crt == False:
                    result.append(reshape_add)
                reshape_crt = True
            else:
                if len(result_temp) == 4:
                    result.append(result_temp[0])
                    result.append(result_temp[1])
                else:
                    result.append(result_temp)

        result = const_delet(result,ops_dic_bias)
        result = layer_connect(result,operation_name)

        result = slice_delet(result)
        result = div_sub_mul_delect(result)
        result = index_delet(result)
        result = conv_scale(result)
        result = pool_padding(result)  # 处理pool层stride=2时候的pad四个方向上的选择（主要是右边和下边）
        #处理identity
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

        reshape_list = list()#记录那些matmul被修改了
        for i in range(len(result)):
            name = result[i][1]
            if name == 'Reshape':
                result,reshape_list = reshape_conv(result, i,reshape_list)

        #处理未修改的matmul
        for i in range(len(result)):
            name = result[i][1]
            if name == 'MatMul_conv' and not i in reshape_list:
                conv = result[i]
                filter = [conv[3][-1], conv[-1][-1]]
                filter.insert(0, 1)
                filter.insert(0, 1)
                conv[4] = filter
                w_name = conv[2] + '_weight'
                w = layer_w_b[w_name]
                w = w.reshape(filter)
                layer_w_b[w_name] = w
                ################
                result[i] = conv

        length = len(result)
        i = 0
        while i < length:
            n = 0
            if result[i][1] in pool_index and result[i][-1][1] == 1 and result[i][-1][2] == 1:
                if result[i][-3] == 'VALID':
                    if result[i][1] == 'MaxPool':
                        result, n = global_maxpooling_valid(result, i)
                    elif result[i][1] == 'AvgPool':
                        result, n = global_avgpooling_valid(result, i)
                    i = i+n+1
                    length += n
                else:
                    i = i+1
            else:
                i = i+1
        for key in layer_w_b:
            layer_w_b[key] = np.array(layer_w_b[key])
            if len(layer_w_b[key].shape) == 4:
                layer_w_b[key] = layer_w_b[key].transpose((C, W, N, H))
        elapsed = time.time() - start
        print("Number of network nodes:{}\nParsing time:{} s".format(ops_num,elapsed))


    return result,layer_w_b


if __name__ == "__main__":
    model_path = r'mobilenet_v1.pb'
    # model_path = r'tf_keras_test_pad.pb'
    # model_path = r'mnist.pb'
    print('Loading the graph..............')
    graph = load_graph_tf(model_path)
    batch_norm_in = False
    with tf.Session(graph = graph) as sess:
        ops = graph.get_operations()
        # op = ops[5]
        # print(op.outputs[0].eval())


        operation_type = [op.type for op in ops]
        # print(sess.run('MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/batchnorm/add:0'))
        # for name in operation_type:
        #     print(name)
        # operation_type = [op.type for op in ops]
        # for name in operation_type:
        #     print(name)
        name_shape_dic = name_shape(ops)#获取name:shape字典，方便后面出现[None,None,None,n]的情况
        deal_with_op(ops)#查看有没有不能处理的节点
        for op in ops:
            if op.type == 'Conv2D':
                # print(op.get_attr("data_format"))
                if op.get_attr("data_format") == b'NCHW':
                    N = 0
                    H = 2
                    W = 3
                    C = 1
                    break
        operation = [op.type for op in ops]
        # for opr in operation:
        #     print(opr)
        operation_name = [op.name for op in ops]


        if 'Cast' in operation_type:
            warnings.warn(
                "There is a cast operation in your net,we don't support of it, this operation will cause errors in the results")
        if 'Softmax' in operation_type:
            warnings.warn(
                "There is a softmax operation in your net,we don't support of it,you can use logsoftmax instead.")
        # for name in operation_type:
        #     print(name)
        for i in range(len(operation_name)):
            if operation_type[i] == 'MatMul':
                operation_name[i] = operation_name[i].replace('MatMul','dense')

        dict1 = get_const_value(ops)  # 常量的名称和值组成的字典
        dict2 = get_const_value_1(ops)
        ops_dic_bias = dict(dict1, **dict2)
        # 解析网络并根据constant和Identity获取节点以及值，构成字典


        find_op_by_input = find_my_input(ops, operation)  # 节点与节点输入组成的字典
        find_op_by_output,find_op_by_input = find_my_output(ops,find_op_by_input)  # 节点和节点输出组成的字典

        find_identity_name = find_my_identity(ops)
        print('I have stored all layers_data ')
        # op = ops[9]
        # inputs_name = [v.name for v in op.inputs]
        # txt = [v.get_shape().as_list() for v in op.inputs]
        # print(txt)
        # print(op)
        # print(ops_dic_bias['bias1/bias:0'])
        # print(sess.run('strided_slice:0'))
        """
        ops_dic_bias:存储const和name
        find_op_by_input:存储op->(name)
        find_op_by_output:存储(name)->op
        """
        ops_data = [op for op in ops if not op.type=='Identity' and not op.type == 'Squeeze' and  not op.type == 'Const'or not op.type == 'NoOp']
        ops_operation = [op.name for op in ops_data]
        # print(ops_operation)
        layers_op = layer_index(ops_data,ops_dic_bias,find_op_by_output,find_op_by_input)
        if batch_norm_in == False:
            layers_op = layer_batchnorm_out(layers_op)
        # print(layers_op[7][2])
        print(len(layers_op))
        result = []

        reshape_crt = False
        for layer in layers_op:
            if layer[0] == 'Reshape'and len(layer) == 2:
                reshape_crt = True
        for layer in layers_op:
            # print(layer[1])
            result_temp_1 = operation_data(layer,ops_dic_bias,find_op_by_input,find_op_by_output,name_shape_dic)
            if result_temp_1 == None:
                continue
            else:
                result_temp = result_temp_1
            if result_temp[3] == 'Max' or result_temp[3] == 'Mean' :
                reshape_add = reshape_creat(result_temp)
                result.append(result_temp)
                if reshape_crt == False:
                    result.append(reshape_add)
                reshape_crt = True
            else:
                if len(result_temp) == 4:
                    result.append(result_temp[0])
                    result.append(result_temp[1])
                else:
                    result.append(result_temp)
            # print(result_temp)
        for layer in result:
            print(layer)
        # print(layer_w_b['dense_weight'])

        # for layer in result :
        #     # if layer[-1] != 1:
        #     print(layer)
        # print(11111)
        result = const_delet(result,ops_dic_bias)
        # print(22222)
        result = layer_connect(result,operation_name)
        # print(33333)
        result = slice_delet(result)
        # print(44444)

        result = div_sub_mul_delect(result)
        # print(55555)
        result = index_delet(result)
        # print(6666)
        result = conv_scale(result)
        # print(77777)

        result = pool_padding(result)#处理pool层stride=2时候的pad四个方向上的选择（主要是右边和下边）



        #处理identity
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




        # for layer in result:
        #     name = layer[1]
        #     if name == 'Reshape':
        #         result = reshape_delet(result)
        reshape_list = list()#记录那些matmul被修改了
        for i in range(len(result)):
            name = result[i][1]
            if name == 'Reshape' and len(result[i][3]) == 4:
                print(result[i])
                result,reshape_list = reshape_conv(result, i,reshape_list)
        #处理未修改的matmul
        for i in range(len(result)):
            name = result[i][1]
            if name == 'MatMul_conv' and not i in reshape_list:
                conv = result[i]
                filter = [conv[3][-1],conv[-1][-1]]
                filter.insert(0,1)
                filter.insert(0,1)
                conv[4] = filter
                w_name = conv[2] + '_weight'
                w = layer_w_b[w_name]
                w = w.reshape(filter)
                layer_w_b[w_name] = w
                ################
                result[i] = conv
        #处理pool层
        length = len(result)
        i = 0
        while i < length:
            n = 0
            if result[i][1] in pool_index and result[i][-1][1] == 1 and result[i][-1][2] == 1:
                if result[i][-3] == 'VALID':
                    if result[i][1] == 'MaxPool':
                        result, n = global_maxpooling_valid(result, i)
                    elif result[i][1] == 'AvgPool':
                        result, n = global_avgpooling_valid(result, i)
                    i = i+n+1
                    length += n
                # if result[i][-3] == 'SAME':
                #     result, n = global_pooling_same(result, i)
                #     i = i + n + 1
                #     length += n
                else:
                    i = i+1
            else:
                i = i+1


        for key in layer_w_b:
            layer_w_b[key] = np.array(layer_w_b[key])
            if len(layer_w_b[key].shape) == 4:
                layer_w_b[key] = layer_w_b[key].transpose((C,W,N,H))



        print('层数：%d' % len(result))
        for layer in result:
            print(layer)


        print('Key in layer_w_b:')
        #打印存储w/b的字典的键
        for key in layer_w_b:
            print(key)
            print(layer_w_b[key].shape)
            # if len(layer_w_b[key].shape) == 1:
            # if key == 'inception_resnet_v2/block35_1_conv/Conv2D_bias':
            #     print(key,layer_w_b[key].shape)
        # print(layer_w_b['p_re_lu/mul_weight'])
        # print(layer_w_b['dense/dense_weight'].transpose((2,3,1,0)))




        # layer = layers_op[10]
        # result_temp = operation_data(layer, ops_operation, ops_dic_bias, find_op_by_output, find_op_by_input)
        # print(result_temp)
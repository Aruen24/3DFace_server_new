import os
import sys
ndk_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ndk_dir)
from ndk.layers import Layer, get_adjacent_layer_dict
import onnx
import numpy as np
import copy

__all__ = ['load_from_onnx']

PARAM_DICT_WEIGHT_SUFFIX = "_weight"
PARAM_DICT_BIAS_SUFFIX = "_bias"

ONNX_TENSOR_INDEX_BN = 0
ONNX_GAMMR_INDEX_BN = 1
ONNX_BETA_INDEX_BN = 2
ONNX_MEAN_INDEX_BN = 3
ONNX_VAR_INDEX_BN = 4

ONNX_TENSOR_INDEX_POOL = 0
ONNX_TENSOR_INDEX_REDUCE = 0

ONNX_TENSOR_INDEX_CONV = 0
ONNX_WEIGHT_INDEX_CONV = 1
ONNX_BIAS_INDEX_CONV = 2

ONNX_WEIGHT_INDEX_MAT_MUL = 1
ONNX_TENSOR_INDEX_MAT_MUL = 0

ONNX_WEIGHT_INDEX_GEMM = 1
ONNX_TENSOR_INDEX_GEMM = 0
ONNX_BIAS_INDEX_GEMM = 2

ONNX_TENSOR_INDEX_PRELU = 0
ONNX_WEIGHT_INDEX_PRELU = 1

ONNX_NAME_DILATION = 'dilations'
ONNX_NAME_STRIDE = 'strides'
ONNX_NAME_PADDING = 'pads'
ONNX_NAME_GROUP = 'group'
ONNX_NAME_AUTO_PAD = 'auto_pad'
ONNX_NAME_KERNEL_SIZE = 'kernel_shape'
ONNX_NAME_COUNT_INCLUDE_PAD = 'count_include_pad'
ONNX_NAME_EPSILON = 'epsilon'
ONNX_NAME_AXIS = 'axis'
ONNX_NAME_AXES = 'axes'
ONNX_NAME_LEAKY_PARAM = 'alpha'
ONNX_NAME_GEMM_APLHA = 'alpha'
ONNX_NAME_GEMM_BETA = 'beta'
ONNX_NAME_GEMM_TRANS_A = 'transA'
ONNX_NAME_GEMM_TRANS_B = 'transB'
ONNX_NAME_MAX = 'max'
ONNX_NAME_MIN = 'min'
ONNX_NAME_TRANSPOSE_PERM = 'perm'
ONNX_NAME_SPLIT_SIZE = 'split'
ONNX_NAME_PAD_CONSTANT = 'constant'
ONNX_NAME_PAD_MODE = 'mode'
ONNX_NAME_PAD_VALUE = 'value'
ONNX_NAME_PAD_NUMBER = 'pads'

ONNX_NAME_CONSTANT = 'Constant'
ONNX_NAME_CONV = 'Conv'
ONNX_NAME_BN = 'BatchNormalization'
ONNX_NAME_GLB_AVG_POOL = "GlobalAveragePool"
ONNX_NAME_GLB_MAX_POOL = "GlobalMaxPool"
ONNX_NAME_AVG_POOL = "AveragePool"
ONNX_NAME_MAX_POOL = "MaxPool"
ONNX_NAME_REDUCE_MEAN = "ReduceMean"
ONNX_NAME_REDUCE_MAX = "ReduceMax"
ONNX_NAME_ADD = "Add"
ONNX_NAME_SUB = "Sub"
ONNX_NAME_MUL = "Mul"
ONNX_NAME_DIV = "Div"
ONNX_NAME_NEG = "Neg"
ONNX_NAME_ABS = "Abs"
ONNX_NAME_RELU = "Relu"
ONNX_NAME_LEAKY_RELU = "LeakyRelu"
ONNX_NAME_PRELU = "PRelu"
ONNX_NAME_CLIP = "Clip"
ONNX_NAME_SIGMOID = "Sigmoid"
ONNX_NAME_TANH = "Tanh"
ONNX_NAME_SOFTMAX = "Softmax"
ONNX_NAME_LOGSOFTMAX = "LogSoftmax"
ONNX_NAME_SHAPE = "Shape"
ONNX_NAME_GATHER = "Gather"
ONNX_NAME_UNSQUEEZE = "Unsqueeze"
ONNX_NAME_CONCAT = "Concat"
ONNX_NAME_RESHAPE = "Reshape"
ONNX_NAME_FLATTEN = "Flatten"
ONNX_NAME_GEMM = "Gemm"
ONNX_NAME_MAT_MUL = "MatMul"
ONNX_NAME_TRANSPOSE = "Transpose"
ONNX_NAME_SPLIT = "Split"
ONNX_NAME_PAD = "Pad"
ONNX_NAME_DROPOUT = "Dropout"
ONNX_NAME_IDENTITY = "Identity"
#  enum DataType {
#    UNDEFINED = 0;
#    // Basic types.
#    FLOAT = 1;   // float
#    UINT8 = 2;   // uint8_t
#    INT8 = 3;    // int8_t
#    UINT16 = 4;  // uint16_t
#    INT16 = 5;   // int16_t
#    INT32 = 6;   // int32_t
#    INT64 = 7;   // int64_t
#    STRING = 8;  // string
#    BOOL = 9;    // bool
#
#    // IEEE754 half-precision floating-point format (16 bits wide).
#    // This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
#    FLOAT16 = 10;
#
#    DOUBLE = 11;
#    UINT32 = 12;
#    UINT64 = 13;
#    COMPLEX64 = 14;     // complex with float32 real and imaginary components
#    COMPLEX128 = 15;    // complex with float64 real and imaginary components
#
#    // Non-IEEE floating-point format based on IEEE754 single-precision
#    // floating-point number truncated to 16 bits.
#    // This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
#    BFLOAT16 = 16;
#
#    // Future extensions go here.
#  }

data_type_dict_onnx = {1: np.float32, 2: np.uint8, 3: np.int8, 4: np.uint16, 5: np.int16, 6: np.int32, 7: np.int64, 11: np.float64, 12: np.uint32, 13: np.uint64}

#def get_const_from_initializer(initializer):
#    const_dict = {}
#    for tensor_proto in initializer:
#        if hasattr(tensor_proto, 'raw_data'):
#            param = np.frombuffer(tensor_proto.raw_data, dtype = data_type_dict_onnx[tensor_proto.data_type])
#        else:
#            raise Exception("not implemented, info: {}".format(tensor_proto))
#        const_dict[tensor_proto.name] = param
#    return const_dict

def get_const_from_initializer(initializer):
    const_dict = {}
    try:
        from onnx.numpy_helper import to_array
    except ImportError:
        raise ImportError("Onne and protobuf need to be installed. Instructions - https://github.com/onnx/onnx")
    for tensor_proto in initializer:
        if len(tuple(tensor_proto.dims)) > 0:
            np_array = to_array(tensor_proto).reshape(tuple(tensor_proto.dims))
        else:
            np_array = np.array([to_array(tensor_proto)])
        const_dict[tensor_proto.name] = np_array.astype(data_type_dict_onnx[tensor_proto.data_type])
    return const_dict

def get_input_and_param_shape(graph):
    input_and_param_shape = {}
    for tensor_proto in graph.input:
        dim_obj = tensor_proto.type.tensor_type.shape.dim
        shape = [dim.dim_value for dim in dim_obj]
        input_and_param_shape[tensor_proto.name] = shape
    return input_and_param_shape

def get_raw_layer_list_and_param_dict(node, const_dict, input_and_param_shape):
    tensor_shape = {}
    param_dict = {}
    layer_list = []
    for op in node:
        layer_name = op.name if len(op.name) > 0 else "layer" + op.output[0]
        print("get layer {}, type {}, input {}, output {}".format(layer_name, op.op_type, op.input, op.output))
        for input_name in op.input:
            if input_name in input_and_param_shape.keys() and input_name not in const_dict.keys() and input_name not in tensor_shape.keys():
                layer_list.append(Layer(lyr_type = 'input', name = "layer" + input_name, top = input_name, dim = tuple(input_and_param_shape[input_name])))
                tensor_shape[input_name] = copy.deepcopy(input_and_param_shape[input_name])
            else:
                assert input_name in const_dict.keys() or input_name in tensor_shape.keys(), "Nodes should be sorted topologically, but get tensor {} first appears and not in graph.input, please check how you saved this onnx file".format(input_name)
        if op.op_type == ONNX_NAME_CONSTANT:
            attribute = op.attribute[0]
            dim = list(attribute.t.dims) if hasattr(attribute.t, 'dims') else []
            if hasattr(attribute.t, 'raw_data'):
                value = np.frombuffer(attribute.t.raw_data, dtype = data_type_dict_onnx[attribute.t.data_type])
                if len(dim) == 0:
                    const_dict[op.output[0]] = np.array(value[0]) if len(value.shape) != 0 and value.size == 1 else value
                else:
                    const_dict[op.output[0]] = np.array(value).reshape(dim)
            else:
                raise Exception("not implemented, info: {}".format(attribute))
        elif op.op_type == ONNX_NAME_CONV:
            assert op.input[ONNX_TENSOR_INDEX_CONV] not in const_dict.keys(), "Convolution's input feature shouldn't be a constant, check layer {}".format(layer_name)
            ci = tensor_shape[op.input[ONNX_TENSOR_INDEX_CONV]][1]
            hi = tensor_shape[op.input[ONNX_TENSOR_INDEX_CONV]][2]
            wi = tensor_shape[op.input[ONNX_TENSOR_INDEX_CONV]][3]
            bias_term = len(op.input) > ONNX_BIAS_INDEX_CONV
            if op.input[ONNX_WEIGHT_INDEX_CONV] not in input_and_param_shape.keys():
                input_and_param_shape[op.input[ONNX_WEIGHT_INDEX_CONV]] = const_dict[op.input[ONNX_WEIGHT_INDEX_CONV]].shape
            if bias_term:
                if op.input[ONNX_BIAS_INDEX_CONV] not in input_and_param_shape.keys():
                    input_and_param_shape[op.input[ONNX_BIAS_INDEX_CONV]] = const_dict[op.input[ONNX_BIAS_INDEX_CONV]].shape
            kernel_size = input_and_param_shape[op.input[ONNX_WEIGHT_INDEX_CONV]][2:]
            stride = [1, 1]
            dilation = [1, 1]
            pad = [0, 0, 0, 0]
            group = ci // input_and_param_shape[op.input[ONNX_WEIGHT_INDEX_CONV]][1]
            co = input_and_param_shape[op.input[ONNX_WEIGHT_INDEX_CONV]][0]
            for attribute in op.attribute:
                if attribute.name == ONNX_NAME_DILATION:
                    dilation = attribute.ints
                if attribute.name == ONNX_NAME_STRIDE:
                    stride = attribute.ints
                if attribute.name == ONNX_NAME_PADDING:
                    pad = attribute.ints
                if attribute.name == ONNX_NAME_AUTO_PAD:
                    raise Exception("auto_pad is not supported, check layer {}".format(layer_name))
            kh = kernel_size[0]
            kw = kernel_size[1]
            sh = stride[0]
            sw = stride[1]
            dh = dilation[0]
            dw = dilation[1]
            khd = (kh - 1) * dh + 1
            kwd = (kw - 1) * dw + 1
            pn = pad[0]
            ps = pad[2]
            pw = pad[1]
            pe = pad[3]
            ho = (hi + pn + ps - khd) // sh + 1
            wo = (wi + pw + pe - kwd) // sw + 1
            hi_need = (ho - 1) * sh + khd
            wi_need = (wo - 1) * sw + kwd
            ps = int(hi_need - hi - pn)
            pe = int(wi_need - wi - pw)
            
            layer_list.append(Layer(lyr_type = ONNX_NAME_CONV, name = layer_name, top = op.output[0], bottom = op.input[ONNX_TENSOR_INDEX_CONV], 
                              num_output = co, kernel_size_h=kh, kernel_size_w=kw,
                              stride_h=sh, stride_w=sw, pad_n=pn, pad_s=ps, pad_w=pw, pad_e=pe,
                              bias_term=bias_term,
                              dilation_h=dh, dilation_w=dw,
                              group=group))
            
            tensor_shape[op.output[0]] = [tensor_shape[op.input[ONNX_TENSOR_INDEX_CONV]][0], co, ho, wo]
            param_dict[layer_name + PARAM_DICT_WEIGHT_SUFFIX] = const_dict[op.input[ONNX_WEIGHT_INDEX_CONV]].reshape(input_and_param_shape[op.input[ONNX_WEIGHT_INDEX_CONV]])
            if bias_term:
                param_dict[layer_name + PARAM_DICT_BIAS_SUFFIX] = const_dict[op.input[ONNX_BIAS_INDEX_CONV]].reshape(input_and_param_shape[op.input[ONNX_BIAS_INDEX_CONV]])
        elif op.op_type == ONNX_NAME_BN:
            assert op.input[ONNX_TENSOR_INDEX_BN] not in const_dict.keys(), "Batchnormlization's input feature shouldn't be a constant, check layer {}".format(layer_name)
            epsilon = 1e-5
            for attribute in op.attribute:
                if attribute.name == ONNX_NAME_EPSILON:
                    epsilon = attribute.f
            layer_list.append(Layer(lyr_type = ONNX_NAME_BN, name = layer_name, top = op.output[0], bottom = op.input[ONNX_TENSOR_INDEX_BN]))
            std = np.sqrt(const_dict[op.input[ONNX_VAR_INDEX_BN]] + epsilon)
            gamma = const_dict[op.input[ONNX_GAMMR_INDEX_BN]]
            beta = const_dict[op.input[ONNX_BETA_INDEX_BN]]
            mean = const_dict[op.input[ONNX_MEAN_INDEX_BN]]
            param_dict[layer_name + PARAM_DICT_WEIGHT_SUFFIX] = gamma / std
            param_dict[layer_name + PARAM_DICT_BIAS_SUFFIX] = beta - mean * gamma / std
            tensor_shape[op.output[0]] = copy.deepcopy(tensor_shape[op.input[ONNX_TENSOR_INDEX_BN]])
        elif op.op_type == ONNX_NAME_GEMM:
            assert op.input[ONNX_WEIGHT_INDEX_GEMM] in const_dict.keys(), "Gemm operation's input tensor {} should be a constant because it is considered to be the weight of a fully connected layer, check layer {}".format(ONNX_WEIGHT_INDEX_GEMM + 1, layer_name)
            assert op.input[ONNX_TENSOR_INDEX_GEMM] not in const_dict.keys(), "Gemm operation's input tensor {} should be a non-constant because it is considered to be the input feature of a fully connected layer, check layer {}".format(ONNX_TENSOR_INDEX_GEMM + 1, layer_name)
            bias_term = False
            if len(op.input) > ONNX_BIAS_INDEX_GEMM:
                bias_term = True
                assert op.input[ONNX_BIAS_INDEX_GEMM] in const_dict.keys(), "Gemm operation's input tensor {} should be a constant because it is considered to be the bias of a fully connected layer, check layer {}".format(ONNX_BIAS_INDEX_GEMM + 1, layer_name)
            alpha = 1.0
            beta = 1.0
            trans = [0, 0]
            for attribute in op.attribute:
                if attribute.name == ONNX_NAME_GEMM_APLHA:
                    alpha = attribute.f
                if attribute.name == ONNX_NAME_GEMM_BETA:
                    beta = attribute.f
                if attribute.name == ONNX_NAME_GEMM_TRANS_A:
                    trans[0] = attribute.i
                if attribute.name == ONNX_NAME_GEMM_TRANS_B:
                    trans[1] = attribute.i
            assert trans[ONNX_TENSOR_INDEX_GEMM] == 0, "To transpose a non-const tensor in Gemm operation is not supported, check layer {}".format(layer_name)
            if op.input[ONNX_WEIGHT_INDEX_GEMM] not in input_and_param_shape.keys():
                input_and_param_shape[op.input[ONNX_WEIGHT_INDEX_GEMM]] = const_dict[op.input[ONNX_WEIGHT_INDEX_GEMM]].shape
            if bias_term:
                if op.input[ONNX_BIAS_INDEX_GEMM] not in input_and_param_shape.keys():
                    input_and_param_shape[op.input[ONNX_BIAS_INDEX_GEMM]] = const_dict[op.input[ONNX_BIAS_INDEX_GEMM]].shape
            weight = const_dict[op.input[ONNX_WEIGHT_INDEX_GEMM]].reshape(input_and_param_shape[op.input[ONNX_WEIGHT_INDEX_GEMM]])
            if trans[ONNX_WEIGHT_INDEX_GEMM] == 0:
                param_dict[layer_name + PARAM_DICT_WEIGHT_SUFFIX] = weight.transpose((1, 0)) * alpha
            else:
                param_dict[layer_name + PARAM_DICT_WEIGHT_SUFFIX] = weight * alpha
            weight_shape = list(weight.shape)
            weight_shape.extend([1, 1, 1, 1])
            weight_shape = weight_shape[0:4]
            param_dict[layer_name + PARAM_DICT_WEIGHT_SUFFIX] = param_dict[layer_name + PARAM_DICT_WEIGHT_SUFFIX].reshape(weight_shape)
            co = param_dict[layer_name + PARAM_DICT_WEIGHT_SUFFIX].shape[0]
            if bias_term:
                assert const_dict[op.input[ONNX_BIAS_INDEX_GEMM]].size == co, "Gemm operation's input tensor {}'s size should be equal to output's width because it is considered to be the bias of a fully connected layer, check layer {}".format(ONNX_BIAS_INDEX_GEMM + 1, layer_name)
                param_dict[layer_name + PARAM_DICT_BIAS_SUFFIX] = const_dict[op.input[ONNX_BIAS_INDEX_GEMM]].reshape(co)
            tensor_shape[op.output[0]] = [tensor_shape[op.input[ONNX_TENSOR_INDEX_GEMM]][0], co]
            layer_list.append(Layer(lyr_type = 'ip', name = layer_name, bottom = op.input[ONNX_TENSOR_INDEX_GEMM], top = op.output[0], num_output = co, bias_term = bias_term))
        elif op.op_type == ONNX_NAME_MAT_MUL:
            assert op.input[ONNX_WEIGHT_INDEX_MAT_MUL] in const_dict.keys(), "Gemm operation's input tensor {} should be a constant because it is considered to be the weight of a fully connected layer, check layer {}".format(ONNX_WEIGHT_INDEX_GEMM + 1, layer_name)
            assert op.input[ONNX_TENSOR_INDEX_MAT_MUL] not in const_dict.keys(), "Gemm operation's input tensor {} should be a non-constant because it is considered to be the input feature of a fully connected layer, check layer {}".format(ONNX_TENSOR_INDEX_GEMM + 1, layer_name)
            weight = const_dict[op.input[ONNX_WEIGHT_INDEX_MAT_MUL]].reshape(input_and_param_shape[op.input[ONNX_WEIGHT_INDEX_MAT_MUL]])
            param_dict[layer_name + PARAM_DICT_WEIGHT_SUFFIX] = weight.transpose((1, 0))
            co = param_dict[layer_name + PARAM_DICT_WEIGHT_SUFFIX].shape[0]
            tensor_shape[op.output[0]] = [tensor_shape[op.input[ONNX_TENSOR_INDEX_MAT_MUL]][0], co]
            layer_list.append(Layer(lyr_type = 'ip', name = layer_name, bottom = op.input[ONNX_TENSOR_INDEX_MAT_MUL], top = op.output[0], num_output = co, bias_term = False))
        elif op.op_type in [ONNX_NAME_AVG_POOL, ONNX_NAME_MAX_POOL]:
            assert op.input[ONNX_TENSOR_INDEX_POOL] not in const_dict.keys(), "Pooling's input feature shouldn't be a constant, check layer {}".format(layer_name)
            ci = tensor_shape[op.input[ONNX_TENSOR_INDEX_POOL]][1]
            hi = tensor_shape[op.input[ONNX_TENSOR_INDEX_POOL]][2]
            wi = tensor_shape[op.input[ONNX_TENSOR_INDEX_POOL]][3]
            kernel_size = [1, 1]
            stride = [1, 1]
            dilation = [1, 1]
            pad = [0, 0, 0, 0]  
            for attribute in op.attribute:
                if attribute.name == ONNX_NAME_KERNEL_SIZE:
                    kernel_size = attribute.ints
                if attribute.name == ONNX_NAME_DILATION:
                    dilation = attribute.ints
                if attribute.name == ONNX_NAME_STRIDE:
                    stride = attribute.ints
                if attribute.name == ONNX_NAME_PADDING:
                    pad = attribute.ints
                if attribute.name == ONNX_NAME_AUTO_PAD:
                    raise Exception("auto_pad is not supported, check layer {}".format(layer_name))
                if attribute.name == ONNX_NAME_COUNT_INCLUDE_PAD:
                    count_include_pad = attribute.i
                    assert count_include_pad == 0, "count include pad is not supported, check layer {}".format(layer_name)
            kh = kernel_size[0]
            kw = kernel_size[1]
            sh = stride[0]
            sw = stride[1]
            dh = dilation[0]
            dw = dilation[1]
            khd = (kh - 1) * dh + 1
            kwd = (kw - 1) * dw + 1
            pn = pad[0]
            ps = pad[2]
            pw = pad[1]
            pe = pad[3]
            ho = (hi + pn + ps - khd) // sh + 1
            wo = (wi + pw + pe - kwd) // sw + 1
            hi_need = (ho - 1) * sh + khd
            wi_need = (wo - 1) * sw + kwd
            ps = hi_need - hi - pn
            pe = wi_need - wi - pw
            
            layer_list.append(Layer(lyr_type = 'pool', name = layer_name, top = op.output[0], bottom = op.input[0], 
                              kernel_size_h=kh, kernel_size_w=kw, stride_h=sh, stride_w=sw, 
                              pad_n=pn, pad_s=ps, pad_w=pw, pad_e=pe, dilation_h=dh, dilation_w=dw, 
                              pool='ave' if op.op_type == ONNX_NAME_AVG_POOL else 'max'))
            
            tensor_shape[op.output[0]] = [tensor_shape[op.input[ONNX_TENSOR_INDEX_POOL]][0], ci, ho, wo]
        elif op.op_type in [ONNX_NAME_GLB_AVG_POOL, ONNX_NAME_GLB_MAX_POOL]:
            print(tensor_shape)
            assert op.input[ONNX_TENSOR_INDEX_POOL] not in const_dict.keys(), "Pooling's input feature shouldn't be a constant, check layer {}".format(layer_name)
            bi = tensor_shape[op.input[ONNX_TENSOR_INDEX_POOL]][0]
            ci = tensor_shape[op.input[ONNX_TENSOR_INDEX_POOL]][1]
            hi = tensor_shape[op.input[ONNX_TENSOR_INDEX_POOL]][2]
            wi = tensor_shape[op.input[ONNX_TENSOR_INDEX_POOL]][3]
            kernel_size = (int(hi), int(wi))
            layer_list.append(Layer(lyr_type = 'pool', name = layer_name, top = op.output[0], bottom = op.input[0], 
                  kernel_size=kernel_size, stride=1, pad=0, dilation=1, 
                  pool='ave' if op.op_type == ONNX_NAME_GLB_AVG_POOL else 'max'))
            tensor_shape[op.output[0]] = [bi, ci, 1, 1]
        elif op.op_type in [ONNX_NAME_REDUCE_MEAN, ONNX_NAME_REDUCE_MAX]:
            print(tensor_shape)
            for attribute in op.attribute:
                if attribute.name == ONNX_NAME_AXES:
                    assert attribute.ints == [2, 3], "Only ReduceMean on both H and W is supported, but got {} instead, check layer {}".format(attribute.ints, op.name)
            bi = tensor_shape[op.input[ONNX_TENSOR_INDEX_REDUCE]][0]
            ci = tensor_shape[op.input[ONNX_TENSOR_INDEX_REDUCE]][1]
            hi = tensor_shape[op.input[ONNX_TENSOR_INDEX_REDUCE]][2]
            wi = tensor_shape[op.input[ONNX_TENSOR_INDEX_REDUCE]][3]
            kernel_size = (int(hi), int(wi))
            layer_list.append(Layer(lyr_type = 'pool', name = layer_name, top = op.output[0], bottom = op.input[0], 
                  kernel_size=kernel_size, stride=1, pad=0, dilation=1, 
                  pool='ave' if op.op_type == ONNX_NAME_REDUCE_MEAN else 'max'))
            tensor_shape[op.output[0]] = [bi, ci, 1, 1]
        elif op.op_type == ONNX_NAME_NEG:
            if op.input[0] in const_dict.keys():
                const_dict[op.output[0]] = np.negative(const_dict[op.input[0]])
            else:
                ci = tensor_shape[op.input[0]][1]
                param_dict[layer_name + PARAM_DICT_WEIGHT_SUFFIX] = -np.ones(ci)
                layer_list.append(Layer(lyr_type = "scale", name = layer_name, top = op.output[0], bottom = op.input[0], bias_term = False))
                tensor_shape[op.output[0]] = copy.deepcopy(tensor_shape[op.input[0]])
        elif op.op_type == ONNX_NAME_ABS:
            if op.input[0] in const_dict.keys():
                const_dict[op.output[0]] = np.abs(const_dict[op.input[0]], 0, np.inf)
            else:
                print("Abs found, it will be changed to relu with negative slope -1")
                layer_list.append(Layer(lyr_type = ONNX_NAME_RELU, name = layer_name, top = op.output[0], bottom = op.input[0], negative_slope = -1.0))
                tensor_shape[op.output[0]] = copy.deepcopy(tensor_shape[op.input[0]])
        elif op.op_type == ONNX_NAME_RELU:
            if op.input[0] in const_dict.keys():
                const_dict[op.output[0]] = np.clip(const_dict[op.input[0]], 0, np.inf)
            else:
                layer_list.append(Layer(lyr_type = ONNX_NAME_RELU, name = layer_name, top = op.output[0], bottom = op.input[0]))
                tensor_shape[op.output[0]] = copy.deepcopy(tensor_shape[op.input[0]])
        elif op.op_type == ONNX_NAME_LEAKY_RELU:
            alpha = 0.01
            for attribute in op.attribute:
                if attribute.name == ONNX_NAME_LEAKY_PARAM:
                    alpha = attribute.f
            if op.input[0] in const_dict.keys():
                const_dict[op.output[0]] = np.clip(const_dict[op.input[0]], 0, np.inf) + alpha * np.clip(const_dict[op.input[0]], -np.inf, 0)
            else:
                layer_list.append(Layer(lyr_type = ONNX_NAME_RELU, name = layer_name, top = op.output[0], bottom = op.input[0], negative_slope = alpha))
                tensor_shape[op.output[0]] = copy.deepcopy(tensor_shape[op.input[0]])
        elif op.op_type == ONNX_NAME_PRELU:
            assert op.input[ONNX_WEIGHT_INDEX_PRELU] in const_dict.keys(), "Prelu's parameter should be a constant, check layer {}"
            if op.input[ONNX_TENSOR_INDEX_PRELU] in const_dict.keys():
                const_dict[op.output[0]] = np.clip(const_dict[op.input[ONNX_TENSOR_INDEX_PRELU]], 0, np.inf) + const_dict[op.input[ONNX_WEIGHT_INDEX_PRELU]] * np.clip(const_dict[op.input[ONNX_TENSOR_INDEX_PRELU]], -np.inf, 0)
            else:
                ci = tensor_shape[op.input[ONNX_TENSOR_INDEX_PRELU]][1]
                prelu_weight = const_dict[op.input[ONNX_WEIGHT_INDEX_PRELU]]
                assert ci == prelu_weight.size, "Prelu operation's prelu parameter's size should equal to input feature's number of channels because each element in prelu parameter will be operated on the correspond channel of input feature. But got input feature's number of channel is {} and size of prelu parameter is {}. check layer".format(ci, prelu_weight.size, layer_name)
                positive_part_layer_name = layer_name + "/positive_part/relu"
                positive_part_layer_out_name = op.input[0] + "/positive_part/relu"
                negative_part_neg_layer_name = layer_name + "/negative_part/neg"
                negative_part_neg_layer_out_name = op.input[0] + "/negative_part/neg"
                negative_part_relu_layer_name = layer_name + "/negative_part/relu"
                negative_part_relu_layer_out_name = op.input[0] + "/negative_part/relu"
                negative_part_scale_layer_name = layer_name + "/negative_part/scale"
                negative_part_scale_layer_out_name = op.input[0] + "/negative_part/scale"
                add_layer_name = layer_name + "/add"
                param_dict[negative_part_neg_layer_name + PARAM_DICT_WEIGHT_SUFFIX] = -1.0 * np.ones(ci)
                param_dict[negative_part_scale_layer_name + PARAM_DICT_WEIGHT_SUFFIX] = -1.0 * prelu_weight
                layer_list.append(Layer(lyr_type = ONNX_NAME_RELU, name = positive_part_layer_name, top = positive_part_layer_out_name, bottom = op.input[0]))
                layer_list.append(Layer(lyr_type = "scale", name = negative_part_neg_layer_name, top = negative_part_neg_layer_out_name, bottom = op.input[0], bias_term = False))
                layer_list.append(Layer(lyr_type = ONNX_NAME_RELU, name = negative_part_relu_layer_name, top = negative_part_relu_layer_out_name, bottom = negative_part_neg_layer_out_name))
                layer_list.append(Layer(lyr_type = "scale", name = negative_part_scale_layer_name, top = negative_part_scale_layer_out_name, bottom = negative_part_relu_layer_out_name, bias_term = False))
                layer_list.append(Layer(lyr_type = 'eltwise', name = add_layer_name, top = op.output[0], bottom = [positive_part_layer_out_name, negative_part_scale_layer_out_name]))
                tensor_shape[positive_part_layer_out_name] = copy.deepcopy(tensor_shape[op.input[0]])
                tensor_shape[negative_part_neg_layer_out_name] = copy.deepcopy(tensor_shape[op.input[0]])
                tensor_shape[negative_part_relu_layer_out_name] = copy.deepcopy(tensor_shape[op.input[0]])
                tensor_shape[negative_part_scale_layer_out_name] = copy.deepcopy(tensor_shape[op.input[0]])
                tensor_shape[op.output[0]] = copy.deepcopy(tensor_shape[op.input[0]])
        elif op.op_type == ONNX_NAME_CLIP:
            min_value = -np.inf
            max_value = np.inf
            for attribute in op.attribute:
                if attribute.name == ONNX_NAME_MAX:
                    max_value = attribute.f
                if attribute.name == ONNX_NAME_MIN:
                    min_value = attribute.f
            if op.input[0] in const_dict.keys():
                const_dict[op.output[0]] = np.clip(const_dict[op.input[0]], min_value, max_value)
            else:
                assert min_value > -np.inf, "Only clip operation in which min and max are both set is supported, check layer {}".format(layer_name)
                assert max_value < np.inf, "Only clip operation in which min and max are both set is supported, check layer {}".format(layer_name)
                ci = tensor_shape[op.input[0]][1]
                w0 = 6 / (max_value - min_value)
                b0 = -min_value * w0
                w1 = (max_value - min_value) / 6
                b1 = min_value
                if abs(min_value) < 1e-10 and abs(w0 - 1) < 1e-10:
                    print("Clip's max value is 6.0 and min value is 0.0, it will be changed to relu6")
                    layer_list.append(Layer(lyr_type = 'relu6', name = layer_name, bottom = op.input[0], top = op.output[0]))
                    tensor_shape[op.output[0]] = copy.deepcopy(tensor_shape[op.input[0]])
                else:
                    print("Clip operation is found, it will be changed to [scale layer, relu6 layer, scale layer]")
                    scale0_layer_name = layer_name + "/inside_scale0"
                    scale0_out_name = scale0_layer_name + "_output"
                    relu6_layer_name = layer_name + "/inside_relu6"
                    relu6_out_name = relu6_layer_name + "_output"
                    scale1_layer_name = layer_name + "/inside_scale1"
                    param_dict[scale0_layer_name + PARAM_DICT_WEIGHT_SUFFIX] = np.array([w0] * ci)
                    param_dict[scale0_layer_name + PARAM_DICT_BIAS_SUFFIX] = np.array([b0] * ci)
                    param_dict[scale1_layer_name + PARAM_DICT_WEIGHT_SUFFIX] = np.array([w1] * ci)
                    param_dict[scale1_layer_name + PARAM_DICT_BIAS_SUFFIX] = np.array([b1] * ci)
                    layer_list.append(Layer(lyr_type = 'scale', name = scale0_layer_name, bottom = op.input[0], top = scale0_out_name))
                    layer_list.append(Layer(lyr_type = 'relu6', name = relu6_layer_name, bottom = scale0_out_name, top = relu6_out_name))
                    layer_list.append(Layer(lyr_type = 'scale', name = scale1_layer_name, bottom = relu6_out_name, top = op.output[0]))
                    tensor_shape[scale0_out_name] = copy.deepcopy(tensor_shape[op.input[0]])
                    tensor_shape[relu6_out_name] = copy.deepcopy(tensor_shape[op.input[0]])
                    tensor_shape[op.output[0]] = copy.deepcopy(tensor_shape[op.input[0]])
        elif op.op_type == ONNX_NAME_SIGMOID:
            if op.input[0] in const_dict.keys():
                const_dict[op.output[0]] = 1 / (1 + np.exp(-const_dict[op.input[0]]))
            else:
                layer_list.append(Layer(lyr_type = ONNX_NAME_SIGMOID, name = layer_name, top = op.output[0], bottom = op.input[0]))
                tensor_shape[op.output[0]] = copy.deepcopy(tensor_shape[op.input[0]])
        elif op.op_type == ONNX_NAME_TANH:
            if op.input[0] in const_dict.keys():
                const_dict[op.output[0]] = np.tanh(const_dict[op.input[0]])
            else:
                layer_list.append(Layer(lyr_type = ONNX_NAME_TANH, name = layer_name, top = op.output[0], bottom = op.input[0]))
                tensor_shape[op.output[0]] = copy.deepcopy(tensor_shape[op.input[0]])
        elif op.op_type in [ONNX_NAME_SOFTMAX, ONNX_NAME_LOGSOFTMAX]:
            axis = 1
            for attribute in op.attribute:
                if attribute.name == ONNX_NAME_AXIS:
                    axis = attribute.i
            if op.input[0] in const_dict.keys():
                const_dict[op.output[0]] = np.exp(op.input[0]) / np.sum(np.exp(op.input[0]), axis = axis)
            else:
                assert axis == 1, "Only Softmax or Logsoftmax operation on Axis 1 is supported if input is non-constant, but got axis {} in layer {}".format(axis, layer_name)
                if op.op_type == ONNX_NAME_SOFTMAX:
                    print("Warn: Softmax is not supported in NDK, layer {} will be changed to Logsoftmax instead, you could use exp operation to get softmax".format(layer_name))
                layer_list.append(Layer(lyr_type = ONNX_NAME_LOGSOFTMAX, name = layer_name, top = op.output[0], bottom = op.input[0]))
        elif op.op_type == ONNX_NAME_ADD:
            assert len(op.input) == 2, "only adding 2 input tensors is supported, but get {} in layer {}".format(len(op.input), layer_name)
            if op.input[0] in const_dict.keys():
                if op.input[1] in const_dict.keys():
                    const_dict[op.output[0]] = const_dict[op.input[0]] + const_dict[op.input[1]]
                else:
                    const_to_add = const_dict[op.input[0]]
                    ci = tensor_shape[op.input[1]][1]
                    layer_list.append(Layer(lyr_type = 'bias', name = layer_name, top = op.output[0], bottom = op.input[1]))
                    if const_to_add.size == 1:
                        param_dict[layer_name + PARAM_DICT_BIAS_SUFFIX] = np.ones(ci) * const_to_add
                    else:
                        assert const_to_add.size == ci, "only bias on channel is supported, but get constant's size is {} while tensor's number of channel is {} in layer {}".format(const_to_add.size, ci, layer_name)
                        param_dict[layer_name + PARAM_DICT_BIAS_SUFFIX] = const_to_add.reshape(ci)
                    tensor_shape[op.output[0]] = copy.deepcopy(tensor_shape[op.input[1]])
            else:
                if op.input[1] in const_dict.keys():
                    const_to_add = const_dict[op.input[1]]
                    ci = tensor_shape[op.input[0]][1]
                    layer_list.append(Layer(lyr_type = 'bias', name = layer_name, top = op.output[0], bottom = op.input[0]))
                    if const_to_add.size == 1:
                        param_dict[layer_name + PARAM_DICT_BIAS_SUFFIX] = np.ones(ci) * const_to_add
                    else:
                        assert const_to_add.size == ci, "only bias on channel is supported, but get constant's size is {} while tensor's number of channel is {} in layer {}".format(const_to_add.size, ci, layer_name)
                        param_dict[layer_name + PARAM_DICT_BIAS_SUFFIX] = const_to_add.reshape(ci)
                    tensor_shape[op.output[0]] = copy.deepcopy(tensor_shape[op.input[0]])
                else:
                    layer_list.append(Layer(lyr_type = 'eltwise', name = layer_name, top = op.output[0], bottom = list(op.input)))
                    tensor_shape[op.output[0]] = copy.deepcopy(tensor_shape[op.input[0]])
        elif op.op_type == ONNX_NAME_SUB:
            assert len(op.input) == 2, "only 2 input tensors is supported in Sub operation, but get {} in layer {}".format(len(op.input), layer_name)
            if op.input[0] in const_dict.keys():
                if op.input[1] in const_dict.keys():
                    const_dict[op.output[0]] = const_dict[op.input[0]] - const_dict[op.input[1]]
                else:
                    const_to_sub = const_dict[op.input[0]]
                    ci = tensor_shape[op.input[1]][1]
                    layer_list.append(Layer(lyr_type = 'scale', name = layer_name, top = op.output[0], bottom = op.input[1], bias_term = True))
                    if const_to_sub.size == 1:
                        param_dict[layer_name + PARAM_DICT_WEIGHT_SUFFIX] = -np.ones(ci)
                        param_dict[layer_name + PARAM_DICT_BIAS_SUFFIX] = np.ones(ci) * const_to_sub
                    else:
                        assert const_to_sub.size == ci, "only bias on channel is supported, but get constant's size is {} while tensor's number of channel is {} in layer {}".format(const_to_sub.size, ci, layer_name)
                        param_dict[layer_name + PARAM_DICT_WEIGHT_SUFFIX] = -np.ones(ci)
                        param_dict[layer_name + PARAM_DICT_BIAS_SUFFIX] = const_to_sub.reshape(ci)
                    tensor_shape[op.output[0]] = copy.deepcopy(tensor_shape[op.input[1]])
            else:
                if op.input[1] in const_dict.keys():
                    const_to_sub = const_dict[op.input[1]]
                    ci = tensor_shape[op.input[0]][1]
                    layer_list.append(Layer(lyr_type = 'bias', name = layer_name, top = op.output[0], bottom = op.input[0]))
                    if const_to_sub.size == 1:
                        param_dict[layer_name + PARAM_DICT_BIAS_SUFFIX] = np.ones(ci) * (-const_to_sub)
                    else:
                        assert const_to_sub.size == ci, "only bias on channel is supported, but get constant's size is {} while tensor's number of channel is {} in layer {}".format(const_to_sub.size, ci, layer_name)
                        param_dict[layer_name + PARAM_DICT_BIAS_SUFFIX] = -const_to_sub.reshape(ci)
                    tensor_shape[op.output[0]] = copy.deepcopy(tensor_shape[op.input[0]])
                else:
                    raise Exception("Sub operation on 2 non-constant tensor is not supported, check layer {}".format(layer_name))
        elif op.op_type == ONNX_NAME_MUL:
            assert len(op.input) == 2, "only 2 input tensors is supported in Mul operation, but get {} in layer {}".format(len(op.input), layer_name)
            if op.input[0] in const_dict.keys():
                if op.input[1] in const_dict.keys():
                    const_dict[op.output[0]] = const_dict[op.input[0]] * const_dict[op.input[1]]
                else:
                    const_to_mul = const_dict[op.input[0]]
                    ci = tensor_shape[op.input[1]][1]
                    layer_list.append(Layer(lyr_type = 'scale', name = layer_name, top = op.output[0], bottom = op.input[1], bias_term = False))
                    if const_to_mul.size == 1:
                        param_dict[layer_name + PARAM_DICT_WEIGHT_SUFFIX] = np.ones(ci) * const_to_mul
                    else:
                        assert const_to_mul.size == ci, "only bias on channel is supported, but get constant's size is {} while tensor's number of channel is {} in layer {}".format(const_to_mul.size, ci, layer_name)
                        param_dict[layer_name + PARAM_DICT_WEIGHT_SUFFIX] = const_to_mul.reshape(ci)
                    tensor_shape[op.output[0]] = copy.deepcopy(tensor_shape[op.input[1]])
            else:
                if op.input[1] in const_dict.keys():
                    const_to_mul = const_dict[op.input[1]]
                    ci = tensor_shape[op.input[0]][1]
                    layer_list.append(Layer(lyr_type = 'scale', name = layer_name, top = op.output[0], bottom = op.input[0], bias_term = False))
                    if const_to_mul.size == 1:
                        param_dict[layer_name + PARAM_DICT_WEIGHT_SUFFIX] = np.ones(ci) * const_to_mul
                    else:
                        assert const_to_mul.size == ci, "only bias on channel is supported, but get constant's size is {} while tensor's number of channel is {} in layer {}".format(const_to_mul.size, ci, layer_name)
                        param_dict[layer_name + PARAM_DICT_WEIGHT_SUFFIX] = const_to_mul.reshape(ci)
                    tensor_shape[op.output[0]] = copy.deepcopy(tensor_shape[op.input[0]])
                else:
                    _, _, h0, w0 = tensor_shape[op.input[0]]
                    _, _, h1, w1 = tensor_shape[op.input[1]]
                    if h0 == 1 and w0 == 1:
                        layer_list.append(Layer(lyr_type = 'scaleByTensor', name = layer_name, top = op.output[0], bottom = [op.input[1], op.input[0]]))
                        tensor_shape[op.output[0]] = copy.deepcopy(tensor_shape[op.input[1]])
                    elif h1 == 1 and w1 == 1:
                        layer_list.append(Layer(lyr_type = 'scaleByTensor', name = layer_name, top = op.output[0], bottom = [op.input[0], op.input[1]]))
                        tensor_shape[op.output[0]] = copy.deepcopy(tensor_shape[op.input[0]])
                    else:
                        raise Exception("Mul operation on 2 non-constant tensor is not supported, check layer {}".format(layer_name))
        elif op.op_type == ONNX_NAME_DIV:
            assert len(op.input) == 2, "only 2 input tensors is supported in Div operation, but get {} in layer {}".format(len(op.input), layer_name)
            assert  op.input[1] in const_dict.keys(), "Non-constant divisor is nor supported, check layer {}".format(layer_name)
            if op.input[0] in const_dict.keys():
                const_dict[op.output[0]] = const_dict[op.input[0]] / const_dict[op.input[1]]
            else:
                const_to_div = const_dict[op.input[1]]
                ci = tensor_shape[op.input[0]][1]
                layer_list.append(Layer(lyr_type = 'scale', name = layer_name, top = op.output[0], bottom = op.input[0], bias_term = False))
                if const_to_div.size == 1:
                    param_dict[layer_name + PARAM_DICT_WEIGHT_SUFFIX] = np.ones(ci) / const_to_div
                else:
                    assert const_to_div.size == ci, "only bias on channel is supported, but get constant's size is {} while tensor's number of channel is {} in layer {}".format(const_to_div.size, ci, layer_name)
                    param_dict[layer_name + PARAM_DICT_WEIGHT_SUFFIX] = 1.0 / const_to_div.reshape(ci)
                tensor_shape[op.output[0]] = copy.deepcopy(tensor_shape[op.input[0]])
        elif op.op_type == ONNX_NAME_CONCAT:
            assert len(op.input) > 1, "Concat operation's inputs should be at least 2 tensors, check layers {}".format(layer_name)
            const_flag = set()
            for input_name in op.input:
                const_flag.add(input_name in const_dict.keys())
            assert len(const_flag) == 1, "Concat operation's inputs should be all constants or all non-constants, check layer {}".format(layer_name)
            axis = op.attribute[0].i
            if list(const_flag)[0]:
                const_dict[op.output[0]] = np.concatenate([const_dict[input_name] for input_name in op.input], axis = axis)
            else:
                layer_list.append(Layer(lyr_type = ONNX_NAME_CONCAT, name = layer_name, top = op.output[0], bottom = list(op.input), axis = axis))
                dim_along_concat = [tensor_shape[input_name][axis] for input_name in op.input]
                tensor_shape[op.output[0]] = copy.deepcopy(tensor_shape[op.input[0]])
                tensor_shape[op.output[0]][axis] = sum(dim_along_concat)                
        elif op.op_type == ONNX_NAME_SPLIT:
            axis = 0
            split = []
            for attribute in op.attribute:
                if attribute.name == ONNX_NAME_AXIS:
                    axis = attribute.i
                if attribute.name == ONNX_NAME_SPLIT_SIZE:
                    split = attribute.ints
            if op.input[0] in const_dict:
                if len(split) > 0:
                    assert len(split) == len(op.output), "Split operation's split parameter should be a list whose length equals to the length of op.output, but got len(split) = {} and len(op.output) = {} in layer {}".format(len(split), len(op.output), layer_name)
                    assert len(split) > 1, "Split operation should have more than one output, check layer {}".format(layer_name)
                    slice_point = [sum(split[:(i + 1)]) for i in range(len(split) - 1)]
                    split_result = np.split(const_dict[op.input[0]], slice_point, axis)
                    for i in range(len(split)):
                        const_dict[op.output[i]] = split_result[i]
                else:
                    split_result = np.split(const_dict[op.input[0]], len(op.output[0]), axis)
                    for i in range(len(split)):
                        const_dict[op.output[i]] = split_result[i]
            else:
                slice_point = None
                if len(split) > 0:
                    assert len(split) == len(op.output), "Split operation's split parameter should be a list whose length equals to the length of op.output, but got len(split) = {} and len(op.output) = {} in layer {}".format(len(split), len(op.output), layer_name)
                    assert len(split) > 1, "Split operation should have more than one output, check layer {}".format(layer_name)
                    slice_point = [sum(split[:(i + 1)]) for i in range(len(split) - 1)]
                    for i in range(len(op.output)):
                        output_name = op.output[i]
                        tensor_shape[output_name] = copy.deepcopy(tensor_shape[op.input[0]])
                        tensor_shape[output_name][axis] = split[i]
                else:
                    assert tensor_shape[op.input[0]][axis] % len(op.output[0]) == 0, "Split does not result in an equal division, input tensor's shape is {} and split axis is {} while you want to split it into {} parts, check layer {}".format(tensor_shape[op.input[0]], axis, len(op.output), layer_name)
                    for i in range(len(op.output)):
                        output_name = op.output[i]
                        tensor_shape[output_name] = copy.deepcopy(tensor_shape[op.input[0]])
                        tensor_shape[output_name][axis] = tensor_shape[op.input[0]][axis] // len(op.output[0])
                layer_list.append(Layer(lyr_type = ONNX_NAME_SPLIT, name = layer_name, top = list(op.output), bottom = op.input[0], axis = axis, slice_point = slice_point))
        elif op.op_type == ONNX_NAME_SHAPE:
            const_dict[op.output[0]] = np.array(tensor_shape[op.input[0]])
        elif op.op_type == ONNX_NAME_GATHER:
            for input_name in op.input:
                assert input_name in const_dict.keys(), "Gather operation on non-constant tensor is not supported, check layer {}".format(layer_name)
            axis = 0
            if hasattr(op, 'attribute') and len(op.attribute) > 0 and op.attribute[0].name == ONNX_NAME_AXIS:
                axis = op.attribute[0].i
            const_dict[op.output[0]] = np.take(const_dict[op.input[0]], const_dict[op.input[1]], axis = axis)
        elif op.op_type == ONNX_NAME_UNSQUEEZE:
            assert op.input[0] in const_dict, "Unsqueeze operation on a non-constant tensor is not supported, check layer {}".format(layer_name)
            axes = list(op.attribute[0].ints)
            axes.sort()
            axis = axes.pop(0)
            const_dict[op.output[0]] = np.expand_dims(const_dict[op.input[0]], axis = axis)
            for axis in axes:
                const_dict[op.output[0]] = np.expand_dims(const_dict[op.output[0]], axis = axis)
        elif op.op_type == ONNX_NAME_RESHAPE:
            assert op.input[1] in const_dict, "Shape parameter in reshape operation should be a constant, check layer {}".format(layer_name)
            if op.input[0] in const_dict:
                const_dict[op.output[0]] = np.reshape(const_dict[op.input[0]], const_dict[op.input[1]])
            else:
                fake_layer = Layer()
                fake_layer.name = layer_name
                fake_layer.type = ONNX_NAME_RESHAPE
                fake_layer.bottom = op.input[0]
                fake_layer.top = op.output[0]
                layer_list.append(fake_layer)
                fake_input = np.zeros((tensor_shape[op.input[0]]))
                fake_input = fake_input.reshape(const_dict[op.input[1]])
                tensor_shape[op.output[0]] = list(fake_input.shape)
        elif op.op_type == ONNX_NAME_FLATTEN:
            axis = 1
            for attribute in op.attribute:
                if attribute.name == ONNX_NAME_AXIS:
                    axis = attribute.i
            if op.input[0] in const_dict:
                shape_in = const_dict[op.input[0]].shape
                shape_out = [1, 1]
                for i in range(axis):
                    shape_out[0] = shape_out[0] * shape_in[i]
                for i in range(axis, len(shape_in)):
                    shape_out[1] = shape_out[1] * shape_in[i]
                const_dict[op.output[0]] = np.reshape(const_dict[op.input[0]], shape_out)
            else:
                fake_layer = Layer()
                fake_layer.name = layer_name
                fake_layer.type = ONNX_NAME_RESHAPE
                fake_layer.bottom = op.input[0]
                fake_layer.top = op.output[0]
                layer_list.append(fake_layer)
                shape_in = tensor_shape[op.input[0]]
                shape_out = [1, 1]
                for i in range(axis):
                    shape_out[0] = shape_out[0] * shape_in[i]
                for i in range(axis, len(shape_in)):
                    shape_out[1] = shape_out[1] * shape_in[i]
                tensor_shape[op.output[0]] = shape_out
        elif op.op_type == ONNX_NAME_TRANSPOSE:
            if op.input[0] in const_dict:
                perm = list(range(len(const_dict[op.input[0]].shape)))
                perm.reverse()
                const_dict[op.output[0]] = np.transpose(const_dict[op.input[0]], perm)
                input_and_param_shape[op.output[0]] = const_dict[op.output[0]].shape
            else:
                perm = list(range(len(tensor_shape[op.input[0]])))
                perm.reverse()
                for attribute in op.attribute:
                    if attribute.name == ONNX_NAME_TRANSPOSE_PERM:
                        perm = attribute.ints
                fake_layer = Layer()
                fake_layer.name = layer_name
                fake_layer.type = ONNX_NAME_TRANSPOSE
                fake_layer.bottom = op.input[0]
                fake_layer.top = op.output[0]
                fake_layer.perm = perm
                layer_list.append(fake_layer)
                tensor_shape[op.output[0]] = [tensor_shape[op.input[0]][i] for i in perm]
        elif op.op_type == ONNX_NAME_PAD:
            pads = []
            pad_value = 0.0
            pad_mode = ONNX_NAME_PAD_CONSTANT
            for attribute in op.attribute:
                if attribute.name == ONNX_NAME_PAD_MODE:
                    pad_mode = attribute.s
                if attribute.name == ONNX_NAME_PAD_VALUE:
                    pad_value = attribute.f
                if attribute.name == ONNX_NAME_PAD_NUMBER:
                    pads = attribute.ints
            assert len(pads) > 1, "Attribute pads is required in Pad operation, check layer {}".format(layer_name)
            pads_numpy = np.transpose(np.reshape(np.array(pads), (2, -1)))
            if op.input[0] in const_dict.keys():
                const_dict[op.output[0]] = np.pad(const_dict[op.input[0]], pad_width=pads_numpy, mode=pad_mode, constant_values=pad_value)
            else:
                assert len(tensor_shape[op.input[0]]) == 4, "Pad operation on a non-constant tensor whose dimension is not 4 is not supported, get input is a {}D tensor, check layer {}".format(len(tensor_shape[op.input[0]]), layer_name)
                assert pads_numpy[0, 0] == 0, "Pad operation on a non-constant tensor's batch is not supported, check layer {}".format(layer_name)
                assert pads_numpy[0, 1] == 0, "Pad operation on a non-constant tensor's batch is not supported, check layer {}".format(layer_name)
                assert pads_numpy[1, 0] == 0, "Pad operation on a non-constant tensor's channel is not supported, check layer {}".format(layer_name)
                assert pads_numpy[1, 1] == 0, "Pad operation on a non-constant tensor's channel is not supported, check layer {}".format(layer_name)
                fake_layer = Layer(lyr_type = None)
                fake_layer.name = layer_name
                fake_layer.type = ONNX_NAME_PAD
                fake_layer.bottom = op.input[0]
                fake_layer.top = op.output[0]
                fake_layer.pad_n = int(pads_numpy[2, 0])
                fake_layer.pad_s = int(pads_numpy[2, 1])
                fake_layer.pad_w = int(pads_numpy[3, 0])
                fake_layer.pad_e = int(pads_numpy[3, 1])
                layer_list.append(fake_layer)
                tensor_shape[op.output[0]] = [tensor_shape[op.input[0]][0], \
                             tensor_shape[op.input[0]][1], \
                             tensor_shape[op.input[0]][2] + fake_layer.pad_n + fake_layer.pad_s, \
                             tensor_shape[op.input[0]][3] + fake_layer.pad_w + fake_layer.pad_e]
        elif op.op_type in [ONNX_NAME_IDENTITY, ONNX_NAME_DROPOUT]:
            if op.input[0] in const_dict.keys():
                const_dict[op.output[0]] = const_dict[op.input[0]] * 1.0
            else:
                print("Layer {} is of type {}, it will be changed to a scale layer whose weight is 1. This is equivalent to the original type in testing stage".format(layer_name, op.op_type))
                ci = tensor_shape[op.input[0]][1]
                layer_list.append(Layer(lyr_type = 'scale', name = layer_name, top = op.output[0], bottom = op.input[0], bias_term = False))
                param_dict[layer_name + PARAM_DICT_WEIGHT_SUFFIX] = np.ones(ci)
                tensor_shape[op.output[0]] = copy.deepcopy(tensor_shape[op.input[0]])                
        else:
            raise Exception("unsupported op_type {} in layer {}".format(op.op_type, layer_name))
                    
    return layer_list, param_dict, tensor_shape

def deal_with_reshape_before_2D_layer_or_output(layer_list, param_dict, tensor_shape):
    adjacent_dict = get_adjacent_layer_dict(layer_list)
    layer_list_result = [layer for layer in layer_list]
    for layer in layer_list:
        if layer.type == ONNX_NAME_RESHAPE:
            if len(adjacent_dict[layer.name]) == 0:
                layer_list_result.remove(layer)
            else:
                before_ip = True
                for next_layer in adjacent_dict[layer.name]:
                    if next_layer.type not in ['InnerProduct', 'LogSoftmax']:
                        before_ip = False
                        break
                if before_ip:
                    reshape_in_shape = tensor_shape[layer.bottom]
                    if len(reshape_in_shape) == 4:
                        hi = reshape_in_shape[2]
                        wi = reshape_in_shape[3]
                        for next_layer in adjacent_dict[layer.name]:
                            next_layer.bottom = layer.bottom
                            if next_layer.type == 'InnerProduct':
                                co = param_dict[next_layer.name + PARAM_DICT_WEIGHT_SUFFIX].shape[0]
                                param_dict[next_layer.name + PARAM_DICT_WEIGHT_SUFFIX] = param_dict[next_layer.name + PARAM_DICT_WEIGHT_SUFFIX].reshape((co, -1, hi, wi))
                        layer_list_result.remove(layer)
                    else:
                        raise Exception("A reshape operation before fully connect layers or logsoftmax layers should reshape a 4D tensor to a 2D tensor, but get a {}D input tensor in layer {}".format(len(reshape_in_shape), layer.name))
    return layer_list_result, param_dict

def deal_with_reshape_after_fc(layer_list, param_dict, tensor_shape):
    adjacent_dict_next = get_adjacent_layer_dict(layer_list, reverse = False)
    layer_list_result = [layer for layer in layer_list]
    for layer in layer_list:
        if layer.type == ONNX_NAME_RESHAPE:
            print(tensor_shape[layer.bottom])
            reshape_out_shape = tensor_shape[layer.top]
            if len(reshape_out_shape) == 4:
                ho = reshape_out_shape[2]
                wo = reshape_out_shape[3]
                assert ho == 1 and wo == 1, "reshaped tensor's shape after fully connect should be (N, C, 1, 1), but get H = {} and W = {} instead".format(ho, wo)
                for next_layer in adjacent_dict_next[layer.name]:
                    if type(next_layer.bottom)==list:
                        for i in range(len(next_layer.bottom)):
                            if next_layer.bottom[i] == layer.top:
                                next_layer.bottom[i] = layer.bottom
                    else:
                        next_layer.bottom = layer.bottom
                layer_list_result.remove(layer)
            else:
                raise Exception("A reshape operation after fully connect layer should reshape a 4D tensor to a 2D tensor, but get a {}D input tensor in layer {}".format(len(reshape_out_shape), layer.name))
    return layer_list_result, param_dict
                
def deal_with_channel_shuffle(layer_list, tensor_shape):
    adjacent_dict = get_adjacent_layer_dict(layer_list)
    layer_list_result = [layer for layer in layer_list]
    for layer in layer_list:
        if layer in layer_list_result:
            if layer.type == ONNX_NAME_RESHAPE:
                if len(adjacent_dict[layer.name]) == 1 \
                and len(tensor_shape[layer.top]) == 5 \
                and len(tensor_shape[layer.bottom]) == 4 \
                and tensor_shape[layer.top][0] == tensor_shape[layer.bottom][0] \
                and tensor_shape[layer.top][-2] == tensor_shape[layer.bottom][-2] \
                and tensor_shape[layer.top][-1] == tensor_shape[layer.bottom][-1]:
                    next_layer = adjacent_dict[layer.name][0]
                    if next_layer.type == ONNX_NAME_TRANSPOSE:
                        if len(adjacent_dict[next_layer.name]) == 1 \
                        and next_layer.perm == [0, 2, 1, 3, 4]:
                            next_next_layer = adjacent_dict[next_layer.name][0]
                            if next_next_layer.type == ONNX_NAME_RESHAPE \
                            and tensor_shape[layer.bottom][0] == tensor_shape[next_next_layer.top][0]:
                                group = tensor_shape[layer.top][1]
                                layer.type = "ShuffleChannel"
                                print("Find layers {}, {}, {}. They are equivalent to a shuffle channel layer".format(layer.name, next_layer.name, next_next_layer.name))
                                layer.group = group
                                layer.top = next_next_layer.top
                                layer_list_result.remove(next_layer)
                                layer_list_result.remove(next_next_layer)
    return layer_list_result

def merge_pad_before(layer_list):
    adjacent_dict = get_adjacent_layer_dict(layer_list)
    layer_list_result = [layer for layer in layer_list]
    for layer in layer_list:
        if layer in layer_list_result:
            if layer.type == ONNX_NAME_PAD:
                for next_layer in adjacent_dict[layer.name]:
                    assert next_layer.type in ['Convolution', 'Pooling'], "Only Pad operation before Convolution or Pooling is supported, but got Pad operation {} before layer {} of type {}".format(layer.name, next_layer.name, next_layer.type)
                    print("Info: Pad layer named {} is merged into {} layer named {} for running on out chip, this might cause a little error if pad mode is not equivalent to the default pad mode in type {}".format(layer.name, next_layer.type, next_layer.name, next_layer.type))
                    next_layer.bottom = layer.bottom
                    next_layer.pad = (next_layer.pad[0] + layer.pad_n, next_layer.pad[1] + layer.pad_s, next_layer.pad[2] + layer.pad_w, next_layer.pad[3] + layer.pad_e)
                layer_list_result.remove(layer)
    return layer_list_result

def deal_with_large_global_pool(layer_list, tensor_shape):
    layer_list_result = []
    for layer in layer_list:
        if layer.type == 'Pooling' \
        and max(tensor_shape[layer.bottom][2:]) > 16 \
        and max(tensor_shape[layer.top][2:]) == 1 \
        and max(layer.pad) == 0:
            bi = tensor_shape[layer.bottom][0]
            ci = tensor_shape[layer.bottom][1]
            hi = tensor_shape[layer.bottom][2]
            wi = tensor_shape[layer.bottom][3]
            print("Warn: input of global pooling layer name {} is too large. It will be seperated into more than 1 step, which might cause a little error".format(layer.name))
            hi_old = hi
            wi_old = wi
            input_name_old = layer.bottom
            step_index = 0
            while max(hi_old, wi_old) > 16:
                kh = 6 if hi_old > 16 else hi_old
                kw = 6 if wi_old > 16 else wi_old
                ph = 6 - hi_old % 6 if hi_old > 16 and hi_old % 6 != 0 else 0
                pw = 6 - wi_old % 6 if wi_old > 16 and wi_old % 6 != 0 else 0
                hi_new = (hi_old - kh + ph) // 6 + 1
                wi_new = (wi_old - kw + pw) // 6 + 1
                input_name_new = layer.top + "/step{}".format(step_index)
                current_layer_name = layer.name + "/step{}".format(step_index)
                layer_list_result.append(Layer(lyr_type = 'pool', name = current_layer_name, top = input_name_new, bottom = input_name_old, 
                                  kernel_size = (int(kh), int(kw)), stride = 6, pad_n = 0, pad_s = int(ph), pad_w = int(pw), pad_e = 0, dilation = 1, 
                                  pool = layer.pool))
                tensor_shape[input_name_new] = [bi, ci, hi_new, wi_new]
                input_name_old = input_name_new
                hi_old = hi_new
                wi_old = wi_new
                step_index += 1
            kernel_size = (int(hi_old), int(wi_old))
            layer_list_result.append(Layer(lyr_type = 'pool', name = layer.name, top = layer.top, bottom = input_name_old, 
                  kernel_size=kernel_size, stride=1, pad=0, dilation=1, pool = layer.pool))
        else:
            layer_list_result.append(layer)
    return layer_list_result
    
def check_layer_type(layer_list):
    for layer in layer_list:
        assert layer.type in layer.type_dict.values(), "Unsupported layer type {} in layer {}".format(layer.type, layer.name)

def load_from_onnx(onnx_file):
    model = onnx.load(onnx_file)
    const_dict = get_const_from_initializer(model.graph.initializer)
    input_and_param_shape = get_input_and_param_shape(model.graph)
    layer_list, param_dict, tensor_shape = get_raw_layer_list_and_param_dict(model.graph.node, const_dict, input_and_param_shape)
    layer_list_result = deal_with_channel_shuffle(layer_list, tensor_shape)
    layer_list_result, param_dict = deal_with_reshape_before_2D_layer_or_output(layer_list_result, param_dict, tensor_shape)
    layer_list_result, param_dict = deal_with_reshape_after_fc(layer_list_result, param_dict, tensor_shape)
    layer_list_result = merge_pad_before(layer_list_result)
    layer_list_result = deal_with_large_global_pool(layer_list_result, tensor_shape)
    check_layer_type(layer_list_result)
    return layer_list_result, param_dict

if __name__ == "__main__":
    import torchvision
    import torch
    squeezenet1_0 = torchvision.models.squeezenet1_0(pretrained=True)
    squeezenet1_0.eval()
    trainset = torchvision.datasets.ImageFolder('D:/AI_Images/ILSVRC2012/train',
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.Resize((256, 256)),
                                                    torchvision.transforms.RandomCrop(224),
                                                    torchvision.transforms.ToTensor(),
#                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                    ])
                                                )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                              shuffle=True, num_workers=2)
    torch.onnx.export(squeezenet1_0, torch.ones((1, 3, 224, 224)), "squeezenet1_0.onnx")
    layer_list, param_dict = load_from_onnx("squeezenet1_0.onnx")
    from ndk.quant_tools.numpy_net import run_layers
    from ndk.quantize import quantize_model
    from ndk.layers import get_net_input_output
    from ndk.quant_train_torch.quant_layer import QuantizedNet
    from ndk.optimize import add_pre_norm, merge_layers
    from ndk.examples.data_generator_imagenet_partial import data_generator_imagenet_partial
    data_generator_quant = data_generator_imagenet_partial(
                             imagenet_dirname='../examples/imagenet_partial',
                             filenames_to_class='filenames_to_class_by_number.json',
                             batch_size=10,
                             random_order=True, num_class = 1000)
    weight = np.array([1/0.229/255, 1/0.224/255, 1/0.225/255], dtype = np.float32)
    bias = np.array([-0.485/0.229, -0.456/0.224, -0.406/0.225], dtype = np.float32)
    add_pre_norm(layer_list, param_dict, weight, bias)
    layer_list, param_dict = merge_layers(layer_list, param_dict)
    net = QuantizedNet(layer_list, param_dict, False, False, False, False)
    net.cuda()
    input_tensor_name, output_tensor_name = get_net_input_output(layer_list)
#    for i, data in enumerate(trainloader, 0):
    for i in range(100):
#        inputs, labels = data
        data = next(data_generator_quant)
        inputs = torch.FloatTensor(data['input'])
        labels = torch.FloatTensor(data['output'])
        labels = torch.max(labels, 1)[1]
#        print(labels)
#        data_dict = run_layers(inputs.numpy(), layer_list, output_tensor_name, param_dict, hw_aligned = False, quant=False, use_ai_framework=False)
        inputs = inputs.cuda()
        outputs = net(inputs)
        outputs = outputs.cpu().reshape((10, 1000))
#        outputs = data_dict[output_tensor_name[0]].reshape((10, 1000))
#        outputs = torch.from_numpy(outputs)
        predictions = torch.max(outputs, 1)[1]
        print(torch.sum(predictions == labels))
    quant_layer_list0, quant_param_dict0 = quantize_model(
    layer_list=layer_list,
    param_dict=param_dict,
    bitwidth=8,
    data_generator=data_generator_quant,
    aggressive=True,
    factor_num_bin=8,
    num_step_pre=20,
    num_step=40,
    priority_mode='fwb',
    )
    quant_net = QuantizedNet(quant_layer_list0, quant_param_dict0)
    quant_net.cuda()
    quant_net.double()
    for i in range(100):
#        inputs, labels = data
        data = next(data_generator_quant)
        inputs = torch.FloatTensor(data['input'])
        labels = torch.FloatTensor(data['output'])
        labels = torch.max(labels, 1)[1]
#        print(labels)
#        data_dict = run_layers(inputs.numpy(), layer_list, output_tensor_name, param_dict, hw_aligned = False, quant=False, use_ai_framework=False)
        inputs = inputs.cuda().double()
        outputs = quant_net(inputs)
        outputs = outputs.cpu().reshape((10, 1000))
#        outputs = data_dict[output_tensor_name[0]].reshape((10, 1000))
#        outputs = torch.from_numpy(outputs)
        predictions = torch.max(outputs, 1)[1]
        print(torch.sum(predictions == labels))
#    import torch
#    import torch.nn as nn
#    
#    
#    
#    class SELayer(torch.nn.Module):
#        def __init__(self, channel, reduction = 8):
#            super(SELayer, self).__init__()
#            self.conv = torch.nn.Conv2d(3, channel, 3)
#            self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
#            self.fc = torch.nn.Sequential(
#                    torch.nn.Conv2d(channel, channel // reduction, 1),
#                    torch.nn.ReLU(),
#                    torch.nn.Conv2d(channel // reduction, channel, 1),
#                    torch.nn.Sigmoid()
#            )
#        def forward(self, x):
#            x = self.conv(x)
#            y = self.avg_pool(x)
#            y = self.fc(y)
#            z = x * y
#            return z
#
#    from ndk.examples.data_generator_imagenet_partial import data_generator_imagenet_partial
#    data_generator_quant = data_generator_imagenet_partial(
#                             imagenet_dirname='../examples/imagenet_partial',
#                             filenames_to_class='filenames_to_class_by_number.json',
#                             batch_size=10,
#                             random_order=True, num_class = 1000)
#    
#    se = SELayer(16)
#    torch.onnx.export(torch.nn.PReLU(3), torch.ones((1, 3, 224, 224)), "prelu.onnx")
#    se.eval()
#    se.double()
#    x_numpy = np.random.normal(size=(1,3,224,224))
#    x_torch = torch.DoubleTensor(x_numpy)
#    y_torch = se(x_torch)
#    y_numpy = y_torch.data.numpy()
#    layer_list, param_dict = load_from_onnx("prelu.onnx")
#    assert False
#    from ndk.quant_tools.numpy_net import run_layers
#    from ndk.layers import get_net_input_output
#    from ndk.quantize import quantize_model, quantize_model_with_training
#    from ndk.modelpack import modelpack, save_to_file
#    import tensorflow as tf
#    input_tensor_name, output_tensor_name = get_net_input_output(layer_list)
#    quant_layer_list0, quant_param_dict0 = quantize_model(
#        layer_list=layer_list,
#        param_dict=param_dict,
#        bitwidth=8,
#        data_generator=data_generator_quant,
#        aggressive=True,
#        factor_num_bin=8,
#        num_step_pre=20,
#        num_step=40,
#        priority_mode='fwb',
#    )
#    save_to_file(quant_layer_list0, 'se.prototxt', param_dict=quant_param_dict0, fname_npz='se.npz')
##    quant_layer_list, quant_param_dict = quantize_model_with_training(layer_list=quant_layer_list0,
##                                                                           bitwidth=8,
##                                                                           data_generator=data_generator_quant,
##                                                                           param_dict=quant_param_dict0,
##                                                                           log_dir='log',
##                                                                           loss_fn=tf.losses.softmax_cross_entropy,
##                                                                           optimizer=tf.train.AdamOptimizer(1e-5),
##                                                                           num_step_train=200,
##                                                                           rnd_seed = None)#rnd_seed = 961
#    net_name = "se"
#    modelpack(8, quant_layer_list0, quant_param_dict0, net_name + "_quant", model_name = net_name)
#    data_dict = run_layers(x_numpy, layer_list, output_tensor_name, param_dict, hw_aligned = False, quant=False, use_ai_framework=False)
#    datq_dict = run_layers(x_numpy, layer_list, [layer.top for layer in layer_list], quant_param_dict0, hw_aligned = True , quant=True , use_ai_framework=False)
#    z_numpy = data_dict[output_tensor_name[0]]
#    error = (y_numpy - z_numpy)[0]
#    s0 = datq_dict['input.1'][0]
#    s1 = datq_dict['9'][0]
#    s2 = datq_dict['10'][0]
#    s3 = s0 * 1
#    for i in range(16):
#        s3[i] = np.floor((s0[i] * s1[i]) * 16) / 16
#    errors = s3 - s2


#    for layer in layer_list:
#        print(layer)
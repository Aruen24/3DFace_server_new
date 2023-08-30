"""
Copyright(c) 2020 by WuQi Technologies. ALL RIGHTS RESERVED.

This Information is proprietary to WuQi Technologies and MAY NOT
be copied by any method or incorporated into another program without
the express written consent of WuQi. This Information or any portion
thereof remains the property of WuQi. The Information contained herein
is believed to be accurate and WuQi assumes no responsibility or
liability for its use in any way and conveys no license or title under
any patent or copyright and makes no representation or warranty that this
Information is free from patent or copyright infringement.
"""
import os
import sys

import numpy as np
import torch

ndk_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ndk_dir)

from ndk.layers import Layer, get_tensor_shape, get_network_output, multi_output_layer, multi_input_layer
from ndk.quant_tools.quant_func import quant2float, quantized_value

from ndk.quant_train_torch.quantize_tensor import QuantizeFeature, QuantizeParam

def read_vector_from_file_hex(filename):
    table = {}
    contents = []
    with open(filename, 'r') as f:
        contents = f.read().rstrip().split('\n')
    for i in range(len(contents)):
        if i < 32768:
            table[i] = float(int(contents[i], 16)) / 32768
        else:
            table[i - 65536] = float(int(contents[i], 16)) / 32768
    return table

look_up_sigmoid = read_vector_from_file_hex("../quant_tools/sigmoid_sim_out.dat")

class QuantizedConv2D(torch.nn.Module):
    def __init__(self, layer, param_dict, quant_output = True, quant_weight = True, quant_bias = True, bitwidth = 8, for_training = True):
        super(QuantizedConv2D, self).__init__()
        assert isinstance(layer, Layer), "layer should be an instance of ndk.layers.Layer, but got {} instead".format(type(layer))
        assert isinstance(param_dict, dict), "param_dict should be a dict, but got {} instead".format(type(param_dict))
        assert bitwidth in [8, 16], "bitwidth should be 8 or 16, but got {} instead".format(bitwidth)
        self.quant_output = quant_output
        self.quant_weight = quant_weight
        self.quant_bias = quant_bias
        self.bitwidth = bitwidth
        self.params = torch.nn.ParameterDict({})
        self.bias_term = layer.bias_term
        self.name = layer.name
        self.stride = layer.stride
        self.dilation = layer.dilation
        self.group = layer.group
        
        if max(layer.pad) > 0:
            self.need_pad = True
            self.pad_layer = torch.nn.ZeroPad2d((layer.pad[2], max(layer.pad[3], 0), layer.pad[0], max(layer.pad[1], 0)))
        else:
            self.need_pad = False
        
        assert layer.name + "_weight" in param_dict.keys(), "weight is not found in param_dict"
        self.params.update({'weight' : torch.nn.Parameter(torch.from_numpy(param_dict[layer.name + "_weight"]))})
        if self.quant_weight:
            assert layer.name + "_quant_weight" in param_dict.keys(), "quant weight is not found in param_dict"
            assert layer.name + "_frac_weight" in param_dict.keys(), "weight's frac is not found in param_dict"
            if not for_training:
                self.params.update({'weight' : torch.nn.Parameter(torch.from_numpy(quant2float(param_dict[layer.name + "_quant_weight"], self.bitwidth, param_dict[layer.name + "_frac_weight"])))})
            self.weight_frac = np.copy(param_dict[layer.name + "_frac_weight"])
            self.weight_quantizer = QuantizeParam(self.bitwidth, self.weight_frac, type = 0)
#        else:
            
        if layer.bias_term:
            assert layer.name + "_bias" in param_dict.keys(), "bias is not found in param_dict"
            self.params.update({'bias' : torch.nn.Parameter(torch.from_numpy(param_dict[layer.name + "_bias"]))})
            if self.quant_bias:
                assert layer.name + "_quant_bias" in param_dict.keys(), "bias is not found in param_dict"
                assert layer.name + "_frac_bias" in param_dict.keys(), "bias's frac is not found in param_dict"
                if not for_training:
                    self.params.update({'bias' : torch.nn.Parameter(torch.from_numpy(quant2float(param_dict[layer.name + "_quant_bias"], self.bitwidth, param_dict[layer.name + "_frac_bias"])))})
                self.bias_frac = np.copy(param_dict[layer.name + "_frac_bias"])
                self.bias_quantizer = QuantizeParam(self.bitwidth, self.bias_frac, type = 1)
#            else:
        
        if self.quant_output:
            assert layer.top + "_frac" in param_dict.keys(), "output's fraction is not found in param_dict"
            self.feature_frac = np.copy(param_dict[layer.top + "_frac"])
            self.quant_out_layer = QuantizeFeature(bitwidth, self.feature_frac)
            self.quant_out_layer.name = layer.name
            if layer.bias_term and for_training:
                self.params['bias'].data = self.params['bias'].data + 2 ** (-1 - int(self.feature_frac))
        
    def forward(self, x):
        if self.need_pad:
            x = self.pad_layer(x)
        weight = self.weight_quantizer(self.params['weight']) if self.quant_weight else self.params['weight']
        bias = None
        if self.bias_term:
            bias = self.bias_quantizer(self.params['bias']) if self.quant_bias else self.params['bias']
        y = torch.nn.functional.conv2d(x, weight, bias, stride=self.stride, dilation=self.dilation, groups=self.group)
        if self.quant_output:
            z = self.quant_out_layer(y)
            return z
        else:
            return y
    def cuda(self):
        if self.need_pad:
            self.pad_layer.cuda()
        self.params.cuda()
        if self.quant_weight:
            self.weight_quantizer.cuda()
        if self.bias_term and self.quant_bias:
            self.bias_quantizer.cuda()
    def double(self):
        if self.need_pad:
            self.pad_layer.double()
        self.params.double()
        if self.quant_weight:
            self.weight_quantizer.double()
        if self.bias_term and self.quant_bias:
            self.bias_quantizer.double()

class QuantizedInnerproduct(torch.nn.Module):
    def __init__(self, layer, param_dict, quant_output = True, quant_weight = True, quant_bias = True, bitwidth = 8, for_training = True):
        super(QuantizedInnerproduct, self).__init__()
        assert isinstance(layer, Layer), "layer should be an instance of ndk.layers.Layer, but got {} instead".format(type(layer))
        assert isinstance(param_dict, dict), "param_dict should be a dict, but got {} instead".format(type(param_dict))
        assert bitwidth in [8, 16], "bitwidth should be 8 or 16, but got {} instead".format(bitwidth)
        self.quant_output = quant_output
        self.quant_weight = quant_weight
        self.quant_bias = quant_bias
        self.bitwidth = bitwidth
        self.bias_term = layer.bias_term
        self.name = layer.name
        self.params = torch.nn.ParameterDict({})

        assert layer.name + "_weight" in param_dict.keys(), "weight is not found in param_dict"
        self.params.update({'weight' : torch.nn.Parameter(torch.from_numpy(param_dict[layer.name + "_weight"]))})
        if self.quant_weight:
            assert layer.name + "_quant_weight" in param_dict.keys(), "quant weight is not found in param_dict"
            assert layer.name + "_frac_weight" in param_dict.keys(), "weight's frac is not found in param_dict"
            if not for_training:
                self.params.update({'weight' : torch.nn.Parameter(torch.from_numpy(quant2float(param_dict[layer.name + "_quant_weight"], self.bitwidth, param_dict[layer.name + "_frac_weight"])))})
            self.weight_frac = np.copy(param_dict[layer.name + "_frac_weight"])
            self.weight_quantizer = QuantizeParam(self.bitwidth, self.weight_frac, type = 0)
#        else:

        if layer.bias_term:
            assert layer.name + "_bias" in param_dict.keys(), "bias is not found in param_dict"
            self.params.update({'bias' : torch.nn.Parameter(torch.from_numpy(param_dict[layer.name + "_bias"]))})
            if self.quant_bias:
                assert layer.name + "_quant_bias" in param_dict.keys(), "bias is not found in param_dict"
                assert layer.name + "_frac_bias" in param_dict.keys(), "bias's frac is not found in param_dict"
                if not for_training:
                    self.params.update({'bias' : torch.nn.Parameter(torch.from_numpy(quant2float(param_dict[layer.name + "_quant_bias"], self.bitwidth, param_dict[layer.name + "_frac_bias"])))})
                self.bias_frac = np.copy(param_dict[layer.name + "_frac_bias"])
                self.bias_quantizer = QuantizeParam(self.bitwidth, self.bias_frac, type = 1)
#            else:

        if self.quant_output:
            assert layer.top + "_frac" in param_dict.keys(), "output's fraction is not found in param_dict"
            self.feature_frac = np.copy(param_dict[layer.top + "_frac"])
            self.quant_out_layer = QuantizeFeature(bitwidth, self.feature_frac)
            self.quant_out_layer.name = layer.name
            if layer.bias_term and for_training:
                self.params['bias'].data = self.params['bias'].data + 2 ** (-1 - int(self.feature_frac))
        
    def forward(self, x):
        weight = self.weight_quantizer(self.params['weight']) if self.quant_weight else self.params['weight']
        bias = None
        if self.bias_term:
            bias = self.bias_quantizer(self.params['bias']) if self.quant_bias else self.params['bias']
        y = torch.nn.functional.conv2d(x, weight, bias, stride=1, dilation=1, groups=1)
        if self.quant_output:
            z = self.quant_out_layer(y)
            return z
        else:
            return y
    def cuda(self):
        self.params.cuda()
        if self.quant_weight:
            self.weight_quantizer.cuda()
        if self.bias_term and self.quant_bias:
            self.bias_quantizer.cuda()
    def double(self):
        self.params.double()
        if self.quant_weight:
            self.weight_quantizer.double()
        if self.bias_term and self.quant_bias:
            self.bias_quantizer.double()

class QuantizedScale(torch.nn.Module):
    def __init__(self, layer, param_dict, input_shape, quant_output = True, quant_weight = True, quant_bias = True, bitwidth = 8, for_training = True):
        super(QuantizedScale, self).__init__()
        assert isinstance(layer, Layer), "layer should be an instance of ndk.layers.Layer, but got {} instead".format(type(layer))
        assert isinstance(param_dict, dict), "param_dict should be a dict, but got {} instead".format(type(param_dict))
        assert bitwidth in [8, 16], "bitwidth should be 8 or 16, but got {} instead".format(bitwidth)
        self.bias_term = layer.type == "BatchNorm" or layer.bias_term
        self.quant_output = quant_output
        self.quant_weight = quant_weight
        self.quant_bias = quant_bias
        self.bitwidth = bitwidth
        self.name = layer.name
        self.params = torch.nn.ParameterDict({})
        self.ci = input_shape[1]

        assert layer.name + "_weight" in param_dict.keys(), "weight is not found in param_dict"
        self.params.update({'weight' : torch.nn.Parameter(torch.from_numpy(param_dict[layer.name + "_weight"].reshape(1,self.ci,1,1)))})
        if self.quant_weight:
            assert layer.name + "_quant_weight" in param_dict.keys(), "quant weight is not found in param_dict"
            assert layer.name + "_frac_weight" in param_dict.keys(), "weight's frac is not found in param_dict"
            self.weight_frac = np.copy(param_dict[layer.name + "_frac_weight"])
            if not for_training:
                self.params.update({'weight' : torch.nn.Parameter(torch.from_numpy(quant2float(param_dict[layer.name + "_quant_weight"], self.bitwidth, param_dict[layer.name + "_frac_weight"]).reshape(1,self.ci,1,1)))})
            self.weight_frac = np.copy(param_dict[layer.name + "_frac_weight"])
            self.weight_quantizer = QuantizeParam(self.bitwidth, self.weight_frac, type = 2)
#        else:

        if self.bias_term:
            assert layer.name + "_bias" in param_dict.keys(), "bias is not found in param_dict"
            self.params.update({'bias' : torch.nn.Parameter(torch.from_numpy(param_dict[layer.name + "_bias"].reshape(1,self.ci,1,1)))})
            if self.quant_bias:
                assert layer.name + "_quant_bias" in param_dict.keys(), "bias is not found in param_dict"
                assert layer.name + "_frac_bias" in param_dict.keys(), "bias's frac is not found in param_dict"
                self.bias_frac = np.copy(param_dict[layer.name + "_frac_bias"])
                if not for_training:
                    self.params.update({'bias' : torch.nn.Parameter(torch.from_numpy(quant2float(param_dict[layer.name + "_quant_bias"], self.bitwidth, param_dict[layer.name + "_frac_bias"]).reshape(1,self.ci,1,1)))})
                self.bias_frac = np.copy(param_dict[layer.name + "_frac_bias"])
                self.bias_quantizer = QuantizeParam(self.bitwidth, self.bias_frac, type = 2)
#            else:

        if self.quant_output:
            assert layer.top + "_frac" in param_dict.keys(), "output's fraction is not found in param_dict"
            self.feature_frac = np.copy(param_dict[layer.top + "_frac"])
            self.quant_out_layer = QuantizeFeature(bitwidth, self.feature_frac)
            self.quant_out_layer.name = layer.name
            if layer.bias_term and for_training:
                self.params['bias'].data = self.params['bias'].data + 2 ** (-1 - int(self.feature_frac))

    def forward(self, x):
        weight = self.weight_quantizer(self.params['weight']) if self.quant_weight else self.params['weight']
        y = x * weight
        if self.bias_term:
            bias = self.bias_quantizer(self.params['bias']) if self.quant_bias else self.params['bias']
            y = y + bias
        if self.quant_output:
            z = self.quant_out_layer(y)
            return z
        else:
            return y
    def cuda(self):
        self.params.cuda()
        if self.quant_weight:
            self.weight_quantizer.cuda()
        if self.bias_term and self.quant_bias:
            self.bias_quantizer.cuda()
    def double(self):
        self.params.double()
        if self.quant_weight:
            self.weight_quantizer.double()
        if self.bias_term and self.quant_bias:
            self.bias_quantizer.double()

class QuantizedScaleByTensor(torch.nn.Module):
    def __init__(self, layer, param_dict, quant_output = True, bitwidth = 8):
        super(QuantizedScaleByTensor, self).__init__()
        assert isinstance(layer, Layer), "layer should be an instance of ndk.layers.Layer, but got {} instead".format(type(layer))
        assert isinstance(param_dict, dict), "param_dict should be a dict, but got {} instead".format(type(param_dict))
        assert bitwidth in [8, 16], "bitwidth should be 8 or 16, but got {} instead".format(bitwidth)
        self.quant_output = quant_output
        self.name = layer.name

        if self.quant_output:
            assert layer.top + "_frac" in param_dict.keys(), "output's fraction is not found in param_dict"
            self.feature_frac = np.copy(param_dict[layer.top + "_frac"])
            self.quant_out_layer = QuantizeFeature(bitwidth, self.feature_frac)

    def forward(self, x):
        z = x[0] * x[1]
        if self.quant_output:
            z = self.quant_out_layer(z)
        return z

class QuantizedBias(torch.nn.Module):
    def __init__(self, layer, param_dict, input_shape, quant_output = True, quant_bias = True, bitwidth = 8, for_training = True):
        super(QuantizedBias, self).__init__()
        assert isinstance(layer, Layer), "layer should be an instance of ndk.layers.Layer, but got {} instead".format(type(layer))
        assert isinstance(param_dict, dict), "param_dict should be a dict, but got {} instead".format(type(param_dict))
        assert bitwidth in [8, 16], "bitwidth should be 8 or 16, but got {} instead".format(bitwidth)
        self.ci = input_shape[1]
        self.bitwidth = bitwidth
        self.quant_bias = quant_bias
        self.params = torch.nn.ParameterDict({})

        assert layer.name + "_bias" in param_dict.keys(), "bias is not found in param_dict"
        self.params.update({'bias' : torch.nn.Parameter(torch.from_numpy(param_dict[layer.name + "_bias"].reshape(1,self.ci,1,1)))})
        if self.quant_bias:
            assert layer.name + "_quant_bias" in param_dict.keys(), "bias is not found in param_dict"
            assert layer.name + "_frac_bias" in param_dict.keys(), "bias's frac is not found in param_dict"
            if not for_training:
                self.params.update({'bias' : torch.nn.Parameter(torch.from_numpy(quant2float(param_dict[layer.name + "_quant_bias"].reshape(1,self.ci,1,1), self.bitwidth, param_dict[layer.name + "_frac_bias"])))})
            self.bias_frac = np.copy(param_dict[layer.name + "_frac_bias"])
            self.bias_quantizer = QuantizeParam(self.bitwidth, self.bias_frac, type = 2)
#        else:

        if self.quant_output:
            assert layer.top + "_frac" in param_dict.keys(), "output's fraction is not found in param_dict"
            self.feature_frac = np.copy(param_dict[layer.top + "_frac"])
            self.quant_out_layer = QuantizeFeature(bitwidth, self.feature_frac)
            self.quant_out_layer.name = layer.name
            if layer.bias_term and for_training:
                self.params['bias'].data = self.params['bias'].data + 2 ** (-1 - int(self.feature_frac))

    def forward(self, x):
        bias = self.bias_quantizer(self.params['bias']) if self.quant_bias else self.params['bias']
        x = x + bias
        if self.quant_output:
            x = self.quant_out_layer(x)
        return x
    def cuda(self):
        self.params.cuda()
        if self.quant_bias:
            self.bias_quantizer.cuda()
    def double(self):
        self.params.double()
        if self.quant_bias:
            self.bias_quantizer.double()

class QuantizedEltwiseAdd(torch.nn.Module):
     def __init__(self, layer, param_dict, quant_output = True, bitwidth = 8):
        super(QuantizedEltwiseAdd, self).__init__()
        assert isinstance(layer, Layer), "layer should be an instance of ndk.layers.Layer, but got {} instead".format(type(layer))
        if type(param_dict) != type(None):
            assert isinstance(param_dict, dict), "param_dict should be a dict, but got {} instead".format(type(param_dict))
        assert bitwidth in [8, 16], "bitwidth should be 8 or 16, but got {} instead".format(bitwidth)
        self.quant_output = quant_output
        self.bitwidth = bitwidth
        if self.quant_output:
            assert layer.top + "_frac" in param_dict.keys(), "output's fraction is not found in param_dict"
            self.feature_frac = np.copy(param_dict[layer.top + "_frac"])
            self.quant_out_layer = QuantizeFeature(bitwidth, self.feature_frac)
            self.quant_out_layer.name = layer.name
    
     def forward(self, inputs):
        output = torch.zeros_like(inputs[0])
        for input_ in inputs:
            output = output + input_
        if self.quant_output:
            output = self.quant_out_layer(output)
        return output

class QuantizedAvgpool2D(torch.nn.Module):
    def __init__(self, layer, param_dict, input_shape, output_shape, hw_aligned = True, quant_weight = True, quant_output = True, bitwidth = 8):
        super(QuantizedAvgpool2D, self).__init__()
        assert isinstance(layer, Layer), "layer should be an instance of ndk.layers.Layer, but got {} instead".format(type(layer))
        if type(param_dict) != type(None):
            assert isinstance(param_dict, dict), "param_dict should be a dict, but got {} instead".format(type(param_dict))
        assert bitwidth in [8, 16], "bitwidth should be 8 or 16, but got {} instead".format(bitwidth)
        self.quant_output = quant_output
        self.quant_weight = quant_weight
        self.bitwidth = bitwidth
        self.hw_aligned = hw_aligned
        self.input_channel = input_shape[1]
        self.stride = layer.stride
        self.dilation = layer.dilation
        if self.quant_output:
            assert layer.top + "_frac" in param_dict.keys(), "output's fraction is not found in param_dict"
            self.feature_frac = np.copy(param_dict[layer.top + "_frac"])
            self.quant_out_layer = QuantizeFeature(bitwidth, self.feature_frac)
            self.quant_out_layer.name = layer.name
        if self.hw_aligned:
            if max(layer.pad) > 0:
                self.need_pad = True
                self.pad_layer = torch.nn.ZeroPad2d((layer.pad[2], max(layer.pad[3], 0), layer.pad[0], max(layer.pad[1], 0)))
            else:
                self.need_pad = False
            self.weight_for_add = torch.ones((input_shape[1], 1, layer.kernel_size[0], layer.kernel_size[1]), requires_grad = False)
#            self.pool = torch.nn.Conv2d(input_shape[1], input_shape[1], layer.kernel_size, layer.stride, 0, layer.dilation, input_shape[1], False)
#            self.pool.eval()
#            self.pool.register_forward_pre_hook(hook_avgpool_forward)
            h_in = input_shape[2]
            w_in = input_shape[3]
            if layer.pad[1] < 0:
                h_in += layer.pad[1]
            if layer.pad[3] < 0:
                w_in += layer.pad[3]
            # center
            active_h = min(layer.kernel_size[0], h_in)
            active_w = min(layer.kernel_size[1], w_in)
            elem_cnt = np.ones(output_shape) * active_h * active_w
            # left edge
            active_h = min(layer.kernel_size[0], h_in)
            active_w = min(layer.kernel_size[1]-layer.pad[2], w_in)
            elem_cnt[:,0] = active_h * active_w
            # upper edge
            active_h = min(layer.kernel_size[0]-layer.pad[0], h_in)
            active_w = min(layer.kernel_size[1], w_in)
            elem_cnt[0,:] = active_h * active_w
            # lower edge
            active_h = min(layer.kernel_size[0]-max(layer.pad[1],0), h_in)
            active_w = min(layer.kernel_size[1], w_in)
            elem_cnt[-1,:] = active_h * active_w
            # upper-left corner
            active_h = min(layer.kernel_size[0]-layer.pad[0], h_in)
            active_w = min(layer.kernel_size[1]-layer.pad[2], w_in)
            elem_cnt[0, 0] = active_h * active_w
            # lower-left corner
            active_h = min(layer.kernel_size[0]-max(layer.pad[1],0), h_in)
            active_w = min(layer.kernel_size[1]-layer.pad[2], w_in)
            elem_cnt[-1, 0] = active_h * active_w  
            scale = 1 / elem_cnt
            if quant_weight:
                min_elem_cnt = np.min(elem_cnt)
                if self.bitwidth==8:
                    # when 8-bit allows unsigned number
                    scale_frac = self.bitwidth + int(np.ceil(np.log2(min_elem_cnt))) - 1
                    scale = quantized_value(scale, bitwidth=self.bitwidth, frac=scale_frac, floor=False, signed=False)
                else:
                    scale_frac = self.bitwidth + int(np.ceil(np.log2(min_elem_cnt))) - 2
                    scale = quantized_value(scale, bitwidth=self.bitwidth, frac=scale_frac, floor=False, signed=True)
            self.pool_scale = torch.from_numpy(scale).float()
        else:
            pu = layer.pad[0]
            pd = layer.pad[1]
            pl = layer.pad[2]
            pr = layer.pad[3]
            self.pool = torch.nn.AvgPool2d(layer.kernel_size, layer.stride, (pu, pl), pu < pd or pl < pr)
    def forward(self, x):
        if self.hw_aligned:
            if self.need_pad:
                x = self.pad_layer(x)
            x = torch.nn.functional.conv2d(x, self.weight_for_add, stride = self.stride, dilation = self.dilation, groups = self.input_channel)
            x = x * self.pool_scale
        else:
            x = self.pool(x)
        if self.quant_output:
            x = self.quant_out_layer(x)
        return x
    def cuda(self):
        if self.hw_aligned:
            if self.need_pad:
                self.pad_layer.cuda()
            self.weight_for_add = self.weight_for_add.cuda()
            self.pool_scale = self.pool_scale.cuda()
        else:
            self.pool.cuda()
    def double(self):
        if self.hw_aligned:
            if self.need_pad:
                self.pad_layer.double()
            self.weight_for_add = self.weight_for_add.double()
            self.pool_scale = self.pool_scale.double()
        else:
            self.pool.double()

class QuantizedSigmoidFunction(torch.autograd.Function):
    def forward(self, in_):
        batch, channel, height, width = in_.shape
        output = in_.clone()
        for n in range(batch):
            for c in range(channel):
                for h in range(height):
                    for w in range(width):
                        index = int(in_[n, c, h, w] * 4096)
                        index = max(-32768, min(32767, index))
                        output[n, c, h, w] = look_up_sigmoid[index]
        self.save_for_backward(output)
        return output
    def backward(self, grad_output):
        output = self.saved_tensors
        return grad_output * (1.0 - output) * output

class QuantizedSigmoid(torch.nn.Module):
    def __init__(self, quant_output = True, bitwidth = 8):
        super(QuantizedSigmoid, self).__init__()
        assert bitwidth in [8, 16], "bitwidth should be 8 or 16, but got {} instead".format(bitwidth)
        self.quant_output = quant_output
        self.bitwidth = bitwidth
        if quant_output:
            self.quant_out_layer = QuantizeFeature(bitwidth, bitwidth - 1)
        else:
            self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        if self.quant_output:
            y = QuantizedSigmoidFunction.apply(x)
            return self.quant_out_layer(y)
        else:
            return self.sigmoid(x)
    def cuda(self):
        if not self.quant_output:
            self.sigmoid.cuda()
    def double(self):
        if not self.quant_output:
            self.sigmoid.cuda()

class QuantizedTanhFunction(torch.autograd.Function):
    def forward(self, in_):
        batch, channel, height, width = in_.shape
        output = in_.clone()
        for n in range(batch):
            for c in range(channel):
                for h in range(height):
                    for w in range(width):
                        index = int(in_[n, c, h, w] * 8192)
                        index = max(-32768, min(32767, index))
                        output[n, c, h, w] = look_up_sigmoid[index] * 2.0 - 1.0
        self.save_for_backward(output)
        return output
    def backward(self, grad_output):
        output = self.saved_tensors
        return grad_output * (1.0 - output * output)

class QuantizedTanh(torch.nn.Module):
    def __init__(self, quant_output = True, bitwidth = 8):
        super(QuantizedTanh, self).__init__()
        assert bitwidth in [8, 16], "bitwidth should be 8 or 16, but got {} instead".format(bitwidth)
        self.quant_output = quant_output
        self.bitwidth = bitwidth
        if quant_output:
            self.quant_out_layer = QuantizeFeature(bitwidth, bitwidth - 1)
        else:
            self.tanh = torch.nn.Tanh()
    def forward(self, x):
        if self.quant_output:
            y = QuantizedTanhFunction.apply(x)
            return self.quant_out_layer(y)
        else:
            return self.sigmoid(x)
    def cuda(self):
        if not self.quant_output:
            self.tanh.cuda()
    def double(self):
        if not self.quant_output:
            self.tanh.cuda()

class QuantizedLeakyrelu(torch.nn.Module):
    def __init__(self, layer, param_dict, quant_output = True, quant_weight = True, bitwidth = 8):
        super(QuantizedLeakyrelu, self).__init__()
        assert isinstance(layer, Layer), "layer should be an instance of ndk.layers.Layer, but got {} instead".format(type(layer))
        if type(param_dict) != type(None):
            assert isinstance(param_dict, dict), "param_dict should be a dict, but got {} instead".format(type(param_dict))
        assert bitwidth in [8, 16], "bitwidth should be 8 or 16, but got {} instead".format(bitwidth)
        self.quant_weight = quant_weight
        self.quant_output = quant_output
        self.bitwidth = bitwidth
        if self.quant_weight:
            self.leaky = torch.nn.LeakyReLU(min(127, max(-128, round(layer.negative_slope * 64))) / 64)
        else:
            self.leaky = torch.nn.LeakyReLU(layer.negative_slope)
        if self.quant_output:
            assert layer.top + "_frac" in param_dict.keys(), "output's fraction is not found in param_dict"
            self.feature_frac = np.copy(param_dict[layer.top + "_frac"])
            self.quant_out_layer = QuantizeFeature(bitwidth, self.feature_frac)
    def forward(self, x):
        x = self.leaky(x)
        if self.quant_output:
            x = self.quant_out_layer(x)
        return x
    def cuda(self):
        self.leaky.cuda()
    def double(self):
        self.leaky.double()

class QuantizedLogSoftmaxFunction(torch.autograd.Function):
    def __init__(self, bitwidth):
        assert bitwidth in [8, 16], "bitwidth should be 8 or 16, but got {} instead".format(bitwidth)
        self.bitwidth = bitwidth
        self.look_up = read_vector_from_file_hex("../quant_tools/soft_out_result")
        self.sum_exp_debug = []
        self.look_up_result = []
        self.quant_out_layer = QuantizeFeature(bitwidth, bitwidth - 4)
        self.output = None
    def forward(self, in_):
        batch, channel, height, width = in_.shape
        sub = in_.clone()
        for y in range(width):
            for x in range(height):
                temp = in_[:,:,x,y].view(-1)
                sum_exp = -8.0
                for i in range(len(temp)):
                    diff = torch.abs(sum_exp - temp[i])
                    diff_int = int(diff * 4096)
                    if diff_int > 32767:
                        diff_int = 32767
                    look_up = self.look_up[diff_int] / 2
                    self.look_ups.append(int(look_up * 65536))
                    if sum_exp > temp[i]:
                        sum_exp += look_up
                    else:
                        sum_exp = temp[i] + look_up
                    if sum_exp * 2 ** (self.bitwidth - 4) > 2 ** (self.bitwidth - 1) - 1:
                        sum_exp = (2 ** (self.bitwidth - 1) - 1) / 2 ** (self.bitwidth - 4)
                    if sum_exp < -8:
                        sum_exp = -8.0
                    if sum_exp >= 0:
                        self.sum_exp_debug.append(int(sum_exp * 4096))
                    else:
                        self.sum_exp_debug.append(int(sum_exp * 4096) + 65536)
                sub[:,:,x,y].fill_(sum_exp)
        self.output = self.quant_out_layer(in_ - sub)
        return self.output
    def backward(self, grad_output):
        return grad_output * (1.0 - torch.exp(self.output))

class QuantizedLogSoftmax(torch.nn.Module):
    def __init__(self, quant_output = True, bitwidth = 8):
        super(QuantizedLogSoftmax, self).__init__()
        self.quant_output = quant_output
        assert bitwidth in [8, 16], "bitwidth should be 8 or 16, but got {} instead".format(bitwidth)
        self.bitwidth = bitwidth
        if self.quant_output:
            self.logSoftmax = QuantizedLogSoftmaxFunction(bitwidth)
        else:
            self.logSoftmax = torch.nn.LogSoftmax(dim = 1)
    def forward(self, x):
        return self.logSoftmax(x)
    def cuda(self):
        if not self.quant_output:
            self.logSoftmax.cuda()
    def double(self):
        if not self.quant_output:
            self.logSoftmax.cuda()

class QuantizedConcat(torch.nn.Module):
    def __init__(self, layer):
        super(QuantizedConcat, self).__init__()
        assert isinstance(layer, Layer), "layer should be an instance of ndk.layers.Layer, but got {} instead".format(type(layer))
        self.axis = layer.axis
    def forward(self, input_list):
        return torch.cat(input_list, self.axis)

class QuantizedSlice(torch.nn.Module):
    def __init__(self, layer):
        super(QuantizedSlice, self).__init__()
        assert isinstance(layer, Layer), "layer should be an instance of ndk.layers.Layer, but got {} instead".format(type(layer))
        self.axis = layer.axis
        self.slice_point = layer.slice_point
    def forward(self, x):
        size = x.shape[self.axis]
        slice_point_full = [0] + self.slice_point + [size]
        slice_size = []
        for i in range(len(slice_point_full) - 1):
            slice_size.append(slice_point_full[i + 1] - slice_point_full[i])
        return list(torch.split(x, slice_size, self.axis))

class QuantizedShuffleChannel(torch.nn.Module):
    def __init__(self, layer):
        super(QuantizedShuffleChannel, self).__init__()
        assert isinstance(layer, Layer), "layer should be an instance of ndk.layers.Layer, but got {} instead".format(type(layer))
        self.group = layer.group
    def forward(self, x):
        n, c, h, w = x.shape
        x = x.reshape((n, self.group, c // self.group, h, w))
        x = x.transpose(1, 2)
        x = x.reshape((n, c, h, w))
        return x
        
class QuantizedNet(torch.nn.Module):
    def __init__(self, layer_list, param_dict, quant_output = True, quant_weight = True, quant_bias = True, hw_aligned = True, bitwidth = 8, for_training = True):
        super(QuantizedNet, self).__init__()
        assert isinstance(param_dict, dict), "param_dict should be a dict, but got {} instead".format(type(param_dict))
        assert bitwidth in [8, 16], "bitwidth should be 8 or 16, but got {} instead".format(bitwidth)
        self.input_name = []
        self.data_dict = {}
        self.quant_output = quant_output
        self.quant_weight = quant_weight
        self.quant_bias = quant_bias
        self.hw_aligned = hw_aligned
        self.bitwidth = bitwidth
        self.output_name = get_network_output(layer_list)
        self.layer_list = [layer for layer in layer_list if layer.type != "Input"]
        self.tensor_shape = get_tensor_shape(layer_list)
        self.torch_layers = {}
        self.build_layers(layer_list, param_dict, for_training)
        self.param_dict = param_dict
    def build_layers(self, layer_list, param_dict, for_training = True):
        for layer in layer_list:
            assert isinstance(layer, Layer), "layer should be an instance of ndk.layers.Layer, but got {} instead".format(type(layer))
            print("constructing layer {}, type {}".format(layer.name, layer.type))
            if layer.type == "Input":
                self.input_name.append(layer.top)
            elif layer.type == "Convolution":
                self.torch_layers[layer.name] = QuantizedConv2D(layer, param_dict, self.quant_output, self.quant_weight, self.quant_bias, self.bitwidth, for_training)
            elif layer.type == "InnerProduct":
                self.torch_layers[layer.name] = QuantizedInnerproduct(layer, param_dict, self.quant_output, self.quant_weight, self.quant_bias, self.bitwidth, for_training)
            elif layer.type in ["BatchNorm", "Scale"]:
                self.torch_layers[layer.name] = QuantizedScale(layer, param_dict, self.tensor_shape[layer.bottom], self.quant_output, self.quant_weight, self.quant_bias, self.bitwidth, for_training)
            elif layer.type == "ScaleByTensor":
                self.torch_layers[layer.name] = QuantizedScaleByTensor(layer, param_dict, self.quant_output, self.bitwidth)
            elif layer.type in ["Bias"]:
                self.torch_layers[layer.name] = QuantizedBias(layer, param_dict, self.tensor_shape[layer.bottom], self.quant_output, self.quant_bias, self.bitwidth, for_training)
            elif layer.type == "Pooling":
                if layer.pool == "ave":
                    self.torch_layers[layer.name] = QuantizedAvgpool2D(layer, param_dict, self.tensor_shape[layer.bottom], self.tensor_shape[layer.top], self.hw_aligned, self.quant_weight, self.quant_output, self.bitwidth)
                else:
                    pu = layer.pad[0]
                    pd = layer.pad[1]
                    pl = layer.pad[2]
                    pr = layer.pad[3]
                    self.torch_layers[layer.name] = torch.nn.MaxPool2d(layer.kernel_size, layer.stride, (pu, pl), layer.dilation, False, pu < pd or pl < pr)
            elif layer.type == "ReLU6":
                self.torch_layers[layer.name] = torch.nn.ReLU6()
            elif layer.type == "ReLU":
                if layer.negative_slope == 0:
                    self.torch_layers[layer.name] = torch.nn.ReLU()
                else:
                    self.torch_layers[layer.name] = QuantizedLeakyrelu(layer, param_dict, self.quant_output, self.quant_weight, self.bitwidth)
            elif layer.type == "Sigmoid":
                self.torch_layers[layer.name] = QuantizedSigmoid(self.quant_output, self.bitwidth)
            elif layer.type == "Tanh":
                self.torch_layers[layer.name] = QuantizedTanh(self.quant_output, self.bitwidth)
            elif layer.type == "Eltwise":
                if layer.operation == "sum":
                    self.torch_layers[layer.name] = QuantizedEltwiseAdd(layer, param_dict, self.quant_output, self.bitwidth)
                else:
                    raise Exception("Eltwise layer whose operation is {} is not supported".format(layer.operation))
            elif layer.type == "Slice":
                self.torch_layers[layer.name] = QuantizedSlice(layer)
            elif layer.type == "Concat":
                self.torch_layers[layer.name] = QuantizedConcat(layer)
            elif layer.type == "ShuffleChannel":
                self.torch_layers[layer.name] = QuantizedShuffleChannel(layer)
            elif layer.type == "LogSoftmax":
                self.torch_layers[layer.name] = QuantizedLogSoftmax(self.quant_output, self.bitwidth)
            elif layer.type == "Softmax":
                assert not self.quant_output, "Softmax in quant_output mode is not suppored"
                self.torch_layers[layer.name] = torch.nn.Softmax(dim = 1)
            else:
                raise Exception("not supported error of type {}".format(layer.type))
    def forward(self, input_):
        if isinstance(input_, dict):
            self.data_dict.update(input_)
        else:
            assert len(self.input_name) == 1, "more than 1 input tensor is needed"
            self.data_dict[self.input_name[0]] = input_
        for index, layer in enumerate(self.layer_list):
            if layer.type in multi_output_layer:
                layer_output = self.torch_layers[layer.name](self.data_dict[layer.bottom])
                for i in range(len(layer.top)):
                    self.data_dict[layer.top[i]] = layer_output[i]
            elif layer.type in multi_input_layer:
                self.data_dict[layer.top] = self.torch_layers[layer.name]([self.data_dict[bottom] for bottom in layer.bottom])
            else:
                self.data_dict[layer.top] = self.torch_layers[layer.name](self.data_dict[layer.bottom])
        return self.data_dict[self.output_name[0]]
    def cuda(self):
        for key in self.torch_layers.keys():
            torch_layer = self.torch_layers[key]
            torch_layer.cuda()
    def double(self):
        for key in self.torch_layers.keys():
            torch_layer = self.torch_layers[key]
            torch_layer.double()
    def get_param_dict(self):
        param_dict = {}
        for layer in self.layer_list:
            torch_layer = self.torch_layers[layer.name]
            if layer.type in ["Convolution", "InnerProduct", "BatchNorm", "Scale", "Bias"]:
                if layer.type != "Bias":
                    param_dict[layer.name + "_weight"] = torch_layer.params['weight'].data.cpu().numpy()
                    if layer.type in ["BatchNorm", "Scale"]:
                        param_dict[layer.name + "_weight"] = param_dict[layer.name + "_weight"].reshape(-1)
                    if torch_layer.quant_weight:
                        if layer.type in ["Convolution", "InnerProduct"]:
                            param_dict[layer.name + "_quant_weight"] = (torch_layer.weight_quantizer(torch_layer.params['weight'])).data.cpu().numpy()
                            for i in range(len(torch_layer.weight_frac)):
                                param_dict[layer.name + "_quant_weight"][i] = param_dict[layer.name + "_quant_weight"][i] * 2 ** torch_layer.weight_frac[i]
                        else:
                            param_dict[layer.name + "_quant_weight"] = (torch_layer.weight_quantizer(torch_layer.params['weight'])).data.cpu().numpy().reshape(-1)
                            for i in range(len(torch_layer.weight_frac)):
                                param_dict[layer.name + "_quant_weight"][i] = param_dict[layer.name + "_quant_weight"][i] * 2 ** torch_layer.weight_frac[i]
                        param_dict[layer.name + "_quant_weight"] = param_dict[layer.name + "_quant_weight"].astype(np.int32)
                        param_dict[layer.name + "_frac_weight"] = np.array(torch_layer.weight_frac)
                if layer.type == "Bias" or torch_layer.bias_term:
                    param_dict[layer.name + "_bias"] = torch_layer.params['bias'].data.cpu().numpy().reshape(-1)
                    if torch_layer.quant_bias:
                        param_dict[layer.name + "_quant_bias"] = (torch_layer.bias_quantizer(torch_layer.params['bias'])).data.cpu().numpy().reshape(-1)
                        for i in range(len(torch_layer.bias_frac)):
                            param_dict[layer.name + "_quant_bias"][i] = param_dict[layer.name + "_quant_bias"][i] * 2 ** torch_layer.bias_frac[i]
                        param_dict[layer.name + "_quant_bias"] = param_dict[layer.name + "_quant_bias"].astype(np.int32)
                        param_dict[layer.name + "_frac_bias"] = np.array(torch_layer.bias_frac)
        for key in self.param_dict.keys():
            if key.endswith("_frac"):
                param_dict[key] = self.param_dict[key]
            if key.endswith("_signed"):
                param_dict[key] = self.param_dict[key]
        return param_dict
    
if __name__ == "__main__":
    from ndk.onnx_parser.onnx_parser import load_from_onnx
    from ndk.examples.data_generator_imagenet_partial import data_generator_imagenet_partial
    from ndk.optimize import add_pre_norm, merge_layers
    from ndk.quantize import quantize_model
    from ndk.layers import get_net_input_output
    import torchvision
    from ndk.quant_tools.numpy_net import run_layers
    net_name = 'shufflenet_v2_x1_0'
    net_path = './../examples/{}.onnx'.format(net_name)
    train_data_path = 'D:/AI_Images/ILSVRC2012/train'
    valid_data_path = '/Data-pool/data/ILSVRC2012/val/'
    
    layer_list, param_dict = load_from_onnx(net_path)
    data_generator_quant = data_generator_imagenet_partial(
                             imagenet_dirname='../examples/imagenet_partial',
                             filenames_to_class='filenames_to_class_by_number.json',
                             batch_size=10,target_size=(224,224),
                             random_order=True, num_class = 1000)
    weight = np.array([1/0.229/255, 1/0.224/255, 1/0.225/255], dtype = np.float32)
    bias = np.array([-0.485/0.229, -0.456/0.224, -0.406/0.225], dtype = np.float32)
    add_pre_norm(layer_list, param_dict, weight, bias)
    layer_list, param_dict = merge_layers(layer_list, param_dict)
    input_tensor_name, output_tensor_name = get_net_input_output(layer_list)
    quant_layer_list0, quant_param_dict0 = quantize_model(
        layer_list=layer_list,
        param_dict=param_dict,
        bitwidth=8,
        data_generator=data_generator_quant,
        method_dict = "MSE",
        aggressive=True,
        factor_num_bin=8,
        num_step_pre=2,
        num_step=4,
        priority_mode='fwb',
    )
    quant_net = QuantizedNet(quant_layer_list0, quant_param_dict0, for_training = False)
    quant_net.cuda()
    quant_net.double()
    data = next(data_generator_quant)
    inputs = torch.FloatTensor(data['input'])
    inputs = inputs.cuda().double()
    outputs = quant_net(inputs)
    outputs = outputs.cpu().reshape((10, -1))
    test_output = run_layers(input_data_batch=data['input'], layer_list=quant_layer_list0, target_feature_tensor_list=output_tensor_name,
                                          param_dict=quant_param_dict0, bitwidth=8, quant=True, hw_aligned=True, numpy_layers=None, log_on=False)
    out_torch_numpy = outputs.data.cpu().numpy()
    error0 = out_torch_numpy - test_output[output_tensor_name[0]].reshape((10, -1))
    quant_net.train()
    trainset = torchvision.datasets.ImageFolder(train_data_path,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.Resize((256, 256)),
                                                    torchvision.transforms.RandomCrop(224),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0] * 3, std=[1/255] * 3),
                                                    ])
                                                )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                          shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam([{'params': torch_layer.parameters(), 'lr': 1e-5} for torch_layer in quant_net.torch_layers.values()])
    loss_fn = torch.nn.CrossEntropyLoss()
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs = inputs.double()
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = quant_net(inputs)
        net_out = outputs.reshape((10, -1))
        prediction = torch.max(net_out, 1)[1]
        correct = int(torch.sum(prediction == labels).cpu().numpy())
        print(correct)
        loss = loss_fn(net_out, labels)
        print('loss =', float(loss.data))
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if i > 10:
            break
    quant_param_dict1 = quant_net.get_param_dict()
    data = next(data_generator_quant)
    inputs = torch.FloatTensor(data['input'])
    inputs = inputs.cuda().double()
    outputs = quant_net(inputs)
    outputs = outputs.cpu().reshape((10, -1))
    test_output = run_layers(input_data_batch=data['input'], layer_list=quant_layer_list0, target_feature_tensor_list=output_tensor_name,
                                          param_dict=quant_param_dict1, bitwidth=8, quant=True, hw_aligned=True, numpy_layers=None, log_on=False)
    out_torch_numpy = outputs.data.cpu().numpy()
    error1 = out_torch_numpy - test_output[output_tensor_name[0]].reshape((10, -1))
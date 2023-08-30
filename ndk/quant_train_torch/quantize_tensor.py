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

import math
import numpy as np
import torch

ndk_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ndk_dir)

class Floor(torch.autograd.Function):
    @staticmethod
    def forward(self, input_):
        self.save_for_backward(input_)
        output = torch.floor(input_)
        return output
    @staticmethod
    def backward(self, grad_output):
        return grad_output * 1.0

class RoundAndClipInt8(torch.autograd.Function):
    @staticmethod
    def forward(self, input_):
        self.save_for_backward(input_)
        output = torch.round(input_)
        output = torch.clamp(output, -128, 127)
        return output
    @staticmethod
    def backward(self, grad_output):
        return grad_output * 1.0    

class RoundAndClipUint8(torch.autograd.Function):
    @staticmethod
    def forward(self, input_):
        self.save_for_backward(input_)
        output = torch.round(input_)
        output = torch.clamp(output, 0, 255)
        return output
    @staticmethod
    def backward(self, grad_output):
        return grad_output * 1.0    

class RoundAndClipInt16(torch.autograd.Function):
    @staticmethod
    def forward(self, input_):
        self.save_for_backward(input_)
        output = torch.round(input_)
        output = torch.clamp(output, -32768, 32767)
        return output
    @staticmethod
    def backward(self, grad_output):
        return grad_output * 1.0    

class RoundAndClipUint16(torch.autograd.Function):
    @staticmethod
    def forward(self, input_):
        self.save_for_backward(input_)
        output = torch.round(input_)
        output = torch.clamp(output, 0, 65536)
        return output
    @staticmethod
    def backward(self, grad_output):
        return grad_output * 1.0    

class QuantizeFeature(torch.nn.Module):
    def __init__(self, bitwidth, frac, signed = True):
        super(QuantizeFeature, self).__init__()
        assert bitwidth in [8, 16], "bitwidth should be 8 or 16, but got {} instead".format(bitwidth)
        self.bitwidth = bitwidth
        try:
            self.frac = int(frac)
        except:
            raise Exception("frac should be an integer but got type {} instead".format(type(frac)))
        self.signed = signed
        self.max = math.pow(2.0, bitwidth - 1) - 1 if signed else math.pow(2.0, bitwidth) - 1
        self.min = -math.pow(2.0, bitwidth - 1) if signed else 0
    def forward(self, input_):
        input_ = input_ * math.pow(2.0, self.frac)
        output_int = Floor.apply(input_)
        output_int = torch.clamp(output_int, self.min, self.max)
        output = output_int / math.pow(2.0, self.frac)
        return output

'''
type: 0 - conv weight, shape (c_out, c_in, h, w)
      1 - conv bias, shape (c_out)
      2 - scale, shape (1, c_out, 1, 1)
'''
class QuantizeParam(torch.nn.Module):
    def __init__(self, bitwidth, frac, signed = True, type = 0):
        super(QuantizeParam, self).__init__()
        assert bitwidth in [8, 16], "bitwidth should be 8 or 16, but got {} instead".format(bitwidth)
        self.bitwidth = bitwidth
        frac_multiplier_np = np.power(2.0, frac)
        if type == 0:
            frac_multiplier_np = np.expand_dims(frac_multiplier_np, axis=-1)
            frac_multiplier_np = np.expand_dims(frac_multiplier_np, axis=-1)
            frac_multiplier_np = np.expand_dims(frac_multiplier_np, axis=-1)
        elif type == 1:
            pass
        elif type == 2:
            frac_multiplier_np = np.expand_dims(frac_multiplier_np, axis= 0)
            frac_multiplier_np = np.expand_dims(frac_multiplier_np, axis=-1)
            frac_multiplier_np = np.expand_dims(frac_multiplier_np, axis=-1)
        else:
            raise Exception("wrong type : {}".format(type))
        self.frac_multiplier = torch.from_numpy(frac_multiplier_np)
        self.signed = signed
        self.max = math.pow(2.0, bitwidth - 1) - 1 if signed else math.pow(2.0, bitwidth) - 1
        self.min = -math.pow(2.0, bitwidth - 1) if signed else 0
        if signed:
            if bitwidth == 8:
                self.roundAndClip = RoundAndClipInt8
            else:
                self.roundAndClip = RoundAndClipInt16
        else:
            if bitwidth == 8:
                self.roundAndClip = RoundAndClipUint8
            else:
                self.roundAndClip = RoundAndClipUint16
    def forward(self, input_):
        output = input_ * self.frac_multiplier
        output_int = self.roundAndClip.apply(output)
        output = output_int / self.frac_multiplier
        return output
    def double(self):
        self.frac_multiplier = self.frac_multiplier.double()
    def cuda(self):
        self.frac_multiplier = self.frac_multiplier.cuda()

if __name__ == "__main__":
    pass

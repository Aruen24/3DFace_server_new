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
import numpy as np
import math
import time

"Assorted class utilities and tools"
class AttrDisplay:
    """
    Provides an inheritable print overload method that displays
    instansces with their class names and a name=value pair for
    each attribute stored on the instance itself (but not attrs
    inherited from its classes). Can be mixed into any class,
    and will work on any instance.
    """
    def gatherAttrs(self):
        attrs = []
        for key in sorted(self.__dict__):
            if key != 'type_dict':
                attrs.append('%s=%s'%(key, getattr(self,key)))
        return ', '.join(attrs)
    def __str__(self):
        return '[%s: %s]' %(self.__class__.__name__, self.gatherAttrs())


'''
This function counts the number of data samples falls in a set of bins.
bin_values is a 1-dim np.array which determines the representing values of these bins, the values must be sorted in increasing order.
'''
def hist_func(data, bin_values, uniform_bin=True):
    num_bin = len(bin_values)
    counts = np.zeros(num_bin, dtype=int)
    if uniform_bin:
        min_bin = bin_values[0]
        step = 1/(bin_values[1] - bin_values[0])
        quant_data = np.clip(np.round((data.reshape(-1) - min_bin) * step).astype(int), 0, num_bin-1)
        counts = np.zeros(num_bin, dtype=int)
        for x in quant_data:
            counts[x] += 1
    else:
        sorted_data = np.sort(np.array(data).reshape(-1))
        division_values = (bin_values[:-1] + bin_values[1:])/2
        idx_bin = 0
        for x in sorted_data:
            while idx_bin < num_bin-1 and x > division_values[idx_bin]:
                idx_bin += 1
            counts[idx_bin] += 1
    return counts

def format_prob_distribution(bins, probs, target_bins):
    division_values = (target_bins[:-1] + target_bins[1:])/2
    num_bin = len(division_values) + 1
    target_probs = np.zeros(num_bin)
    idx_bin = 0
    for i, x in enumerate(bins):
        while idx_bin < num_bin-1 and x > division_values[idx_bin]:
            idx_bin += 1
        target_probs[idx_bin] += probs[i]
    return target_probs

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


def print_log(msg):
    print('{}: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), msg))

# softmax
def softmax(x):
    assert isinstance(x, np.ndarray), 'input should be a one-dim numpy.ndarray, but get {}'.format(type(x))
    assert len(x.shape)==1, 'input should be a one-dim numpy.ndarray, but get {}'.format(x.shape)
    shifted_x = np.array(x) - np.max(x) # make it stable
    exp_x = np.exp(shifted_x)
    return exp_x / np.sum(exp_x)

# cross entropy, H(p,q) = sum(p*log(1/q))
def cross_entropy(p, q):
    assert isinstance(p, np.ndarray), 'input should be a one-dim numpy.ndarray, but get {}'.format(type(p))
    assert isinstance(q, np.ndarray), 'input should be a one-dim numpy.ndarray, but get {}'.format(type(q))
    assert len(p.shape)==1, 'input should be a one-dim numpy.ndarray, but get {}'.format(p.shape)
    assert len(q.shape)==1, 'input should be a one-dim numpy.ndarray, but get {}'.format(q.shape)
    return np.sum(np.nan_to_num(-p*np.log(q)))

# Cosine distance
def cos_distance(x, y):
    assert isinstance(x, np.ndarray), 'input should be a one-dim numpy.ndarray, but get {}'.format(type(x))
    assert isinstance(y, np.ndarray), 'input should be a one-dim numpy.ndarray, but get {}'.format(type(y))
    assert len(x.shape)==1, 'input should be a one-dim numpy.ndarray, but get {}'.format(x.shape)
    assert len(y.shape)==1, 'input should be a one-dim numpy.ndarray, but get {}'.format(y.shape)
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# Hellinger distance
def hellinger_distance(p, q):
    assert isinstance(p, np.ndarray), 'input should be a one-dim numpy.ndarray, but get {}'.format(type(p))
    assert isinstance(q, np.ndarray), 'input should be a one-dim numpy.ndarray, but get {}'.format(type(q))
    assert len(p.shape)==1, 'input should be a one-dim numpy.ndarray, but get {}'.format(p.shape)
    assert len(q.shape)==1, 'input should be a one-dim numpy.ndarray, but get {}'.format(q.shape)
    return np.sqrt(1 - np.sum(np.sqrt(p * q)))

# Bhattacharyya distance
def bhattacharyya_distance(p, q):
    assert isinstance(p, np.ndarray), 'input should be a one-dim numpy.ndarray, but get {}'.format(type(p))
    assert isinstance(q, np.ndarray), 'input should be a one-dim numpy.ndarray, but get {}'.format(type(q))
    assert len(p.shape)==1, 'input should be a one-dim numpy.ndarray, but get {}'.format(p.shape)
    assert len(q.shape)==1, 'input should be a one-dim numpy.ndarray, but get {}'.format(q.shape)
    return -np.log(np.sum(np.sqrt(p*q)))

# Kullback-Leibler divergence
def KLD(p, q):
    assert isinstance(p, np.ndarray), 'input should be a one-dim numpy.ndarray, but get {}'.format(type(p))
    assert isinstance(q, np.ndarray), 'input should be a one-dim numpy.ndarray, but get {}'.format(type(q))
    assert len(p.shape)==1, 'input should be a one-dim numpy.ndarray, but get {}'.format(p.shape)
    assert len(q.shape)==1, 'input should be a one-dim numpy.ndarray, but get {}'.format(q.shape)
    p = p + np.spacing(1)
    q = q + np.spacing(1)
    return np.sum(p * np.log2(p/q))

# Jensen-Shannon divergence
def JSD(p, q):
    assert isinstance(p, np.ndarray), 'input should be a one-dim numpy.ndarray, but get {}'.format(type(p))
    assert isinstance(q, np.ndarray), 'input should be a one-dim numpy.ndarray, but get {}'.format(type(q))
    assert len(p.shape)==1, 'input should be a one-dim numpy.ndarray, but get {}'.format(p.shape)
    assert len(q.shape)==1, 'input should be a one-dim numpy.ndarray, but get {}'.format(q.shape)
    m = 0.5*(p+q)
    return 0.5 * KLD(p, m) + 0.5 * KLD(q, m)

"""
This function is for saving a feature (input image or an output tensor of a layer in the net) to a C header file, so as to use the tensor in the chip
A common way to use this function is as the steps below:
    1) call function run_layers with parameter quant = True, hw_aligned = True, and target_feature_tensor_list = the tensors you want to save
    2) for each feature you want to save, call this function, frac is in the param_dict that you used to call function run_layers
"""
def save_quantized_feature_to_header(filename, feature, bit_width, frac, signed = True, name_in_header = 'feature', aligned = True):
    assert bit_width in [8, 16], 'bit_width should be 8 or 16, but got {} instead'.format(bit_width)
    assert isinstance(feature, np.ndarray), 'feature should be a numpy.ndarray, but got {} instead'.format(type(feature))
    assert len(feature.shape) == 4, 'feature should be a 4D array, but got a {}D array instead'.format(len(feature.shape))
    max_value = (math.pow(2.0, bit_width - 1) - 1) / math.pow(2.0, frac) if signed else (math.pow(2.0, bit_width) - 1) / math.pow(2.0, frac)
    min_value = -math.pow(2.0, bit_width - 1) / math.pow(2.0, frac) if signed else 0.0
    if signed:
        print("{}bit signed fix point number could refer to real number in range [{}, {}]".format(bit_width, min_value, max_value))
    else:
        print("{}bit unsigned fix point number could refer to real number in range [{}, {}]".format(bit_width, min_value, max_value))
    batch, channel, height, width = feature.shape
    padded_width = math.ceil(width / (256 // bit_width)) * (256 // bit_width) if aligned else width
    written_lines = []
    with open(filename, 'w') as f:
        f.write('#include "iot_io_api.h"\n')
        if signed:
            f.write('int{}_t {}[] = {{\n'.format(bit_width, name_in_header))
        else:
            f.write('uint{}_t {}[] = {{\n'.format(bit_width, name_in_header))
    with open(filename, 'a') as f:
        for b in range(batch):
            for c in range(channel):
                for h in range(height):
                    for w in range(width):
                        assert feature[b, c, h, w] <= max_value and feature[b, c, h, w] >= min_value, "feature[{}, {}, {}, {}] = {}, out of range [{}, {}]".format(b, c, h, w, feature[b, c, h, w], min_value, max_value)
                        element_int_form = int(math.floor(feature[b, c, h, w] * math.pow(2.0, frac)))
                        if element_int_form < 0:
                            element_int_form += 2 ** bit_width
                        if bit_width == 8:
                            written_lines.append("0x{:02x}".format(element_int_form))
                        else:
                            written_lines.append("0x{:04x}".format(element_int_form))
                    for w in range(int(padded_width - width)):
                        if bit_width == 8:
                            written_lines.append("0x{:02x}".format(0))
                        else:
                            written_lines.append("0x{:04x}".format(0))
                if b != 0 or c != 0:
                    f.write(',\n')
                f.write(',\n'.join(written_lines))
                written_lines.clear()
        f.write('\n};')
            

if __name__=='__main__':
    feature = np.random.uniform(low = -0.5, high = 0.5 * 127 / 128, size = (1, 3, 24, 24))
    save_quantized_feature_to_header('temp.h', feature, 8, 8, signed = True)
    p = np.ones(5) / 5.0
    q = np.array([0.1, 0.3, 0.2, 0.2, 0.2])
    print(JSD(p, q))

    class TopTest(AttrDisplay):
        count = 0
        def __init__(self):
            self.attr1 = TopTest.count
            self.attr2 = TopTest.count+1
            TopTest.count += 2
    class SubTest(TopTest):
        pass

    X,Y = TopTest(),SubTest()
    print(X)
    print(Y)


    data = np.random.rand(100000,100)
    bins = np.linspace(0,1,2**20)
    # tic = time.clock()
    # cnt1 = hist_func(data, bins)
    # print('Elapsed time: {:.2f}'.format(time.clock()-tic))
    tic = time.clock()
    cnt1 = hist_func(data, bins)
    print('Elapsed time: {:.2f}'.format(time.clock()-tic))
    # print(cnt1)
    tic = time.clock()
    cnt2 = hist_func2(data, bins)
    print('Elapsed time: {:.2f}'.format(time.clock()-tic))
    # print(cnt2)
    print(np.sum(cnt1-cnt2))


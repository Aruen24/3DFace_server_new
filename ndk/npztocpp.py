# -*- coding: utf-8 -*-

import numpy as np
#import pickle
import modelpack
import os
import sys
ndk_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ndk_dir)
import ndk
np.set_printoptions(threshold='nan')


def changenpz(prototxt, npz):
    layer_list, param_dict=modelpack.load_from_file(prototxt, npz)
    layer_list = ndk.layers.sort_layers(layer_list)    
    if 1:
        with open("layers.txt", "w", encoding='utf-8') as f:
            for layer in layer_list:
                if hasattr(layer, 'bottom'):
                    ...
                else:
                    layer.bottom = 0
                if hasattr(layer, 'num_output'):
                    ...
                else:
                    layer.num_output = 0
                if hasattr(layer, 'bias_term'):
                    ...
                else:
                    layer.bias_term = True
                if hasattr(layer, 'pad'):
                    ...
                else:
                    layer.pad = (0,0,0,0)
                if hasattr(layer, 'pad_h'):
                    ...
                else:
                    layer.pad_h = 0
                if hasattr(layer, 'pad_w'):
                    ...
                else:
                    layer.pad_w = 0
                if hasattr(layer, 'kernel_size'):
                    ...
                else:
                    layer.kernel_size = (0,0)
                if hasattr(layer, 'kernel_size_h'):
                    ...
                else:
                    layer.kernel_size_h = 0
                if hasattr(layer, 'kernel_size_w'):
                    ...
                else:
                    layer.kernel_size_w = 0
                if hasattr(layer, 'stride'):
                    ...
                else:
                    layer.stride = (0,0)
                if hasattr(layer, 'stride_h'):
                    ...
                else:
                    layer.stride_h = 0
                if hasattr(layer, 'stride_w'):
                    ...
                else:
                    layer.stride_w = 0
                if hasattr(layer, 'group'):
                    ...
                else:
                    layer.group = 0
                if hasattr(layer, 'pool'):
                    ...
                else:
                    layer.pool = 0
                if hasattr(layer, 'operation'):
                    ...
                else:
                    layer.operation = 0
                if hasattr(layer, 'dim'):
                    ...
                else:
                    layer.dim = (0,0,0,0)
                if hasattr(layer, 'slice_point'):
                    ...
                else:
                    layer.slice_point = 0
                if hasattr(layer, 'global_pooling'):
                    ...
                else:
                    layer.global_pooling = 0 
                if hasattr(layer, 'dilation'):
                    ...
                else:
                    layer.dilation = (0,0)
                if hasattr(layer, 'dilation_h'):
                    ...
                else:
                    layer.dilation_h = 0 
                if hasattr(layer, 'dilation_w'):
                    ...
                else:
                    layer.dilation_w = 0
                if hasattr(layer, 'order'):
                    ...
                else:
                    layer.order = (0,0,0,0)
                if hasattr(layer, 'axis'):
                    ...
                else:
                    layer.axis = 0
                if hasattr(layer, 'aspect_ratio'):
                    ...
                else:
                    layer.aspect_ratio = 0
                if hasattr(layer, 'negative_slope'):
                    ...
                else:
                    layer.negative_slope = 0
                f.write(str(layer.name))
                f.write(' ')
                f.write(str(layer.type).lower())
                f.write(' ')
                f.write(str(layer.bottom).replace(' ', '').replace('[','').replace(']','').replace('\'',''))
                f.write(' ')
                f.write(str(layer.dim).replace(' ', '').replace('(','').replace(')',''))
                f.write(' ')
                f.write(str(layer.top).replace(' ', '').replace('[','').replace(']','').replace('\'',''))
                f.write(' ')
                f.write(str(layer.num_output))
                f.write(' ')
                f.write(str(layer.bias_term).lower())
                f.write(' ')
                f.write(str(layer.pad).replace(' ', '').replace('(','').replace(')',''))
                f.write(' ')
                f.write(str(layer.pad_h))
                f.write(' ')
                f.write(str(layer.pad_w))
                f.write(' ')
                f.write(str(layer.kernel_size).replace(' ', '').replace('(','').replace(')',''))
                f.write(' ')
                f.write(str(layer.kernel_size_h))
                f.write(' ')
                f.write(str(layer.kernel_size_w))
                f.write(' ')
                f.write(str(layer.stride).replace(' ', '').replace('(','').replace(')',''))
                f.write(' ')
                f.write(str(layer.stride_h))
                f.write(' ')
                f.write(str(layer.stride_w))
                f.write(' ')
                f.write(str(layer.group))
                f.write(' ')
                f.write(str(layer.pool))
                f.write(' ')
                f.write(str(layer.operation))
                f.write(' ')
                f.write(str(layer.slice_point).replace(' ', '').replace('[','').replace(']',''))
                f.write(' ')
                f.write(str(layer.global_pooling))
                f.write(' ')
                f.write(str(layer.dilation).replace(' ', '').replace('(','').replace(')',''))
                f.write(' ')
                f.write(str(layer.dilation_h))
                f.write(' ')
                f.write(str(layer.dilation_w))
                f.write(' ')
                f.write(str(layer.order).replace(' ', '').replace('(','').replace(')',''))
                f.write(' ')
                f.write(str(layer.axis))
                f.write(' ')
                f.write(str(layer.negative_slope))
                f.write(' ')
                f.write('\n')
               # f.write(str(layer.top))
        f.close()
    
    import struct
    if 1:
        with open("params.dat", "wb") as f:
            for layer in param_dict:
                if type(param_dict[layer]) in [int]:        
                    f.write(struct.pack('f', float(param_dict[layer])) )
                elif type(param_dict[layer]) in [bool]:  
                    ...
                else:
                    param = param_dict[layer].flatten()
                    length = param.size
                    for i in range(length):
                        f.write(struct.pack('f',float(param[i]))  )
        f.close()
        with open("params_name.txt", "w", encoding='utf-8') as f:
            for layer in param_dict:
                f.write(str(layer))
                f.write(' ')
                if type(param_dict[layer]) in [int]:
                    shape = 1
                    f.write(str(shape))
                elif type(param_dict[layer]) in [bool]:  
                    shape = 0
                    f.write(str(shape))
                    f.write(' ')
                    f.write(str(param_dict[layer]).lower())
                else:
                    size = param_dict[layer].shape
                    print(size)
                    if len(size)==1:
                        f.write(str(size).replace(' ', '').replace('(','').replace(')','').replace(',',''))
                    else:
                        f.write(str(size).replace(' ', '').replace('(','').replace(')',''))
                f.write('\n')
        f.close()

if __name__=='__main__':
    print('Not implemenlayersted')
    changenpz('test_code/keypoint_ir.prototxt', 'test_code/keypoint_ir.npz')

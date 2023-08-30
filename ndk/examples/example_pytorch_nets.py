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
import numpy as np
import torch

# add ndk directory to system path, and import ndk.
import sys
ndk_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ndk_dir)

from ndk.onnx_parser.onnx_parser import load_from_onnx
from ndk.optimize import add_pre_norm, merge_layers
from ndk.quantize import quantize_model
from ndk.examples.data_generator_imagenet_partial import data_generator_imagenet_partial
from ndk.quant_train_torch.quant_layer import QuantizedNet
from ndk.modelpack import modelpack

if __name__ == "__main__":
    net_name = 'shufflenet_v2_x1_0'

    layer_list, param_dict = load_from_onnx(net_name + ".onnx")

    weight = np.array([1/0.229/255, 1/0.224/255, 1/0.225/255], dtype = np.float32)
    bias = np.array([-0.485/0.229, -0.456/0.224, -0.406/0.225], dtype = np.float32)
    add_pre_norm(layer_list, param_dict, weight, bias)

    layer_list, param_dict = merge_layers(layer_list, param_dict)

    data_generator_quant = data_generator_imagenet_partial(
                             imagenet_dirname='../examples/imagenet_partial',
                             filenames_to_class='filenames_to_class_by_number.json',
                             batch_size=10,
                             random_order=True, num_class = 1000)
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

    data_generator_train = data_generator_imagenet_partial(
                             imagenet_dirname='../examples/imagenet_partial',
                             filenames_to_class='filenames_to_class_by_number.json',
                             batch_size=10,
                             random_order=True,
                             one_hot = False,
                             num_class = 1000)
    quant_net = QuantizedNet(quant_layer_list0, quant_param_dict0)
    if torch.cuda.is_available():
        quant_net.cuda()
    quant_net.double()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([{'params': torch_layer.parameters(), 'lr': 1e-5} for torch_layer in quant_net.torch_layers.values()])
    for i in range(1000):
        data = next(data_generator_train)
        data_in = data['input']
        labels = data['output'].reshape(-1)
        out_shape = labels.shape
        data_in_torch = torch.from_numpy(data_in).double()
        labels_torch = torch.from_numpy(labels)
        if torch.cuda.is_available():
            data_in_torch = data_in_torch.cuda()
            labels_torch = labels_torch.cuda()
            torch.cuda.empty_cache()
        net_out = quant_net(data_in_torch)
        net_out = net_out.reshape((out_shape[0], -1))
        prediction = torch.max(net_out, 1)[1]
        print(torch.sum(prediction == labels_torch))
        loss = loss_fn(net_out, labels_torch.long())
        print('loss =', float(loss.data))
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    quant_param_dict = quant_net.get_param_dict()
    modelpack(8, quant_layer_list0, quant_param_dict, net_name + "_quant_netpackinfo", model_name = net_name, use_machine_code = False)
    modelpack(8, quant_layer_list0, quant_param_dict, net_name + "_quant_machinecode", model_name = net_name, use_machine_code = True)
#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import sys
import os
import importlib
from tensorflow.python.framework import ops

ndk_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ndk_dir)

import ndk
from ndk.simulated_quant. sim_quant_layers import  SimQuantLayerInput,SimQuantLayerDense,SimQuantLayerConv2d, \
    SimQuantLayerRelu, SimQuantLayerPool, SimQuantLayerBatchNorm
from ndk.simulated_quant.sim_quant_model import SimQuantModel
from ndk.utils import print_log
import ndk.examples.data_generator_mnist as mnist
import ndk.examples.data_generator_imagenet_partial as ln





if __name__ == '__main__':

    FLOAT_MODEL_TRAIN_STEP = 20000
    TRAIN_FLOAT_MODEL = False
    QUANT_GENERAL = True
    QUANT_TRAIN = False
    EVAL_FLOAT = True
    EVAL_QUANT_TRAIN = True
    EXPORT_TO_BIN = True

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    tf.reset_default_graph()
    lr = 0.001

    g_train = ln.data_generator_imagenet_partial(
        imagenet_dirname=r'./liveness/',
        batch_size=512,
        random_order=True,
        n=75776,
        filenames_to_class="train_images.json",
        grayscale=True,
        one_hot=False,
        num_class=2
    )


    if QUANT_GENERAL:
        layer_list, param_dict = ndk.tensorflow_interface.load_from_pb("spc_2d_220715_feathernet_add_epoch150_v2.pb")

        layer_list, param_dict = ndk.optimize.merge_layers(layer_list, param_dict)

        quant_layer_list, quant_param_dict = ndk.quantize.quantize_model(layer_list=layer_list,
                                                                         param_dict=param_dict,
                                                                         bitwidth=16,
                                                                         data_generator=g_train,
                                                                         usr_param_dict=None,
                                                                         num_step_pre=20,
                                                                         num_step=20,
                                                                         gaussian_approx=False
        )

        fname = 'spc_2d_220715_feathernet_add_epoch150_quant_v2'
        ndk.modelpack.save_to_file(layer_list=quant_layer_list, fname_prototxt=fname,
                                   param_dict=quant_param_dict, fname_npz=fname)

    if QUANT_TRAIN:
        fname = 'spc_2d_220715_feathernet_add_epoch150_quant_v2'
        quant_layer_list, quant_param_dict = ndk.modelpack.load_from_file(fname_prototxt=fname, fname_npz=fname)
        quant_layer_list, quant_param_dict = ndk.quantize.quantize_model_with_training(layer_list=quant_layer_list,
                                                                                       bitwidth=16,
                                                                                       param_dict=quant_param_dict,
                                                                                       data_generator=g_train,
                                                                                       loss_fn=tf.losses.softmax_cross_entropy,
                                                                                       optimizer=tf.train.AdamOptimizer(1e-5),
                                                                                       num_step_train=200,
                                                                                       num_step_log=100,
                                                                                       layer_group=1)
        fname = 'spc_2d_220715_feathernet_add_epoch150_quant_train_v2'
        ndk.modelpack.save_to_file(layer_list=quant_layer_list, fname_prototxt=fname,
                                   param_dict=quant_param_dict, fname_npz=fname)

    if EVAL_FLOAT:
        layer_list, param_dict = ndk.tensorflow_interface.load_from_pb("spc_2d_220715_feathernet_add_epoch150_v2.pb")
        g_test = ln.data_generator_imagenet_partial(
                                imagenet_dirname=r'./liveness/',
                                batch_size=3100,
                                random_order=False,
                                n=3100,
                                filenames_to_class="test_images.json",
                                interpolation='bilinear',
                                grayscale=True,
                                one_hot=False,
                                num_class=2
                            )

        data_batch = next(g_test)
        net_output_tensor_name = ndk.layers.get_network_output(layer_list)[0]
        test_output = ndk.quant_tools.numpy_net.run_layers(input_data_batch=data_batch['input'],
                                                           layer_list=layer_list,
                                                           target_feature_tensor_list=[net_output_tensor_name],
                                                           param_dict=param_dict,
                                                           bitwidth=16,
                                                           quant=False,
                                                           hw_aligned=True,
                                                           numpy_layers=None,
                                                           log_on=True)
        correct_prediction = np.equal(np.argmax(test_output[net_output_tensor_name], 1), np.argmax(data_batch['output'], 1))
        accuracy = np.mean(correct_prediction)
        print('Before quantization, model accuracy={:.3f}%'.format(accuracy*100))

    if EVAL_QUANT_TRAIN:

        fname = 'spc_2d_220715_feathernet_add_epoch150_quant_v2'
        quant_layer_list, quant_param_dict = ndk.modelpack.load_from_file(fname_prototxt=fname, fname_npz=fname)

        g_test = ln.data_generator_imagenet_partial(
            imagenet_dirname=r'./liveness/',
            batch_size=3100,
            random_order=False,
            n=3100,
            filenames_to_class="test_images.json",
            interpolation='bilinear',
            grayscale=True,
            one_hot=False,
            num_class=2
                            )

        data_batch = next(g_test)
        net_output_tensor_name = ndk.layers.get_network_output(quant_layer_list)[0]
        test_output = ndk.quant_tools.numpy_net.run_layers(input_data_batch=data_batch['input'],
                                                           layer_list=quant_layer_list,
                                                           target_feature_tensor_list=[net_output_tensor_name],
                                                           param_dict=quant_param_dict,
                                                           bitwidth=16,
                                                           quant=True,
                                                           hw_aligned=True,
                                                           numpy_layers=None,
                                                           log_on=True)

        infer = np.argmax(test_output[net_output_tensor_name], 1)
        labels = np.argmax(data_batch['output'], 1)

        for i in range(3100):
            if infer[i] != labels[i]:
                print(data_batch['filenames'][i])

        correct_prediction = np.equal(np.argmax(test_output[net_output_tensor_name], 1), np.argmax(data_batch['output'], 1))
        accuracy = np.mean(correct_prediction)
        print('After quantization, model accuracy={:.3f}%'.format(accuracy*100))
        

from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow.python.framework import ops

import sys
import os
ndk_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ndk_dir)

from ndk.simulated_quant.sim_quant_layers import SimQuantLayerConv2d,SimQuantLayerRelu,SimQuantLayerPool, \
    SimQuantLayerConcat,SimQuantLayerInput
from ndk.simulated_quant.sim_quant_model import SimQuantModel
from ndk.examples.mtcnn.loss import get_pnet_loss
from ndk.examples.mtcnn.data_process import get_input_fn
from ndk.examples.mtcnn.data_process import get_data_generator
from ndk.examples.mtcnn.loss import count_correct_samples
from ndk.utils import print_log
from ndk.modelpack import save_to_file
import ndk.quantize
from ndk.quant_tools.numpy_net import run_layers

def p_net(input_tensor):
    model = SimQuantModel()

    sim_quant_layer = SimQuantLayerInput((input_tensor, ),
                                     dim=(64, 3, 12, 12), name='input')
    model.add_layer(sim_quant_layer)

    sim_quant_layer = SimQuantLayerConv2d(sim_quant_layer.output, kernel_size=3, num_output=10, stride=1, pad=0, trainable=True, name='conv1')
    model.add_layer(sim_quant_layer)
    sim_quant_layer = SimQuantLayerRelu(sim_quant_layer.output, name='conv1/relu')
    model.add_layer(sim_quant_layer)
    sim_quant_layer = SimQuantLayerPool(sim_quant_layer.output, kernel_size=2, stride=2, pad=0, name='pool1')
    model.add_layer(sim_quant_layer)

    sim_quant_layer = SimQuantLayerConv2d(sim_quant_layer.output, kernel_size=3, num_output=16, stride=1, pad=0, trainable=True, name='conv2')
    model.add_layer(sim_quant_layer)
    sim_quant_layer = SimQuantLayerRelu(sim_quant_layer.output, name='conv2/relu')
    model.add_layer(sim_quant_layer)

    sim_quant_layer = SimQuantLayerConv2d(sim_quant_layer.output, kernel_size=3, num_output=32, stride=1, pad=0, trainable=True, name='conv3')
    model.add_layer(sim_quant_layer)
    sim_quant_layer = SimQuantLayerRelu(sim_quant_layer.output, name='conv3/relu')
    model.add_layer(sim_quant_layer)

    conv4_1 = SimQuantLayerConv2d(sim_quant_layer.output, kernel_size=1, num_output=2, stride=1, pad=0, trainable=True, name='conv4_1')
    model.add_layer(conv4_1)

    conv4_2 = SimQuantLayerConv2d(sim_quant_layer.output, kernel_size=1, num_output=4, stride=1, pad=0, trainable=True, name='conv4_2')
    model.add_layer(conv4_2)

    conv4_3 = SimQuantLayerConv2d(sim_quant_layer.output, kernel_size=1, num_output=10, stride=1, pad=0, trainable=True, name='conv4_3')
    model.add_layer(conv4_3)

    concat = SimQuantLayerConcat(input=([conv4_1.output[0], conv4_2.output[0], conv4_3.output[0]],
                                        [conv4_1.output[1], conv4_2.output[1], conv4_3.output[1]]),
                                 name='concat')
    model.add_layer(concat)
    logits = concat.output_tensor

    return model, logits

if __name__ == '__main__':
    tf.reset_default_graph()
    TRAIN_FLOAT_MODEL = False
    QUANT_GENERAL = True
    QUANT_TRAIN = True
    EVAL_FLOAT = True
    EVAL_GENERAL_QUANT = True
    EVAL_QUANT_TRAIN = True
    BATCH_SIZE = 384
    FLOAT_MODEL_TAIN_STEP = 220000

    if TRAIN_FLOAT_MODEL:
        with ops.Graph().as_default() as g:
            input_fn = get_input_fn(batchsize=BATCH_SIZE)
            features, labels = input_fn()
            model, logits = p_net(features)
            loss_fn = get_pnet_loss(batchsize=BATCH_SIZE)
            total_loss = loss_fn(labels, logits)
            train_op = tf.train.AdamOptimizer(0.001).minimize(total_loss)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                data_generator = get_data_generator(batchsize=BATCH_SIZE)
                for i in range(FLOAT_MODEL_TAIN_STEP):
                    data_batch = next(data_generator)
                    _, run_total_loss = sess.run([train_op, total_loss])

                    if i % 100 == 0 and i != 0:
                        print_log("loss {}, step {}/{}".format(run_total_loss, i, FLOAT_MODEL_TAIN_STEP))

                layer_list, param_dict = model.export_quant_param_dict(sess)

        fname = 'mtcnn/mtcnn_pnet'
        save_to_file(layer_list=layer_list, fname_prototxt=fname, param_dict=param_dict, fname_npz=fname)

    if QUANT_GENERAL:
        fname = 'mtcnn/mtcnn_pnet'
        layer_list, param_dict = ndk.modelpack.load_from_file(fname_prototxt=fname, fname_npz=fname)
        data_generator = get_data_generator(batchsize=BATCH_SIZE)
        quant_layer_list, quant_param_dict = ndk.quantize.quantize_model(layer_list=layer_list,
                                                                         param_dict=param_dict,
                                                                         bitwidth=8,
                                                                         data_generator=data_generator)
        quant_fname = 'mtcnn/mtcnn_pnet_quant'
        save_to_file(layer_list=quant_layer_list, fname_prototxt=quant_fname, param_dict=quant_param_dict, fname_npz=quant_fname)

    if QUANT_TRAIN:
        quant_fname = 'mtcnn/mtcnn_pnet_quant'
        quant_layer_list, quant_param_dict = ndk.modelpack.load_from_file(fname_prototxt=quant_fname, fname_npz=quant_fname)
        train_input_fn = get_input_fn(batchsize=BATCH_SIZE)
        loss_fn = get_pnet_loss(batchsize=BATCH_SIZE)
        quant_layer_list, quant_param_dict = ndk.quantize.quantize_model_with_training(layer_list=quant_layer_list,
                                                                                       bitwidth=8,
                                                                                       param_dict=quant_param_dict,
                                                                                       input_fn=train_input_fn,
                                                                                       loss_fn=loss_fn,
                                                                                       optimizer=tf.train.AdamOptimizer(0.00001),
                                                                                       num_step_train=2000,
                                                                                       num_step_log=100,
                                                                                       layer_group=2)
        quant_fname = 'mtcnn/mtcnn_pnet_quant_train'
        save_to_file(layer_list=quant_layer_list, fname_prototxt=quant_fname, param_dict=quant_param_dict,
                     fname_npz=quant_fname)
    if EVAL_FLOAT:
        fname = 'mtcnn/mtcnn_pnet'
        layer_list, param_dict = ndk.modelpack.load_from_file(fname_prototxt=fname, fname_npz=fname)
        data_generator = get_data_generator(batchsize=1000)
        nb_pos_neg_sample = 0
        nb_correct_pred_sample = 0
        for i in range(50):
            data_batch = next(data_generator)
            logits = run_layers(input_data_batch=data_batch['input'],
                                layer_list=layer_list,
                                target_feature_tensor_list=['concat_out:0'],
                                param_dict=param_dict,
                                quant=False)
            logits = logits['concat_out:0']
            nb_valid_sample, nb_correct_sample = count_correct_samples(labels=data_batch['output'], logits=logits)
            nb_pos_neg_sample = nb_pos_neg_sample + nb_valid_sample
            nb_correct_pred_sample = nb_correct_pred_sample + nb_correct_sample
            print("batch {}, accuracy {}.".format(i, (nb_correct_sample/nb_valid_sample)*100))
        acc = nb_correct_pred_sample/nb_pos_neg_sample
        print("Before quantization, accuracy {}%".format(acc*100))

    if EVAL_GENERAL_QUANT:
        fname = 'mtcnn/mtcnn_pnet_quant'
        quant_layer_list, quant_param_dict = ndk.modelpack.load_from_file(fname_prototxt=fname, fname_npz=fname)
        data_generator = get_data_generator(batchsize=1000)
        nb_pos_neg_sample = 0
        nb_correct_pred_sample = 0
        for i in range(50):
            data_batch = next(data_generator)
            logits = run_layers(input_data_batch=data_batch['input'],
                                layer_list=quant_layer_list,
                                target_feature_tensor_list=['concat_out:0'],
                                param_dict=quant_param_dict,
                                quant=True)
            logits = logits['concat_out:0']
            nb_valid_sample, nb_correct_sample = count_correct_samples(labels=data_batch['output'], logits=logits)
            nb_pos_neg_sample = nb_pos_neg_sample + nb_valid_sample
            nb_correct_pred_sample = nb_correct_pred_sample + nb_correct_sample
            print("batch {}, accuracy {}.".format(i, (nb_correct_sample/nb_valid_sample)*100))
        acc = nb_correct_pred_sample/nb_pos_neg_sample
        print("After general quantization, accuracy {}%".format(acc*100))

    if EVAL_QUANT_TRAIN:
        fname = 'mtcnn/mtcnn_pnet_quant_train'
        quant_layer_list, quant_param_dict = ndk.modelpack.load_from_file(fname_prototxt=fname, fname_npz=fname)
        data_generator = get_data_generator(batchsize=1000)
        nb_pos_neg_sample = 0
        nb_correct_pred_sample = 0
        for i in range(50):
            data_batch = next(data_generator)
            logits = run_layers(input_data_batch=data_batch['input'],
                                layer_list=quant_layer_list,
                                target_feature_tensor_list=['concat_out:0'],
                                param_dict=quant_param_dict,
                                quant=True)
            logits = logits['concat_out:0']
            nb_valid_sample, nb_correct_sample = count_correct_samples(labels=data_batch['output'], logits=logits)
            nb_pos_neg_sample = nb_pos_neg_sample + nb_valid_sample
            nb_correct_pred_sample = nb_correct_pred_sample + nb_correct_sample
            print("batch {}, accuracy {}.".format(i, (nb_correct_sample/nb_valid_sample)*100))
        acc = nb_correct_pred_sample/nb_pos_neg_sample
        print("After quantization training, accuracy {}%".format(acc*100))



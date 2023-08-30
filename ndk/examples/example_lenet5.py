# !/usr/bin/env python3
# -*- coding:utf-8 -*- -
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
import numpy as np

try:
    import tensorflow as tf
    TF_FOUND = True
except ModuleNotFoundError:
    TF_FOUND = False

# add ndk directory to system path, and import ndk.
import sys
ndk_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ndk_dir)
import ndk
import ndk.examples.data_generator_mnist as mnist

if TF_FOUND:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    def LeNet5(input):
        with tf.variable_scope('conv1'):
            conv1_weights = tf.get_variable(
                'weight', [5, 5, 1, 6],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            conv1_biases = tf.get_variable('bias', [6], initializer=tf.constant_initializer(0.1))
            conv1 = tf.nn.conv2d(input, conv1_weights, strides=[1, 1, 1, 1], padding='VALID', name='conv')
            conv1_out= tf.nn.bias_add(conv1, conv1_biases, name='conv_out')
            relu1 = tf.nn.relu(conv1_out, name='relu')
            pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='max_pool')

        with tf.variable_scope('conv2'):
            conv2_weights = tf.get_variable(
                'weight', [5, 5, 6, 16],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            conv2_biases = tf.get_variable('bias', [16], initializer=tf.constant_initializer(0.1))
            conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='VALID', name='conv')
            conv2_out = tf.nn.bias_add(conv2, conv2_biases, name='conv_out')
            relu2 = tf.nn.relu(conv2_out, name='relu')

            pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='max_pool')

            pool_shape = pool2.get_shape().as_list()
            nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
            reshaped = tf.reshape(pool2, [-1, nodes])

        with tf.variable_scope('fc1'):
            fc1_weights = tf.get_variable('weight', [nodes, 120],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            fc1_biases = tf.get_variable('bias', [120], initializer=tf.constant_initializer(0.1))
            fc1 = tf.matmul(reshaped, fc1_weights) + fc1_biases
            fc1_out = tf.identity(fc1, name='fc_out')
            fc1_relu = tf.nn.relu(fc1_out, name='relu')

        with tf.variable_scope('fc2'):
            fc2_weights = tf.get_variable('weight', [120, 84],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            fc2_biases = tf.get_variable('bias', [84], initializer=tf.constant_initializer(0.1))
            fc2 = tf.matmul(fc1_relu, fc2_weights) + fc2_biases
            fc2_out = tf.identity(fc2, name='fc_out')
            fc2_relu = tf.nn.relu(fc2_out, name='relu')

        with tf.variable_scope('fc3'):
            fc3_weights = tf.get_variable('weight', [84, 10],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            fc3_biases = tf.get_variable('bias', [10], initializer=tf.constant_initializer(0.1))
            fc3 = tf.matmul(fc2_relu, fc3_weights) + fc3_biases
            fc3_out = tf.identity(fc3, name='fc3_out')

        return fc3_out

    def train_and_freeze(fname_pb, train_steps=5000, batch_size=32, lr=1e-4, rnd_seed=None):
        tf.reset_default_graph()
        if type(rnd_seed)!=type(None):
            np.random.seed(rnd_seed)
            tf.set_random_seed(rnd_seed)
        g = mnist.data_generator_mnist(mnist_dirname='mnist',
                                 batch_size=batch_size, random_order=True, use_test_set=False)
        g_test = mnist.data_generator_mnist(mnist_dirname='mnist',
                                 batch_size=10000, random_order=False, use_test_set=True)

        net_input = tf.placeholder(tf.float32, shape=[None, 32, 32, 1], name='net_input') # NHWC format
        net_output = tf.identity(LeNet5(net_input), name="net_output")
        true_output = tf.placeholder(tf.float32, shape=[None, 10])

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=net_output, labels=true_output))
        train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
        correct_prediction = tf.equal(tf.argmax(net_output, 1), tf.argmax(true_output, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        test = next(g_test)
        test_in = test['input']
        test_out = test['output']
        test_in = test_in.transpose([0, 2, 3, 1])  # NCHW to NHWC
        test_out = test_out.transpose([0, 2, 3, 1]).reshape((-1, 10))

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(train_steps):
                batch = next(g)
                batch_in = batch['input']
                batch_out = batch['output']
                batch_in = batch_in.transpose([0,2,3,1]) # NCHW to NHWC
                batch_out = batch_out.transpose([0,2,3,1]).reshape((-1, 10))
                if i % 1000 == 0:
                    train_accuracy = accuracy.eval(feed_dict={net_input: batch_in, true_output: batch_out})
                    train_loss = loss.eval(feed_dict={net_input: batch_in, true_output: batch_out})
                    print("step %d\n  training accuracy %g\n  training loss %g" % (i, train_accuracy, train_loss))
                    test_accuracy = accuracy.eval(feed_dict={net_input: test_in, true_output: test_out})
                    test_loss = loss.eval(feed_dict={net_input: test_in, true_output: test_out})
                    print("  test accuracy %g\n  test loss %g" % (test_accuracy, test_loss))

                train_step.run(
                    feed_dict={net_input: batch_in, true_output: batch_out})

            # freeze the graph and save to pb file
            constant_graph = tf.graph_util.convert_variables_to_constants(sess=sess, input_graph_def=sess.graph.as_graph_def(), output_node_names=['net_output'])
            with tf.gfile.GFile(fname_pb, "wb") as f:
                f.write(constant_graph.SerializeToString())

    def load_and_test(fname_pb):
        tf.reset_default_graph()
        graph_def = tf.GraphDef()
        with open(fname_pb, "rb") as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

        print('model is recovered from {}'.format(fname_pb))

        g_test = mnist.data_generator_mnist(mnist_dirname='mnist',
                                                batch_size=10000, random_order=False, use_test_set=True)
        test = next(g_test)
        test_in = test['input']
        test_out = test['output']
        test_in = test_in.transpose([0, 2, 3, 1])  # NCHW to NHWC
        test_out = test_out.transpose([0, 2, 3, 1]).reshape((-1, 10))

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            net_input_tensor = sess.graph.get_tensor_by_name("net_input:0")
            net_output_tensor = sess.graph.get_tensor_by_name("net_output:0")
            net_output = sess.run(net_output_tensor, feed_dict={net_input_tensor: test_in})

        correct_prediction = np.equal(np.argmax(net_output, 1), np.argmax(test_out, 1))
        accuracy = np.mean(correct_prediction)
        print('Prediction accuracy: {:g}'.format(accuracy))

if __name__ == '__main__':

    TRAIN_TF_NET = False # training a new model using tensorflow
    EVAL_TF_NET = True # evaluating the tensorflow model
    SHOW_LAYER_INFO = True # show the information of loaded model (from a .pb file)
    RUN_NUMPY_LAYER_FLOAT = True # evaluating the model from the loaded model, using numpy layers
    DO_QUANT_GENERAL = True # do general quantization
    DO_QUANT_TRAINING = True # do quantized training
    RUN_NUMPY_LAYER_QUANT = True # evaluating the quantized model, using numpy layers.
    TENSOR_ANALYSIS = True # analyze the float/quant model
    EXPORT_TO_BIN = True # export to bin for SDK

    model_name = 'lenet5_model'

    # train a float-point model, and export to pb file
    if TRAIN_TF_NET:
        print('Training a Tensorflow model and export to pb file...')
        assert TF_FOUND, 'need tensorflow'
        if TF_FOUND:
            train_and_freeze(fname_pb=model_name+'.pb', train_steps=200000, batch_size=32, lr=1e-4, rnd_seed=190605)

    # evaluation the performance on test set
    if EVAL_TF_NET:
        print('Evaluating the Tensorflow model...')
        assert TF_FOUND, 'need tensorflow'
        if TF_FOUND:
            load_and_test(fname_pb=model_name+'.pb')


    if TF_FOUND:
        # load the model from pb file
        layer_list, param_dict = ndk.tensorflow_interface.load_from_pb(model_name+'.pb')
        ndk.modelpack.save_to_file(layer_list, model_name, param_dict, model_name)
    else:
        # load the model from previously saved ndk model (prototxt and npz file pair)
        layer_list, param_dict = ndk.modelpack.load_from_file(fname_prototxt=model_name,
                                                                fname_npz=model_name)


    # check the validity of the parameters
    ndk.layers.check_layers(layer_list, param_dict)

    # merge layers if possible
    layer_list, param_dict = ndk.optimize.merge_layers(layer_list, param_dict)

    # get and display network information
    if SHOW_LAYER_INFO:
        # network output tensor name
        output_tensor = ndk.layers.get_network_output(layer_list)
        print('\nOutput Tensor name:')
        print(output_tensor)

        # show all the layer names
        print('\nLayer name list:')
        print(ndk.layers.get_layer_name_list(layer_list))

        # show all the tensor names
        print('\nTensor name list:')
        print(ndk.layers.get_tensor_name_list(layer_list, feature_only=False))

        # show the layer properties
        print('\nlayers:')
        for layer in layer_list:
           print(layer)

        # export layer_list to prototxt file
        ndk.modelpack.save_to_file(layer_list=layer_list, fname_prototxt='loaded_model.prototxt')

    # run float-point performance using numpy layers, before quantization
    if RUN_NUMPY_LAYER_FLOAT:
        print('Running the loaded float-point model...')
        g_test = mnist.data_generator_mnist(mnist_dirname='mnist',
                                            batch_size=10000,
                                            random_order=False,
                                            use_test_set=True)
        data_batch = next(g_test)

        numpy_layer = ndk.quant_tools.numpy_net.build_numpy_layers(layer_list=layer_list,
                                                                   param_dict=param_dict,
                                                                   quant=False)
        test_output = ndk.quant_tools.numpy_net.run_layers(input_data_batch=data_batch['input'],
                                                           layer_list=layer_list,
                                                           target_feature_tensor_list=['net_output:0'],
                                                           param_dict=None,
                                                           bitwidth=8,
                                                           quant=False,
                                                           hw_aligned=False,
                                                           numpy_layers=numpy_layer,
                                                           log_on=True)
        correct_prediction = np.equal(np.argmax(test_output['net_output:0'], 1), np.argmax(data_batch['output'], 1))
        accuracy = np.mean(correct_prediction)
        print('Before quantization, model accuracy: {:.3f}%'.format(accuracy*100))

    # do general quantization
    if DO_QUANT_GENERAL:
        print('General quantization process...')
        np.random.seed(528)
        g_train = mnist.data_generator_mnist(mnist_dirname='mnist',
                                             batch_size=32,
                                             random_order=True,
                                             use_test_set=False)
        method_dict = {}
        # for key in ndk.layers.get_tensor_name_list(layer_list, feature_only=False):
        #     method_dict[key] = 'MSE'
        quant_layer_list, quant_param_dict = ndk.quantize.quantize_model(layer_list=layer_list,
                                                                         param_dict=param_dict,
                                                                         bitwidth=8,
                                                                         method_dict=method_dict,
                                                                         data_generator=g_train,
                                                                         aggressive=True,
                                                                         num_step_pre=20,
                                                                         num_step=100,
                                                                         gaussian_approx=False
                                                                         )
        fname_model_quant = model_name+'_quant'
        ndk.modelpack.save_to_file(layer_list=quant_layer_list,
                                   fname_prototxt=fname_model_quant,
                                   param_dict=quant_param_dict,
                                   fname_npz=fname_model_quant)
    
    # do model quantization with training
    if DO_QUANT_TRAINING:
        print('Training quantized model...')
        np.random.seed(528)
        g_train = mnist.data_generator_mnist(mnist_dirname='mnist',
                                             batch_size=32,
                                             random_order=True,
                                             use_test_set=False)
        quant_layer_list, quant_param_dict = ndk.quantize.quantize_model_with_training(layer_list=layer_list,
                                                                                       bitwidth=8,
                                                                                       data_generator=g_train,
                                                                                       param_dict=param_dict,
                                                                                       log_dir='log',
                                                                                       loss_fn=tf.losses.softmax_cross_entropy,
                                                                                       optimizer=tf.train.AdamOptimizer(1e-5),
                                                                                       num_step_train=1000,
                                                                                       num_step_log=100,
                                                                                       rnd_seed=961,
                                                                                       layer_group=10)
        fname_model_quant = model_name+'_quant_train'
        ndk.modelpack.save_to_file(layer_list=layer_list,
                                   fname_prototxt=fname_model_quant,
                                   param_dict=quant_param_dict,
                                   fname_npz=fname_model_quant)

    if DO_QUANT_TRAINING:
        fname_model_quant = model_name+'_quant_train'
    else:
        fname_model_quant = model_name + '_quant'

    # run fix-point performance using numpy layers
    if RUN_NUMPY_LAYER_QUANT:
        print('Running quantized model...')
        quant_layer_list, quant_param_dict = ndk.modelpack.load_from_file(fname_prototxt=fname_model_quant,
                                                                          fname_npz=fname_model_quant)

        g_test = mnist.data_generator_mnist(mnist_dirname='mnist',
                                            batch_size=10000,
                                            random_order=False,
                                            use_test_set=True)
        data_batch = next(g_test)
        net_output_tensor_name = ndk.layers.get_network_output(quant_layer_list)[0]
        test_output = ndk.quant_tools.numpy_net.run_layers(input_data_batch=data_batch['input'],
                                                           layer_list=quant_layer_list,
                                                           target_feature_tensor_list=[net_output_tensor_name],
                                                           param_dict=quant_param_dict,
                                                           bitwidth=8,
                                                           quant=True,
                                                           hw_aligned=True,
                                                           numpy_layers=None,
                                                           log_on=True)
        correct_prediction = np.equal(np.argmax(test_output[net_output_tensor_name], 1), np.argmax(data_batch['output'], 1))
        accuracy = np.mean(correct_prediction)
        print('After quantization, model accuracy={:.3f}%'.format(accuracy*100))


    # analyze the model
    if TENSOR_ANALYSIS:
        print('Analyzing model...')
        quant_layer_list, quant_param_dict = ndk.modelpack.load_from_file(fname_prototxt=fname_model_quant,
                                                                          fname_npz=fname_model_quant)

        g_test = mnist.data_generator_mnist(mnist_dirname='mnist',
                                            batch_size=100,
                                            random_order=False,
                                            use_test_set=True)

        res_dirname = 'analysis_result'

        float_stats =  ndk.quantize.analyze_tensor_distribution(layer_list=quant_layer_list,
                                                                param_dict=quant_param_dict,
                                                                target_tensor_list = ['net_input:0',
                                                                                      'conv1/conv_weight',
                                                                                      'conv1/conv_bias',
                                                                                      'conv1/conv_out:0',
                                                                                      'conv1/max_pool:0',
                                                                                      'net_output:0'],
                                                                output_dirname=os.path.join(res_dirname, 'float'),
                                                                data_generator=g_test,
                                                                quant=False,
                                                                bitwidth=8,
                                                                num_batch=100,
                                                                hw_aligned=True,
                                                                num_bin=51,
                                                                max_abs_val=None,
                                                                log_on=False)

        quant_stats =  ndk.quantize.analyze_tensor_distribution(layer_list=quant_layer_list,
                                                                param_dict=quant_param_dict,
                                                                target_tensor_list = ['net_input:0',
                                                                                      'conv1/conv_weight',
                                                                                      'conv1/conv_bias',
                                                                                      'conv1/conv_out:0',
                                                                                      'conv1/max_pool:0',
                                                                                      'net_output:0'],
                                                                output_dirname=os.path.join(res_dirname, 'quant'),
                                                                data_generator=g_test,
                                                                quant=True,
                                                                bitwidth=8,
                                                                num_batch=100,
                                                                hw_aligned=True,
                                                                num_bin=51,
                                                                max_abs_val=None,
                                                                log_on=False)

        compare_dict = ndk.quantize.compare_float_and_quant(layer_list=quant_layer_list,
                                                            param_dict=quant_param_dict,
                                                            target_tensor_list = ['net_input:0',
                                                                                  'conv1/conv_weight',
                                                                                  'conv1/conv_bias',
                                                                                  'conv1/conv_out:0',
                                                                                  'conv1/max_pool:0',
                                                                                  'net_output:0'],
                                                            bitwidth=8,
                                                            output_dirname=res_dirname,
                                                            data_generator=g_test,
                                                            num_batch=100,
                                                            hw_aligned=True,
                                                            num_bin=51,
                                                            max_abs_val=None,
                                                            log_on=False)
    if EXPORT_TO_BIN:
        print('Export to binary image...')
        ndk.modelpack.modelpack_from_file(bitwidth=8,
                                          fname_prototxt=fname_model_quant,
                                          fname_npz=fname_model_quant,
                                          out_file_path='out',
                                          model_name=model_name)

    print('Done.')


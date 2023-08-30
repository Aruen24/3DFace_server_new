from datetime import datetime
import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
import time

import h5py
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import pickle
# from queue import Queue
import random,threading,time
import utli
import train_function
import dataPath
from multiprocessing import Process,Queue

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def main(args):

    a = np.zeros((1,2,3))
    b = a.shape
    c = b[0]

    true_data_dirs = dataPath.trueValidFilePaths
    false_data_dirs = dataPath.falseValidFilePaths

    N = len(true_data_dirs)
    true_paths_raw = []
    for i in range(N):
        true_paths_raw += utli.get_dataset_common(true_data_dirs[i])
    num_true = len(true_paths_raw)
    N = len(false_data_dirs)
    false_paths_raw = []
    for i in range(N):
        false_paths_raw += utli.get_dataset_common(false_data_dirs[i])
    num_false = len(false_paths_raw)
    all_paths = true_paths_raw+false_paths_raw
    print("true num:%d  false num:%d"%(num_true,num_false))


    args.batch_size = num_true + num_false
    img_placeholder = tf.placeholder(tf.float32, [args.batch_size, args.image_h_size, args.image_w_size, 1])
    phase_train_placeholder = tf.placeholder(tf.bool)
    label_placeholder = tf.placeholder(tf.int32, [None], name='label')
    lr = tf.placeholder(tf.float32, name="learningrate")


    network = importlib.import_module(args.model_def)
    logits, Predictions = network.inference(img_placeholder,  phase_train=phase_train_placeholder, )
    # total_loss, t, cost = train_function.loss(Predictions, label_placeholder)
    train_acc = train_function.get_accuracy(Predictions, label_placeholder)

    vars_to_restore = utli.get_vars_to_restore(args.model_ckpt)
    saver = tf.train.Saver(vars_to_restore)
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())
        saver.restore(sess, args.model_ckpt)

        label = np.zeros((num_false+num_true))
        label[0:num_true] = 1
        data = np.zeros((num_false+num_true,args.image_h_size,args.image_w_size,1))
        print("start read true data..")
        data[0:num_true, :, :, :] = utli.get_batch_data_process(true_paths_raw,num_process=25)
        print("start read false data..")
        data[num_true:, :, :, :] = utli.get_batch_data_process(false_paths_raw,num_process=25)

        print("start cal data..")
        feed_dict = {img_placeholder: data,
                     label_placeholder: label,
                     phase_train_placeholder: False}

        trainAcc,predection= sess.run([train_acc,Predictions], feed_dict=feed_dict)
        # print(trainAcc)
        # print('update time:%.4f' % (time.time() - end_time))

        print("trainAcc:%.4f   "%(trainAcc))
        err_index = utli.get_predection_err_index(label,predection)
        for i in err_index:
            print(all_paths[i])







def parse_arguments(argv):
    parser = argparse.ArgumentParser()


    #parser.add_argument('--model_ckpt', type=str,
    #                    default='/home/data02_disk/tao/3DFace_server/log/faceFlat3/20191107_024039_96/-15484')
    #parser.add_argument('--model_ckpt', type=str,
    #                    default='./log/faceFlat3/20210917_054659_0.6/-57618')
    parser.add_argument('--model_ckpt', type=str,
                        default='./log/faceFlat3/20210919_160916/-57216')   
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='models.faceFlat3')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=200)
    parser.add_argument('--image_w_size', type=int,
                        help='Image size (height, width) in pixels.', default=96)
    parser.add_argument('--image_h_size', type=int,
                        help='Image size (height, width) in pixels.', default=112)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
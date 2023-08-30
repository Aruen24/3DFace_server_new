from __future__ import absolute_import, division, print_function
import random
import os

import tensorflow as tf
# tf.enable_eager_execution()
import numpy as np
# import cv2

# def random_flip_images(image_batch, label_batch, landmark_batch):
#     # mirror
#     if random.choice([0, 1]) > 0:
#         num_images = image_batch.shape[0]
#         fliplandmarkindexes = np.where(label_batch == -2)[0]
#         flipposindexes = np.where(label_batch == 1)[0]
#         # only flip
#         flipindexes = np.concatenate((fliplandmarkindexes, flipposindexes))
#         # random flip
#         for i in flipindexes:
#             cv2.flip(image_batch[i], 1, image_batch[i])
#
#             # pay attention: flip landmark
#         for i in fliplandmarkindexes:
#             landmark_ = landmark_batch[i].reshape((-1, 2))
#             landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark_])
#             landmark_[[0, 1]] = landmark_[[1, 0]]  # left eye<->right eye
#             landmark_[[3, 4]] = landmark_[[4, 3]]  # left mouth<->right mouth
#             landmark_batch[i] = landmark_.ravel()
#
#     return image_batch, landmark_batch
#
#
# def image_color_distort(inputs):
#     inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)
#     inputs = tf.image.random_brightness(inputs, max_delta=0.2)
#     inputs = tf.image.random_hue(inputs, max_delta=0.2)
#     inputs = tf.image.random_saturation(inputs, lower=0.5, upper=1.5)
#
#     return inputs

def _parse_image(image_string):
    image = tf.decode_raw(image_string, tf.uint8)
    image = tf.reshape(image, [12, 12, 3])
    image = (tf.cast(image, tf.float32) - 127.5) / 128
    return image

def _parse_function(example_proto):
    features = {
        'image/encoded': tf.FixedLenFeature([], tf.string),  # one image  one record
        'image/label': tf.FixedLenFeature([], tf.int64),
        'image/roi': tf.FixedLenFeature([4], tf.float32),
        'image/landmark': tf.FixedLenFeature([10], tf.float32)
    }

    parsed_features = tf.parse_single_example(example_proto, features)
    image = _parse_image(parsed_features['image/encoded'])
    # image = image_color_distort(image)
    label = tf.cast(parsed_features['image/label'], tf.float32)
    roi = tf.cast(parsed_features['image/roi'],tf.float32)
    landmark = tf.cast(parsed_features['image/landmark'],tf.float32)
    # image, landmark = random_flip_images(image, label, landmark)
    image = tf.transpose(image, perm=(2, 0, 1))

    features = image
    labels = tf.concat([tf.expand_dims(label, axis=0), roi, landmark], axis=-1)

    return features, labels

def get_input_fn(batchsize):
    def train_input_fn():
        label_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train_PNet_landmark.txt')
        f = open(label_file, 'r')
        num = len(f.readlines())
        print("Total size of the dataset is: ", num)
        dataset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train_PNet_landmark_small.tfrecord_shuffle')
        dataset = tf.data.TFRecordDataset(dataset_dir)
        dataset = dataset.map(_parse_function, num_parallel_calls=6)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(20000)
        dataset = dataset.batch(batch_size=batchsize)
        dataset = dataset.prefetch(4000)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return (features, labels)

    return train_input_fn

def get_data_generator(batchsize):
    sess = tf.Session()
    train_input_fn = get_input_fn(batchsize)
    def data_generator():
        features, labels = train_input_fn()
        data_batch = {}
        while True:
            run_features, run_labels = sess.run([features, labels])
            data_batch['input'] = run_features
            data_batch['output'] = run_labels
            yield data_batch
    return data_generator()

if __name__ == '__main__':
    input_fn = get_input_fn(batchsize=2)
    features, labels = input_fn()
    with tf.Session() as sess:
        for i in range(10):
            run_features, run_labels = sess.run([features, labels])
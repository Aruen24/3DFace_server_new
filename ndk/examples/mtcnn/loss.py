import tensorflow as tf
import numpy as np

num_keep_radio = 0.7

def cls_ohem(cls_prob, label, batch_size):
    zeros = tf.zeros_like(label)
    # label=-1 --> label=0net_factory

    # pos -> 1, neg -> 0, others -> 0
    label_filter_invalid = tf.where(tf.less(label, 0), zeros, label)
    num_cls_prob = tf.size(cls_prob)
    cls_prob_reshape = tf.reshape(cls_prob, [num_cls_prob, -1])
    label_int = tf.cast(label_filter_invalid, tf.int32)
    # get the number of rows of class_prob
    # num_row = tf.to_int32(cls_prob.get_shape()[0])
    # row = [0,2,4.....]
    row = tf.reshape(tf.range(batch_size) * 2, shape=[-1, 1])
    indices_ = row + label_int
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
    loss = -tf.log(label_prob + 1e-10)
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob, dtype=tf.float32)
    # set pos and neg to be 1, rest to be 0
    valid_inds = tf.where(label[:, 0] < zeros, zeros, ones)
    # get the number of POS and NEG examples
    num_valid = tf.reduce_sum(valid_inds)

    keep_num = tf.cast(num_valid * num_keep_radio, dtype=tf.int32)
    # FILTER OUT PART AND LANDMARK DATA
    loss = loss * valid_inds
    loss, _ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_mean(loss)


def bbox_ohem_smooth_L1_loss(bbox_pred, bbox_target, label):
    sigma = tf.constant(1.0)
    threshold = 1.0 / (sigma ** 2)
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    valid_inds = tf.where(label != zeros_index, tf.ones_like(label, dtype=tf.float32), zeros_index)
    abs_error = tf.abs(bbox_pred - bbox_target)
    loss_smaller = 0.5 * ((abs_error * sigma) ** 2)
    loss_larger = abs_error - 0.5 / (sigma ** 2)
    smooth_loss = tf.reduce_sum(tf.where(abs_error < threshold, loss_smaller, loss_larger), axis=1)
    keep_num = tf.cast(tf.reduce_sum(valid_inds) * num_keep_radio, dtype=tf.int32)
    smooth_loss = smooth_loss * valid_inds
    _, k_index = tf.nn.top_k(smooth_loss, k=keep_num)
    smooth_loss_picked = tf.gather(smooth_loss, k_index)
    return tf.reduce_mean(smooth_loss_picked)


def bbox_ohem_orginal(bbox_pred, bbox_target, label):
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    # pay attention :there is a bug!!!!
    valid_inds = tf.where(label != zeros_index, tf.ones_like(label, dtype=tf.float32), zeros_index)
    # (batch,)
    square_error = tf.reduce_sum(tf.square(bbox_pred - bbox_target), axis=1)
    # keep_num scalar
    keep_num = tf.cast(tf.reduce_sum(valid_inds) * num_keep_radio, dtype=tf.int32)
    # keep valid index square_error
    square_error = square_error * valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)


# label=1 or label=-1 then do regression
def bbox_ohem(bbox_pred, bbox_target, label):
    '''

    :param bbox_pred:
    :param bbox_target:
    :param label: class label
    :return: mean euclidean loss for all the pos and part examples
    '''
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    ones_index = tf.ones_like(label, dtype=tf.float32)
    # keep pos and part examples
    valid_inds = tf.where(tf.equal(tf.abs(label), 1), ones_index, zeros_index)
    # (batch,)
    # calculate square sum
    square_error = tf.square(bbox_pred - bbox_target)
    square_error = tf.reduce_sum(square_error, axis=1)
    # keep_num scalar
    num_valid = tf.reduce_sum(valid_inds)
    # keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    # count the number of pos and part examples
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    # keep valid index square_error
    square_error = square_error * valid_inds
    # keep top k examples, k equals to the number of positive examples
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)

    return tf.reduce_mean(square_error)


def landmark_ohem(landmark_pred, landmark_target, label):
    '''

    :param landmark_pred:
    :param landmark_target:
    :param label:
    :return: mean euclidean loss
    '''
    # keep label =-2  then do landmark detection
    ones = tf.ones_like(label, dtype=tf.float32)
    zeros = tf.zeros_like(label, dtype=tf.float32)
    valid_inds = tf.where(tf.equal(label, -2), ones, zeros)
    square_error = tf.square(landmark_pred - landmark_target)
    square_error = tf.reduce_sum(square_error, axis=1)
    num_valid = tf.reduce_sum(valid_inds)
    # keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    square_error = square_error * valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)

def get_pnet_loss(batchsize):
    def pnet_loss(labels, logits):
        cls_pred, bbox_pred, landmark_pred = logits[:, :2], logits[:, 2:6], logits[:, 6:]
        cls_pred = tf.nn.softmax(cls_pred, axis=1)
        cls_label, bbox_target, landmark_target = labels[:, :1], labels[:, 1:5], labels[:, 5:]
        cls_prob = tf.squeeze(cls_pred, [2, 3], name='cls_prob')

        cls_loss = cls_ohem(cls_prob, cls_label, batch_size=batchsize)

        bbox_pred = tf.squeeze(bbox_pred, [2, 3], name='bbox_pred')
        bbox_loss = bbox_ohem(bbox_pred, bbox_target, cls_label)

        landmark_pred = tf.squeeze(landmark_pred, [2, 3], name="landmark_pred")
        landmark_loss = landmark_ohem(landmark_pred, landmark_target, cls_label)

        lossL2 = tf.add_n(tf.losses.get_regularization_losses())
        total_loss = cls_loss + 0.5*bbox_loss + 0.5*landmark_loss + lossL2

        return total_loss

    return pnet_loss

def count_correct_samples(labels, logits):
    '''
    :param logits:
    :param label:
    '''
    # get the index of maximum value along axis one from cls_prob
    # 0 for negative 1 for positive
    cls_label, bbox_target, landmark_target = labels[:, :1], labels[:, 1:5], labels[:, 5:]
    cls_pred, bbox_pred, landmark_pred = logits[:, :2, 0, 0], logits[:, 2:6, 0, 0], logits[:, 6:, 0, 0]
    # cls_pred = tf.squeeze(cls_pred, [2, 3], name='cls_prob')
    pred = np.argmax(cls_pred, axis=1)
    label_int = cls_label.astype(np.int64)
    # return the index of pos and neg examples
    cond = np.where(label_int >= 0)[0]
    label_picked = pred[cond]
    pred_picked = label_int[cond]
    nb_valid_sample = len(label_picked)
    nb_correct_sample = len(label_picked[label_picked==pred_picked[:,0]])
    return nb_valid_sample, nb_correct_sample

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
def loss(logits,label_batches):
    with tf.name_scope('loss'):

        logits = slim.softmax(logits)
        t = logits[:,1]
        cost = focal_loss(logits[:,1],label_batches)
        regularization_loss =  ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)

        total_loss=tf.add_n([cost]+regularization_loss,name='total_loss')

        tf.summary.scalar('loss_cross', cost)
        tf.summary.scalar('loss_L2', tf.add_n(regularization_loss))
        tf.summary.scalar('loss_total', total_loss)
    return total_loss,t,cost



def focal_loss(prediction_tensor, target_tensor):
    gama = 2.0
    epsilon = 1e-8
    alpha = 0.3
    target_tensor = tf.cast(target_tensor, dtype=tf.float32)
    fl = -(alpha*target_tensor*tf.pow(1-prediction_tensor,gama)*tf.log(prediction_tensor+epsilon)+
           (1-alpha)*(1-target_tensor)*tf.pow(prediction_tensor,gama)*tf.log(1-prediction_tensor+epsilon))
    loss = tf.reduce_mean(fl)
    return loss


def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers

def get_accuracy_softmax(prediction,labels):
    with tf.name_scope('softmaxaccuracy'):
        correct_prediction = tf.equal(tf.argmax(prediction, 1),labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        _ = tf.summary.scalar('train_accuracy', accuracy)
    return accuracy
def get_accuracy(logits,labels):
    with tf.name_scope('accuracy'):
        acc1 = tf.nn.in_top_k(logits,labels,1)
        acc2 = tf.cast(acc1,tf.float32)
        acc = tf.reduce_mean(acc2)
        _ = tf.summary.scalar('train_accuracy', acc)
    return acc
def get_accuracy_test(logits,labels):
    with tf.name_scope('testaccuracy'):
        acc1 = tf.nn.in_top_k(logits,labels,1)
        acc2 = tf.cast(acc1,tf.float32)
        acc = tf.reduce_mean(acc2)
        _ = tf.summary.scalar('test_accuracy', acc)
    return acc
def get_accuracy_val(logits,labels):
    with tf.name_scope('valaccuracy'):
        acc1 = tf.nn.in_top_k(logits,labels,1)
        acc2 = tf.cast(acc1,tf.float32)
        acc = tf.reduce_mean(acc2)

    return acc

def training(loss,lr,global_step):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)

    return train_op

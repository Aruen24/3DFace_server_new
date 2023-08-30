#!/usr/bin/env python3
"""
Gradients for inner product.
"""
import os
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
fake_quant_with_d_grad_module = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                                              'fake_quant_with_d_per_out_channel_grad.so'))

@ops.RegisterGradient("FakeQuantWithD")
def _inner_product_grad_cc(op, grad):
    """
    The gradient for `inner_product` using the operation implemented in C++.
    
    :param op: `inner_product` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
    :param grad: gradient with respect to the output of the `inner_product` op.
    :return: gradients with respect to the input of `inner_product`.
    """
    
    return [fake_quant_with_d_grad_module.fake_quant_with_d_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2]),
            tf.zeros_like(op.inputs[1]), 
            tf.constant(0, dtype=tf.float32),
            tf.constant(0, dtype=tf.float32)]    
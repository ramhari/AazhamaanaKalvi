import numpy as np
import tensorflow as tf


# =========================================FULLY CONNECTED LAYERS=======================================================

# FC Forward Layers:

def affine_layer_variable_definitions(input_dim, output_dim, init_weights=None, init_bias=None):
# Advanced Xavier initialization for weights 
    if init_weights is None:
        init_weights = tf.truncated_normal([input_dim, output_dim], mean=0, stddev=np.sqrt(2.0/input_dim))
		
# TODO: Bias initialization (Check if small positive bias needed)
    if init_bias is None:
        init_bias = tf.zeros([output_dim])

    weights = tf.Variable(init_weights)
    bias = tf.Variable(init_bias)

    return weights, bias


def affine_relu_forward(weights, bias, x):
    return tf.nn.relu(tf.matmul(x, weights) + bias)


def affine_forward(weights, bias, x):
    return tf.matmul(x, weights) + bias


def deep_softmax_forward_pass(input_activation, input_layer_dim, hidden_layer_dims, output_dim):

    cached_data = {}
    params = {}

    for hidden_layer_id in range(len(hidden_layer_dims)):

        with tf.name_scope('scope' + str(hidden_layer_id)):
            [weights, bias] = \
                affine_layer_variable_definitions(input_layer_dim, hidden_layer_dims[hidden_layer_id])
            output_activation = affine_relu_forward(weights, bias, input_activation)
            cached_data[hidden_layer_id] = {'weights': weights, 'bias': bias, 'input_activation': input_activation}
            params[hidden_layer_id] = {'weights': weights, 'bias': bias}

        input_activation = output_activation
        input_layer_dim = hidden_layer_dims[hidden_layer_id]

    with tf.name_scope('scope_output'):
        [weights, bias] = affine_layer_variable_definitions(input_layer_dim, output_dim)
        logit = affine_forward(weights, bias, input_activation)
        cached_data[len(hidden_layer_dims)] = {'weights': weights, 'bias': bias, 'input_activation': input_activation}
        params[len(hidden_layer_dims)] = {'weights': weights, 'bias': bias}

    return logit, cached_data, params


# FC Backward Layers:

def affine_backward(dl_by_dy, cached_data):
    gradient_ops = {
        'weights': tf.matmul(tf.transpose(cached_data['input_activation']), dl_by_dy),
        'bias': tf.reduce_sum(dl_by_dy, 0),
        'dl_by_dx': tf.matmul(dl_by_dy, tf.transpose(cached_data['weights']))
    }

    return gradient_ops


def relu_backward(dl_by_dy, input_activation):
    gradient_ops = {
        'dl_by_dx': tf.mul(dl_by_dy, tf.to_float(input_activation > 0))
    }
    return gradient_ops


def affine_relu_backward(dl_by_dy, cached_data):
    input_activation_to_relu = \
        affine_forward(cached_data['weights'], cached_data['bias'], cached_data['input_activation'])
    relu_gradient = relu_backward(dl_by_dy, input_activation_to_relu)

    gradient_ops = affine_backward(relu_gradient['dl_by_dx'], cached_data)
    return gradient_ops


def deep_softmax_backward_pass(dl_by_dy, cached_data_dict):

    gradient_ops = {}

    for layer in sorted(cached_data_dict.keys(), reverse=True):
        cached_data = cached_data_dict[layer]

        if layer == len(cached_data_dict)-1:
            gradient_ops[layer] = affine_backward(dl_by_dy, cached_data)
        else:
            input_gradient = gradient_ops[layer+1]['dl_by_dx']
            gradient_ops[layer] = affine_relu_backward(input_gradient, cached_data)

    return gradient_ops


# =========================================CONVOLUTION LAYERS==========================================================
def init_conv_params(filter_size, num_input_channels, num_filters, strides, padding='SAME'):
    init_kernel_weights = tf.truncated_normal([filter_size, filter_size, num_input_channels, num_filters],
                                              stddev=1.0/filter_size)
    init_kernel_bias = tf.zeros([num_filters])
    params = {
        'filter_weights': tf.Variable(init_kernel_weights),
        'filter_bias': tf.Variable(init_kernel_bias),
        'strides': strides,
        'padding': padding
    }
    return params


def init_max_pool_params(pooling_kernal_size, strides, padding='VALID'):
    params = {
        'pooling_kernel_size': pooling_kernal_size,
        'strides': strides,
        'padding': padding
    }
    return params


def vectorize_input(x, batch_size):
    vectorize_op = tf.reshape(x, [batch_size, -1])
    return vectorize_op


def conv_layer_forward(kernel_params,  x):

    conv_op = \
        tf.nn.conv2d(x, kernel_params['filter_weights'],
                     strides=kernel_params['strides'], padding=kernel_params['padding'])
    add_bias_op = tf.nn.bias_add(conv_op, kernel_params['filter_bias'])

    return add_bias_op


def max_pool_forward(params, x):
    max_pool_op = tf.nn.max_pool(x, ksize=params['pooling_kernel_size'],
                                 strides=params['strides'], padding=params['padding'])
    return max_pool_op


def deep_conv_forward(input_images, hidden_layer_filter_params):
    pass


# =========================================UTIL/COMMON LAYERS==========================================================

def update_params_sgd(params, gradient_ops, learning_rate):

    update_ops = {}

    for layer in sorted(gradient_ops.keys(), reverse=True):
        update_ops[layer] = {}

        new_weights = params[layer]['weights'] - learning_rate*gradient_ops[layer]['weights']
        new_bias = params[layer]['bias'] - learning_rate*gradient_ops[layer]['bias']

        update_ops[layer]['weights'] = tf.assign(params[layer]['weights'], new_weights)
        update_ops[layer]['bias'] = tf.assign(params[layer]['bias'], new_bias)

    return update_ops


def compute_data_loss(logit, labels):
    loss = tf.constant(1)
    dl_by_dlogit = tf.ones_like(logit)
    return loss, dl_by_dlogit


def compute_regularization_loss(params, reg_rate):
    layer_reg_loss = [reg_rate * .5 * tf.reduce_sum(tf.square(params[layer]['weights'])) for layer in params]
    accumulate_loss = tf.add_n(layer_reg_loss)

    return accumulate_loss

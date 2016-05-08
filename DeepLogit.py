from NetworkTFLayers import *
from DataParser import *


def deep_logit(input_dim, hidden_layer_dims, output_dim, learning_rate, reg_rate, batch_size, max_num_iters):

    x_train = tf.placeholder(tf.float32)
    y_labels = tf.placeholder(tf.float32)
    x_train_flattened = vectorize_input(x_train, batch_size)

    # params_conv = init_conv_params(filter_size=3, num_input_channels=1, num_filters=1, strides=[1, 1, 1, 1])
    # conv_op = conv_layer_forward(params_conv, x_train)

    vectorised_conv_output = vectorize_input(x_train_flattened, batch_size)
    [logit, _, params] = deep_softmax_forward_pass(vectorised_conv_output, input_dim, hidden_layer_dims, output_dim)

    data_loss = tf.nn.softmax_cross_entropy_with_logits(logit, y_labels)
    regularization_loss = compute_regularization_loss(params, reg_rate)
    loss = tf.add(data_loss, regularization_loss)

    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=.9)

    train_op = optimizer.minimize(loss)

    db = DataParser()

    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        for _ in range(max_num_iters):

            # Get batch data:
            [batch_x, batch_labels] = db.get_next_batch(batch_size)

            # Single back propagation + update operation
            [_, curr_loss, _] = sess.run([train_op, loss, logit], {x_train: batch_x, y_labels: batch_labels})

    return curr_loss

deep_logit(input_dim=1024, hidden_layer_dims=[100, 50, 60, 40], output_dim=2,
           learning_rate=.01, reg_rate=.01, batch_size=100, max_num_iters=10)

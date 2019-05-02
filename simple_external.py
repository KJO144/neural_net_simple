import tensorflow as tf


def loss_and_grads_from_tf(X, Y, params, layer_dims, activation_functions):

    n_x, m = X.shape
    n_y = Y.shape[0]

    # placeholders for the input x and output y
    x_ph = tf.placeholder(tf.float64, shape=(n_x, m))
    y_ph = tf.placeholder(tf.float64, shape=(n_y, m))

    # tensor variables for the parameters
    tensor_params = {}
    for name, value in params.items():
        tensor_params[name] = tf.Variable(value)

    num_layers = len(layer_dims)

    # define dict of activation functions
    avs = {'relu': tf.nn.relu}

    a_prev = x_ph
    for i in range(1, num_layers+1):
        w = tensor_params['W'+str(i)]
        b = tensor_params['b'+str(i)]
        z = tf.matmul(w, a_prev) + b
        if i is num_layers:  # because tf.nn.sigmoid_cross_entropy_with_logits takes logits
            a = tf.identity(z)
        else:
            a = avs[activation_functions[i-1]](z)
        a_prev = a

    # cross-entropy loss
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.transpose(a), labels=tf.transpose(y_ph)))

    # track gradients too
    param_names = params.keys()
    tensors = [tensor_params[pn] for pn in param_names]
    grads_list = tf.gradients(loss, tensors)
    grads_dict = dict(zip(param_names, grads_list))

    return loss, grads_dict, x_ph, y_ph


def calculate_loss_and_grads_tf(x, y, x_ph, y_ph, loss_tensor, grads_tensors):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_value, grads = sess.run([loss_tensor, grads_tensors], feed_dict={x_ph: x, y_ph: y})
    return loss_value, grads

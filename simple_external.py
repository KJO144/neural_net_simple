def loss_and_grads_from_tf(X, Y, params, layer_dims, activation_functions):
    import tensorflow as tf

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

    a_prev = x_ph
    for i in range(1, num_layers+1):
        w = tensor_params['W'+str(i)]
        b = tensor_params['b'+str(i)]
        z = tf.matmul(w, a_prev) + b
        activation_function = activation_functions[i-1]
        if activation_function == 'relu':
            a = tf.nn.relu(z)
        elif activation_function == 'sigmoid':
            a = tf.nn.sigmoid(z)
        else:
            raise ValueError('unknown activation function')
        a_prev = a

    # cross-entropy loss
    assert m is 1, "only support m=1 for now"    
    loss = - tf.matmul(y_ph, tf.math.log(a)) - tf.matmul(1-y_ph, tf.math.log(1-a))

    # track gradients too
    param_names = params.keys()
    tensors = [tensor_params[pn] for pn in param_names]
    grads_list = tf.gradients(loss, tensors)
    grads_dict = dict(zip(param_names, grads_list))

    # run it
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_value, grads = sess.run([loss, grads_dict], feed_dict={x_ph: X, y_ph: Y})

    return loss_value, grads

import numpy as np


def relu(x):
    ret = np.maximum(0, x)
    return ret


def sigmoid(x):
    ret = 1 + np.exp(-x)
    return 1 / ret


def compute_cost(y_label, yhat):
    size = y_label.shape[1]
    ret = np.dot(y_label, np.log(yhat).T) + np.dot((1-y_label), np.log(1-yhat).T)
    ret = ret.squeeze()
    return -ret / size


def initialize_params(n_x, layer_dims):
    np.random.seed(0)
    layer_dims_temp = [n_x] + layer_dims
    ret = {}
    for i in range(1, len(layer_dims_temp)):
        n_h = layer_dims_temp[i]
        w = np.random.rand(n_h, layer_dims_temp[i-1])
        b = np.zeros((n_h, 1))
        ret['W'+str(i)] = w
        ret['b'+str(i)] = b
    return ret


def forward_prop(x, params, activation_functions):
    assert (len(params) % 2 == 0)
    num_layers = int(len(params) / 2)
    cache = {}
    a_prev = x
    for i in range(1, num_layers + 1):
        w = params['W'+str(i)]
        b = params['b'+str(i)]
        z = np.dot(w, a_prev) + b
        activation_function = activation_functions[i-1]
        if activation_function == 'relu':
            a = relu(z)
        elif activation_function == 'sigmoid':
            a = sigmoid(z)
        else:
            raise ValueError('unknown activation function')
        cache['Z'+str(i)] = z
        cache['A'+str(i)] = a
        a_prev = a
    return a, cache


def relu_deriv(v):
    return np.greater(v, 0)


def compute_grads(al, params, cache, x, y):
    # x is n_x x m
    # y is 1 x m
    m = x.shape[1]
    assert(m is y.shape[1])
    cache['A0'] = x
    assert (len(params) % 2 == 0)
    num_layers = int(len(params) / 2)

    grads = {}
    dz = al - y
    for i in reversed(range(1, num_layers+1)):
        al_next = cache['A' + str(i - 1)]
        dw = np.dot(dz, al_next.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        grads['W' + str(i)] = dw
        grads['b' + str(i)] = db
        if i > 1:
            w = params['W' + str(i)]
            z_next = cache['Z' + str(i - 1)]
            dz = np.dot(w.T, dz) * relu_deriv(z_next)

    return grads


def update_parameters(parameters, gradients, learning_rate):
    new_params = {}
    for param_name, param_value in parameters.items():
        grad = gradients[param_name]
        new_params[param_name] = param_value - learning_rate * grad
        # print( "updating {} from\n {} \nto\n {}".format(param_name, param_value, new_params[param_name] ) )
    return new_params


def learn_params(x_train, y_train, layer_dims, activation_functions, learning_rate, epochs):
    # initialize params
    n_x = x_train.shape[0]
    params = initialize_params(n_x, layer_dims)

    cost = None
    for i in range(epochs):
        # forward prop
        final_activation, cache = forward_prop(x_train, params, activation_functions)

        # compute cost
        cost = compute_cost(y_train, final_activation)
        # print("Cost after iteration {}: {}".format( i, cost) )

        # backward prop to calculate gradients
        grads = compute_grads(final_activation, params, cache, x_train, y_train)

        params = update_parameters(params, grads, learning_rate)
    return params, cost

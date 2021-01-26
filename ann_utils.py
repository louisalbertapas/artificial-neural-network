import numpy as np

# Helper function to calculate sigmoid activation
def sigmoid(z):
    activation = 1 / (1 + np.exp(-z))
    cache = z
    
    return activation, cache

# Helper function to calculate the derivative of sigmoid activation
def sigmoid_backward(dA, cache):
    Z = cache
    activation = 1 / (1 + np.exp(-Z))
    dZ = dA * activation * (1 - activation)
    
    return dZ

# Helper function to calculate ReLU activation
def relu(z):
    activation = np.maximum(0, z)
    cache = z
    
    return activation, cache

# Helper function to calculate the derivative of relu activation
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    
    dZ[Z <= 0] = 0
    
    return dZ

# Helper function to initialize weights and bias based on initializer_type
def initialize_parameters(layer_dims, initializer_type='random'):
    parameters = {}
    np.random.seed(1)
    num_layers = len(layer_dims)
    for l in range(1, num_layers):
        if initializer_type == 'random':
            parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        elif initializer_type == 'xavier':
            parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(1 / layer_dims[l-1])
        elif initializer_type == 'he':
            parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    
    return parameters

# Function to calculate the linear part (Z) of a single forward propagation
def linear_forward(Activations, Weights, bias):
    Z = np.dot(Weights, Activations) + bias
    linear_cache = (Activations, Weights, bias)
    
    return Z, linear_cache

# Function to calculate the activation part (A) of a single forward propagation
def activation_forward(Activations, Weights, bias, activation_fn='relu'):
    Z, linear_cache = linear_forward(Activations, Weights, bias)
    if activation_fn == 'relu':
        A, activation_cache = relu(Z)
    elif activation_fn == 'sigmoid':
        A, activation_cache = sigmoid(Z)
        
    cache = (linear_cache, activation_cache)
        
    return A, cache

# Function to calculate the forward propagation of the network
def forward_propagate(X, parameters):
    A = X
    num_layers = len(parameters) // 2
    caches = []
    
    for l in range(1, num_layers):
        A_prev = A
        A, cache = activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(cache)
    
    # Compute for the last layer using sigmoid
    A_output, cache = activation_forward(A, parameters['W' + str(num_layers)], parameters['b' + str(num_layers)], 'sigmoid')
    caches.append(cache)
    
    return A_output, caches

def compute_cost(A_output, Y):
    m = Y.shape[1]
    cost = (-1/m) * np.sum(Y * np.log(A_output) + (1 - Y) * (np.log(1- A_output)))
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost

# Function to calculate the linear part (Z) of a single backward propagation
def linear_backward(dZ, linear_cache):
    (A_prev, Weights, bias) = linear_cache
    m = A_prev.shape[1]
    
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(Weights.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == Weights.shape)
    assert (db.shape == bias.shape)
    
    return dA_prev, dW, db

# Function to calculate the activation part (A) of a single backward propagation
def activation_backward(dA, cache, activation_fn='relu'):
    linear_cache, activation_cache = cache
    
    if activation_fn == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation_fn == 'relu':
        dZ = relu_backward(dA, activation_cache)
    
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db
    

# Function to calculate the backward propagation of the network
def backward_propagate(A_output, Y, caches):
    gradients = {}
    num_layers = len(caches)
    m = A_output.shape[1]
    Y = Y.reshape(A_output.shape)
    
    dAoutput = - (np.divide(Y, A_output) - np.divide(1 - Y, 1 - A_output))
    current_cache = caches[num_layers - 1]
    gradients["dA" + str(num_layers - 1)], gradients["dW" + str(num_layers)], gradients["db" + str(num_layers)] = activation_backward(dAoutput, current_cache, 'sigmoid')
    
    for l in reversed(range(num_layers - 1)):
        current_cache = caches[l]
        dA_prev, dW_prev, db_prev = activation_backward(gradients["dA" + str(l + 1)], current_cache, 'relu')
        gradients["dA" + str(l)] = dA_prev
        gradients["dW" + str(l + 1)] = dW_prev
        gradients["db" + str(l + 1)] = db_prev
        
    return gradients

# Function to optimize parameters using gradient descent
def optimize_parameters(parameters, gradients, learning_rate):
    num_layers = len(parameters) // 2
    
    for l in range(num_layers):
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * gradients['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * gradients['db' + str(l + 1)]
        
    return parameters

def predict(X, y, parameters, threshold=0.5):
    m = X.shape[1]
    prob = np.zeros((1,m))
    probs, _ = forward_propagate(X, parameters)
    
    for i in range(probs.shape[1]):
        if probs[0,i] >= threshold:
            prob[0,i] = 1
        else:
            prob[0,i] = 0
    
    return prob

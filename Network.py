import random
import Constants
import numpy as np


def build_network(input, output, number_of_hidden_layers, number_of_neurons_per_hidden_layer, learning_rate, epochs,
                  is_biased,
                  activation_func):
    base_inputs = input
    # hidden_layer_weights = gen_random_weights(number_of_hidden_layers, len(input), number_of_neurons_per_hidden_layer)
    # output_layer_weights = gen_random_weights(Constants.OUTPUT_LAYERS_CNT, number_of_neurons_per_hidden_layer,
    #                                           Constants.OUTPUT_NEURONS_CNT)
    #
    # hidden_layer_bias = gen_bias(number_of_hidden_layers, number_of_neurons_per_hidden_layer)
    # output_layer_bias = gen_bias(Constants.OUTPUT_LAYERS_CNT, Constants.OUTPUT_NEURONS_CNT)

    hidden_layer_weights = [[[0.21, 0.15], [-0.4, 0.1]]]
    output_layer_weights = [[[-0.2, 0.3]]]

    hidden_layer_bias = [[-0.3, 0.25]]
    output_layer_bias = [[-0.4]]

    print(hidden_layer_weights)
    print(output_layer_weights)

    outputs = []

    for layer in range(number_of_hidden_layers):
        x = []
        for neuron in range(number_of_neurons_per_hidden_layer):
            if is_biased:
                x.append(apply_activation_func(activation_func, get_neuron_activation(len(base_inputs), base_inputs,
                                                                                      hidden_layer_weights[layer][
                                                                                          neuron]) +
                                               hidden_layer_bias[layer][neuron]))
            else:
                x.append(apply_activation_func(activation_func, get_neuron_activation(len(base_inputs), base_inputs,
                                                                                      hidden_layer_weights[layer][
                                                                                          neuron])))
        base_inputs = x
        outputs.append(base_inputs)

    x = []
    for neuron in range(Constants.OUTPUT_NEURONS_CNT):
        if is_biased:
            x.append(apply_activation_func(activation_func, get_neuron_activation(len(base_inputs), base_inputs,
                                                                                  output_layer_weights[0][neuron]) +
                                           output_layer_bias[0][neuron]))
        else:
            x.append(apply_activation_func(activation_func, get_neuron_activation(len(base_inputs), base_inputs,
                                                                                  output_layer_weights[0][neuron])))
        base_inputs = x
        outputs.append(base_inputs)

    print("forward propagation")
    print(outputs)

    back_propagate(outputs, output, activation_func, number_of_neurons_per_hidden_layer, hidden_layer_weights,
                   output_layer_weights, number_of_hidden_layers, hidden_layer_bias, output_layer_bias)


def gen_random_weights(layers_num, input_layer_size, layer_size):
    res = []
    for layer in range(layers_num):
        layer_weights = []
        if layer == 0:
            inputs = input_layer_size
        else:
            inputs = layer_size
        for neuron in range(layer_size):
            neuron_weights = []
            for weight in range(inputs):
                neuron_weights.append(random.random())
            layer_weights.append(neuron_weights)
        res.append(layer_weights)
    return res


def gen_bias(n, size):
    res = []
    for layer in range(n):
        layer_bias = []
        for neuron in range(size):
            layer_bias.append(random.random())
        res.append(layer_bias)
    return res


def get_neuron_activation(n, input, weight):
    res = 0
    for i in range(n):
        res += input[i] * weight[i]
    return res


def get_neuron_error(n, error, weight):
    res = 0
    for i in range(n):
        res += error[i] * weight[i]
    return res


def sigmoid(x):
    # assumed that a = 1
    a = 1
    return 1 / (1 + np.exp(-a * x))


def tanh(x):
    # assumed that a = 1
    a = 1
    return np.tanh(a * x)


def apply_activation_func(activation_func, activation):
    if activation_func == 'sigmoid':
        return sigmoid(activation)
    elif activation_func == 'tanh':
        return tanh(activation)


def apply_activation_derv(activation_func, value):
    if activation_func == 'sigmoid':
        return sigmoid_derivative(value)
    elif activation_func == 'tanh':
        return tanh_derivative(value)


def sigmoid_derivative(x):
    return x * (1 - x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def back_propagate(actual, target, activation_func, number_of_neurons_per_hidden_layer, hidden_weights, output_weights,
                   number_of_hidden_layers, hidden_layer_bias, output_layer_bias):
    outputs_errors = []
    output_layer_index = number_of_hidden_layers
    for output_neuron in range(Constants.OUTPUT_NEURONS_CNT):
        # output_neuron => 0 Bombay
        if output_neuron == target:
            derv_result = apply_activation_derv(activation_func, actual[output_layer_index][output_neuron])
            neuron_error = (1 - actual[output_layer_index][output_neuron]) * derv_result
        else:
            derv_result = apply_activation_derv(activation_func, actual[output_layer_index][output_neuron])
            neuron_error = (0 - actual[output_layer_index][output_neuron]) * derv_result
        outputs_errors.append(neuron_error)

    base_outputs = outputs_errors

    # print("base_outputs")
    # print(base_outputs)

    hidden_errors = []
    unreversed_hidden_weights = hidden_weights.copy()
    unreversed_output_weights = output_weights.copy()
    hidden_weights_reversed = reverse_hidden_to_hidden(hidden_weights, number_of_hidden_layers)
    output_weights_reversed = [reverse_output_to_hidden(output_weights, number_of_neurons_per_hidden_layer)]
    print("output_weights")
    print(output_weights)

    print("before reverse")
    print(unreversed_hidden_weights)
    print("after reverse")
    print(hidden_weights_reversed)
    for layer in range(number_of_hidden_layers):
        x = []
        for neuron in range(number_of_neurons_per_hidden_layer):
            if layer == 0:
                x.append(get_neuron_error(len(base_outputs), base_outputs,
                                          output_weights_reversed[layer][
                                              neuron]) * apply_activation_derv(activation_func, actual[layer][neuron]))
            else:
                x.append(get_neuron_error(len(base_outputs), base_outputs,
                                          hidden_weights_reversed[layer][
                                              neuron]) * apply_activation_derv(activation_func, actual[layer][neuron]))
        hidden_errors.append(x)
        base_outputs = x

        hidden_errors.append(outputs_errors)
        print("backward propagation")
        print(hidden_errors)

        update_weights(0.001, unreversed_hidden_weights, unreversed_output_weights, hidden_errors, [0, 0], actual,
                       number_of_hidden_layers,
                       number_of_neurons_per_hidden_layer, hidden_layer_bias, output_layer_bias, True)


def reverse_hidden_to_hidden(weights, number_of_hidden_layers):
    for layer in range(number_of_hidden_layers):
        weights[layer] = list(zip(*weights[layer]))
    return weights


def reverse_output_to_hidden(weights, number_of_neurons_per_hidden_layer):
    res = np.reshape(weights[0], (Constants.OUTPUT_NEURONS_CNT, number_of_neurons_per_hidden_layer)).T.tolist()
    return res


def update_weights(learning_rate, hidden_weights, output_weights, error_signals, input_layer, actual,
                   number_hidden_layers,
                   number_neurons_hidden_layer, hidden_layer_bias, output_layer_bias, is_biased):
    print("inside")
    print(hidden_weights)
    print(output_weights)
    print("bias")
    print(hidden_layer_bias)
    print(output_layer_bias)
    add_bias_to_weights(hidden_weights, hidden_layer_bias)
    add_bias_to_weights(output_weights, output_layer_bias)
    for layer in range(number_hidden_layers):
        new_layer_weights = []
        for neuron in range(number_neurons_hidden_layer):
            new_neuron_weights = []
            if layer == 0:
                for f in range(len(input_layer) + is_biased):
                    if is_biased and f == len(input_layer):
                        new_neuron_weights.append(
                            hidden_weights[layer][neuron][f] + learning_rate * error_signals[layer][neuron] * 1)
                    else:
                        new_neuron_weights.append(
                            hidden_weights[layer][neuron][f] + learning_rate * error_signals[layer][neuron] *
                            input_layer[
                                f])
            else:
                for x in range(actual[layer - 1] + is_biased):
                    if is_biased and x == actual[layer - 1]:
                        new_neuron_weights.append(
                            hidden_weights[layer][neuron] + learning_rate * error_signals[layer][neuron] *
                            1)
                    else:
                        new_neuron_weights.append(
                            hidden_weights[layer][neuron] + learning_rate * error_signals[layer][neuron] *
                            actual[layer - 1][x])
            new_layer_weights.append(new_neuron_weights)

        print(new_layer_weights)

    new_output_layer_weights = []
    for output_neuron in range(Constants.OUTPUT_NEURONS_CNT):
        new_output_weights = []
        for hidden_layer_neuron in range(number_neurons_hidden_layer + is_biased):
            if is_biased and hidden_layer_neuron == number_neurons_hidden_layer:
                new_output_weights.append(
                    output_weights[0][output_neuron][hidden_layer_neuron] + learning_rate *
                    error_signals[number_hidden_layers][output_neuron] * 1)
            else:
                new_output_weights.append(
                    output_weights[0][output_neuron][hidden_layer_neuron] + actual[number_hidden_layers - 1][
                        hidden_layer_neuron] * learning_rate * error_signals[number_hidden_layers][output_neuron])
        new_output_layer_weights.append(new_output_weights)

    print(new_output_layer_weights)


def add_bias_to_weights(list1, list2):
    for sublist1, element2 in zip(list1[0], list2[0]):
        sublist1.append(element2)
    return list1


# def train_neural_network(inputs, output, number_of_hidden_layers, number_of_neurons_per_hidden_layer, learning_rate, epochs,
#                   is_biased,
#                   activation_func):
#     # for

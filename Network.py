import random
import Constants
import numpy as np


def forward_propagation(inputs, hidden_layer_weights, output_layer_weights, hidden_layer_bias, output_layer_bias,
                        number_of_hidden_layers, number_of_neurons_per_hidden_layer, number_of_outputs,
                        is_biased,
                        activation_func, is_test=False):
    if is_test:
        print("sample input: ", inputs)
    base_inputs = inputs
    outputs = []
    for layer in range(number_of_hidden_layers):
        x = []
        for neuron in range(number_of_neurons_per_hidden_layer):
            if is_biased:
                x.append(apply_activation_func(activation_func, get_neuron_net(len(base_inputs), base_inputs,
                                                                               hidden_layer_weights[layer][
                                                                                   neuron]) +
                                               hidden_layer_bias[layer][neuron]))
            else:
                x.append(apply_activation_func(activation_func, get_neuron_net(len(base_inputs), base_inputs,
                                                                               hidden_layer_weights[layer][
                                                                                   neuron])))
        base_inputs = x
        outputs.append(base_inputs)

    x = []
    for neuron in range(number_of_outputs):
        if is_biased:
            x.append(apply_activation_func(activation_func, get_neuron_net(len(base_inputs), base_inputs,
                                                                           output_layer_weights[neuron]) +
                                           output_layer_bias[neuron]))
        else:
            x.append(apply_activation_func(activation_func, get_neuron_net(len(base_inputs), base_inputs,
                                                                           output_layer_weights[neuron])))
    base_inputs = x
    outputs.append(base_inputs)

    # print("outputs")
    # print(outputs)

    """
        [
            [0.84884938389734, 0.8785326453677295], 
            [0.6513062763326295, 0.7411387403053826], 
            [0.8426063005367096, 0.6877155240707966, 0.834744855846409]
        ]
    """
    return outputs


def back_propagate(actual, target, activation_func, number_of_neurons_per_hidden_layer, hidden_weights, output_weights,
                   number_of_hidden_layers, number_of_outputs):
    outputs_errors = []
    for output_neuron in range(number_of_outputs):
        # output_neuron => 0 Bombay
        if output_neuron == target:
            derv_result = apply_activation_derv(activation_func, actual[-1][output_neuron])
            neuron_error = (1 - actual[-1][output_neuron]) * derv_result
        else:
            derv_result = apply_activation_derv(activation_func, actual[-1][output_neuron])
            neuron_error = (0 - actual[-1][output_neuron]) * derv_result
        outputs_errors.append(neuron_error)

    # print("outputs_errors")
    # print(outputs_errors)

    """
        output_errors: [sigmoids]
        [
            0.09227852334130661, 
            -0.07994095098585972, 
            -0.08855861967125905
        ]
    """

    base_outputs = outputs_errors

    hidden_errors = []

    hidden_weights_reversed = reverse_hidden_to_hidden(hidden_weights)
    output_weights_reversed = reverse_output_to_hidden(output_weights, number_of_neurons_per_hidden_layer,
                                                       number_of_outputs)

    # print("hidden_weights_reversed")
    # print(hidden_weights_reversed)

    for layer in range(number_of_hidden_layers - 1, -1, -1):
        x = []
        for neuron in range(number_of_neurons_per_hidden_layer):
            if layer == number_of_hidden_layers - 1:
                derv = apply_activation_derv(activation_func, actual[layer][neuron])
                summation = get_neuron_error(len(base_outputs), base_outputs,
                                             output_weights_reversed[
                                                 neuron])
                x.append(summation * derv)
            else:
                derv = apply_activation_derv(activation_func, actual[layer][neuron])
                summation = get_neuron_error(len(base_outputs), base_outputs,
                                             hidden_weights_reversed[layer][
                                                 neuron])
                x.append(summation * derv)
        hidden_errors.append(x)
        base_outputs = x

    hidden_errors.reverse()
    hidden_errors.append(outputs_errors)

    # print("error signal")
    # print(hidden_errors)

    """
        [
            [-0.015822542637296905, -0.00634654084876852], 
            [-0.00029850113945332184, -0.0006973515621079713], 
            [0.02369789977448652, -0.10063479267027352, -0.136780978933255]
        ]
    """

    return hidden_errors


def update_weights(learning_rate, hidden_weights, output_weights, error_signals, input_layer, actual,
                   number_hidden_layers,
                   number_neurons_hidden_layer, hidden_layer_bias, output_layer_bias, is_biased):
    add_bias_to_weights(hidden_weights, hidden_layer_bias)
    add_bias_to_weights(output_weights, output_layer_bias)

    new_hidden_weights = []
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
                for x in range(number_neurons_hidden_layer + is_biased):
                    if is_biased and x == number_neurons_hidden_layer:
                        new_neuron_weights.append(
                            hidden_weights[layer][neuron][x] + learning_rate * error_signals[layer][neuron] *
                            1)
                    else:
                        new_neuron_weights.append(
                            hidden_weights[layer][neuron][x] + learning_rate * error_signals[layer][neuron] *
                            actual[layer - 1][x])
            new_layer_weights.append(new_neuron_weights)

        new_hidden_weights.append(new_layer_weights)

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
                    output_weights[0][output_neuron][hidden_layer_neuron] + learning_rate *
                    error_signals[number_hidden_layers][output_neuron] * actual[number_hidden_layers - 1][
                        hidden_layer_neuron])
        new_output_layer_weights.append(new_output_weights)

    return new_hidden_weights, new_output_layer_weights


def update_weights_and_bias(inputs, output_values, error_signal, number_of_hidden_layers, number_of_neurons,
                            number_of_outputs, learning_rate, hidden_weights, output_weights, hidden_bias, output_bias,
                            is_biased):
    output_values.insert(0, inputs)
    new_hidden_weights = []
    for layer in range(number_of_hidden_layers):
        new_layer_weights = []
        for neuron in range(number_of_neurons):
            new_neuron_weights = []
            for weight in range(len(hidden_weights[layer][neuron])):
                delta = learning_rate * output_values[layer][weight] * error_signal[layer][neuron]
                new_weight = hidden_weights[layer][neuron][weight] + delta
                new_neuron_weights.append(new_weight)
            new_layer_weights.append(new_neuron_weights)
        new_hidden_weights.append(new_layer_weights)

    # print("new_hidden_weights")
    # print(new_hidden_weights)

    new_output_weights = []
    for neuron in range(number_of_outputs):
        new_neuron_weights = []
        for weight in range(len(output_weights[neuron])):
            delta = learning_rate * output_values[-2][weight] * error_signal[-1][neuron]
            new_weight = output_weights[neuron][weight] + delta
            new_neuron_weights.append(new_weight)
        new_output_weights.append(new_neuron_weights)

    # print("new_output_weights")
    # print(new_output_weights)
    #
    # print("hidden_bias")
    # print(hidden_bias)

    new_hidden_bias = []
    new_output_bias = []

    if is_biased:
        for layer in range(number_of_hidden_layers):
            new_layer_bias = []
            for neuron in range(number_of_neurons):
                delta = learning_rate * error_signal[layer][neuron]
                new_bias = hidden_bias[layer][neuron] + delta
                new_layer_bias.append(new_bias)
            new_hidden_bias.append(new_layer_bias)

        # print("new_hidden_bias")
        # print(new_hidden_bias)

        # print("output_bias")
        # print(output_bias)
        for neuron in range(number_of_outputs):
            delta = learning_rate * error_signal[-1][neuron]
            new_bias = output_bias[neuron] + delta
            new_output_bias.append(new_bias)

        # print("new_output_bias")
        # print(new_output_bias)

    return new_hidden_weights, new_output_weights, new_hidden_bias, new_output_bias


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
                neuron_weights.append(random.uniform(-1, 1))
            layer_weights.append(neuron_weights)
        res.append(layer_weights)
    return res


def gen_output_weights(layers_num, input_layer_size, layer_size):
    layer_weights = []
    for layer in range(layers_num):
        if layer == 0:
            inputs = input_layer_size
        else:
            inputs = layer_size
        for neuron in range(layer_size):
            neuron_weights = []
            for weight in range(inputs):
                neuron_weights.append(random.uniform(-1, 1))
            layer_weights.append(neuron_weights)
    return layer_weights


def gen_bias(n, size):
    res = []
    for layer in range(n):
        layer_bias = []
        for neuron in range(size):
            layer_bias.append(random.uniform(-1, 1))
        res.append(layer_bias)
    return res


def gen_output_bias(n):
    layer_bias = []
    for layer in range(n):
        layer_bias.append(random.uniform(-1, 1))
    return layer_bias


def get_neuron_net(n, inputs, weight):
    res = 0
    for i in range(n):
        res += inputs[i] * weight[i]
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


def reverse_hidden_to_hidden(weights):
    modified_list = [[list(i) for i in zip(*sublist)] for sublist in weights]
    modified_list.reverse()
    return modified_list


def reverse_output_to_hidden(weights, number_of_neurons_per_hidden_layer, number_of_output):
    res = np.reshape(weights, (number_of_output, number_of_neurons_per_hidden_layer)).T.tolist()
    return res


def add_bias_to_weights(weights, bias):
    flatten_bias = [num for sublist in bias for num in sublist]
    cnt = 0
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            weights[i][j].append(flatten_bias[0])
            cnt += 1
    return weights


def initialize_weights_and_biases(inputs, number_of_outputs, number_of_hidden_layers,
                                  number_of_neurons_per_hidden_layer):
    hidden_layer_weights = gen_random_weights(number_of_hidden_layers, len(inputs), number_of_neurons_per_hidden_layer)
    output_layer_weights = gen_output_weights(Constants.OUTPUT_LAYERS_CNT, number_of_neurons_per_hidden_layer,
                                              number_of_outputs)

    hidden_layer_bias = gen_bias(number_of_hidden_layers, number_of_neurons_per_hidden_layer)
    output_layer_bias = gen_output_bias(number_of_outputs)
    return hidden_layer_weights, output_layer_weights, hidden_layer_bias, output_layer_bias


def split_weights_and_bias(merged, number_of_layers, number_of_neurons):
    res = []
    for layer in range(number_of_layers):
        layer_biases = []
        for neuron in range(number_of_neurons):
            layer_biases.append(merged[layer][neuron][-1])
            merged[layer][neuron].pop()
        res.append(layer_biases)
    return merged, res


def split_output_weights_and_bias(merged, output_cnt):
    res = []
    for neuron in range(output_cnt):
        res.append(merged[neuron][-1])
        merged[neuron].pop()
    return [merged], [res]


def train_neural_network(inputs, output, number_of_outputs, number_of_hidden_layers, number_of_neurons_per_hidden_layer,
                         learning_rate,
                         epochs,
                         is_biased,
                         activation_func,
                         hidden_layer_weights,
                         output_layer_weights,
                         hidden_layer_bias,
                         output_layer_bias):
    last_hidden_weights = hidden_layer_weights
    last_output_weights = output_layer_weights

    last_hidden_bias = hidden_layer_bias
    last_output_bias = output_layer_bias

    for epoch in range(epochs):
        # print('========== Epoch#', epoch)

        values = forward_propagation(inputs, last_hidden_weights, last_output_weights, last_hidden_bias,
                                     last_output_bias,
                                     number_of_hidden_layers, number_of_neurons_per_hidden_layer, number_of_outputs,
                                     is_biased, activation_func)

        error_signal = back_propagate(values, output, activation_func=activation_func,
                                      number_of_neurons_per_hidden_layer=number_of_neurons_per_hidden_layer,
                                      hidden_weights=last_hidden_weights, output_weights=last_output_weights,
                                      number_of_hidden_layers=number_of_hidden_layers,
                                      number_of_outputs=number_of_outputs)

        (last_hidden_weights, last_output_weights,
         last_hidden_bias, last_output_bias) = update_weights_and_bias(inputs,
                                                                       values,
                                                                       error_signal,
                                                                       number_of_hidden_layers,
                                                                       number_of_neurons_per_hidden_layer,
                                                                       number_of_outputs,
                                                                       learning_rate,
                                                                       hidden_layer_weights,
                                                                       output_layer_weights,
                                                                       hidden_layer_bias,
                                                                       output_layer_bias,
                                                                       is_biased)
        #
        # print("============= After Epoch ===============")
        # print("last_hidden_weights", last_hidden_weights)
        # print("last_hidden_bias", last_hidden_bias)
        # print("last_output_weights", last_output_weights)
        # print("last_output_bias", last_output_bias)

        # print(values[number_of_hidden_layers])
        # print(output)

    return last_hidden_weights, last_output_weights, last_hidden_bias, last_output_bias


def get_max_index(outputs):
    result = max(outputs)
    for i in range(len(outputs)):
        if result == outputs[i]:
            return i


def test_neural_network(x_test, y_test, is_biased, activation_func, number_of_hidden_layers,
                        number_of_neurons_per_hidden_layer, number_of_outputs, last_hidden_weights, last_hidden_bias,
                        last_output_weights, last_output_bias):
    real_results = []
    N = len(x_test.index.tolist())
    # N = 2
    for i in range(N):
        print("X")
        print(list(x_test.iloc[i]))
        print("Y")
        print("Actual Output: ", y_test[i])
        # output = forward_propagation(list(x_test.iloc[i]), last_hidden_weights, last_output_weights, last_hidden_bias,
        #                              last_output_bias, number_of_hidden_layers,
        #                              number_of_neurons_per_hidden_layer,
        #                              is_biased, activation_func, True, is_test=True)

        values = forward_propagation(list(x_test.iloc[i]), last_hidden_weights, last_output_weights, last_hidden_bias,
                                     last_output_bias,
                                     number_of_hidden_layers,
                                     number_of_neurons_per_hidden_layer, number_of_outputs, is_biased, activation_func,
                                     is_test=True)

        print("Output Neurons(Classes): ", values[number_of_hidden_layers])
        real_results.append(get_max_index(values[number_of_hidden_layers]))
    predicted = y_test.copy()
    return predicted, y_test

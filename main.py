import Network
from gui import draw

draw()


# inputs = [0, 0]  # input cnt = 2
# hidden_layer_weights = [[[0.21, 0.15], [-0.4, 0.1]], [[-0.3, 0.2], [-0.1, 0.46]]]
# output_layer_weights = [[0.1, 0.3], [0.2, 0.4]]
# hidden_layer_bias = [[-0.3, 0.25], [0.41, -0.12]]
# output_layer_bias = [-0.4, -0.1]
# n_hidden_layers = 2
# n_hidden_neurons = 2
# n_output = 2
# learning_rate = 0.1
# is_biased = True
# target = 0

# inputs = [1, 0, 1]  # input cnt = 2
# hidden_layer_weights = [[[0.2, 0.4, -0.5], [-0.3, 0.1, 0.2]]]
# output_layer_weights = [[-0.3, -0.2]]
# hidden_layer_bias = [[-0.4, 0.2]]
# output_layer_bias = [0.1]
# n_hidden_layers = 1
# n_hidden_neurons = 2
# n_output = 1
# learning_rate = 0.9
# is_biased = True
# target = 0

# inputs = [0, 0]  # input cnt = 2
# hidden_layer_weights = [[[0.21, 0.15], [-0.4, 0.1]]]
# output_layer_weights = [[-0.2, 0.3]]
# hidden_layer_bias = [[-0.3, 0.25]]
# output_layer_bias = [-0.4]
# n_hidden_layers = 1
# n_hidden_neurons = 2
# n_output = 1
# learning_rate = 0.9
# is_biased = True
# target = 1

# values = Network.forward_propagation(inputs, hidden_layer_weights, output_layer_weights, hidden_layer_bias,
#                                      output_layer_bias,
#                                      n_hidden_layers, n_hidden_neurons, n_output, is_biased, 'sigmoid')
#
# error_signal = Network.back_propagate(values, target, activation_func="sigmoid",
#                                       number_of_neurons_per_hidden_layer=n_hidden_neurons,
#                                       hidden_weights=hidden_layer_weights, output_weights=output_layer_weights,
#                                       number_of_hidden_layers=n_hidden_layers, number_of_outputs=n_output)
#
# Network.update_weights_and_bias(inputs, values, error_signal, n_hidden_layers, n_hidden_neurons, n_output,
#                                 learning_rate,
#                                 hidden_layer_weights,
#                                 output_layer_weights, hidden_layer_bias, output_layer_bias, is_biased)

"""
    Class A Sample => 0
    Class B Sample => 40
    Class C Sample => 80
"""

# print("sample input")
# print(sample_features)
# print(sample_y)

# NN.initialize_maps(1,2, 1, True)
#####
# preprocessing:
# - Normalization
# - fill with average
# - Encoding
# - Train/Test

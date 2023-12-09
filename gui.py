import tkinter as tk

from matplotlib import pyplot as plt

import Network
from FileInput import import_excel
from Preprocessing import get_train_test
from sklearn.metrics import ConfusionMatrixDisplay


def calculate_confusion_matrix(predicted, actual, num_classes=3):
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    for pred, actual_val in zip(predicted, actual):
        matrix[pred][actual_val] += 1
    print(matrix)
    return matrix


def train_and_test(number_hidden_layers, number_neurons, number_of_output, learning_rate, epochs, is_biased,
                   activation_function):
    df = import_excel()

    x_train, y_train, x_test, y_test = get_train_test(df)

    y_train = y_train.tolist()

    train_N = len(y_train)

    last_hidden_weights, last_output_weights, last_hidden_bias, last_output_bias = Network.initialize_weights_and_biases(
        x_train.iloc[0], number_of_output, number_hidden_layers, number_neurons)

    print("============= Before Train ===============")
    print("last_hidden_weights", last_hidden_weights)
    print("last_hidden_bias", last_hidden_bias)
    print("last_output_weights", last_output_weights)
    print("last_output_bias", last_output_bias)

    print("Model is Currently Training..")
    for i in range(train_N):
        last_hidden_weights, last_output_weights, last_hidden_bias, last_output_bias = Network.train_neural_network(
            x_train.iloc[i], y_train[i], number_of_output, number_hidden_layers, number_neurons, learning_rate, epochs,
            is_biased,
            activation_function, last_hidden_weights, last_output_weights,
            last_hidden_bias, last_output_bias)

    print("============= After Train ===============")
    print("last_hidden_weights", last_hidden_weights)
    print("last_hidden_bias", last_hidden_bias)
    print("last_output_weights", last_output_weights)
    print("last_output_bias", last_output_bias)

    predicted, actual = Network.test_neural_network(x_test, y_test, is_biased, activation_function,
                                                    number_hidden_layers, number_neurons, number_of_output,
                                                    last_hidden_weights, last_hidden_bias, last_output_weights,
                                                    last_output_bias)

    # display_accuracy_confusion(predicted, actual)


def draw():
    def get_user_input():
        hidden_layers = int(hidden_layers_entry.get())
        neurons_in_layers = int(neuron_entry.get())
        learning_rate = float(learning_rate_entry.get())
        epochs = int(epochs_entry.get())
        use_bias = bias_var.get()
        activation_function = activation_var.get()

        print("Hidden Layers:", hidden_layers)
        print("Neurons in Each Hidden Layer:", neurons_in_layers)
        print("Learning Rate:", learning_rate)
        print("Epochs:", epochs)
        print("Use Bias:", use_bias)
        print("Activation Function:", activation_function)

        train_and_test(hidden_layers, neurons_in_layers, 3, learning_rate, epochs, use_bias, activation_function)

    root = tk.Tk()
    root.title("Neural Network Configuration")

    # Adding margin from the top
    margin_top = 10
    tk.Label(root, text="").grid(row=0, column=0, pady=margin_top)

    # Adding padding to the layout
    padx_value = 10
    pady_value = 5

    # Labels
    tk.Label(root, text="Number of Hidden Layers:").grid(row=1, column=0, padx=padx_value, pady=pady_value)
    tk.Label(root, text="Neurons in Each Hidden Layer:").grid(row=2, column=0, padx=padx_value, pady=pady_value)
    tk.Label(root, text="Learning Rate (eta):").grid(row=3, column=0, padx=padx_value, pady=pady_value)
    tk.Label(root, text="Number of Epochs (m):").grid(row=4, column=0, padx=padx_value, pady=pady_value)
    tk.Label(root, text="Add Bias:").grid(row=5, column=0, padx=padx_value, pady=pady_value)
    tk.Label(root, text="Activation Function:").grid(row=6, column=0, padx=padx_value, pady=pady_value)

    # Entries
    hidden_layers_entry = tk.Entry(root)
    hidden_layers_entry.grid(row=1, column=1, padx=padx_value, pady=pady_value)

    neuron_entry = tk.Entry(root)
    neuron_entry.grid(row=2, column=1, padx=padx_value, pady=pady_value)

    learning_rate_entry = tk.Entry(root)
    learning_rate_entry.grid(row=3, column=1, padx=padx_value, pady=pady_value)

    epochs_entry = tk.Entry(root)
    epochs_entry.grid(row=4, column=1, padx=padx_value, pady=pady_value)

    bias_var = tk.BooleanVar()
    bias_checkbox = tk.Checkbutton(root, variable=bias_var)
    bias_checkbox.grid(row=5, column=1, padx=padx_value, pady=pady_value)

    activation_var = tk.StringVar(value="sigmoid")
    activation_dropdown = tk.OptionMenu(root, activation_var, "sigmoid", "tanh")
    activation_dropdown.grid(row=6, column=1, padx=padx_value, pady=pady_value)

    # Button
    submit_button = tk.Button(root, text="Submit", command=get_user_input)
    submit_button.grid(row=7, columnspan=2, padx=padx_value, pady=pady_value)

    root.mainloop()

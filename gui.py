import tkinter as tk

def draw():
    def get_user_input():
        hidden_layers = int(hidden_layers_entry.get())
        neurons_in_layers = [int(neuron_entry.get()) for _ in range(hidden_layers)]
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

    activation_var = tk.StringVar(value="Sigmoid")
    activation_dropdown = tk.OptionMenu(root, activation_var, "Sigmoid", "Hyperbolic Tangent")
    activation_dropdown.grid(row=6, column=1, padx=padx_value, pady=pady_value)

    # Button
    submit_button = tk.Button(root, text="Submit", command=get_user_input)
    submit_button.grid(row=7, columnspan=2, padx=padx_value, pady=pady_value)

    root.mainloop()
import Network
from FileInput import import_excel
from Preprocessing import get_train_test

# draw()

df = import_excel()

x_train, y_train, x_test, y_test = get_train_test(df)

sample_features = x_train.iloc[80].tolist()
sample_y = y_train[80]

print(len(x_train.index.tolist()))
"""
    Class A Sample => 0
    Class B Sample => 40
    Class C Sample => 80
"""

# print("sample input")
# print(sample_features)
# print(sample_y)

train_N = len(y_train)
# train_N = 1
test_N = len(y_test)

last_hidden_weights, last_output_weights, last_hidden_bias, last_output_bias = Network.initialize_weights_and_biases(
    x_train.iloc[0], 4, 4)

print("Model is Currently Training..")
for i in range(train_N):
    last_hidden_weights, last_hidden_bias, last_output_weights, last_output_bias = Network.train_neural_network(
        x_train.iloc[i], y_train[i], 4, 4, 0.1, 1000, True, 'sigmoid', last_hidden_weights, last_output_weights,
        last_hidden_bias, last_output_bias)


print("============= After Train ===============")
print("last_hidden_weights", last_hidden_weights)
print("last_hidden_bias", last_hidden_bias)
print("last_output_weights", last_output_weights)
print("last_output_bias", last_output_bias)

#
Network.test_neural_network(x_test, y_test, True, 'sigmoid', 4, 4,
                            last_hidden_weights, last_hidden_bias, last_output_weights,
                            last_output_bias)

# NN.initialize_maps(1,2, 1, True)
#####
# preprocessing:
# - Normalization
# - fill with average
# - Encoding
# - Train/Test

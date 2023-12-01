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

print("sample input")
print(sample_features)
print(sample_y)

# Network.train_neural_network(sample_features, sample_y, 4, 5, 0.1, 1000, True, 'sigmoid')
Network.test_neural_network(x_test, y_test, True, 'sigmoid')

# NN.initialize_maps(1,2, 1, True)
#####
# preprocessing:
# - Normalization
# - fill with average
# - Encoding
# - Train/Test

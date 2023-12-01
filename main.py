import Network
from FileInput import import_excel
from Preprocessing import get_train_test

# draw()

df = import_excel()

x_train, y_train, x_test, y_test = get_train_test(df)

sample_features = x_train.iloc[0].tolist()
sample_y = y_train[0]

print("sample input")
print(sample_features)
print(sample_y)

Network.train_neural_network(sample_features, sample_y, 4, 4, 0.001, 1, True, 'sigmoid')

# NN.initialize_maps(1,2, 1, True)
#####
# preprocessing:
# - Normalization
# - fill with average
# - Encoding
# - Train/Test

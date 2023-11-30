import Network
from gui import draw
from FileInput import import_excel
from Preprocessing import get_train_test

# draw()

df = import_excel()

x_train, y_train, x_test, y_test = get_train_test(df)

# print(x_train.iloc[0])

Network.train_neural_network([0, 0], 1, 1, 2, 0.001, 11, True, 'sigmoid')

#####
# preprocessing:
# - Normalization
# - fill with average
# - Encoding
# - Train/Test

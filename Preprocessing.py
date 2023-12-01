from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import ClassesEncoding


def get_train_test(df):
    # split features

    output_features = df['Class']
    df = df.drop('Class', axis=1)

    # fill with average
    df = df.fillna(df.mean())

    # normalization 0 to 1
    scaler = MinMaxScaler()

    normalized_features = scaler.fit_transform(df)

    df = pd.DataFrame(normalized_features, columns=df.columns)

    # encoding
    output_features = ClassesEncoding.encode(output_features)

    # print(output_features)

    class1 = df[:50]
    class2 = df[50:100]
    class3 = df[100:]

    y1 = output_features[:50]
    y2 = output_features[50:100]
    y3 = output_features[100:]

    class1_x_train, class1_x_test, class1_y_train, class1_y_test = train_test_split(class1, y1, test_size=0.4,
                                                                                    random_state=42)
    class2_x_train, class2_x_test, class2_y_train, class2_y_test = train_test_split(class2, y2, test_size=0.4,
                                                                                    random_state=42)
    class3_x_train, class3_x_test, class3_y_train, class3_y_test = train_test_split(class3, y3, test_size=0.4,
                                                                                    random_state=42)

    train_data_x = pd.concat([class1_x_train, class2_x_train, class3_x_train])
    train_data_y = class1_y_train + class2_y_train + class3_y_train

    shuffled_df = train_data_x
    shuffled_df['Class'] = train_data_y
    shuffled_df = shuffled_df.sample(frac=1)

    shuffled_y = shuffled_df['Class']
    shuffled_x = shuffled_df.drop('Class', axis=1)

    test_data_x = pd.concat([class1_x_test, class2_x_test, class3_x_test])
    test_data_y = class1_y_test + class2_y_test + class3_y_test

    return shuffled_x, shuffled_y, test_data_x, test_data_y

# def compare(categs, output):
#     bombay = 'BOMBAY'
#     cali = 'CALI'
#     sira = 'SIRA'
#     for i in range(len(output)):
#         if categs[i] == bombay and output == 0:
#             continue
#         if categs[i] == cali and output == 1:
#             continue
#         if categs[i] == sira and output == 2:
#             continue
#         else:
#             print("some value doesn't match")
#     if len(categs) != len(output):
#         print("lenght doesn't match")

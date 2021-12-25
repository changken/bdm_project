import numpy as np
import pandas as pd


def remove_dump_values(data, cols):
    for col in cols:
        data[col] = np.where(data[col] == '-', 'None', data[col])
    return data


def normalize(data):
    return (data - np.min(data)) / np.std(data)


def preprocess(data):
    cols = data.columns
    cols_cat = data.select_dtypes('object').columns
    cols_numeric = data._get_numeric_data().columns

    # remove dump value
    data_bin = remove_dump_values(data, cols)

    # remove unnecessary features
    cols_cat = cols_cat.drop(['attack_cat'])
    cols_numeric = cols_numeric.drop(['id', 'label'])

    # one hot encoding category feature
    data_bin_hot = pd.get_dummies(data_bin, columns=cols_cat)

    # normalize numeric features
    data_bin_hot[cols_numeric] = data_bin_hot[cols_numeric].astype('float')
    data_bin_hot[cols_numeric] = normalize(data_bin_hot[cols_numeric])

    return data_bin_hot

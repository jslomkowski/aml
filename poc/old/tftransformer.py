from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.multioutput import RegressorChain
from xgboost import XGBRegressor
# from sklearn.model_selection import TimeSeriesSplit, train_test_split
import pandas as pd
import numpy as np


def make_dataset(leave_uldl=False):
    df = pd.read_pickle('data_AL_processed_tf_lite.pkl')
    if leave_uldl:
        df = df[['col2', 'cell_name', 'prb', 'fn1', 'uldl']]
    else:
        df = df[df['uldl'] == 'dl']
        df = df[['date', 'cell_name', 'prb', 'fn1']]
    df = df.replace({df['cell_name'].unique()[0]: 'A',
                     df['cell_name'].unique()[1]: 'B'}).reset_index(drop=True)
    df = df.sort_values(['cell_name', 'date'])
    return df


def make_dataset2():
    in_seq1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    in_seq2 = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
    in_seq3 = np.array(range(1, 41))
    in_seq4 = np.array(range(41, 81))
    out_seq = np.array(range(81, 121))
    df = pd.DataFrame([in_seq1, in_seq2, in_seq3, in_seq4, out_seq],
                      index=['date', 'cell_name', 'uldl', 'fn1', 'prb']).T
    return df


def ts_transform(df, split_by, split_value, steps_in, steps_out, steps,
                 add_split, target, forecast_type):

    train = df.loc[df[split_by] < split_value]
    test = df.loc[df[split_by] >= split_value]

    def transform(df):
        try:
            df_dict = {k: v for k, v in df.groupby(add_split)}
        except TypeError:
            df_dict = {1: df}

        X, y = list(), list()

        col_range = range(len(df.columns))
        if forecast_type == 'independent':
            target_index = [i for i in col_range]
            feature_index = [i for i in col_range]
        elif forecast_type == 'dependent':
            target_index = [i for i in col_range if df.columns[i] in target]
            feature_index = [i for i in col_range if i not in target_index]

        for d in df_dict:
            df_d = df_dict[d].values
            for i in range(0, len(df_d), steps):
                if steps_in + i + steps_out > len(df_d):
                    break
                X_ = df_d[np.ix_(range(i, steps_in + i), feature_index)]
                y_ = df_d[steps_in + i:steps_in + i + steps_out, target_index]
                X.append(X_)
                y.append(y_)
        X = np.array(X)
        y = np.array(y)

        n_input = X.shape[1] * X.shape[2]
        X = X.reshape((X.shape[0], n_input))

        n_output = y.shape[1] * y.shape[2]
        y = y.reshape((y.shape[0], n_output))
        return X, y

    X_train, y_train = transform(train)
    X_test, y_test = transform(test)

    return X_train, X_test, y_train, y_test


df = make_dataset2()

split_by = 'date'
# split_value = '2020-05-15 09:00:00'
split_value = 14
steps_in = 3
steps_out = 3
steps = 1
add_split = ['cell_name']
target = ['prb']
forecast_type = 'dependent'

X_train, X_test, y_train, y_test = ts_transform(df, split_by, split_value,
                                                steps_in, steps_out, steps,
                                                add_split, target, forecast_type)

model = XGBRegressor()
encoder = OrdinalEncoder()
chain = RegressorChain(model)

X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

chain.fit(X_train, y_train)
y_pred_train = chain.predict(X_train)
y_pred_test = chain.predict(X_test)

print(mean_absolute_percentage_error(y_train, y_pred_train))
print(mean_absolute_percentage_error(y_test, y_pred_test))

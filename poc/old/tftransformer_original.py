import numpy as np
import pandas as pd

# define input sequence


def make_df():

    in_seq1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    in_seq2 = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                        3, 3, 3, 3, 3, 4, 4, 4, 4, 4])

    in_seq3 = np.array(range(1, 21))
    in_seq4 = np.array(range(21, 41))
    out_seq = np.array(range(41, 61))

    df = pd.DataFrame([in_seq1, in_seq2, in_seq3, in_seq4, out_seq],
                      index=['col1', 'col2', 'col3', 'col4', 'col5']).T
    return df


df = make_df()

steps_in = 3
steps_out = 2
step = 2
grain = ['col1']
target = ['col5']
forecast_type = 'dependent'

try:
    df_dict = {k: v for k, v in df.groupby(grain)}
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
    for i in range(0, len(df_d), step):
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

print(df)
print(X)
print(y)

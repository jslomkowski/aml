import pandas as pd
import numpy as np



def split_df(df, steps_in, steps_out):
    X, y = list(), list()
    for i in range(len(df)):
        print(i)
        # i = 0
        # find the end of this pattern
        end_ix = i + steps_in
        out_end_ix = end_ix + steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(df):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = df.iloc[i:end_ix, :], df.iloc[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    X = np.array(X)
    y = np.array(y)
    n_input = X.shape[1] * X.shape[2]
    X = X.reshape((X.shape[0], n_input))
    return X, y


seq_x.to_numpy().flatten()).T

pd.DataFrame(seq_x.values.flatten()).T

df=make_data()
steps_in, steps_out=3, 2
X, y=split_df(df, steps_in, steps_out)
dfX=pd.concat(X, axis = 1, ignore_index = True)
pd.DataFrame(X).head()

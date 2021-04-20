from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.multioutput import RegressorChain
from xgboost import XGBRegressor
# from sklearn.model_selection import TimeSeriesSplit, train_test_split
import pandas as pd
import numpy as np


def make_dataset():
    df = pd.read_pickle('data_AL_processed_tf.pkl')
    df = df.loc[df['uldl'] == 'dl']
    df['date'] = pd.to_datetime(df['date'])
    # df = df.loc[df['date'] <= '2020-04-28 11:30:00']
    df = df[['date', 'cell_name', 'prb', 'fn1',
             'date_hour']].reset_index(drop=True)
    df = df.replace({df['cell_name'].unique()[0]: 'A',
                     df['cell_name'].unique()[1]: 'B'})
    df['date'] = df['date'].astype('str')
    return df


df = make_dataset()
split_date = '2020-05-15 09:00:00'
train = df.loc[df['date'] < split_date]
test = df.loc[df['date'] >= split_date]


steps_in = 24 * 2 * 7
steps_out = 24 * 2 * 3
step = 24 * 2 * 7
grain = ['cell_name']
target = ['prb']
forecast_type = 'dependent'


def ts_transform(df, steps_in, steps_out, step, grain, target, forecast_type):

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
    return X, y


X_train, y_train = ts_transform(
    train, steps_in, steps_out, step, grain, target, forecast_type)

model = XGBRegressor()
encoder = OrdinalEncoder()
chain = RegressorChain(base_estimator=model)

X_train = encoder.fit_transform(X_train)
chain.fit(X_train, y_train)
y_pred_train = chain.predict(X_train)

print(mean_absolute_percentage_error(y_train, y_pred_train))


X_test, y_test = ts_transform(
    test, steps_in, steps_out, step, grain, target, forecast_type)

X_test = encoder.fit_transform(X_test)
y_pred_test = chain.predict(X_test)
print(mean_absolute_percentage_error(y_test, y_pred_test))

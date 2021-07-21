import itertools
import random
import string
from copy import deepcopy

import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import (GridSearchCV, ParameterGrid,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

X, y = load_boston(return_X_y=True)
X = pd.DataFrame(X)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def create_model(optimizer='adagrad', dropout=0.2):
    model = Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_absolute_error',
                  optimizer='adam')

    return model


model = KerasRegressor(build_fn=create_model, verbose=0)


param_grid = {
    # 'model1__optimizer': ['rmsprop', 'adam', 'adagrad'],
    'model1__epochs': [4, 8],
    'model1__dropout': [0.1, 0.2],
}


pipeline = Pipeline([
    ('ss1', StandardScaler()),
    ('ss2', Normalizer()),
    ('model1', model)
    # ('model1', RandomForestRegressor())
])


def make_aml_combinations(pipeline, param_grid):

    # creates dictionary with unique steps and list of trans and models
    #  per step
    # out:
    # {'imp': [MeanMedianImputer()],
    #  'disc': [EqualFrequencyDiscretiser(), EqualWidthDiscretiser()],
    #  'model': [LinearRegression(), RandomForestRegressor()]}
    fd = {}
    st = dict(pipeline.steps)
    ts = {v: k for k, v in st.items()}
    for k, v in st.items():
        # print(k, v)
        k = ''.join([i for i in k if not i.isdigit()])
        if k not in fd.keys():
            fd[k] = [v]
        else:
            fd[k].append(v)

    # creates all combination of pipeline steps
    # out:
    # [{'imp': MeanMedianImputer(),
    # 'disc': EqualFrequencyDiscretiser(),
    # 'model': LinearRegression()},
    # {'imp': MeanMedianImputer(),
    # 'disc': EqualFrequencyDiscretiser(),
    # 'model': RandomForestRegressor()},
    # {'imp': MeanMedianImputer(),
    # 'disc': EqualWidthDiscretiser(),
    # 'model': LinearRegression()},
    # {'imp': MeanMedianImputer(),
    # 'disc': EqualWidthDiscretiser(),
    # 'model': RandomForestRegressor()}]
    def _product_dict(**kwargs):
        for instance in itertools.product(*kwargs.values()):
            yield dict(zip(kwargs.keys(), instance))
    pipelines_dict = list(_product_dict(**fd))

    # this will attach numbers at the end of step string
    # [{'imp1': MeanMedianImputer(),
    # 'disc1': EqualFrequencyDiscretiser(),
    # 'model1': LinearRegression()},
    # {'imp1': MeanMedianImputer(),
    # 'disc1': EqualFrequencyDiscretiser(),
    # 'model2': RandomForestRegressor()},
    # {'imp1': MeanMedianImputer(),
    # 'disc2': EqualWidthDiscretiser(),
    # 'model1': LinearRegression()},
    # {'imp1': MeanMedianImputer(),
    # 'disc2': EqualWidthDiscretiser(),
    # 'model2': RandomForestRegressor()}]
    for p in pipelines_dict:
        for k, v in list(p.items()):
            p[ts[v]] = p.pop(k)

    final_pipes = []
    for pipe_dict in pipelines_dict:

        # creates Pipeline object from pipelines_dict
        pipe = Pipeline([(k, v) for k, v in pipe_dict.items()])

        # hard copies param_grid
        clone_grid = deepcopy(param_grid)

        # creates list with keys to delete from the param_grid
        delete_indexes = []
        for g in clone_grid:
            if g.split('__')[0] not in pipe_dict:
                delete_indexes.append(g)

        # deletes those keys from cloned param_grid
        for k in delete_indexes:
            clone_grid.pop(k, None)

        # creates ParameterGrid combination and sets it to each pipeline
        clone_grid_list = list(ParameterGrid(clone_grid))
        for c in clone_grid_list:
            clone_pipe = deepcopy(pipe)
            clone_pipe.set_params(**c)
            final_pipes.append(clone_pipe)
    return final_pipes


def fit(final_pipes, metric):
    results = []
    for f in final_pipes:
        print(f)
        f.fit(X_train, y_train)
        y_pred_train = f.predict(X_train)
        y_pred_test = f.predict(X_test)
        letters = string.ascii_lowercase
        pipe_name = ''.join(random.choice(letters) for i in range(10))
        error_train = mean_absolute_error(y_train, y_pred_train)
        error_test = mean_absolute_error(y_test, y_pred_test)
        res = {'name': pipe_name,
               'params': f,
               'error_train': round(error_train, 2),
               'error_test': round(error_test, 2),
               'train_test_dif': round(error_test / error_train, 2),
               }
        results.append(res)
    results = pd.DataFrame.from_dict(results)

    return results


final_pipes = make_aml_combinations(pipeline, param_grid)
results = fit(final_pipes, mean_absolute_error)

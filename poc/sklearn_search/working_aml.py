import random
import string
from sklearn.metrics import mean_absolute_error
from multiprocessing import Pool
import itertools
from copy import deepcopy

import pandas as pd
from feature_engine.discretisation import (EqualFrequencyDiscretiser,
                                           EqualWidthDiscretiser)
from feature_engine.imputation import MeanMedianImputer
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline


def _validate_steps(self):
    names, estimators = zip(*self.steps)
    self._validate_names(names)
    transformers = estimators[:-1]
    for t in transformers:
        if t is None or t == 'passthrough':
            continue


Pipeline._validate_steps = _validate_steps

X, y = load_boston(return_X_y=True)
X = pd.DataFrame(X)
y = pd.Series(y)

for i in range(10):
    X = X.append(X)
    y = y.append(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline([
    ('imp1', MeanMedianImputer()),
    ('disc1', EqualFrequencyDiscretiser()),
    ('disc2', EqualWidthDiscretiser()),
    ('model1', LinearRegression()),
    ('model2', RandomForestRegressor())
])

grid = {
    'disc1__q': [5, 15],
    'model2__n_estimators': [50, 150]
}


def make_aml_combinations(pipeline, grid):
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

    def _product_dict(**kwargs):
        for instance in itertools.product(*kwargs.values()):
            yield dict(zip(kwargs.keys(), instance))

    pipelines_dict = list(_product_dict(**fd))

    for p in pipelines_dict:
        for k, v in p.items():
            p[ts[v]] = p.pop(k)

    final_pipes = []
    for pipe_dict in pipelines_dict:

        pipe = Pipeline([(k, v) for k, v in pipe_dict.items()])

        clone_grid = deepcopy(grid)

        delete_indexes = []
        for g in clone_grid:
            if g.split('__')[0] not in pipe_dict:
                delete_indexes.append(g)

        for k in delete_indexes:
            clone_grid.pop(k, None)

        clone_grid_list = list(ParameterGrid(clone_grid))
        for c in clone_grid_list:
            clone_pipe = deepcopy(pipe)
            clone_pipe.set_params(**c)
            final_pipes.append(clone_pipe)
    return final_pipes


def fit(f):
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
           'error_train': round(error_train, 2),
           'error_test': round(error_test, 2)}
    results.append(res)
    return results


if __name__ == '__main__':
    final_pipes = make_aml_combinations(pipeline, grid)
    pool = Pool()
    results = pool.map(fit, final_pipes)
    df_results = pd.DataFrame.from_dict(results)
    print('done')

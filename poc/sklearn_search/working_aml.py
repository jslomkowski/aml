import itertools

import pandas as pd
from feature_engine.discretisation import (EqualFrequencyDiscretiser,
                                           EqualWidthDiscretiser)
from feature_engine.imputation import MeanMedianImputer
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline
from copy import deepcopy


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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline([
    ('imp1', MeanMedianImputer()),
    ('disc1', EqualFrequencyDiscretiser()),
    ('disc2', EqualWidthDiscretiser()),
    ('model1', LinearRegression()),
    ('model2', RandomForestRegressor())
])

param_grid = {
    'disc1__q': [5, 15],
    'model2__n_estimators': [50, 150]
}


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

    clone_param_grid = deepcopy(param_grid)

    delete_indexes = []
    for g in clone_param_grid:
        if g.split('__')[0] not in pipe_dict:
            delete_indexes.append(g)

    for k in delete_indexes:
        clone_param_grid.pop(k, None)

    clone_param_grid_list = list(ParameterGrid(clone_param_grid))
    for c in clone_param_grid_list:
        clone_pipe = deepcopy(pipe)
        clone_pipe.set_params(**c)
        final_pipes.append(clone_pipe)

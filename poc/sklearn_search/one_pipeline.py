# from pipeline import AMLSearchCV
from itertools import product
from copy import deepcopy
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from feature_engine.discretisation import EqualFrequencyDiscretiser, EqualWidthDiscretiser
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.pipeline import Pipeline
from feature_engine.imputation import MeanMedianImputer
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV


def _validate_steps(self):
    names, estimators = zip(*self.steps)
    # validate names
    self._validate_names(names)
    # validate estimators
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
    'disc1__q': [5, 10],
    'model2__n_estimators': [50, 100]
}


# steps_lst = []
# for p in pipeline.steps:
#     if type(p) == tuple:
#         steps_lst.append(p)
#     elif type(p) == list:
#         for pi in p:
#             steps_lst.append(pi)
# pipeline = Pipeline(steps_lst)

cfg = []
for k, v in pipeline.get_params().items():
    if k.find('__') > 0:
        if k in param_grid:
            cfg.append([k, param_grid[k]])
        else:
            cfg.append([k, v])


cfg = pd.DataFrame(cfg, columns=['config', 'value'])
cfg[['block', 'config']] = cfg['config'].str.split(
    '__', 1, expand=True)
cfg['block2'] = cfg['block'].str.replace('\d+', '', regex=True)

cfg2 = cfg[['block', 'block2']].drop_duplicates()

prods = []
for _, d in cfg2.groupby("block2"):
    prods.append([s for _, s in d.iterrows()])
dfs = [pd.concat(ss, axis=1).T for ss in product(*prods)]

configs = []
for i in range(len(dfs)):
    # i=0
    t = pd.merge(dfs[i], cfg, how='left', on='block')[
        ['block', 'config', 'value']]
    configs.append(t)

t.explode('value')
prods = []
for _, d in t.explode('value').groupby("config"):
    prods.append([s for _, s in d.iterrows()])

dfs_list = []
for c in configs:
    # c = configs[0]
    prods = []
    for _, d in c.groupby("config"):
        prods.append([s for _, s in d.iterrows()])
    dfs = [pd.concat(ss, axis=1).T for ss in product(*prods)]
    print(dfs, '\n')
    dfs_list = dfs_list + dfs

pipes = []

for d in dfs_list:
    d['config'] = d['block'] + '__' + d['config']
    pipe_copy = deepcopy(pipeline)
    params = d[['config', 'value']].set_index(
        'config').T.to_dict('records')[0]
    pipe_copy.set_params(**params)

    delete_indexes = []
    for s in range(len(pipe_copy.steps)):
        if pipe_copy.steps[s][0] not in d['block'].unique():
            delete_indexes.append(s)

    pipe_copy.steps = [i for j, i in enumerate(
        pipe_copy.steps) if j not in delete_indexes]
    pipes.append(pipe_copy)

# grid = GridSearchCV(pipeline, param_grid)
# grid.fit(X_train, y_train)

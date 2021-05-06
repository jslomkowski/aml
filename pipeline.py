from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np
from itertools import product
import itertools
import pandas as pd
from copy import deepcopy
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Binarizer
from sklearn.ensemble import RandomForestRegressor


def _alt_validate_steps(self):
    names, estimators = zip(*self.steps)

    # validate names
    self._validate_names(names)

    # validate estimators
    transformers = estimators[:-1]

    for t in transformers:
        if t is None or t == 'passthrough':
            continue


class MyPipe(Pipeline):
    pass


MyPipe._validate_steps = _alt_validate_steps


pipe = MyPipe([
    ('imp', SimpleImputer(strategy=('mean', 'most_frequent'))),
    ('imp2', KNNImputer(n_neighbors=(6, 7))),
    ('sca', StandardScaler()),
    ('sca2', Binarizer()),
    ('mod', DecisionTreeRegressor()),
    ('mod2', RandomForestRegressor())
])


cfg = []
for k, v in pipe.get_params().items():
    if k.find('__') > 0:
        if type(pipe.get_params()[k]) == tuple:
            for i in range(len(v)):
                cfg.append([k, v[i]])
        else:
            cfg.append([k, v])

cfg = pd.DataFrame(cfg, columns=['config', 'value'])
cfg[['block', 'config']] = cfg['config'].str.split('__', 1, expand=True)
cfg['block2'] = cfg['block'].str.replace('\d+', '', regex=True)


cfg2 = cfg[['block', 'block2']].drop_duplicates()

prods = []
for _, d in cfg2.groupby("block2"):
    prods.append([s for _, s in d.iterrows()])
dfs = [pd.concat(ss, axis=1).T for ss in product(*prods)]

configs = []
for i in range(len(dfs)):
    t = pd.merge(dfs[i], cfg, how='left', on='block')[
        ['block', 'config', 'value']]
    configs.append(t)

dfs_list = []
for c in configs:
    prods = []
    for _, d in c.groupby("config"):
        prods.append([s for _, s in d.iterrows()])
    dfs = [pd.concat(ss, axis=1).T for ss in product(*prods)]
    dfs_list = dfs_list + dfs

pipes = []

for d in dfs_list:
    d['config'] = d['block'] + '__' + d['config']
    pipe_copy = deepcopy(pipe)
    params = d[['config', 'value']].set_index('config').T.to_dict('records')[0]
    pipe_copy.set_params(**params)

    delete_indexes = []
    for s in range(len(pipe_copy.steps)):
        if pipe_copy.steps[s][0] not in d['block'].unique():
            delete_indexes.append(s)

    pipe_copy.steps = [i for j, i in enumerate(
        pipe_copy.steps) if j not in delete_indexes]
    pipes.append(pipe_copy)

pipes

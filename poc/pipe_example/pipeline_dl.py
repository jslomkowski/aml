# to istnieje dla tego że pipeline oryginalny nie można przesyłać dowolnej
# wybranej przez siebie konfiguracji dla estymatorów lub transformerów np.:
# EqualFrequencyDiscretiser((50,100,150)) się wywali dlatego że to tuple a
# musi być int. Zadanie polega na tym żeby zrobić pipeline ele bez wartości.
# Wartości będą podane później w słowniku config

# import math
# import collections
import datetime
import itertools
# import os
# import random
# import re
# import string
# from copy import deepcopy
from itertools import product

# import numpy as np
import pandas as pd
# from joblib import dump
# from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.datasets import load_boston
# from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
# from tensorflow.keras.models import Sequential

# from monkey import _validate_steps

# from tensorflow.keras.layers import Dense, Dropout, Input


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

timenow = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')


class NoTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input, y=None):
        return self

    def transform(self, input, y=None):
        return input * 1


config = {
    'scal1__with_mean': [True, False],
    'layer_a1__units': [10, 15],
    'layer_a1__activation': ['relu', 'sigmoid'],
}


pipeline = Pipeline([
    ('scal1', StandardScaler()),
    ('layer_a1', Dense(50)),
    ('layer_a2', NoTransformer()),
    ('layer_b1', Dense(50)),
    ('layer_b2', NoTransformer()),
    ('layer_c1', Dense(1))
])

steps_lst = pd.DataFrame(pipeline.get_params()[
                         'steps'], columns=['block', 'class'])
steps_lst['block_clean'] = steps_lst['block'].str.replace(
    '\d+', '', regex=True)
steps_lst_trim = steps_lst[['block', 'block_clean']].drop_duplicates()

prods = []
for _, d in steps_lst_trim.groupby("block_clean"):
    prods.append([s for _, s in d.iterrows()])
dfs = [pd.concat(ss, axis=1).T.sort_index() for ss in product(*prods)]


pipelines = []
for i in range(len(dfs)):
    t = pd.merge(dfs[i], steps_lst, how='left', on='block')[
        ['block', 'class']]
    t['class_str'] = t['class'].astype(str)
    t = t.drop(t[t['class_str'] == 'NoTransformer()'].index)
    t = t[['block', 'class']]
    pipelines.append(t)

pipelines[0]
#########################################################################

config = {
    'scal1__with_mean': [True, False],
    'layer_a1__units': [10, 15],
    'layer_a1__activation': ['relu', 'sigmoid']
}


def _product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


configs = list(_product_dict(**config))

# build custom config for specific pipeline
ready_pipes = []
for p in pipelines:
    p = pipelines[-1]
    c = pd.DataFrame(configs).T.reset_index()
    c[['block', 'config']] = c['index'].str.split('__', expand=True)
    keys = list(p.columns.values)[0]
    i1 = c.set_index(keys).index
    i2 = p.set_index(keys).index
    c = c[i1.isin(i2)].T.drop_duplicates().T
    c.index = c['index']
    c = c.drop(['index', 'block', 'config'], axis=1).T
    c = c.to_dict('records')

    # convert single config to dataframe ready to be merged with pipeline
    for c1 in c:
        # c1 = c[1]
        cfg = pd.DataFrame(c1, index=[0]).T.reset_index()
        cfg[['block', 'config']] = cfg['index'].str.split('__', 1, expand=True)
        cfg['value'] = cfg.apply(lambda row: {row['config']: row[0]}, axis=1)
        cfg = cfg[['block', 'value']]
        cfg = cfg.groupby('block')['value'].apply(list)
        # layers in neural network neet to be in one dictionary
        for y in range(len(cfg)):
            x = cfg[y]
            res = {}
            for d in x:
                res.update(d)
            cfg[y] = res
        cfg = cfg.reset_index()

        # merge pipeline with config tailored for that pipeline
        ready_pipe = pd.merge(p, cfg, how='left', on='block')

        # iterate through pipeline dataframe and change config to new one
        for i in range(len(ready_pipe)):
            # i = 1
            if type(ready_pipe.loc[i, :]['value']) == dict:
                if "sklearn" in str(type(ready_pipe.iloc[i]['class'])):
                    ready_pipe.iloc[i]['class'].set_params(
                        **ready_pipe.iloc[i]['value'])
                elif "tensorflow" in str(type(ready_pipe.iloc[i]['class'])):
                    org_cfg = ready_pipe['class'][i].get_config()
                    replace = ready_pipe['value'][i]
                    new_cfg = {key: replace.get(
                        key, org_cfg[key]) for key in org_cfg}
                    ready_pipe['class'][i] = ready_pipe['class'][i].from_config(
                        new_cfg)
        print(ready_pipe[['block', 'class']])
        ready_pipes.append(ready_pipe[['block', 'class']])

ready_pipes[0]['class'][0].get_params()
ready_pipes[2]['class'][1].get_config()
ready_pipe['class'][1].get_config()
ready_pipe['class'][2]


# from aml import MakePipeline
from itertools import product
import collections
import datetime
import itertools
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
# from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.models import Sequential

# from aml.config.validation import (validate_model_config,
#                                    validate_transformer_config,
#                                    validate_yml_config, validate_config)
from config.connector import default_connector


class MakePipeline():

    def _what_time_is_it(self):
        return datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')

    # open user yml
    def read_user_yml(self):
        with open(Path(__file__).parents[1] / 'aml_config.yml') as file:
            cfg = yaml.full_load(file)
        return cfg

    def _flatten(self, d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            if v is None:
                v = 'None'
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(self._flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _to_list(self, cfg):
        for k in cfg.keys():
            if type(cfg[k]) != list:
                cfg[k] = [cfg[k]]
        return cfg

    def _product_dict(self, **kwargs):
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in itertools.product(*vals):
            yield dict(zip(keys, instance))

    def _connect_values_with_keys(self, cfg):
        cfg2 = []
        for i in cfg:
            cfg2_dict = {}
            for k, v in i.items():
                if k.find('_') == -1:
                    new_key = (k + '_' + i[k])
                    cfg2_dict.update({new_key: 'default'})
                else:
                    cfg2_dict.update({k: v})
            cfg2.append(cfg2_dict)
        return cfg2

    def _wh_list(self, cfg):
        wh_list = []
        for j in cfg:
            d = {}
            for i in j.items():
                t_class = '_'.join(i[0].split("_")[0:2])
                d.setdefault(t_class)
                fd = {k: v for k, v in j.items() if t_class in k}
                df = {}
                for i in fd.items():
                    try:
                        param_name = i[0].split("_")[2:]
                        if len(param_name) > 1:
                            param_name = ['_'.join(param_name)]
                        df.setdefault(param_name[0], i[1])
                    except IndexError:
                        param_name = i[0].split("_")[2:3]
                d[t_class] = df
            wh_list.append(d)
        return wh_list

    def _final_list(self, wh_list):
        final_list = []
        for w in wh_list:
            df = pd.DataFrame([w.keys(), w.values()],
                              index=['keys', 'values']).T
            df[['block', 'name']] = df['keys'].str.split('_', expand=True)
            prods = []
            for _, d in df.groupby("block"):
                prods.append([s for _, s in d.iterrows()])
            dfs = [pd.concat(ss, axis=1).T for ss in product(*prods)]
            for d in dfs:
                d = d[['block', 'name', 'values']]
                d = d.sort_index()
                final_list.append(d)
        return final_list

    # join user settings with default settings and add classes
    def _merge_usser_dict_connector(self, cfg, connector):
        for i in cfg:
            for j in i.itertuples():
                i.at[j[0], 'clas'] = connector[j[2]]
        return cfg


self = MakePipeline()

connector = default_connector()

timenow = self._what_time_is_it()

cfg = self.read_user_yml()

# cfg = validate_config(cfg)

cfg = self._flatten(cfg)

cfg = self._to_list(cfg)

cfg = list(self._product_dict(**cfg))

cfg = self._connect_values_with_keys(cfg)

cfg = self._wh_list(cfg)

cfg = self._final_list(cfg)

cfg = self._merge_usser_dict_connector(cfg, connector)

for c in cfg:
    print(c)
    for i in c.itertuples():
        print(i)
        i.clas.set_params(**i.values)

c[-1]['clas'][3].set_params(**c[-1]['values'][3])

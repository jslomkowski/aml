
from sklearn.datasets import load_boston
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
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from aml.config.validation import (validate_model_config,
                                   validate_transformer_config,
                                   validate_yml_config)
from aml.config.connector import default_connector


class MakePipeline():

    def __init__(self, arch):
        self.arch = arch

    def _net(self, X_train, hidden_layers, neurons1, neurons2, neurons3):

        model = Sequential()
        model.add(
            Dense(neurons1, input_dim=X_train.shape[1], activation='relu'))
        if hidden_layers >= 2:
            model.add(Dense(neurons2, activation='relu'))
        if hidden_layers >= 3:
            model.add(Dense(neurons3, activation='relu'))
        model.add(Dense(1, activation='relu'))
        return model

    def _model_name(self, arch, m, timenow):

        if self.arch == 'net':
            name = timenow
            for i in list(m.values()):
                name = name + ' ' + i[0]

        elif self.arch == 'sk':
            name = m[2]

        return name

    def _net_err_lst(self, history):

        hist_lst = list(history.history.keys())
        if len(hist_lst) == 2:
            err_train_loss = history.history[hist_lst[0]][-1]
            err_test_loss = history.history[hist_lst[1]][-1]
            err_train_metrics = float('NaN')
            err_test_metrics = float('NaN')

        else:
            err_train_loss = history.history[hist_lst[0]][-1]
            err_test_loss = history.history[hist_lst[2]][-1]
            err_train_metrics = history.history[hist_lst[1]][-1]
            err_test_metrics = history.history[hist_lst[3]][-1]
        err_lst = {'err_train_loss': round(err_train_loss, 2),
                   'err_train_metrics': round(err_train_metrics, 2),
                   'err_test_loss': round(err_test_loss, 2),
                   'err_test_metrics': round(err_test_metrics, 2)}
        return err_lst

    def _sk_err_lst(self, m, y_train, y_pred_train, y_test, y_pred_test,
                    y_pred_train_opt=None, y_pred_test_opt=None):
        print(m)
        metrics = m['metrics'][1]
        err_train_metrics = metrics(y_train, y_pred_train)
        err_test_metrics = metrics(y_test, y_pred_test)
        try:
            err_train_metrics_opt = metrics(y_train, y_pred_train_opt)
            err_test_metrics_opt = metrics(y_test, y_pred_test_opt)
        except TypeError:
            err_train_metrics_opt = float('NaN')
            err_test_metrics_opt = float('NaN')

        err_lst = {'err_train_metrics': round(err_train_metrics, 2),
                   'err_test_metrics': round(err_test_metrics, 2),
                   'err_train_metrics_opt': round(err_train_metrics_opt, 2),
                   'err_test_metrics_opt': round(err_test_metrics_opt, 2)}
        return err_lst

    def _save_results_report(self, res, timenow):

        if not os.path.exists('results'):
            os.makedirs('results')
        res = pd.DataFrame(res).sort_values('err_test_metrics')
        res.to_excel('results/res_list_' + timenow + '.xlsx', index=False)

    # def _optimize_model(self, m, y_train, y_pred_train, y_pred_test):

    #     metrics = m['metrics'][1]
    #     optimize = m['optimize'][1]
    #     if optimize:
    #         def objective(x):
    #             return metrics(y_train, y_pred_train * x)
    #         x0 = 1.0
    #         sol = minimize(objective, x0)
    #         weight = round(sol.x[0], 2)
    #         y_pred_train_opt = y_pred_train * weight
    #         y_pred_test_opt = y_pred_test * weight

    #     else:
    #         y_pred_train_opt, y_pred_test_opt, weight = 0, 0, float('NaN')

    #     return y_pred_train_opt, y_pred_test_opt, weight

    def _save_log(self, name):

        if not os.path.exists('tb_logs'):
            os.makedirs('tb_logs')
        logdir = 'tb_logs/' + name
        return logdir

    def _net_fit_predict(self, m, name, X_train, y_train, X_test, y_test):

        tb = TensorBoard(log_dir=self._save_log(name),
                         histogram_freq=0,
                         profile_batch=0)
        es = EarlyStopping(monitor='val_loss',
                           mode='auto',
                           verbose=0,
                           patience=5)

        # if user specified less than 3 neurons remove excess
        if m['layers'][1] == 1:
            m['neurons2'] = [None, None]
            m['neurons3'] = [None, None]
        if m['layers'][1] == 2:
            m['neurons3'] = [None, None]

        model = self._net(X_train, m['layers'][1], m['neurons1'][1],
                          m['neurons2'][1], m['neurons3'][1])
        model.compile(optimizer=m['optimizer'][1],
                      loss=m['loss'][1],
                      metricss=[m['metrics'][1]])

        fit_time = time.time()
        history = model.fit(X_train, y_train,
                            batch_sizes=m['batch_sizes'][1],
                            epochs=m['epochs'][1],
                            verbose=1,
                            validation_data=(X_test, y_test),
                            callbacks=[
                                tb,
                                es
                            ],
                            )
        fit_time = round(time.time() - fit_time, 0)
        return fit_time, history, model

    def _sk_fit_predict(self, m, X_train, y_train, X_test):

        fit_time = time.time()
        m[4].fit(X_train, y_train)
        fit_time = round(time.time() - fit_time, 0)
        y_pred_train = m[4].predict(X_train)
        y_pred_test = m[4].predict(X_test)
        return fit_time, y_pred_train, y_pred_test

    # join user settings with default settings and add classes
    def _merge_usser_dict_w_default_config(self, cfg, connector):
        for i in cfg:
            for j in i.itertuples():
                i.at[j[0], 'clas'] = connector[j[2]]
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

    def _product_dict(self, **kwargs):

        # to jest scikit learn
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in itertools.product(*vals):
            yield dict(zip(keys, instance))

    def _to_list(self, cfg):

        for k in cfg.keys():
            if type(cfg[k]) != list:
                cfg[k] = [cfg[k]]
        return cfg

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

    def _model_and_transformer_settings(self):

        connector = default_connector()

        # open user yml
        with open(Path(__file__).parents[1] / 'aml_config.yml') as file:
            cfg = yaml.full_load(file)

        # # validate dictionary for errors
        # validate_yml_config(yml_model_config)
        # yml_model_config = validate_model_config(
        #     yml_model_config, model_default_config, self.arch)
        # yml_transformer_config = validate_transformer_config(
        #     yml_transformer_config, trans_default_config)

        cfg = self._flatten(cfg)
        cfg = self._to_list(cfg)
        cfg = list(self._product_dict(**cfg))
        cfg = self._connect_values_with_keys(cfg)
        cfg = self._wh_list(cfg)
        cfg = self._final_list(cfg)

        # connects user dict with default dict
        cfg = self._merge_usser_dict_w_default_config(
            cfg, connector)

        # TODO trzeba ztrobić jakiś test czy wszędzie są pary nazwa -> obiekt

        return cfg

    def _fit_transformers(self, value, X_train, y_train, X_test):

        transformer = value[4]
        transformer.fit(X_train, y_train)
        X_train_index_col = [X_train.index, X_train.columns]
        X_test_index_col = [X_test.index, X_test.columns]
        X_train = transformer.transform(X_train)
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train,
                                   index=X_train_index_col[0],
                                   columns=X_train_index_col[1])
        X_test = transformer.transform(X_test)
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test,
                                  index=X_test_index_col[0],
                                  columns=X_test_index_col[1])
        del transformer
        return X_train, X_test

    def evaluate(self, X, y):

        cfg = self._model_and_transformer_settings()

        timenow = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')

        res = []
        for t in cfg:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=0)

            for value in t.loc[t['block'] != 'models'].itertuples():
                X_train, X_test = self._fit_transformers(
                    value, X_train, y_train, X_test)

            for m in t.loc[t['block'] == 'models'].itertuples():
                print(m)
                name = self._model_name(self.arch, m, timenow)

                if self.arch == 'net':

                    # fit model and make predictions
                    fit_time, history, model = self._net_fit_predict(
                        m, name, X_train, y_train, X_test, y_test)

                    # create list with errors
                    err_lst = self._net_err_lst(history)

                    # build case
                    case = {
                        'loss': m['loss'][0],
                        'metrics': m['metrics'][0],
                        'optimizer': m['optimizer'][0],
                        'batch_sizes': m['batch_sizes'][1],
                        'epochs': m['epochs'][1],
                        'transformations': transformations,
                        'fit_time_sec': fit_time,
                        'err_train_loss': err_lst['err_train_loss'],
                        'err_train_metrics': err_lst['err_train_metrics'],
                        'err_test_loss': err_lst['err_test_loss'],
                        'err_test_metrics': err_lst['err_test_metrics'],
                        'test/train ratio': round(err_lst['err_test_metrics'] / err_lst['err_train_metrics'], 2) if err_lst['err_train_metrics'] else 0,
                        'layers': m['layers'][1],
                        'neurons1': m['neurons1'][1],
                        'neurons2': m['neurons2'][1],
                        'neurons3': m['neurons3'][1]
                    }

                elif self.arch == 'sk':

                    # fit model and make predictions
                    fit_time, y_pred_train, y_pred_test = self._sk_fit_predict(
                        m, X_train, y_train, X_test)

                    # # optimize model
                    # y_pred_train_opt, y_pred_test_opt, weight = self._optimize_model(
                    #     m, y_train, y_pred_train, y_pred_test)

                    # create list with errors
                    err_lst = self._sk_err_lst(m, y_train, y_pred_train, y_test,
                                               y_pred_test)
                    err_lst['weight'] = weight

                    # build case
                    case = {
                        'model_name': name,
                        'transformations': transformations,
                        'metrics': m['metrics'][0],
                        'fit_time_sec': fit_time,
                        'err_train_metrics': err_lst['err_train_metrics'],
                        'err_test_metrics': err_lst['err_test_metrics'],
                        'err_train_metrics_opt': err_lst['err_train_metrics_opt'],
                        'err_test_metrics_opt': err_lst['err_test_metrics_opt'],
                        'test/train ratio': round(err_lst['err_test_metrics'] / err_lst['err_train_metrics'], 2) if err_lst['err_train_metrics'] else 0,
                        'weight': err_lst['weight']
                    }

                # TODO add save model functionality
                res.append(case)
        self._save_results_report(res, timenow)


mp = MakePipeline('sk')

X = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
y = pd.Series(load_boston().target, name='TARGET')

mp.evaluate(X, y)

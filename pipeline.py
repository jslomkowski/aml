from joblib import dump
import string
import random
import os
import datetime
import numpy as np
from itertools import product
import pandas as pd
from copy import deepcopy
from sklearn.pipeline import Pipeline


def _alt_validate_steps(self):
    names, estimators = zip(*self.steps)

    # validate names
    self._validate_names(names)

    # validate estimators
    transformers = estimators[:-1]

    for t in transformers:
        if t is None or t == 'passthrough':
            continue


class AMLPipeline(Pipeline):
    def __init__(self, pipeline, metric, save_performance=False,
                 save_values=False, save_pipelines=False):
        self.pipeline = Pipeline(pipeline)
        self.metric = metric
        self.save_performance = save_performance
        self.save_values = save_values
        self.save_pipelines = save_pipelines
        self.timenow = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')

    def _make(self):
        cfg = []
        for k, v in self.pipeline.get_params().items():
            if k.find('__') > 0:
                if type(self.pipeline.get_params()[k]) == tuple:
                    for i in range(len(v)):
                        cfg.append([k, v[i]])
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
            pipe_copy = deepcopy(self.pipeline)
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

        return pipes

    def _save_report(self, report, report_name):
        if not os.path.exists('reports'):
            os.mkdir('reports')
        report.to_csv(
            'reports/' + self.timenow + '_' + report_name + '.csv', index=False)

    def fit(self, X, y):
        self.pipe_lst = []
        self.fit_time_list = []
        pipes = self._make()
        for p in pipes:
            then = datetime.datetime.now()
            p.fit(X, y)
            now = datetime.datetime.now()
            time_delta = now - then
            minutes, seconds = time_delta.seconds // 60 % 60, time_delta.seconds
            fit_time = str(minutes) + ':' + str(seconds)
            self.pipe_lst.append(p)
            self.fit_time_list.append(fit_time)
        return self

    def validate(self, X_train, y_train,
                 X_test=None, y_test=None,
                 X_val=None, y_val=None):

        scores = np.array([])
        pipes = []
        pipe_names = []
        preds = []
        values_report = []  # ! TODO

        # train pipelines on data provided
        for p in self.pipe_lst:
            y_pred_train = p.predict(X_train)
            preds.append(y_pred_train)
            _scores_train = round(self.metric(y_train, y_pred_train), 2)
            scores = np.append(scores, _scores_train)
            if X_test is not None:
                y_pred_test = p.predict(X_test)
                preds.append(y_pred_test)
                _scores_test = round(self.metric(y_test, y_pred_test), 2)
                scores = np.append(scores, _scores_test)
            if X_val is not None:
                y_pred_val = p.predict(X_val)
                preds.append(y_pred_val)
                _scores_val = round(self.metric(y_val, y_pred_val), 2)
                scores = np.append(scores, _scores_val)
            pipes.append(p)
            letters = string.ascii_lowercase
            pipe_name = ''.join(random.choice(letters) for i in range(10))
            pipe_names.append(pipe_name)
            if self.save_pipelines:
                if not os.path.exists('pipes'):
                    os.mkdir('pipes')
                dump(p, f'pipes/{pipe_name}.joblib')

        # build performance report based on scores, pipes and fit time
        scores = scores.reshape(len(self.pipe_lst), -1)
        scores_names = ['train_score', 'test_score',
                        'val_score'][:scores.shape[1]]
        scores = pd.DataFrame(scores, columns=scores_names)
        scores['train_test_dif'] = round(
            scores['train_score'] / scores['test_score'], 2)
        scores['train_val_dif'] = round(
            scores['train_score'] / scores['val_score'], 2)
        pipes = pd.Series(pipes, name='pipeline')
        pipe_names = pd.Series(pipe_names, name='pipe_name')
        fit_time = pd.Series(self.fit_time_list, name='fit_time(H:M)')
        performance_report = pd.concat(
            [pipe_names, pipes, fit_time, scores], axis=1)

        # save if needed
        if self.save_performance is True:
            self._save_report(performance_report, 'performance_report')
        if self.save_values is True:
            self._save_report(values_report, 'values_report')

        return performance_report


Pipeline._validate_steps = _alt_validate_steps

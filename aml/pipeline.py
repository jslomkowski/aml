import datetime
import os
import itertools
import random
import string
import time
from copy import deepcopy

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from aml.config_template import config_dict


def _validate_steps(self):
    names, estimators = zip(*self.steps)
    self._validate_names(names)
    transformers = estimators[:-1]
    for t in transformers:
        if t is None or t == 'passthrough':
            continue


Pipeline._validate_steps = _validate_steps

# self = AMLGridSearchCV(pipeline, param_grid)


class AMLGridSearchCV:

    def __init__(self, pipeline, param_grid, scoring=None):
        self.pipeline = pipeline
        self.param_grid = param_grid
        if scoring is None:
            self.scoring = mean_absolute_error

    def _models_template_check(self, pipeline):
        # ! TODO
        """Checks if user has provided nested sk-learn classes, for example
        nested models from models_template
        Returns:
            [list]: list with unpacked pipeline steps.
        """
        pipeline_steps_list = []
        for p in pipeline.steps:
            if isinstance(p, list):
                for i in p:
                    pipeline_steps_list.append(i)
            else:
                pipeline_steps_list.append(p)
        return pipeline_steps_list

    def _check_def_config(self, pipeline_steps_list, param_grid):
        """Checks if user provided '*' symbol for whole param_grid or one of
        it's classes. If so, default param_dict from config_template.py will be
        used for whole dictionary or just that class
        Example for one class:

            param_grid = {
                'disc1__q': [5, 15],
                'model2__*': []
            }

        Here, only model2__ will have its default config dictionary found in
        config_template.py

        Example for whole config_dict:

        param_grid = {'*'}

        Here, _check_def_config mehod will find all classes provided by the
        user in pipeline and all of their hyperparameters will be
        extracted from config_template.py

        Args:
            pipeline_steps_list (list): list of pipeline steps from
            _models_template_check list
            param_grid (dictionary): dictionary with hyperparameters

        Returns:
            [dictionary]: dictionary with hyperparameters
        """
        # ! todo

        # Checks if user has provided a star in param_dict.
        is_star_list = []
        for k in param_grid:
            if len(k) > 1 and k[-1] == '*':
                is_star_list.append(k.split('__')[0])
            elif k == '*':
                for p in pipeline_steps_list:
                    is_star_list.append(p[0])

        # If so then find class name for that class that has star attached to
        # it.
        search_list = {}
        for p in pipeline_steps_list:
            if p[0] in is_star_list:
                search_list[p[0]] = str(p[1].__class__)[8:][:-2]

        # For that class you found, search for its parameters in
        # config_template.py and attach those to a new dict
        param_grid_mod = {}
        for k, v in search_list.items():
            try:
                for c in config_dict[v]:
                    param_grid_mod[k + '__' + c] = config_dict[v][c]
            except KeyError:
                print(
                    f'Unable to find config for {k} in config_template')
                continue

        # Find and delete config with * and update param_grid_mod to param_grid
        if param_grid != {'*'}:
            for kp in list(param_grid.keys()):
                if kp[:-3] in search_list.keys():
                    del param_grid[kp]
            param_grid.update(param_grid_mod)
        else:
            param_grid = param_grid_mod
        return param_grid

    def _make_aml_combinations(self, pipeline, param_grid):

        pipeline_steps_list = self._models_template_check(pipeline)
        param_grid = self._check_def_config(pipeline_steps_list, param_grid)

        fd = {}
        st = dict(pipeline_steps_list)
        ts = {v: k for k, v in st.items()}
        for k, v in st.items():
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
            for k, v in list(p.items()):
                p[ts[v]] = p.pop(k)

        final_pipes = []
        for pipe_dict in pipelines_dict:
            pipe = Pipeline([(k, v) for k, v in pipe_dict.items()])
            clone_grid = deepcopy(param_grid)
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

    def _worker(self, final_pipes, X_train, y_train, X_test=None, y_test=None):
        results = []
        now = time.time()
        final_pipes.fit(X_train, y_train)
        run_time = int(time.time() - now)
        y_pred_train = final_pipes.predict(X_train)
        if X_test is not None:
            y_pred_test = final_pipes.predict(X_test)
        letters = string.ascii_lowercase
        pipe_name = ''.join(random.choice(letters) for i in range(10))
        error_train = self.scoring(y_train, y_pred_train)
        if X_test is not None:
            error_test = self.scoring(y_test, y_pred_test)
        else:
            error_test = np.nan
        res = {'name': pipe_name,
               'params': final_pipes.named_steps,
               'train_time (sec)': run_time,
               'error_train': round(error_train, 2),
               'error_test': round(error_test, 2),
               'train_test_dif': round(error_test / error_train, 2),
               }
        results.append(res)
        return results

    def fit(self, X_train, y_train, X_test=None, y_test=None, n_jobs=None,
            prefer='processes', save_report=True, report_format='xlsx'):
        results = Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(self._worker)(i, X_train, y_train, X_test, y_test) for i in
            self._make_aml_combinations(self.pipeline, self.param_grid))
        results = pd.DataFrame.from_dict([i[0] for i in results])
        if save_report:
            today = datetime.datetime.now()
            today = today.strftime("%Y-%m-%d %H-%M-%S")
            if not os.path.exists('aml_reports/'):
                os.mkdir('aml_reports/')
            if report_format == 'csv':
                results.to_csv(f'aml_reports/{today}.csv', index=False)
            elif report_format == 'xlsx':
                results.to_excel(f'aml_reports/{today}.xlsx')
            else:
                print(f'{report_format} - report format unrecognised.')
        return results

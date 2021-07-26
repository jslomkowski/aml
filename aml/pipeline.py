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
# from aml import config_dict


def _validate_steps(self):
    names, estimators = zip(*self.steps)
    self._validate_names(names)
    transformers = estimators[:-1]
    for t in transformers:
        if t is None or t == 'passthrough':
            continue


Pipeline._validate_steps = _validate_steps

self = AMLGridSearchCV(pipeline, param_grid)


class AMLGridSearchCV:

    def __init__(self, pipeline, param_grid, scoring=None):
        self.pipeline = pipeline
        self.param_grid = param_grid
        if scoring is None:
            self.scoring = mean_absolute_error

    def _models_template_check(self, pipeline):
        """Checks if user has provided nested sk-learn classes, for example
        nested models from models_template
        Returns:
            [list]: list with unpacked pipeline steps.
        """
        pipeline_steps_list = []
        for p in pipeline.steps:
            if isinstance(p, list):
                for _ in p:
                    pipeline_steps_list.append(_)
            else:
                pipeline_steps_list.append(p)
        return pipeline_steps_list

    # def _check_def_config(self, pipeline_steps_list, param_grid):
    #     # ! todo
    #     something_list = []
    #     for k in param_grid:
    #         if len(k) > 1 and k[-1] == '*':
    #             something_list.append(k.split('__')[0])
    #         elif k == '*':
    #             for p in pipeline_steps_list:
    #                 something_list.append(p[0])

    #     search_list = []
    #     for p in pipeline_steps_list:
    #         if p[0] in something_list:
    #             search_list.append(str(p[1].__class__)[8:][:-2])

    #     param_grid_mod = {}
    #     for s in search_list:
    #         print(s)
    #         s = search_list[2]
    #         try:
    #             for c in config_dict[s]:
    #                 print(c)
    #                 param_grid_mod[pipeline_steps_list[0] + '__' + c] = config_dict[s][c]
    #         except KeyError:
    #             continue

    #     return param_grid_mod

    def _make_aml_combinations(self, pipeline, param_grid):

        pipeline_steps_list = self._models_template_check(pipeline)
        # param_grid = self._check_def_config(pipeline_steps_list, param_grid)

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
        final_pipes.fit(X_train, y_train)
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
               'error_train': round(error_train, 2),
               'error_test': round(error_test, 2),
               'train_test_dif': round(error_test / error_train, 2),
               }
        results.append(res)
        return results

    def fit(self, X_train, y_train, X_test=None, y_test=None, n_jobs=None,
            prefer='processes'):
        now = time.time()
        results = Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(self._worker)(i, X_train, y_train, X_test, y_test) for i in
            self._make_aml_combinations(self.pipeline, self.param_grid))
        results = pd.DataFrame.from_dict([i[0] for i in results])
        print(time.time() - now)
        return results

from joblib import Parallel, delayed
import time
import numpy as np
import random
import string
from sklearn.metrics import mean_absolute_error
import itertools
from copy import deepcopy

import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline


def _validate_steps(self):
    names, estimators = zip(*self.steps)
    self._validate_names(names)
    transformers = estimators[:-1]
    for t in transformers:
        if t is None or t == 'passthrough':
            continue


Pipeline._validate_steps = _validate_steps


class AMLGridSearchCV:

    def __init__(self, pipeline, param_grid, scoring=None):
        self.pipeline = pipeline
        self.param_grid = param_grid
        if scoring is None:
            self.scoring = mean_absolute_error

    def _make_aml_combinations(self, pipeline, param_grid):
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

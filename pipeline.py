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
    def __init__(self, pipeline):
        self.pipeline = Pipeline(pipeline)

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

    def fit(self, X, y):
        self.lst = []
        pipes = self._make()
        for p in pipes:
            p.fit(X, y)
            self.lst.append(p)
        return self

    def validate(self, X):
        for p in self.lst:
            print(p.predict(X))


Pipeline._validate_steps = _alt_validate_steps

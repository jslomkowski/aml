from copy import deepcopy
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# pipe = Pipeline([
#     ('imp', SimpleImputer()),
#     ('imp2', KNNImputer()),
#     ('sca', StandardScaler()),
#     ('dtr', DecisionTreeRegressor(min_samples_leaf=(
#         4, 6, 8), splitter=('best', 'random')))
# ])


pipe = Pipeline([
    ('imp', SimpleImputer(strategy=('mean', 'most_frequent'))),
    ('imp2', KNNImputer(n_neighbors=(6, 7))),
    # ('sca', StandardScaler()),
    ('mod', DecisionTreeRegressor(min_samples_leaf=(
        6, 8), splitter=('besd', 'random'))),
    ('mod2', XGBRegressor(n_estimators=(98, 99)))
])

cfg = []
for k, v in pipe.get_params().items():
    if type(pipe.get_params()[k]) == tuple:
        for i in range(len(v)):
            cfg.append([k, v[i]])

cfg[0]


pipes = []

for k in pipe.get_params().keys():
    print(k)
    # if type(pipe.get_params()[k]) == range:
    #     for i in list(pipe.get_params()[k]):
    #         pipe_copy = deepcopy(pipe)
    #         params = {k: i}
    #         pipe_copy.set_params(**params)
    #         pipes.append(pipe_copy)
    if type(pipe.get_params()[k]) == tuple:
        print('y')
        for i in list(pipe.get_params()[k]):
            print(i)
            pipe_copy = deepcopy(pipe)
            params = {k: i}
            pipe_copy.set_params(**params)
            pipes.append(pipe_copy)

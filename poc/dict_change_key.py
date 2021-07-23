
import inspect
import sys
import numpy as np

regressor_config_dict = {

    'sklearn.linear_model.ElasticNetCV': {
        'l1_ratio': np.arange(0.0, 1.01, 0.05),
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    },

    'sklearn.ensemble.ExtraTreesRegressor': {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.GradientBoostingRegressor': {
        'n_estimators': [100],
        'loss': ["ls", "lad", "huber", "quantile"],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05),
        'alpha': [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    },

    'sklearn.ensemble.AdaBoostRegressor': {
        'n_estimators': [100],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'loss': ["linear", "square", "exponential"]
    },

    'sklearn.tree.DecisionTreeRegressor': {
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },

    'sklearn.neighbors.KNeighborsRegressor': {
        'n_neighbors': range(1, 101),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },

    'sklearn.linear_model.LassoLarsCV': {
        'normalize': [True, False]
    },

    'sklearn.svm.LinearSVR': {
        'loss': ["epsilon_insensitive", "squared_epsilon_insensitive"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1.]
    },

    'sklearn.ensemble.RandomForestRegressor': {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.linear_model.RidgeCV': {
    },

    'xgboost.XGBRegressor': {
        'n_estimators': [100],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'n_jobs': [1],
        'verbosity': [0],
        'objective': ['reg:squarederror']
    },

    'sklearn.linear_model.SGDRegressor': {
        'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
        'penalty': ['elasticnet'],
        'alpha': [0.0, 0.01, 0.001],
        'learning_rate': ['invscaling', 'constant'],
        'fit_intercept': [True, False],
        'l1_ratio': [0.25, 0.0, 1.0, 0.75, 0.5],
        'eta0': [0.1, 1.0, 0.01],
        'power_t': [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0]
    },

    # Preprocessors
    'sklearn.preprocessing.Binarizer': {
        'threshold': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.decomposition.FastICA': {
        'tol': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.cluster.FeatureAgglomeration': {
        'linkage': ['ward', 'complete', 'average'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    },

    'sklearn.preprocessing.MaxAbsScaler': {
    },

    'sklearn.preprocessing.MinMaxScaler': {
    },

    'sklearn.preprocessing.Normalizer': {
        'norm': ['l1', 'l2', 'max']
    },

    'sklearn.kernel_approximation.Nystroem': {
        'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
        'gamma': np.arange(0.0, 1.01, 0.05),
        'n_components': range(1, 11)
    },

    'sklearn.decomposition.PCA': {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },

    'sklearn.preprocessing.PolynomialFeatures': {
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [False]
    },

    'sklearn.kernel_approximation.RBFSampler': {
        'gamma': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.preprocessing.RobustScaler': {
    },

    'sklearn.preprocessing.StandardScaler': {
    },

    # Selectors
    'sklearn.feature_selection.SelectFwe': {
        'alpha': np.arange(0, 0.05, 0.001),
        'score_func': {
            'sklearn.feature_selection.f_regression': None
        }
    },

    'sklearn.feature_selection.SelectPercentile': {
        'percentile': range(1, 100),
        'score_func': {
            'sklearn.feature_selection.f_regression': None
        }
    },

    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },

    'sklearn.feature_selection.SelectFromModel': {
        'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesRegressor': {
                'n_estimators': [100],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    }

}

# for k in regressor_config_dict:
#     # print(k)
#     print(k.split('.')[-1])


# from sklearn.linear_model import ElasticNetCV
# from sklearn.ensemble import ExtraTreesRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.ensemble import AdaBoostRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.linear_model import LassoLarsCV
# from sklearn.svm import LinearSVR
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import RidgeCV
# from xgboost import XGBRegressor
# from sklearn.linear_model import SGDRegressor
# from sklearn.preprocessing import Binarizer
# from sklearn.decomposition import FastICA
# from sklearn.cluster import FeatureAgglomeration
# from sklearn.preprocessing import MaxAbsScaler
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import Normalizer
# from sklearn.kernel_approximation import Nystroem
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.kernel_approximation import RBFSampler
# from sklearn.preprocessing import RobustScaler
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import SelectFwe
# from sklearn.feature_selection import SelectPercentile
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.feature_selection import SelectFromModel

# import sys, inspect

# clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)


new_dict = {
    'AMLGridSearchCV': '__main__.AMLGridSearchCV',
    'AdaBoostRegressor': 'sklearn.ensemble._weight_boosting.AdaBoostRegressor',
    'Binarizer': 'sklearn.preprocessing._data.Binarizer',
    'DecisionTreeRegressor': 'sklearn.tree._classes.DecisionTreeRegressor',
    'ElasticNetCV': 'sklearn.linear_model._coordinate_descent.ElasticNetCV',
    'EqualFrequencyDiscretiser': 'feature_engine.discretisation.equal_frequency.EqualFrequencyDiscretiser',
    'EqualWidthDiscretiser': 'feature_engine.discretisation.equal_width.EqualWidthDiscretiser',
    'ExtraTreesRegressor': 'sklearn.ensemble._forest.ExtraTreesRegressor',
    'FastICA': 'sklearn.decomposition._fastica.FastICA',
    'FeatureAgglomeration': 'sklearn.cluster._agglomerative.FeatureAgglomeration',
    'GradientBoostingRegressor': 'sklearn.ensemble._gb.GradientBoostingRegressor',
    'KNeighborsRegressor': 'sklearn.neighbors._regression.KNeighborsRegressor',
    'LassoLarsCV': 'sklearn.linear_model._least_angle.LassoLarsCV',
    'LinearRegression': 'sklearn.linear_model._base.LinearRegression',
    'LinearSVR': 'sklearn.svm._classes.LinearSVR',
    'MaxAbsScaler': 'sklearn.preprocessing._data.MaxAbsScaler',
    'MinMaxScaler': 'sklearn.preprocessing._data.MinMaxScaler',
    'Normalizer': 'sklearn.preprocessing._data.Normalizer',
    'Nystroem': 'sklearn.kernel_approximation.Nystroem',
    'PCA': 'sklearn.decomposition._pca.PCA',
    'Parallel': 'joblib.parallel.Parallel',
    'ParameterGrid': 'sklearn.model_selection._search.ParameterGrid',
    'Pipeline': 'sklearn.pipeline.Pipeline',
    'PolynomialFeatures': 'sklearn.preprocessing._data.PolynomialFeatures',
    'RBFSampler': 'sklearn.kernel_approximation.RBFSampler',
    'RandomForestRegressor': 'sklearn.ensemble._forest.RandomForestRegressor',
    'RidgeCV': 'sklearn.linear_model._ridge.RidgeCV',
    'RobustScaler': 'sklearn.preprocessing._data.RobustScaler',
    'SGDRegressor': 'sklearn.linear_model._stochastic_gradient.SGDRegressor',
    'SelectFromModel': 'sklearn.feature_selection._from_model.SelectFromModel',
    'SelectFwe': 'sklearn.feature_selection._univariate_selection.SelectFwe',
    'SelectPercentile': 'sklearn.feature_selection._univariate_selection.SelectPercentile',
    'StandardScaler': 'sklearn.preprocessing._data.StandardScaler',
    'VarianceThreshold': 'sklearn.feature_selection._variance_threshold.VarianceThreshold',
    'XGBRegressor': 'xgboost.sklearn.XGBRegressor',
    'self': '__main__.AMLGridSearch',
}

for k in list(regressor_config_dict):
    regressor_config_dict[new_dict[k.split(
        '.')[-1]]] = regressor_config_dict[k]

# import numpy as np

# regressor_config_dict = {
#     'sklearn.preprocessing.MaxAbsScaler': {},
#     'sklearn.preprocessing.MinMaxScaler': {},
#     'sklearn.preprocessing.Normalizer': {'norm': ['l1', 'l2', 'max']},
#     'sklearn.kernel_approximation.Nystroem': {'kernel': ['rbf',
#                                                          'cosine',
#                                                          'chi2',
#                                                          'laplacian',
#                                                          'polynomial',
#                                                          'poly',
#                                                          'linear',
#                                                          'additive_chi2',
#                                                          'sigmoid'],
#                                               'gamma': array([0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
#                                                               0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]),
#                                               'n_components': range(1, 11)},
#     'sklearn.decomposition.PCA': {'svd_solver': ['randomized'],
#                                   'iterated_power': range(1, 11)},
#     'sklearn.preprocessing.PolynomialFeatures': {'degree': [2],
#                                                  'include_bias': [False],
#                                                  'interaction_only': [False]},
#     'sklearn.kernel_approximation.RBFSampler': {'gamma': array([0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
#                                                                 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.])},
#     'sklearn.preprocessing.RobustScaler': {},
#     'sklearn.preprocessing.StandardScaler': {},
#     'sklearn.feature_selection.SelectFwe': {'alpha': array([0., 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008,
#                                                             0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017,
#                                                             0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026,
#                                                             0.027, 0.028, 0.029, 0.03, 0.031, 0.032, 0.033, 0.034, 0.035,
#                                                             0.036, 0.037, 0.038, 0.039, 0.04, 0.041, 0.042, 0.043, 0.044,
#                                                             0.045, 0.046, 0.047, 0.048, 0.049]),
#                                             'score_func': {'sklearn.feature_selection.f_regression': None}},
#     'sklearn.feature_selection.SelectPercentile': {'percentile': range(1, 100),
#                                                    'score_func': {'sklearn.feature_selection.f_regression': None}},
#     'sklearn.feature_selection.VarianceThreshold': {'threshold': [0.0001,
#                                                                   0.0005,
#                                                                   0.001,
#                                                                   0.005,
#                                                                   0.01,
#                                                                   0.05,
#                                                                   0.1,
#                                                                   0.2]},
#     'sklearn.feature_selection.SelectFromModel': {'threshold': array([0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
#                                                                       0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]),
#                                                   'estimator': {'sklearn.ensemble.ExtraTreesRegressor': {'n_estimators': [100],
#                                                                                                          'max_features': array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
#                                                                                                                                 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.])}}},
#     'sklearn.linear_model._coordinate_descent.ElasticNetCV': {'l1_ratio': array([0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
#                                                                                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]),
#                                                               'tol': [1e-05, 0.0001, 0.001, 0.01, 0.1]},
#     'sklearn.ensemble._gb.GradientBoostingRegressor': {'n_estimators': [100],
#                                                        'loss': ['ls', 'lad', 'huber', 'quantile'],
#                                                        'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0],
#                                                        'max_depth': range(1, 11),
#                                                        'min_samples_split': range(2, 21),
#                                                        'min_samples_leaf': range(1, 21),
#                                                        'subsample': array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
#                                                                            0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]),
#                                                        'max_features': array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
#                                                                               0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]),
#                                                        'alpha': [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]},
#     'sklearn.ensemble._weight_boosting.AdaBoostRegressor': {'n_estimators': [100],
#                                                             'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0],
#                                                             'loss': ['linear', 'square', 'exponential']},
#     'sklearn.tree._classes.DecisionTreeRegressor': {'max_depth': range(1, 11),
#                                                     'min_samples_split': range(2, 21),
#                                                     'min_samples_leaf': range(1, 21)},
#     'sklearn.neighbors._regression.KNeighborsRegressor': {'n_neighbors': range(1, 101),
#                                                           'weights': ['uniform', 'distance'],
#                                                           'p': [1, 2]},
#     'sklearn.linear_model._least_angle.LassoLarsCV': {'normalize': [True, False]},
#     'sklearn.svm._classes.LinearSVR': {'loss': ['epsilon_insensitive',
#                                                 'squared_epsilon_insensitive'],
#                                        'dual': [True, False],
#                                        'tol': [1e-05, 0.0001, 0.001, 0.01, 0.1],
#                                        'C': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0],
#                                        'epsilon': [0.0001, 0.001, 0.01, 0.1, 1.0]},
#     'sklearn.ensemble._forest.RandomForestRegressor': {'n_estimators': [100],
#                                                        'max_features': array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
#                                                                               0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]),
#                                                        'min_samples_split': range(2, 21),
#                                                        'min_samples_leaf': range(1, 21),
#                                                        'bootstrap': [True, False]},
#     'sklearn.linear_model._ridge.RidgeCV': {},
#     'xgboost.sklearn.XGBRegressor': {'n_estimators': [100],
#                                      'max_depth': range(1, 11),
#                                      'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0],
#                                      'subsample': array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
#                                                          0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]),
#                                      'min_child_weight': range(1, 21),
#                                      'n_jobs': [1],
#                                      'verbosity': [0],
#                                      'objective': ['reg:squarederror']},
#     'sklearn.linear_model._stochastic_gradient.SGDRegressor': {'loss': ['squared_loss',
#                                                                         'huber',
#                                                                         'epsilon_insensitive'],
#                                                                'penalty': ['elasticnet'],
#                                                                'alpha': [0.0, 0.01, 0.001],
#                                                                'learning_rate': ['invscaling', 'constant'],
#                                                                'fit_intercept': [True, False],
#                                                                'l1_ratio': [0.25, 0.0, 1.0, 0.75, 0.5],
#                                                                'eta0': [0.1, 1.0, 0.01],
#                                                                'power_t': [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0]},
#     'sklearn.preprocessing._data.Binarizer': {'threshold': array([0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
#                                                                   0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.])},
#     'sklearn.decomposition._fastica.FastICA': {'tol': array([0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
#                                                              0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.])},
#     'sklearn.cluster._agglomerative.FeatureAgglomeration': {'linkage': ['ward',
#                                                                         'complete',
#                                                                         'average'],
#                                                             'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']},
#     'sklearn.preprocessing._data.MaxAbsScaler': {},
#     'sklearn.preprocessing._data.MinMaxScaler': {},
#     'sklearn.preprocessing._data.Normalizer': {'norm': ['l1', 'l2', 'max']},
#     'sklearn.decomposition._pca.PCA': {'svd_solver': ['randomized'],
#                                        'iterated_power': range(1, 11)},
#     'sklearn.preprocessing._data.PolynomialFeatures': {'degree': [2],
#                                                        'include_bias': [False],
#                                                        'interaction_only': [False]},
#     'sklearn.preprocessing._data.RobustScaler': {},
#     'sklearn.preprocessing._data.StandardScaler': {},
#     'sklearn.feature_selection._univariate_selection.SelectFwe': {'alpha': array([0., 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008,
#                                                                                   0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017,
#                                                                                   0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026,
#                                                                                   0.027, 0.028, 0.029, 0.03, 0.031, 0.032, 0.033, 0.034, 0.035,
#                                                                                   0.036, 0.037, 0.038, 0.039, 0.04, 0.041, 0.042, 0.043, 0.044,
#                                                                                   0.045, 0.046, 0.047, 0.048, 0.049]),
#                                                                   'score_func': {'sklearn.feature_selection.f_regression': None}},
#     'sklearn.feature_selection._univariate_selection.SelectPercentile': {'percentile': range(1, 100),
#                                                                          'score_func': {'sklearn.feature_selection.f_regression': None}},
#     'sklearn.feature_selection._variance_threshold.VarianceThreshold': {'threshold': [0.0001,
#                                                                                       0.0005,
#                                                                                       0.001,
#                                                                                       0.005,
#                                                                                       0.01,
#                                                                                       0.05,
#                                                                                       0.1,
#                                                                                       0.2]},
#     'sklearn.feature_selection._from_model.SelectFromModel': {'threshold': array([0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
#                                                                                   0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]),
#                                                               'estimator': {'sklearn.ensemble.ExtraTreesRegressor': {'n_estimators': [100],
#                                                                                                                      'max_features': array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
#                                                                                                                                             0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.])}}}}


g = locals()
for name, obj in g.iteritems():

modulenames = set(sys.modules) & set(globals())
allmodules = [sys.modules[name] for name in modulenames]

clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)

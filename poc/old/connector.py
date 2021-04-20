
import itertools

import sklearn
import tensorflow as tf
import xgboost
from feature_engine.discretisation import *
from feature_engine.encoding import *
from feature_engine.imputation import *
from feature_engine.outliers import *
from feature_engine.selection import *
from feature_engine.transformation import *
# from sklearn import *
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import *


def default_connector():

    connector = {

        # models
        'AdaBoostRegressor': sklearn.ensemble._weight_boosting.AdaBoostRegressor(),
        'ARDRegression': sklearn.linear_model._bayes.ARDRegression(),
        'BaggingRegressor': sklearn.ensemble._bagging.BaggingRegressor(),
        'BayesianRidge': sklearn.linear_model._bayes.BayesianRidge(),
        'CCA': sklearn.cross_decomposition._pls.CCA(),
        'DecisionTreeRegressor': sklearn.tree._classes.DecisionTreeRegressor(),
        'DummyRegressor': sklearn.dummy.DummyRegressor(),
        'ElasticNet': sklearn.linear_model._coordinate_descent.ElasticNet(),
        'ElasticNetCV': sklearn.linear_model._coordinate_descent.ElasticNetCV(n_jobs=-1),
        'ExtraTreeRegressor': sklearn.tree._classes.ExtraTreeRegressor(),
        'ExtraTreesRegressor': sklearn.ensemble._forest.ExtraTreesRegressor(n_jobs=-1),
        'GammaRegressor': sklearn.linear_model._glm.glm.GammaRegressor(),
        'GaussianProcessRegressor': sklearn.gaussian_process._gpr.GaussianProcessRegressor(),  # performance!
        'GradientBoostingRegressor': sklearn.ensemble._gb.GradientBoostingRegressor(),
        'HistGradientBoostingRegressor': sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingRegressor(),
        'HuberRegressor': sklearn.linear_model._huber.HuberRegressor(),
        'KNeighborsRegressor': sklearn.neighbors._regression.KNeighborsRegressor(n_jobs=-1),
        'KernelRidge': sklearn.kernel_ridge.KernelRidge(),  # performance!
        'Lars': sklearn.linear_model._least_angle.Lars(),
        'LarsCV': sklearn.linear_model._least_angle.LarsCV(n_jobs=-1),
        'Lasso': sklearn.linear_model._coordinate_descent.Lasso(),
        'LassoCV': sklearn.linear_model._coordinate_descent.LassoCV(n_jobs=-1),
        'LassoLars': sklearn.linear_model._least_angle.LassoLars(),
        'LassoLarsCV': sklearn.linear_model._least_angle.LassoLarsCV(n_jobs=-1),
        'LassoLarsIC': sklearn.linear_model._least_angle.LassoLarsIC(),
        'LinearRegression': sklearn.linear_model._base.LinearRegression(n_jobs=-1),
        'LinearSVR': sklearn.svm._classes.LinearSVR(),  # performance!
        'MLPRegressor': sklearn.neural_network._multilayer_perceptron.MLPRegressor(),  # performance!
        'NuSVR': sklearn.svm._classes.NuSVR(),  # performance!
        'OrthogonalMatchingPursuit': sklearn.linear_model._omp.OrthogonalMatchingPursuit(),
        'OrthogonalMatchingPursuitCV': sklearn.linear_model._omp.OrthogonalMatchingPursuitCV(n_jobs=-1),
        'PLSCanonical': sklearn.cross_decomposition._pls.PLSCanonical(),
        'PLSRegression': sklearn.cross_decomposition._pls.PLSRegression(),
        'PassiveAggressiveRegressor': sklearn.linear_model._passive_aggressive.PassiveAggressiveRegressor(),
        'PoissonRegressor': sklearn.linear_model._glm.glm.PoissonRegressor(),
        'RANSACRegressor': sklearn.linear_model._ransac.RANSACRegressor(),
        'RandomForestRegressor': sklearn.ensemble._forest.RandomForestRegressor(n_jobs=-1),
        'Ridge': sklearn.linear_model._ridge.Ridge(),
        'RidgeCV': sklearn.linear_model._ridge.RidgeCV(),
        'SGDRegressor': sklearn.linear_model._stochastic_gradient.SGDRegressor(),  # performance!
        'SVR': sklearn.svm._classes.SVR(),  # performance!
        'TheilSenRegressor': sklearn.linear_model._theil_sen.TheilSenRegressor(n_jobs=-1),
        'TransformedTargetRegressor': sklearn.compose._target.TransformedTargetRegressor(),
        'TweedieRegressor': sklearn.linear_model._glm.glm.TweedieRegressor(),
        'XGBRegressor': xgboost.XGBRegressor(n_jobs=-1),


        # metrics
        'evs': sklearn.metrics.explained_variance_score,
        'me': sklearn.metrics.max_error,
        'mse': sklearn.metrics.mean_squared_error,
        'msle': sklearn.metrics.mean_squared_log_error,
        'medae': sklearn.metrics.median_absolute_error,
        'mae': sklearn.metrics.mean_absolute_error,
        'mape': sklearn.metrics.mean_absolute_percentage_error,
        'r2': sklearn.metrics.r2_score,
        'mpd': sklearn.metrics.mean_poisson_deviance,
        'mgd': sklearn.metrics.mean_gamma_deviance,


        # optimize
        'True': True,
        'False': False,


        # losses
        'l_mean_squared_error': tf.keras.losses.MSE,
        'l_mean_absolute_error': tf.keras.losses.MAE,
        'l_mean_absolute_percentage_error': tf.keras.losses.MAPE,
        'l_mean_squared_logarithmic_error': tf.keras.losses.MSLE,
        'l_cosine_similarity': tf.keras.losses.cosine_similarity,
        'l_logcosh': tf.keras.losses.log_cosh,


        # metrics
        'm_mean_squared_error': tf.keras.metrics.MeanSquaredError(),
        'm_mean_absolute_error': tf.keras.metrics.MeanAbsoluteError(),
        'm_mean_absolute_percentage_error': tf.keras.metrics.MeanAbsolutePercentageError(),
        'm_mean_squared_logarithmic_error': tf.keras.metrics.MeanSquaredLogarithmicError(),
        'm_cosine_similarity': tf.keras.metrics.CosineSimilarity(),
        'm_logcosh': tf.keras.metrics.LogCoshError(),


        # optimizers
        'adam': tf.keras.optimizers.Adam(),
        'sgd': tf.keras.optimizers.SGD(),


        # batch_sizes
        # 1,
        # 2,
        # 4,
        # 8,
        # 16,
        # 32,
        # 64,
        # 128,
        # 256


        # epochs
        # 1,
        # 2,
        # 4,
        # 8,
        # 16,
        # 32,
        # 64,
        # 128,
        # 256


        # neurons1
        # 1,
        # 4,
        # 16,
        # 64,
        # 256


        # neurons2
        # 1,
        # 4,
        # 16,
        # 64,
        # 256


        # neurons3
        # 1,
        # 4,
        # 16,
        # 64,
        # 256


        # layers
        # 1,
        # 2,
        # 3

        # spliters
        # 'train_test_split': train_test_split,

        # imputers
        'MeanMedianImputer': MeanMedianImputer(),
        'ArbitraryNumberImputer': ArbitraryNumberImputer(),
        'EndTailImputer': EndTailImputer(),
        'CategoricalImputer': CategoricalImputer(),
        'RandomSampleImputer': RandomSampleImputer(),


        # encoders
        'OneHotEncoder': OneHotEncoder(),
        'CountFrequencyEncoder': CountFrequencyEncoder(),
        'OrdinalEncoder': OrdinalEncoder(),
        'MeanEncoder': MeanEncoder(),
        'DecisionTreeEncoder': DecisionTreeEncoder(),
        'RareLabelEncoder': RareLabelEncoder(),


        # transformers
        'LogTransformer': LogTransformer(),
        'ReciprocalTransformer': ReciprocalTransformer(),
        'BoxCoxTransformer': BoxCoxTransformer(),
        'YeoJohnsonTransformer': YeoJohnsonTransformer(),


        # discretisers
        'EqualFrequencyDiscretiser': EqualFrequencyDiscretiser(),
        'EqualWidthDiscretiser': EqualWidthDiscretiser(),
        'DecisionTreeDiscretiser': DecisionTreeDiscretiser(),


        # # outlier_handlers
        # 'Winsorizer': Winsorizer(),
        # 'OutlierTrimmer': OutlierTrimmer(),


        # scalers
        # 'FunctionTransformer': FunctionTransformer(),
        # 'StandardScaler': StandardScaler(),
        # 'Normalizer': Normalizer(),
    }

    return connector

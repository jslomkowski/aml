
import xgboost
import sklearn
import itertools
from sklearn import *


def sk_settings():

    optimize = [
        True,
        # False
    ]

    metrics = [
        # sklearn.metrics.explained_variance_score,
        # sklearn.metrics.max_error,
        sklearn.metrics.mean_absolute_error,
        # sklearn.metrics.mean_squared_error,
        # sklearn.metrics.mean_squared_log_error,
        # sklearn.metrics.median_absolute_error,
        # sklearn.metrics.mean_absolute_percentage_error,
        # sklearn.metrics.r2_score,
        # sklearn.metrics.mean_poisson_deviance,
        # sklearn.metrics.mean_gamma_deviance,
        # sklearn.metrics.mean_tweedie_devianc,
    ]

    model = [
        # sklearn.linear_model._bayes.ARDRegression(),
        # sklearn.ensemble._weight_boosting.AdaBoostRegressor(),
        # sklearn.ensemble._bagging.BaggingRegressor(),
        # sklearn.linear_model._bayes.BayesianRidge(),
        # sklearn.cross_decomposition._pls.CCA(),
        # sklearn.tree._classes.DecisionTreeRegressor(),
        # sklearn.dummy.DummyRegressor(),
        # sklearn.linear_model._coordinate_descent.ElasticNet(),
        # sklearn.linear_model._coordinate_descent.ElasticNetCV(n_jobs=-1),
        # sklearn.tree._classes.ExtraTreeRegressor(),
        sklearn.ensemble._forest.ExtraTreesRegressor(n_jobs=-1),
        # sklearn.linear_model._glm.glm.GammaRegressor(),
        # # sklearn.gaussian_process._gpr.GaussianProcessRegressor(), #performance!
        # sklearn.ensemble._gb.GradientBoostingRegressor(),
        # sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingRegressor(),
        # sklearn.linear_model._huber.HuberRegressor(),
        # sklearn.neighbors._regression.KNeighborsRegressor(n_jobs=-1),
        # # sklearn.kernel_ridge.KernelRidge(), #performance!
        # sklearn.linear_model._least_angle.Lars(),
        # sklearn.linear_model._least_angle.LarsCV(n_jobs=-1),
        # sklearn.linear_model._coordinate_descent.Lasso(),
        # sklearn.linear_model._coordinate_descent.LassoCV(n_jobs=-1),
        # sklearn.linear_model._least_angle.LassoLars(),
        # sklearn.linear_model._least_angle.LassoLarsCV(n_jobs=-1),
        # sklearn.linear_model._least_angle.LassoLarsIC(),
        # sklearn.linear_model._base.LinearRegression(n_jobs=-1),
        # #sklearn.svm._classes.LinearSVR(), # performance!
        # # sklearn.neural_network._multilayer_perceptron.MLPRegressor(), #performance!
        # # sklearn.svm._classes.NuSVR(), #performance!
        # sklearn.linear_model._omp.OrthogonalMatchingPursuit(),
        # sklearn.linear_model._omp.OrthogonalMatchingPursuitCV(n_jobs=-1),
        # sklearn.cross_decomposition._pls.PLSCanonical(),
        # sklearn.cross_decomposition._pls.PLSRegression(),
        # sklearn.linear_model._passive_aggressive.PassiveAggressiveRegressor(),
        # sklearn.linear_model._glm.glm.PoissonRegressor(),
        # sklearn.linear_model._ransac.RANSACRegressor(),
        # sklearn.ensemble._forest.RandomForestRegressor(n_jobs=-1),
        # sklearn.linear_model._ridge.Ridge(),
        # sklearn.linear_model._ridge.RidgeCV(),
        # # sklearn.linear_model._stochastic_gradient.SGDRegressor(), # performance!
        # # sklearn.svm._classes.SVR(), #performance!
        # # sklearn.linear_model._theil_sen.TheilSenRegressor(n_jobs = -1), #performance!
        # sklearn.compose._target.TransformedTargetRegressor(),
        # sklearn.linear_model._glm.glm.TweedieRegressor(),
        # xgboost.XGBRegressor(n_jobs=-1)
    ]

    # combine all variables into single dict
    settings = {'metrics': metrics, 'model': model, 'optimize': optimize}

    if not settings['metrics']:
        settings['metrics'] = ['mean_squared_error']

    # unpack that dict
    settings = list((dict(zip(settings, x))
                     for x in itertools.product(*settings.values())))

    return settings

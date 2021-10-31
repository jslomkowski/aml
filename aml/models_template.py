# This is custom models or transformers template you can use in pipeline so you
#  don't have to type that much ;)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


aml_basic_regressors = [
    ('model1', LinearRegression()),
    ('model2', Lasso()),
    ('model3', Ridge()),
    ('model4', ElasticNet()),
    ('model5', Lars()),
    ('model6', LassoLars()),
    ('model7', OrthogonalMatchingPursuit()),
    ('model8', BayesianRidge()),
    ('model9', ARDRegression()),
    ('model10', PassiveAggressiveRegressor()),
    ('model11', RANSACRegressor()),
    ('model12', TheilSenRegressor()),
    ('model13', HuberRegressor()),
    ('model14', KernelRidge()),
    ('model15', SVR()),
    ('model16', KNeighborsRegressor()),
    ('model17', DecisionTreeRegressor()),
    ('model18', RandomForestRegressor()),
    ('model19', ExtraTreesRegressor()),
    ('model20', AdaBoostRegressor()),
    ('model21', GradientBoostingRegressor()),
    ('model22', MLPRegressor()),
    ('model23', XGBRegressor()),
]

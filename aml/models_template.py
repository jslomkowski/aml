# This is custom models or transformers template you can use in pipeline so you
#  don't have to type that much ;)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


aml_basic_regressors = [
    ('model1', LinearRegression()),
    ('model2', RandomForestRegressor())
]

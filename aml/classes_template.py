
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

templates = [
    {'basic_regresors': [
        ('model1', LinearRegression()),
        ('model2', RandomForestRegressor())
    ]
    }
]

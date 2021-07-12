from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, ParameterGrid
import pandas as pd

X, y = load_boston(return_X_y=True)
X = pd.DataFrame(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# it takes a list of tuples as parameter
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LinearRegression())
])

param_grid = {
    'scaler__copy': [True, False],
    'some__C': [0.1, 1.0]
}

steps_simplified = []
for s in pipeline.steps:
    print(s[0])
    steps_simplified.append(s[0])


# grid = list(ParameterGrid(param_grid))

delete_indexes = []
for g in param_grid:
    print(g)
    if g.split('__')[0] not in steps_simplified:
        delete_indexes.append(g)

for key in param_grid.keys():
    if key in delete_indexes:
        del param_grid[key]
        break

grid = GridSearchCV(pipeline, cv=3, param_grid=param_grid)
grid.fit(X_train, y_train)

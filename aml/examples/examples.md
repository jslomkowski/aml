
- [Examples:](#examples)
  - [Simple pipeline example](#simple-pipeline-example)
  - [Simple pipeline gridsearch](#simple-pipeline-gridsearch)
  - [Simple AML example](#simple-aml-example)
  - [AML without grid search #1](#aml-without-grid-search-1)
  - [AML without grid search #2](#aml-without-grid-search-2)
  - [AML without grid search #3](#aml-without-grid-search-3)
  - [AML with grid search #1](#aml-with-grid-search-1)
  - [AML with grid search #2](#aml-with-grid-search-2)
  - [AML with grid search #3](#aml-with-grid-search-3)
  - [AML multiprocessing](#aml-multiprocessing)
  - [AML with function transformer](#aml-with-function-transformer)
  - [AML with custom transformer](#aml-with-custom-transformer)
  - [AML with identity transformer](#aml-with-identity-transformer)
  - [AML with neural networks](#aml-with-neural-networks)

## Examples:

### Simple pipeline example
```
import pandas as pd
from feature_engine.discretisation import EqualFrequencyDiscretiser
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

X, y = load_boston(return_X_y=True)
X = pd.DataFrame(X)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline([
    ('disc1', EqualFrequencyDiscretiser()),
    ('model1', LinearRegression()),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```
### Simple pipeline gridsearch
```
import pandas as pd
from feature_engine.discretisation import EqualFrequencyDiscretiser
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

X, y = load_boston(return_X_y=True)
X = pd.DataFrame(X)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline([
    ('disc1', EqualFrequencyDiscretiser()),
    ('model1', LinearRegression()),
])

param_grid = {
    'disc1__q': [5, 15],
    'model1__normalize': [True, False]
}

gs = GridSearchCV(pipeline, param_grid)
gs.fit(X_train, y_train)
```
### Simple AML example
```
import pandas as pd
from feature_engine.discretisation import EqualFrequencyDiscretiser
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from aml import AMLGridSearchCV

X, y = load_boston(return_X_y=True)
X = pd.DataFrame(X)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline([
    ('disc1', EqualFrequencyDiscretiser()),
    ('model1', LinearRegression()),
])

param_grid = {
    'disc1__q': [5, 15],
    'model1__normalize': [True, False]
}

gs = AMLGridSearchCV(pipeline, param_grid)
results = gs.fit(X_train, y_train)
```
### AML without grid search #1
```
import pandas as pd
from feature_engine.discretisation import (EqualFrequencyDiscretiser,
                                           EqualWidthDiscretiser)
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from aml import AMLGridSearchCV

X, y = load_boston(return_X_y=True)
X = pd.DataFrame(X)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline([
    ('disc1', EqualFrequencyDiscretiser()),
    ('disc2', EqualWidthDiscretiser()),
    ('model1', LinearRegression()),
    ('model2', RandomForestRegressor())
])

param_grid = {}

aml = AMLGridSearchCV(pipeline, param_grid)
results = aml.fit(X_train, y_train, X_test, y_test)
```
### AML without grid search #2
```
import pandas as pd
from feature_engine.discretisation import (EqualFrequencyDiscretiser,
                                           EqualWidthDiscretiser)
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from aml import AMLGridSearchCV

X, y = load_boston(return_X_y=True)
X = pd.DataFrame(X)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline([
    ('disc1', EqualFrequencyDiscretiser()),
    ('disc2', EqualWidthDiscretiser()),
    ('rle1', RareLabelEncoder()),
    ('ohe1', OneHotEncoder()),
    ('model1', LinearRegression()),
    ('model2', RandomForestRegressor())
])

param_grid = {}

aml = AMLGridSearchCV(pipeline, param_grid)
results = aml.fit(X_train, y_train, X_test, y_test)
```
### AML without grid search #3
```
import pandas as pd
from feature_engine.discretisation import (EqualFrequencyDiscretiser,
                                           EqualWidthDiscretiser)
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from aml import AMLGridSearchCV

X, y = load_boston(return_X_y=True)
X = pd.DataFrame(X)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline([
    ('disc1', EqualFrequencyDiscretiser()),
    ('disc2', EqualWidthDiscretiser()),
    'basic_regresors'
])

param_grid = {}

aml = AMLGridSearchCV(pipeline, param_grid)
results = aml.fit(X_train, y_train, X_test, y_test)
```
### AML with grid search #1
```
import pandas as pd
from feature_engine.discretisation import (EqualFrequencyDiscretiser,
                                           EqualWidthDiscretiser)
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from aml import AMLGridSearchCV

X, y = load_boston(return_X_y=True)
X = pd.DataFrame(X)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline([
    ('disc1', EqualFrequencyDiscretiser()),
    ('disc2', EqualWidthDiscretiser()),
    ('model1', LinearRegression()),
    ('model2', RandomForestRegressor())
])

param_grid = {
    'disc1__q': [5, 15],
    'model2__n_estimators': [50, 150]
}

aml = AMLGridSearchCV(pipeline, param_grid)
results = aml.fit(X_train, y_train, X_test, y_test)
```
### AML with grid search #2
```
import pandas as pd
from feature_engine.discretisation import (EqualFrequencyDiscretiser,
                                           EqualWidthDiscretiser)
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from aml import AMLGridSearchCV

X, y = load_boston(return_X_y=True)
X = pd.DataFrame(X)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline([
    ('disc1', EqualFrequencyDiscretiser()),
    ('disc2', EqualWidthDiscretiser()),
    ('model1', LinearRegression()),
    ('model2', RandomForestRegressor())
])

param_grid = {
    'disc1__q': [5, 15],
    'model2__*': []
}

aml = AMLGridSearchCV(pipeline, param_grid)
results = aml.fit(X_train, y_train, X_test, y_test)
```
### AML with grid search #3
```
import pandas as pd
from feature_engine.discretisation import (EqualFrequencyDiscretiser,
                                           EqualWidthDiscretiser)
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from aml import AMLGridSearchCV

X, y = load_boston(return_X_y=True)
X = pd.DataFrame(X)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline([
    ('disc1', EqualFrequencyDiscretiser()),
    ('disc2', EqualWidthDiscretiser()),
    ('model1', LinearRegression()),
    ('model2', RandomForestRegressor())
])

param_grid = {'*'}

aml = AMLGridSearchCV(pipeline, param_grid)
results = aml.fit(X_train, y_train, X_test, y_test)
```
### AML multiprocessing
```
import pandas as pd
from feature_engine.discretisation import (EqualFrequencyDiscretiser,
                                           EqualWidthDiscretiser)
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from aml import AMLGridSearchCV

X, y = load_boston(return_X_y=True)
X = pd.DataFrame(X)
y = pd.Series(y)

for i in range(5):
    X = X.append(X)
    y = y.append(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline([
    ('disc1', EqualFrequencyDiscretiser()),
    ('disc2', EqualWidthDiscretiser()),
    ('model1', LinearRegression()),
    ('model2', RandomForestRegressor())
])

param_grid = {
    'disc1__q': [5, 15],
    'model2__n_estimators': [50, 150]
}

aml = AMLGridSearchCV(pipeline, param_grid)
results = aml.fit(X_train, y_train, X_test, y_test, n_jobs=-1)
```
### AML with function transformer
```
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from aml import AMLGridSearchCV

X, y = load_boston(return_X_y=True)
X = pd.DataFrame(X)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def some_transformer(X):
    X = X * 100
    return X


pipeline = Pipeline([
    ('ft1', FunctionTransformer(some_transformer)),
    ('model1', LinearRegression()),
    ('model2', RandomForestRegressor())
])

param_grid = {}

aml = AMLGridSearchCV(pipeline, param_grid)
results = aml.fit(X_train, y_train, X_test, y_test)
```
### AML with custom transformer
```
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from aml import AMLGridSearchCV

X, y = load_boston(return_X_y=True)
X = pd.DataFrame(X)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


class SomeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X * 100
        return X


pipeline = Pipeline([
    ('st', SomeTransformer()),
    ('model1', LinearRegression()),
    ('model2', RandomForestRegressor())
])

param_grid = {}

aml = AMLGridSearchCV(pipeline, param_grid)
results = aml.fit(X_train, y_train, X_test, y_test)
```
### AML with identity transformer
```
import pandas as pd
from feature_engine.discretisation import (EqualFrequencyDiscretiser,
                                           EqualWidthDiscretiser)
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from aml import AMLGridSearchCV, IdentityTransformer

X, y = load_boston(return_X_y=True)
X = pd.DataFrame(X)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline([
    ('disc1', EqualFrequencyDiscretiser()),
    ('disc2', EqualWidthDiscretiser()),
    ('disc3', IdentityTransformer()),
    ('model1', LinearRegression())
])

param_grid = {}

aml = AMLGridSearchCV(pipeline, param_grid)
results = aml.fit(X_train, y_train, X_test, y_test)
```
### AML with neural networks
```
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aml import AMLGridSearchCV

X, y = load_boston(return_X_y=True)
X = pd.DataFrame(X)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def nn_model(first_lr, optimizer):
    model = Sequential()
    model.add(Dense(first_lr, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_absolute_error',
                  optimizer=optimizer)
    return model


model = KerasRegressor(build_fn=nn_model, verbose=0)

pipeline = Pipeline([
    ('ss1', StandardScaler()),
    ('model1', model)
])

param_grid = {
    'model1__optimizer': ['rmsprop', 'adam', 'adagrad'],
    'model1__first_lr': [10, 20, 30],
    'model1__epochs': [4, 8],
}

aml = AMLGridSearchCV(pipeline, param_grid)
results = aml.fit(X_train, y_train, X_test, y_test)

```
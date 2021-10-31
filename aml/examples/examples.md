
- [Examples:](#examples)
  - [Simple pipeline example](#simple-pipeline-example)
  - [Simple pipeline gridsearch](#simple-pipeline-gridsearch)
  - [Simple AML example](#simple-aml-example)
  - [AML without grid search #1](#aml-without-grid-search-1)
  - [AML without grid search #2](#aml-without-grid-search-2)
  - [AML without grid search #3](#aml-without-grid-search-3)
  - [AML with injected default models](#aml-with-injected-default-models)
  - [AML with injected custom models](#aml-with-injected-custom-models)
  - [AML with grid search basic](#aml-with-grid-search-basic)
  - [AML with grid search for one model](#aml-with-grid-search-for-one-model)
  - [AML with grid search for whole pipeline](#aml-with-grid-search-for-whole-pipeline)
  - [AML multiprocessing](#aml-multiprocessing)
  - [AML with function transformer](#aml-with-function-transformer)
  - [AML with custom transformer](#aml-with-custom-transformer)
  - [AML with identity transformer](#aml-with-identity-transformer)
  - [AML with neural networks #1](#aml-with-neural-networks-1)
  - [AML with neural networks #2](#aml-with-neural-networks-2)

## Examples:

### Simple pipeline example

Through the tutorials below imports and cutting data into train and test, sets
is a standard procedure, the only thing that is changing is the pipeline and param_grid. Below is the sklearn.pipeline from which AML originated

```
import pandas as pd
from feature_engine.discretisation import EqualFrequencyDiscretiser
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

X, y = fetch_california_housing(return_X_y=True)
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

But what if you wanted to optimize hyperparameters of it? You can use sklearn
GridSearchCV
```
import pandas as pd
from feature_engine.discretisation import EqualFrequencyDiscretiser
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

X, y = fetch_california_housing(return_X_y=True)
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

Above examples are usage of scikit-learn. You can do the same thing with using AML. All you have to do is import AMLGridsearchCV from aml and change GridsearchCV to AMLGridsearchCV 
```
import pandas as pd
from feature_engine.discretisation import EqualFrequencyDiscretiser
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from aml import AMLGridSearchCV

X, y = fetch_california_housing(return_X_y=True)
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

If you don't want to provide any parameters then simply pass empty param_grid dictionary, all scikit-learn transformers and estimators have default parameters, also notice that you can use multiple models in the pipeline ('model1 and model2), functionality that is not available in scikit-learn. Here AML will give you all of the combinations from inside of the pipeline 
```
disc1 with model1
disc1 with model2
disc2 with model1
disc2 with model2
```
```
import pandas as pd
from feature_engine.discretisation import (EqualFrequencyDiscretiser,
                                           EqualWidthDiscretiser)
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from aml import AMLGridSearchCV

X, y = fetch_california_housing(return_X_y=True)
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
Another example of only pipeline object without param_grid. The key to understand AML is to understand block objects (those strings before class, like 'model1' or 'disc1) it is up to you to decide how you name them, just make sure that if you want to create the grouping then you have to use the same string with a number in the end. In this example, 'pow1' and 'out1' have no other transformers so the final pipes will look like so:
```
disc1, pow1, out1, model1
disc2, pow1, out1, model1
disc1, pow1, out1, model2
disc2, pow1, out1, model2
```
```
import pandas as pd
from feature_engine.discretisation import (EqualFrequencyDiscretiser,
                                           EqualWidthDiscretiser)
from feature_engine.transformation import PowerTransformer
from feature_engine.outliers import Winsorizer
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from aml import AMLGridSearchCV

X, y = fetch_california_housing(return_X_y=True)
X = pd.DataFrame(X)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline([
    ('disc1', EqualFrequencyDiscretiser()),
    ('disc2', EqualWidthDiscretiser()),
    ('pow1', PowerTransformer()),
    ('out1', Winsorizer()),
    ('model1', LinearRegression()),
    ('model2', RandomForestRegressor())
])

param_grid = {}

aml = AMLGridSearchCV(pipeline, param_grid)
results = aml.fit(X_train, y_train, X_test, y_test)
```
### AML without grid search #3
Below is an example with 5 different models. Combinations that will be created:
```
disc1 with model1
disc1 with model2
disc1 with model3
disc1 with model4
disc1 with model5
```
```
import pandas as pd
from feature_engine.discretisation import EqualFrequencyDiscretiser
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from aml import AMLGridSearchCV

X, y = fetch_california_housing(return_X_y=True)
X = pd.DataFrame(X)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline([
    ('disc1', EqualFrequencyDiscretiser()),
    ('model1', LinearRegression()),
    ('model2', RandomForestRegressor()),
    ('model3', DecisionTreeRegressor()),
    ('model4', GradientBoostingRegressor()),
    ('model5', KNeighborsRegressor()),
])

param_grid = {}

aml = AMLGridSearchCV(pipeline, param_grid)
results = aml.fit(X_train, y_train, X_test, y_test)
```
### AML with injected default models
if you don't want to type too much you can use a predefined template from AML
```
import pandas as pd
from feature_engine.discretisation import (EqualFrequencyDiscretiser,
                                           EqualWidthDiscretiser)
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from aml import AMLGridSearchCV
from aml.models_template import aml_basic_regressors

X, y = fetch_california_housing(return_X_y=True)
X = pd.DataFrame(X)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline([
    ('disc1', EqualFrequencyDiscretiser()),
    ('disc2', EqualWidthDiscretiser()),
    aml_basic_regressors
])

param_grid = {}

aml = AMLGridSearchCV(pipeline, param_grid)
results = aml.fit(X_train, y_train, X_test, y_test)
```
### AML with injected custom models
or create your own template and inject it into the pipeline.
```
import pandas as pd
from feature_engine.discretisation import (EqualFrequencyDiscretiser,
                                           EqualWidthDiscretiser)
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from aml import AMLGridSearchCV

X, y = fetch_california_housing(return_X_y=True)
X = pd.DataFrame(X)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressors = [
    ('model1', LinearRegression()),
    ('model2', RandomForestRegressor())
]

pipeline = Pipeline([
    ('disc1', EqualFrequencyDiscretiser()),
    ('disc2', EqualWidthDiscretiser()),
    regressors
])

param_grid = {}

aml = AMLGridSearchCV(pipeline, param_grid)
results = aml.fit(X_train, y_train, X_test, y_test)

```
### AML with grid search basic
Here is the example of simple grid search. Naming convention should be: 
```
block_string_from_pipeline(dunder)hyperparameter_of_that_class
```
```
import pandas as pd
from feature_engine.discretisation import (EqualFrequencyDiscretiser,
                                           EqualWidthDiscretiser)
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from aml import AMLGridSearchCV

X, y = fetch_california_housing(return_X_y=True)
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
### AML with grid search for one model
Like in the previous example you can use custom hyperparameters but if you want AML to do the work just provide * (star) instead of hyperparameter name. In the example below * means:
```
'n_estimators': [100],
'max_features': np.arange(0.05, 1.01, 0.05),
'min_samples_split': range(2, 21),
'min_samples_leaf': range(1, 21),
'bootstrap': [True, False]
```
see the documentation or config_template module for supported templates.
```
import pandas as pd
from feature_engine.discretisation import (EqualFrequencyDiscretiser,
                                           EqualWidthDiscretiser)
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from aml import AMLGridSearchCV

X, y = fetch_california_housing(return_X_y=True)
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
### AML with grid search for whole pipeline
The below example is pretty hardcore. You can inject hyperparameters to every supported class by providing '*' in param_grid.
```
import pandas as pd
from feature_engine.discretisation import (EqualFrequencyDiscretiser,
                                           EqualWidthDiscretiser)
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from aml import AMLGridSearchCV

X, y = fetch_california_housing(return_X_y=True)
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
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from aml import AMLGridSearchCV

X, y = fetch_california_housing(return_X_y=True)
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
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from aml import AMLGridSearchCV

X, y = fetch_california_housing(return_X_y=True)
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
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from aml import AMLGridSearchCV

X, y = fetch_california_housing(return_X_y=True)
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
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from aml import AMLGridSearchCV, IdentityTransformer

X, y = fetch_california_housing(return_X_y=True)
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
### AML with neural networks #1
```
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aml import AMLGridSearchCV

X, y = fetch_california_housing(return_X_y=True)
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
### AML with neural networks #2
```
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import (SGD, Adadelta, Adagrad, Adam, Adamax,
                                         RMSprop)
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from aml import AMLGridSearchCV

X, y = fetch_california_housing(return_X_y=True)
X = pd.DataFrame(X)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def nn_model(optimizer, ler1_act, ler2_act, ler1_neur, ler2_neur):
    model = Sequential()
    model.add(Dense(ler1_neur, activation=ler1_act))
    model.add(Dense(ler2_neur, activation=ler2_act))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_absolute_error',
                  optimizer=optimizer)
    return model


es = EarlyStopping(monitor='val_loss', mode='min',
                   verbose=1, patience=10)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss',
                     mode='min', verbose=1)
model = KerasRegressor(build_fn=nn_model, verbose=0, callbacks=[es, mc])

pipeline = Pipeline([
    ('ss1', StandardScaler()),
    ('model1', model)
])

param_grid = {
    'model1__epochs': [10, 50, 100],
    'model1__batch_size': [10, 20, 40, 60, 80, X_train.shape[0]],
    'model1__optimizer': [SGD(), Adadelta(), Adagrad(), Adam(0.1), Adamax(),
                 s         RMSprop()],
    'model1__ler1_act': ['softmax', 'softplus', 'softsign', 'relu', 'tanh',
                         'sigmoid', 'hard_sigmoid', 'linear'],
    'model1__ler2_act': ['softmax', 'softplus', 'softsign', 'relu', 'tanh',
                         'sigmoid', 'hard_sigmoid', 'linear'],
    'model1__ler1_neur': [8, int(X_train.shape[1] * 1.5), int(X_train.shape[1] * 2)],
    'model1__ler2_neur': [8, int(X_train.shape[1] * 1.5), int(X_train.shape[1] * 2)],
}

aml = AMLGridSearchCV(pipeline, param_grid)
results = aml.fit(X_train, y_train, X_test, y_test)
```
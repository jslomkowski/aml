# Use scikit-learn to grid search the number of neurons
from sklearn.pipeline import Pipeline
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
# Function to create model, required for KerasClassifier
from sklearn.datasets import load_boston


def create_model(neurons=1):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=8, kernel_initializer='uniform',
              activation='linear', kernel_constraint=maxnorm(4)))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


X, y = load_boston(return_X_y=True)

# create model
model = KerasClassifier(build_fn=create_model,
                        epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
neurons = [1, 5, 10, 15, 20, 25, 30]
param_grid = dict(neurons=neurons)


pipe = Pipeline([
    ('model1', KerasClassifier(create_model))
])

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, Binarizer
from sklearn.ensemble import RandomForestRegressor
from pipeline import AMLPipeline
from sklearn.pipeline import Pipeline

X, y = make_regression(random_state=1, n_samples=300, n_features=10, noise=100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

pipe = AMLPipeline([
    # ('imp', SimpleImputer(strategy=('mean', 'most_frequent'))),
    # ('imp2', KNNImputer(n_neighbors=(6, 7))),
    ('sca', StandardScaler()),
    # ('sca2', Binarizer()),
    ('mod', DecisionTreeRegressor()),
    ('mod2', RandomForestRegressor())
])

pipe.fit(X_train, y_train)

pipe.validate(X_test)

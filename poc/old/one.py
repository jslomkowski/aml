import numpy as np
from aml.pipeline2 import MakePipeline
from sklearn.datasets import load_boston
import pandas as pd

X = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
y = pd.Series(load_boston().target, name='TARGET')

self = MakePipeline('sk')
# mp = MakePipeline('sk')
# mp.evaluate(X, y)

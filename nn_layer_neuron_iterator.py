
import itertools
import pandas as pd
from tensorflow.keras.layers import Dense

# lst = [
# Dense(10),
# Dense(20),
# Dense(30),
# ]

lst = [10, 20, 30]


czwarta = list(itertools.product(lst, repeat=2))
print(czwarta)

import numpy as np
import pandas as pd

arr = pd.Series([3, 1, 2, 3, 4])

x = arr.value_counts(normalize=True)

print(x)
print(x.loc[3])
print(x.loc[x.index[0]])
print(len(x))

for items in x:
    print(items)

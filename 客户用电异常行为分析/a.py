import pandas as pd
import numpy as np
from sklearn.datasets import dump_svmlight_file

df = pd.DataFrame()
df['Id'] = np.arange(10)
df['F1'] = np.random.rand(10,)
df['F2'] = np.random.rand(10,)
df['Target'] = map(lambda x: -1 if x < 0.5 else 1, np.random.rand(10,))

print (df.columns)
print (np.setdiff1d(df.columns,['Id','Target']))


X = df[np.setdiff1d(df.columns,['Id','Target'])]
y = df.Target

print (df)

dump_svmlight_file(X,y,'smvlight.dat',zero_based=True,multilabel=False)

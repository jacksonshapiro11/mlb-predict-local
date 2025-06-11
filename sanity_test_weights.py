import pandas as pd
from collections import Counter

y = pd.Series(['FF','FS','FF','KC','OTHER','FC'])
class_factors = {'FS':2,'OTHER':2,'KC':1.5,'FC':1.3}
row_w = y.map(class_factors).fillna(1)
print("Row weights ->", row_w.tolist(), "   OK" if row_w.max()==2 else "ERROR") 
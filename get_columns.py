import pandas as pd
df = pd.read_parquet("data/raw/statcast_2023.parquet")
print(df.columns) 
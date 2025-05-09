import pandas as pd

category = "relationship"

df = pd.read_csv(f"output/{category}/qwq-32b_results.csv")

print(len(df))

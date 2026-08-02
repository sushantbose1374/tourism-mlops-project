
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("Tourism Dataset Registered Successfully")
print("=" * 60)

print(f"Dataset Shape : {df.shape}")
print(f"Columns       : {list(df.columns)}")
print(f"Missing Values:\n{df.isnull().sum()}")

print("\nRegistration Completed Successfully.")

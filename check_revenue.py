
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "monthly_customer_revenue_clean.csv"

revenue = pd.read_csv(DATA_PATH)

print("Dataset shape:", revenue.shape)
print("\nColumn names:")
print(revenue.columns.tolist())

print("\nFirst five rows:")
print(revenue.head().to_string())

print("\nMissing values:")
print(revenue.isnull().sum())

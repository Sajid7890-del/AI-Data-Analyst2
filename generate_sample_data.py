import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Seed for reproducibility
np.random.seed(42)

# Generate sample transactions
n_rows = 200

start_date = datetime(2026, 1, 1)
dates = [start_date + timedelta(days=int(np.random.randint(0, 180))) for _ in range(n_rows)]

products = ["iPhone 15", "MacBook Air", "iPad Pro", "AirPods Pro", "Apple Watch S9"]
categories = ["Smartphones", "Laptops", "Tablets", "Audio", "Wearables"]
product_cat_map = dict(zip(products, categories))

regions = ["North", "South", "East", "West"]

# Generate data fields
row_data = []
for i in range(n_rows):
    date = dates[i].strftime("%Y-%m-%d")
    product = np.random.choice(products)
    category = product_cat_map[product]
    region = np.random.choice(regions)
    
    # Quantity between 1 and 5
    quantity = int(np.random.randint(1, 6))
    
    # Base prices
    base_prices = {
        "iPhone 15": 799.00,
        "MacBook Air": 999.00,
        "iPad Pro": 799.00,
        "AirPods Pro": 249.00,
        "Apple Watch S9": 399.00
    }
    price = base_prices[product]
    
    # Introduce small random variation to price (discounts/markup)
    price = round(price * np.random.uniform(0.9, 1.05), 2)
    sales = round(quantity * price, 2)
    
    # Customer rating 3.5 to 5.0
    rating = round(np.random.uniform(3.5, 5.0), 1)
    
    row_data.append([date, product, category, region, quantity, price, sales, rating])

# Create DataFrame
df = pd.DataFrame(row_data, columns=["Date", "Product", "Category", "Region", "Quantity", "Price", "Sales", "Rating"])

# Introduce a few missing values in 'Rating' and 'Region' for quality test demonstration
for idx in np.random.choice(df.index, size=5, replace=False):
    df.loc[idx, 'Rating'] = np.nan
    
for idx in np.random.choice(df.index, size=3, replace=False):
    df.loc[idx, 'Region'] = np.nan

# Save to CSV
df.to_csv("sample_sales_data.csv", index=False)
print("Sample dataset 'sample_sales_data.csv' successfully created with 200 rows.")

import pandas as pd
import numpy as np
import json

# Load the locations we already extracted
with open('locations.json', 'r') as f:
    locations = json.load(f)

# Define options for our features
locations_list = list(locations.keys())
unit_types = ["Studio (0 bedroom)", "1 bedroom", "2 bedroom", "3 bedroom", "4 bedroom", "5 bedroom"]
yes_no = ['Yes', 'No']

# Number of sample data points to create
n_samples = 200

# Create a dictionary to hold our simulated data
data = {
    'Location': np.random.choice(locations_list, n_samples),
    'Unit_Type': np.random.choice(unit_types, n_samples),
    'Parking': np.random.choice(yes_no, n_samples),
    'Laundry': np.random.choice(yes_no, n_samples),
    'AC': np.random.choice(yes_no, n_samples),
    'Price': np.random.randint(800, 5000, n_samples) # Base random price
}

# Create the DataFrame
df = pd.DataFrame(data)

# Make the price somewhat dependent on the number of bedrooms for realism
def adjust_price(row):
    if 'Studio' in row['Unit_Type']:
        return max(800, row['Price'] - 1000)
    bedrooms = int(row['Unit_Type'].split()[0])
    # Add price based on bedrooms, plus some noise
    new_price = 800 + (bedrooms * 500) + np.random.randint(-200, 200)
    return new_price

df['Price'] = df.apply(adjust_price, axis=1)

# Save to CSV
df.to_csv('df_houses.csv', index=False)

print(f"✅ 'df_houses.csv' created successfully with {n_samples} rows.")
import pandas as pd
import numpy as np

# --- Configuration ---
NUM_ROWS = 2000
CITIES = {
    "New York": {"state": "NY", "lat": 40.7128, "lon": -74.0060, "base_price": 3000},
    "Los Angeles": {"state": "CA", "lat": 34.0522, "lon": -118.2437, "base_price": 2500},
    "Chicago": {"state": "IL", "lat": 41.8781, "lon": -87.6298, "base_price": 1800},
    "Houston": {"state": "TX", "lat": 29.7604, "lon": -95.3698, "base_price": 1500},
    "Phoenix": {"state": "AZ", "lat": 33.4484, "lon": -112.0740, "base_price": 1600},
    "Miami": {"state": "FL", "lat": 25.7617, "lon": -80.1918, "base_price": 2800},
}
AMENITIES = ['parking', 'pool', 'ac', 'gym', 'balcony']

# --- Data Generation ---
data = []
for _ in range(NUM_ROWS):
    city_name = np.random.choice(list(CITIES.keys()))
    city_info = CITIES[city_name]
    
    bedrooms = np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.4, 0.3, 0.15, 0.05])
    
    # Price influenced by city base price and number of bedrooms
    price = city_info['base_price'] + (bedrooms * 500) + np.random.randint(-200, 200)
    
    # Unit type mapping
    if bedrooms == 0:
        unit_type = 'Studio'
    elif bedrooms == 1:
        unit_type = '1 Bedroom'
    else:
        unit_type = f'{bedrooms} Bedrooms'
        
    row = {
        'price': price,
        'city': city_name,
        'state': city_info['state'],
        'bedrooms': bedrooms,
        'unit_type': unit_type,
        'latitude': city_info['lat'] + np.random.normal(0, 0.05),
        'longitude': city_info['lon'] + np.random.normal(0, 0.05)
    }
    
    # Add random boolean amenities
    for amenity in AMENITIES:
        row[amenity] = np.random.choice([True, False], p=[0.6, 0.4])
        
    data.append(row)

df = pd.DataFrame(data)

# --- Save to CSV ---
df.to_csv('housing_data.csv', index=False)
print("Generated housing_data.csv successfully!")
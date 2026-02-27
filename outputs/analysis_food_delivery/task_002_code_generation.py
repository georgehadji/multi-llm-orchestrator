"""
data_preprocessing.py
Script for preprocessing restaurant data and generating synthetic order data.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_raw_datasets():
    """
    Load raw datasets from the data directory.
    Returns a dictionary of DataFrames.
    """
    data_dir = "data"
    datasets = {}
    
    # Try to load Zomato dataset
    zomato_path = os.path.join(data_dir, "zomato_raw.csv")
    if os.path.exists(zomato_path):
        print(f"Loading Zomato dataset from {zomato_path}")
        try:
            datasets['zomato'] = pd.read_csv(zomato_path)
            print(f"  Loaded {len(datasets['zomato'])} records")
        except Exception as e:
            print(f"  Error loading Zomato: {e}")
    
    # Try to load Wolt dataset (if exists)
    wolt_path = os.path.join(data_dir, "wolt_raw.csv")
    if os.path.exists(wolt_path):
        print(f"Loading Wolt dataset from {wolt_path}")
        try:
            datasets['wolt'] = pd.read_csv(wolt_path)
            print(f"  Loaded {len(datasets['wolt'])} records")
        except Exception as e:
            print(f"  Error loading Wolt: {e}")
    
    # Try to load efood dataset (if exists)
    efood_path = os.path.join(data_dir, "efood_raw.csv")
    if os.path.exists(efood_path):
        print(f"Loading efood dataset from {efood_path}")
        try:
            datasets['efood'] = pd.read_csv(efood_path)
            print(f"  Loaded {len(datasets['efood'])} records")
        except Exception as e:
            print(f"  Error loading efood: {e}")
    
    if not datasets:
        print("No raw datasets found. Creating synthetic restaurant data...")
        datasets['synthetic'] = create_synthetic_restaurant_data()
    
    return datasets

def create_synthetic_restaurant_data():
    """Create synthetic restaurant data if no real datasets are available."""
    n_restaurants = 250
    
    # Generate synthetic restaurant data
    restaurant_ids = [f"REST_{i:04d}" for i in range(1, n_restaurants + 1)]
    cities = ['Athens', 'Thessaloniki', 'Patras', 'Heraklion', 'Larissa', 'Volos', 'Rhodes']
    cuisines = ['Greek', 'Italian', 'Asian', 'Fast Food', 'Mediterranean', 
                'Seafood', 'Vegetarian', 'Burgers', 'Pizza', 'Sushi']
    
    data = {
        'restaurant_id': restaurant_ids,
        'name': [f"Restaurant {i}" for i in range(1, n_restaurants + 1)],
        'city': np.random.choice(cities, n_restaurants, p=[0.4, 0.2, 0.1, 0.1, 0.08, 0.07, 0.05]),
        'cuisine': np.random.choice(cuisines, n_restaurants),
        'rating': np.round(np.random.uniform(3.0, 5.0, n_restaurants), 1),
        'delivery_fee': np.round(np.random.uniform(0, 5, n_restaurants), 2),
        'min_order': np.round(np.random.uniform(5, 20, n_restaurants), 2),
        'delivery_time': np.random.randint(15, 60, n_restaurants),
        'is_active': np.random.choice([True, False], n_restaurants, p=[0.9, 0.1])
    }
    
    return pd.DataFrame(data)

def standardize_columns(df, source_name):
    """
    Standardize column names across different datasets.
    """
    # Create a copy to avoid modifying the original
    df_std = df.copy()
    
    # Define mapping dictionaries for different sources
    column_mappings = {
        'zomato': {
            'Restaurant Name': 'name',
            'Restaurant ID': 'restaurant_id',
            'City': 'city',
            'Cuisines': 'cuisine',
            'Aggregate rating': 'rating',
            'Average Cost for two': 'avg_cost_for_two',
            'Has Table booking': 'has_table_booking',
            'Has Online delivery': 'has_online_delivery',
            'Is delivering now': 'is_delivering_now',
            'Price range': 'price_range',
            'Votes': 'votes'
        },
        'wolt': {
            'title': 'name',
            'venue_id': 'restaurant_id',
            'city': 'city',
            'tag': 'cuisine',
            'rating': 'rating',
            'delivery_price': 'delivery_fee',
            'short_description': 'description'
        },
        'efood': {
            'restaurant_name': 'name',
            'restaurant_code': 'restaurant_id',
            'location': 'city',
            'category': 'cuisine',
            'score': 'rating',
            'delivery_cost': 'delivery_fee',
            'minimum_order': 'min_order'
        },
        'synthetic': {}  # Already standardized
    }
    
    # Apply mapping if source exists in mappings
    if source_name in column_mappings:
        mapping = column_mappings[source_name]
        df_std = df_std.rename(columns=mapping)
    
    # Ensure standard columns exist (create if missing)
    standard_columns = ['restaurant_id', 'name', 'city', 'cuisine', 'rating', 
                       'delivery_fee', 'min_order', 'delivery_time', 'is_active']
    
    for col in standard_columns:
        if col not in df_std.columns:
            if col == 'restaurant_id':
                df_std[col] = [f"{source_name.upper()}_{i}" for i in range(len(df_std))]
            elif col == 'name':
                df_std[col] = f"Restaurant from {source_name}"
            elif col == 'city':
                df_std[col] = 'Unknown'
            elif col == 'cuisine':
                df_std[col] = 'Mixed'
            elif col == 'rating':
                df_std[col] = np.nan
            elif col == 'delivery_fee':
                df_std[col] = 0.0
            elif col == 'min_order':
                df_std[col] = 0.0
            elif col == 'delivery_time':
                df_std[col] = 30
            elif col == 'is_active':
                df_std[col] = True
    
    return df_std

def handle_missing_values(df):
    """
    Handle missing values in the DataFrame.
    """
    df_clean = df.copy()
    
    # Drop rows with critical missing values
    critical_cols = ['restaurant_id', 'name', 'city']
    df_clean = df_clean.dropna(subset=critical_cols, how='any')
    
    # Impute missing values for other columns
    if 'rating' in df_clean.columns:
        df_clean['rating'] = df_clean['rating'].fillna(df_clean['rating'].median())
    
    if 'delivery_fee' in df_clean.columns:
        df_clean['delivery_fee'] = df_clean['delivery_fee'].fillna(0)
    
    if 'min_order' in df_clean.columns:
        df_clean['min_order'] = df_clean['min_order'].fillna(0)
    
    if 'delivery_time' in df_clean.columns:
        df_clean['delivery_time'] = df_clean['delivery_time'].fillna(30)
    
    if 'cuisine' in df_clean.columns:
        df_clean['cuisine'] = df_clean['cuisine'].fillna('Unknown')
    
    if 'is_active' in df_clean.columns:
        df_clean['is_active'] = df_clean['is_active'].fillna(True)
    
    return df_clean

def create_unified_dataframe(datasets):
    """
    Create a unified DataFrame from multiple datasets.
    Ensures at least 200 restaurant records.
    """
    unified_dfs = []
    
    for source_name, df in datasets.items():
        print(f"\nProcessing {source_name} dataset...")
        
        # Standardize columns
        df_std = standardize_columns(df, source_name)
        print(f"  Standardized columns: {list(df_std.columns)}")
        
        # Handle missing values
        df_clean = handle_missing_values(df_std)
        print(f"  Records after cleaning: {len(df_clean)}")
        
        # Select only the standard columns we want
        standard_cols = ['restaurant_id', 'name', 'city', 'cuisine', 'rating', 
                        'delivery_fee', 'min_order', 'delivery_time', 'is_active']
        
        # Ensure all standard columns exist
        for col in standard_cols:
            if col not in df_clean.columns:
                df_clean[col] = None
        
        df_final = df_clean[standard_cols].copy()
        df_final['source'] = source_name
        unified_dfs.append(df_final)
    
    # Combine all DataFrames
    if unified_dfs:
        unified_df = pd.concat(unified_dfs, ignore_index=True)
    else:
        unified_df = pd.DataFrame()
    
    # Ensure we have at least 200 records
    if len(unified_df) < 200:
        print(f"\nWarning: Only {len(unified_df)} records found. Adding synthetic records...")
        synthetic_df = create_synthetic_restaurant_data()
        synthetic_df['source'] = 'synthetic_additional'
        
        # Select needed columns
        needed_cols = ['restaurant_id', 'name', 'city', 'cuisine', 'rating', 
                      'delivery_fee', 'min_order', 'delivery_time', 'is_active', 'source']
        synthetic_df = synthetic_df[needed_cols]
        
        # Combine with existing data
        unified_df = pd.concat([unified_df, synthetic_df], ignore_index=True)
    
    # Remove duplicates based on restaurant_id
    unified_df = unified_df.drop_duplicates(subset=['restaurant_id'])
    
    print(f"\nFinal unified DataFrame: {len(unified_df)} records")
    return unified_df

def generate_synthetic_order_data(restaurants_df, n_orders=1000):
    """
    Generate synthetic order data based on restaurant information.
    """
    print(f"\nGenerating {n_orders} synthetic orders...")
    
    # Get active restaurants
    active_restaurants = restaurants_df[restaurants_df['is_active'] == True]
    if len(active_restaurants) == 0:
        active_restaurants = restaurants_df
    
    restaurant_ids = active_restaurants['restaurant_id'].values
    restaurant_names = active_restaurants['name'].values
    
    # Generate order data
    order_ids = [f"ORD_{i:06d}" for i in range(1, n_orders + 1)]
    
    # Generate timestamps (last 90 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    timestamps = []
    for _ in range(n_orders):
        random_days = np.random.uniform(0, 90)
        random_hours = np.random.uniform(0, 24)
        random_minutes = np.random.uniform(0, 60)
        random_seconds = np.random.uniform(0, 60)
        
        delta = timedelta(days=random_days, hours=random_hours, 
                         minutes=random_minutes, seconds=random_seconds)
        order_time = start_date + delta
        timestamps.append(order_time)
    
    # Generate other order attributes
    restaurant_choices = np.random.choice(restaurant_ids, n_orders)
    restaurant_names_dict = dict(zip(restaurant_ids, restaurant_names))
    restaurant_names_list = [restaurant_names_dict[rid] for rid in restaurant_choices]
    
    # Generate items (1-5 items per order)
    items_list = []
    item_categories = ['Main Course', 'Appetizer', 'Dessert', 'Drink', 'Side Dish']
    item_names = {
        'Main Course': ['Chicken Souvlaki', 'Moussaka', 'Pizza Margherita', 'Burger', 
                       'Pasta Carbonara', 'Sushi Platter', 'Greek Salad', 'Gyros'],
        'Appetizer': ['Tzatziki', 'Hummus', 'Bruschetta', 'Spring Rolls', 'Garlic Bread'],
        'Dessert': ['Baklava', 'Cheesecake', 'Ice Cream', 'Chocolate Cake', 'Galaktoboureko'],
        'Drink': ['Coke', 'Water', 'Beer', 'Wine', 'Orange Juice', 'Coffee'],
        'Side Dish': ['French Fries', 'Rice', 'Roasted Vegetables', 'Bread']
    }
    
    for _ in range(n_orders):
        n_items = np.random.randint(1, 6)
        items = []
        for _ in range(n_items):
            category = np.random.choice(item_categories)
            item_name = np.random.choice(item_names[category])
            quantity = np.random.randint(1, 4)
            price = np.round(np.random.uniform(3, 20), 2)
            items.append(f"{item_name} x{quantity} (${price})")
        items_list.append("; ".join(items))
    
    # Generate distances (km)
    distances = np.round(np.random.exponential(scale=2.5, size=n_orders), 2)
    distances = np.clip(distances, 0.5, 10)  # Limit between 0.5 and 10 km
    
    # Generate delivery times (minutes) - depends on distance and restaurant delivery time
    delivery_times = []
    for i in range(n_orders):
        rest_id = restaurant_choices[i]
        rest_delivery_time = restaurants_df.loc[
            restaurants_df['restaurant_id'] == rest_id, 'delivery_time'
        ].values[0] if rest_id in restaurants_df['restaurant_id'].values else 30
        
        # Base time + distance factor + random variation
        base_time = rest_delivery_time
        distance_factor = distances[i] * 3  # 3 minutes per km
        random_variation = np.random.uniform(-5, 10)
        total_time = base_time + distance_factor + random_variation
        delivery_times.append(max(15, min(90, int(total_time))))  # Limit 15-90 minutes
    
    # Generate order ratings (1-5)
    ratings = np.random.choice([1, 2, 3, 4, 5], n_orders, p=[0.02, 0.05, 0.13, 0.5, 0.3])
    
    # Generate order values
    order_values = np.round(np.random.uniform(10, 100, n_orders), 2)
    
    # Create DataFrame
    orders_df = pd.DataFrame({
        'order_id': order_ids,
        'timestamp': timestamps,
        'restaurant_id': restaurant_choices,
        'restaurant_name': restaurant_names_list,
        'items': items_list,
        'order_value': order_values,
        'distance_km': distances,
        'delivery_time_minutes': delivery_times,
        'customer_rating': ratings,
        'delivery_status': np.random.choice(['Delivered', 'Cancelled', 'In Progress'], 
                                           n_orders, p=[0.85, 0.05, 0.1])
    })
    
    # Add day of week and hour for analysis
    orders_df['day_of_week'] = orders_df['timestamp'].dt.day_name()
    orders_df['hour_of_day'] = orders_df['timestamp'].dt.hour
    
    print(f"Generated {len(orders_df)} synthetic orders")
    return orders_df

def save_processed_data(restaurants_df, orders_df):
    """
    Save processed data to CSV files.
    """
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Save restaurants data
    restaurants_path = os.path.join(data_dir, "restaurants_processed.csv")
    restaurants_df.to_csv(restaurants_path, index=False)
    print(f"\nSaved processed restaurants data to: {restaurants_path}")
    print(f"  Records: {len(restaurants_df)}")
    print(f"  Columns: {list(restaurants_df.columns)}")
    
    # Save orders data
    orders_path = os.path.join(data_dir, "orders_synthetic.csv")
    orders_df.to_csv(orders_path, index=False)
    print(f"\nSaved synthetic orders data to: {orders_path}")
    print(f"  Records: {len(orders_df)}")
    print(f"  Columns: {list(orders_df.columns)}")
    
    return restaurants_path, orders_path

def main():
    """
    Main function to execute the data preprocessing pipeline.
    """
    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load raw datasets
    print("\n1. Loading raw datasets...")
    datasets = load_raw_datasets()
    
    # Step 2: Create unified DataFrame
    print("\n2. Creating unified DataFrame...")
    restaurants_df = create_unified_dataframe(datasets)
    
    # Display sample of processed data
    print("\nSample of processed restaurants data:")
    print(restaurants_df.head())
    print(f"\nDataFrame shape: {restaurants_df.shape}")
    print(f"\nColumn information:")
    print(restaurants_df.info())
    
    # Step 3: Generate synthetic order data
    print("\n3. Generating synthetic order data...")
    orders_df = generate_synthetic_order_data(restaurants_df, n_orders=1000)
    
    # Display sample of order data
    print("\nSample of synthetic order data:")
    print(orders_df.head())
    
    # Step 4: Save processed data
    print("\n4. Saving processed data...")
    restaurants_path, orders_path = save_processed_data(restaurants_df, orders_df)
    
    # Step 5: Summary statistics
    print("\n5. Summary Statistics:")
    print("-" * 40)
    print(f"Total restaurants: {len(restaurants_df)}")
    print(f"Active restaurants: {restaurants_df['is_active'].sum()}")
    print(f"Unique cities: {restaurants_df['city'].nunique()}")
    print(f"Unique cuisines: {restaurants_df['cuisine'].nunique()}")
    print(f"Average rating: {restaurants_df['rating'].mean():.2f}")
    print(f"Average delivery fee: ${restaurants_df['delivery_fee'].mean():.2f}")
    
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()

"""
seasonal_analysis.py
Seasonal analysis module for vacation rental data analysis.
Provides functions for analyzing seasonal patterns in pricing and occupancy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(filepath: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load and prepare data for seasonal analysis.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing rental data
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns:
    --------
    pd.DataFrame
        Prepared DataFrame with datetime index and filtered date range
    """
    try:
        # Load the data
        df = pd.read_csv(filepath)
        
        # Convert date column to datetime and set as index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        # Filter by date range
        df = df.loc[start_date:end_date]
        
        # Ensure required columns exist
        required_columns = ['price', 'city', 'island', 'occupied']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        return df
    
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def monthly_average_price_per_city(df: pd.DataFrame, 
                                   start_date: str, 
                                   end_date: str) -> Dict[str, Dict]:
    """
    Calculate monthly average price per city for line chart visualization.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with rental data (must contain 'price', 'city', and datetime index)
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns:
    --------
    Dict[str, Dict]
        Dictionary with cities as keys and monthly price data as values
        Format: {city: {'months': [], 'prices': []}}
    """
    # Filter data for the specified date range
    filtered_df = df.loc[start_date:end_date].copy()
    
    # Resample to monthly frequency and calculate average price per city
    monthly_data = filtered_df.groupby('city').resample('M')['price'].mean()
    
    # Convert to dictionary format for easy charting
    result = {}
    
    for city in monthly_data.index.get_level_values(0).unique():
        city_data = monthly_data[city]
        
        # Format months for display
        months = [ts.strftime('%Y-%m') for ts in city_data.index]
        
        result[city] = {
            'months': months,
            'prices': city_data.values.tolist(),
            'average_price': float(city_data.mean())
        }
    
    return result


def occupancy_rate_heatmap_data(df: pd.DataFrame, 
                                start_date: str, 
                                end_date: str) -> Dict[str, Dict]:
    """
    Generate occupancy rate heatmap data (city × month).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with rental data (must contain 'occupied', 'city', and datetime index)
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns:
    --------
    Dict[str, Dict]
        Dictionary with cities as keys and monthly occupancy rates as values
        Format: {city: {month: occupancy_rate}}
    """
    # Filter data for the specified date range
    filtered_df = df.loc[start_date:end_date].copy()
    
    # Ensure 'occupied' column is boolean or numeric
    if filtered_df['occupied'].dtype == 'object':
        filtered_df['occupied'] = filtered_df['occupied'].map({'True': True, 'False': False, 'true': True, 'false': False})
    
    # Convert to numeric if needed
    filtered_df['occupied'] = pd.to_numeric(filtered_df['occupied'], errors='coerce')
    
    # Resample to monthly frequency and calculate occupancy rate per city
    monthly_occupancy = filtered_df.groupby('city').resample('M')['occupied'].mean()
    
    # Create heatmap data structure
    heatmap_data = {}
    
    for city in monthly_occupancy.index.get_level_values(0).unique():
        city_data = monthly_occupancy[city]
        
        # Create dictionary of month: occupancy_rate
        month_rates = {}
        for date, rate in city_data.items():
            month_key = date.strftime('%Y-%m')
            month_rates[month_key] = float(rate)
        
        heatmap_data[city] = {
            'monthly_rates': month_rates,
            'average_occupancy': float(city_data.mean()),
            'peak_month': city_data.idxmax().strftime('%Y-%m') if len(city_data) > 0 else None,
            'peak_rate': float(city_data.max()) if len(city_data) > 0 else None
        }
    
    return heatmap_data


def summer_winter_price_premium(df: pd.DataFrame, 
                                start_date: str, 
                                end_date: str,
                                summer_months: List[int] = [6, 7, 8],
                                winter_months: List[int] = [12, 1, 2]) -> Dict[str, Dict]:
    """
    Calculate summer vs winter price premium per island.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with rental data (must contain 'price', 'island', and datetime index)
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    summer_months : List[int]
        Months considered as summer (default: June, July, August)
    winter_months : List[int]
        Months considered as winter (default: December, January, February)
    
    Returns:
    --------
    Dict[str, Dict]
        Dictionary with islands as keys and price premium data as values
        Format: {island: {'summer_avg': float, 'winter_avg': float, 'premium': float, 'premium_percentage': float}}
    """
    # Filter data for the specified date range
    filtered_df = df.loc[start_date:end_date].copy()
    
    # Extract month from datetime index
    filtered_df['month'] = filtered_df.index.month
    
    # Filter for summer and winter months
    summer_df = filtered_df[filtered_df['month'].isin(summer_months)]
    winter_df = filtered_df[filtered_df['month'].isin(winter_months)]
    
    # Calculate average prices per island for summer and winter
    summer_prices = summer_df.groupby('island')['price'].mean()
    winter_prices = winter_df.groupby('island')['price'].mean()
    
    # Calculate price premium
    result = {}
    
    for island in filtered_df['island'].unique():
        summer_avg = summer_prices.get(island, np.nan)
        winter_avg = winter_prices.get(island, np.nan)
        
        if pd.notna(summer_avg) and pd.notna(winter_avg) and winter_avg > 0:
            premium = summer_avg - winter_avg
            premium_percentage = (premium / winter_avg) * 100
        else:
            premium = np.nan
            premium_percentage = np.nan
        
        result[island] = {
            'summer_avg_price': float(summer_avg) if pd.notna(summer_avg) else None,
            'winter_avg_price': float(winter_avg) if pd.notna(winter_avg) else None,
            'price_premium': float(premium) if pd.notna(premium) else None,
            'premium_percentage': float(premium_percentage) if pd.notna(premium_percentage) else None,
            'summer_months': summer_months,
            'winter_months': winter_months
        }
    
    return result


def generate_seasonal_report(df: pd.DataFrame, 
                            start_date: str, 
                            end_date: str) -> Dict[str, Dict]:
    """
    Generate a comprehensive seasonal analysis report.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with rental data
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns:
    --------
    Dict[str, Dict]
        Comprehensive report containing all seasonal analyses
    """
    # Generate all analyses
    price_analysis = monthly_average_price_per_city(df, start_date, end_date)
    occupancy_analysis = occupancy_rate_heatmap_data(df, start_date, end_date)
    premium_analysis = summer_winter_price_premium(df, start_date, end_date)
    
    # Compile report
    report = {
        'date_range': {
            'start_date': start_date,
            'end_date': end_date
        },
        'price_analysis': price_analysis,
        'occupancy_analysis': occupancy_analysis,
        'premium_analysis': premium_analysis,
        'summary': {
            'total_cities_analyzed': len(price_analysis),
            'total_islands_analyzed': len(premium_analysis),
            'analysis_period_months': len(pd.date_range(start=start_date, end=end_date, freq='M'))
        }
    }
    
    return report


# Example usage and testing function
def example_usage():
    """
    Example usage of the seasonal analysis module.
    """
    # Create sample data for demonstration
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n_records = len(dates)
    
    sample_data = pd.DataFrame({
        'date': dates,
        'price': np.random.uniform(100, 500, n_records),
        'city': np.random.choice(['City_A', 'City_B', 'City_C'], n_records),
        'island': np.random.choice(['Island_X', 'Island_Y'], n_records),
        'occupied': np.random.choice([0, 1], n_records, p=[0.3, 0.7])
    })
    
    # Save sample data
    sample_data.to_csv('sample_rental_data.csv', index=False)
    
    # Load and prepare data
    df = load_and_prepare_data('sample_rental_data.csv', '2023-01-01', '2023-12-31')
    
    # Run analyses
    print("=== Monthly Average Price per City ===")
    price_data = monthly_average_price_per_city(df, '2023-01-01', '2023-12-31')
    for city, data in price_data.items():
        print(f"{city}: Average price = ${data['average_price']:.2f}")
    
    print("\n=== Occupancy Rate Heatmap Data ===")
    occupancy_data = occupancy_rate_heatmap_data(df, '2023-01-01', '2023-12-31')
    for city, data in occupancy_data.items():
        print(f"{city}: Average occupancy = {data['average_occupancy']:.2%}")
    
    print("\n=== Summer vs Winter Price Premium ===")
    premium_data = summer_winter_price_premium(df, '2023-01-01', '2023-12-31')
    for island, data in premium_data.items():
        if data['premium_percentage']:
            print(f"{island}: Premium = {data['premium_percentage']:.1f}%")
    
    print("\n=== Comprehensive Report ===")
    report = generate_seasonal_report(df, '2023-01-01', '2023-12-31')
    print(f"Analysis complete for {report['summary']['total_cities_analyzed']} cities "
          f"and {report['summary']['total_islands_analyzed']} islands")


if __name__ == "__main__":
    # Run example usage when script is executed directly
    example_usage()

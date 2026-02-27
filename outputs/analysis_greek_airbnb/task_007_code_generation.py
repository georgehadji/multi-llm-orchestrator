"""
host_analysis.py - Analysis module for Airbnb host data analysis
Functions for superhost comparison and multi-listing host analysis
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

def compare_superhosts(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compare superhosts vs regular hosts on price, occupancy, and rating statistics
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing Airbnb listing data with columns:
        - 'host_is_superhost': boolean or string indicating superhost status
        - 'price': numeric price value
        - 'availability_365': availability days per year
        - 'review_scores_rating': rating score (0-100)
        
    Returns:
    --------
    Tuple containing:
    1. comparison_df: DataFrame with aggregated statistics by host type
    2. summary_stats: Dictionary with detailed statistical comparisons
    """
    
    # Validate required columns
    required_cols = ['host_is_superhost', 'price', 'availability_365', 'review_scores_rating']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean and prepare data
    df_clean = df.copy()
    
    # Convert superhost to boolean if needed
    if df_clean['host_is_superhost'].dtype == 'object':
        df_clean['host_is_superhost'] = df_clean['host_is_superhost'].map({
            't': True, 'f': False, 'True': True, 'False': False,
            'true': True, 'false': False, '1': True, '0': False
        }).astype(bool)
    
    # Convert price to numeric if needed
    if df_clean['price'].dtype == 'object':
        df_clean['price'] = df_clean['price'].replace('[\$,]', '', regex=True).astype(float)
    
    # Calculate occupancy rate (assuming availability_365 is days available)
    df_clean['occupancy_rate'] = 1 - (df_clean['availability_365'] / 365)
    df_clean['occupancy_rate'] = df_clean['occupancy_rate'].clip(0, 1)  # Ensure between 0-1
    
    # Group by superhost status
    grouped = df_clean.groupby('host_is_superhost')
    
    # Calculate statistics
    comparison_data = []
    
    for is_superhost, group in grouped:
        host_type = 'Superhost' if is_superhost else 'Regular Host'
        
        stats = {
            'host_type': host_type,
            'count': len(group),
            'avg_price': group['price'].mean(),
            'median_price': group['price'].median(),
            'price_std': group['price'].std(),
            'avg_occupancy_rate': group['occupancy_rate'].mean(),
            'median_occupancy_rate': group['occupancy_rate'].median(),
            'avg_rating': group['review_scores_rating'].mean(),
            'median_rating': group['review_scores_rating'].median(),
            'rating_std': group['review_scores_rating'].std(),
            'min_rating': group['review_scores_rating'].min(),
            'max_rating': group['review_scores_rating'].max()
        }
        comparison_data.append(stats)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Calculate additional summary statistics
    superhost_group = df_clean[df_clean['host_is_superhost']]
    regular_group = df_clean[~df_clean['host_is_superhost']]
    
    summary_stats = {
        'total_listings': len(df_clean),
        'superhost_listings': len(superhost_group),
        'regular_host_listings': len(regular_group),
        'superhost_percentage': (len(superhost_group) / len(df_clean)) * 100,
        
        'price_comparison': {
            'price_ratio': superhost_group['price'].mean() / regular_group['price'].mean() 
                          if regular_group['price'].mean() > 0 else np.nan,
            'price_difference': superhost_group['price'].mean() - regular_group['price'].mean(),
            'price_t_test': None  # Placeholder for statistical test
        },
        
        'occupancy_comparison': {
            'occupancy_ratio': superhost_group['occupancy_rate'].mean() / 
                              regular_group['occupancy_rate'].mean() 
                              if regular_group['occupancy_rate'].mean() > 0 else np.nan,
            'occupancy_difference': superhost_group['occupancy_rate'].mean() - 
                                   regular_group['occupancy_rate'].mean()
        },
        
        'rating_comparison': {
            'rating_difference': superhost_group['review_scores_rating'].mean() - 
                                regular_group['review_scores_rating'].mean(),
            'rating_gap_percentage': ((superhost_group['review_scores_rating'].mean() - 
                                      regular_group['review_scores_rating'].mean()) / 
                                     regular_group['review_scores_rating'].mean()) * 100 
                                     if regular_group['review_scores_rating'].mean() > 0 else np.nan
        }
    }
    
    return comparison_df, summary_stats


def analyze_multi_listing_hosts(df: pd.DataFrame, 
                               city_col: str = 'city',
                               host_id_col: str = 'host_id') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify multi-listing hosts and calculate their share per city
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing Airbnb listing data
    city_col : str
        Column name containing city information
    host_id_col : str
        Column name containing host identifier
        
    Returns:
    --------
    Tuple containing:
    1. multi_host_summary: DataFrame with multi-listing host statistics per city
    2. host_listing_counts: DataFrame with listing counts per host
    """
    
    # Validate required columns
    if host_id_col not in df.columns:
        raise ValueError(f"Host ID column '{host_id_col}' not found in DataFrame")
    
    if city_col not in df.columns:
        raise ValueError(f"City column '{city_col}' not found in DataFrame")
    
    # Count listings per host
    host_counts = df.groupby(host_id_col).size().reset_index(name='listing_count')
    
    # Identify multi-listing hosts (more than 2 properties)
    multi_listing_hosts = host_counts[host_counts['listing_count'] > 2]
    
    # Merge with original data to get city information
    df_with_counts = df.merge(host_counts, on=host_id_col, how='left')
    
    # Calculate statistics per city
    city_stats = []
    
    for city in df_with_counts[city_col].unique():
        city_data = df_with_counts[df_with_counts[city_col] == city]
        
        # Total listings in city
        total_listings = len(city_data)
        
        # Multi-listing hosts in city
        multi_hosts_in_city = city_data[city_data['listing_count'] > 2]
        multi_listing_count = len(multi_hosts_in_city)
        
        # Unique hosts in city
        unique_hosts = city_data[host_id_col].nunique()
        
        # Multi-listing unique hosts
        multi_unique_hosts = multi_hosts_in_city[host_id_col].nunique()
        
        # Calculate shares
        multi_listing_share = (multi_listing_count / total_listings * 100) if total_listings > 0 else 0
        multi_host_share = (multi_unique_hosts / unique_hosts * 100) if unique_hosts > 0 else 0
        
        # Average listings per multi-listing host in this city
        avg_multi_listings = (multi_hosts_in_city.groupby(host_id_col)['listing_count']
                             .first().mean()) if multi_unique_hosts > 0 else 0
        
        city_stats.append({
            'city': city,
            'total_listings': total_listings,
            'unique_hosts': unique_hosts,
            'multi_listing_count': multi_listing_count,
            'multi_listing_share_percent': multi_listing_share,
            'multi_host_count': multi_unique_hosts,
            'multi_host_share_percent': multi_host_share,
            'avg_listings_per_multi_host': avg_multi_listings,
            'top_multi_host_listings': multi_hosts_in_city['listing_count'].max() 
                                      if multi_listing_count > 0 else 0
        })
    
    # Create city summary DataFrame
    multi_host_summary = pd.DataFrame(city_stats)
    
    # Sort by multi-listing share (descending)
    multi_host_summary = multi_host_summary.sort_values('multi_listing_share_percent', 
                                                       ascending=False).reset_index(drop=True)
    
    # Create detailed host listing counts DataFrame
    host_listing_counts = host_counts.copy()
    host_listing_counts['is_multi_listing'] = host_listing_counts['listing_count'] > 2
    
    # Add city information for each host (take the first city if host has listings in multiple cities)
    host_cities = df.groupby(host_id_col)[city_col].first().reset_index()
    host_listing_counts = host_listing_counts.merge(host_cities, on=host_id_col, how='left')
    
    # Calculate additional statistics
    host_listing_counts['listing_category'] = pd.cut(
        host_listing_counts['listing_count'],
        bins=[0, 1, 2, 5, 10, 20, float('inf')],
        labels=['Single', 'Double', '3-5', '6-10', '11-20', '20+'],
        right=False
    )
    
    return multi_host_summary, host_listing_counts


def generate_host_analysis_report(df: pd.DataFrame, 
                                 city_col: str = 'city',
                                 host_id_col: str = 'host_id') -> Dict[str, Any]:
    """
    Generate a comprehensive host analysis report combining both analyses
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing Airbnb listing data
    city_col : str
        Column name containing city information
    host_id_col : str
        Column name containing host identifier
        
    Returns:
    --------
    Dictionary containing all analysis results
    """
    
    # Run both analyses
    superhost_comparison, superhost_stats = compare_superhosts(df)
    multi_host_summary, host_listing_counts = analyze_multi_listing_hosts(
        df, city_col, host_id_col
    )
    
    # Calculate overall multi-listing statistics
    total_hosts = host_listing_counts['host_id'].nunique()
    multi_listing_hosts = host_listing_counts[host_listing_counts['is_multi_listing']]
    total_multi_hosts = len(multi_listing_hosts)
    
    # Create comprehensive report
    report = {
        'superhost_analysis': {
            'comparison_table': superhost_comparison,
            'summary_statistics': superhost_stats
        },
        
        'multi_listing_analysis': {
            'city_summary': multi_host_summary,
            'host_details': host_listing_counts,
            'overall_statistics': {
                'total_hosts': total_hosts,
                'multi_listing_hosts': total_multi_hosts,
                'multi_listing_host_percentage': (total_multi_hosts / total_hosts * 100) 
                                                 if total_hosts > 0 else 0,
                'total_listings_by_multi_hosts': multi_listing_hosts['listing_count'].sum(),
                'avg_listings_per_multi_host': multi_listing_hosts['listing_count'].mean() 
                                               if total_multi_hosts > 0 else 0,
                'max_listings_by_single_host': host_listing_counts['listing_count'].max()
            }
        },
        
        'combined_insights': {
            'cities_with_high_multi_listing': multi_host_summary.head(5)[['city', 
                                                                         'multi_listing_share_percent']].to_dict('records'),
            'superhost_prevalence': superhost_stats['superhost_percentage'],
            'recommended_analysis': [
                "Correlation between superhost status and multi-listing ownership",
                "Impact of multi-listing concentration on city-level pricing",
                "Occupancy rates comparison between multi-listing and single-listing hosts"
            ]
        }
    }
    
    return report


# Example usage function
def example_usage():
    """
    Example of how to use the host analysis functions
    """
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'host_id': np.random.choice(range(1, 201), n_samples),
        'host_is_superhost': np.random.choice(['t', 'f'], n_samples, p=[0.3, 0.7]),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Miami'], n_samples),
        'price': np.random.uniform(50, 500, n_samples),
        'availability_365': np.random.randint(0, 365, n_samples),
        'review_scores_rating': np.random.uniform(70, 100, n_samples)
    })
    
    print("=== Host Analysis Module Example ===\n")
    
    # 1. Superhost comparison
    print("1. Superhost vs Regular Host Comparison:")
    comparison_df, stats = compare_superhosts(sample_data)
    print(comparison_df.to_string())
    print(f"\nSuperhost percentage: {stats['superhost_percentage']:.1f}%")
    
    # 2. Multi-listing host analysis
    print("\n2. Multi-listing Host Analysis:")
    multi_summary, host_counts = analyze_multi_listing_hosts(sample_data)
    print("\nTop 5 cities by multi-listing share:")
    print(multi_summary.head().to_string())
    
    # 3. Comprehensive report
    print("\n3. Comprehensive Analysis Report:")
    report = generate_host_analysis_report(sample_data)
    print(f"Total hosts: {report['multi_listing_analysis']['overall_statistics']['total_hosts']}")
    print(f"Multi-listing hosts: {report['multi_listing_analysis']['overall_statistics']['multi_listing_hosts']}")
    
    return sample_data, comparison_df, multi_summary, report


if __name__ == "__main__":
    # Run example when script is executed directly
    example_usage()

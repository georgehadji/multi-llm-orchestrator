"""
price_analysis.py - Analysis module for property price data
Functions for generating plotly-ready data structures for various price analyses
"""

import pandas as pd
import numpy as np
from geopy.distance import geodesic
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
import plotly.express as px


def prepare_neighbourhood_choropleth(df: pd.DataFrame, 
                                     location_col: str = 'neighbourhood',
                                     price_col: str = 'price',
                                     geojson_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Prepare data for neighbourhood average price choropleth map
    
    Args:
        df: DataFrame containing property data
        location_col: Column name for neighbourhood/location
        price_col: Column name for price data
        geojson_data: Optional GeoJSON data for neighbourhood boundaries
    
    Returns:
        Dictionary with plotly-ready data structure for choropleth
    """
    # Calculate average price per neighbourhood
    avg_prices = df.groupby(location_col)[price_col].mean().reset_index()
    avg_prices.columns = [location_col, 'average_price']
    
    # Sort by average price for better visualization
    avg_prices = avg_prices.sort_values('average_price', ascending=False)
    
    # Create color scale based on price percentiles
    price_percentiles = avg_prices['average_price'].quantile([0.25, 0.5, 0.75, 0.9]).values
    
    # Prepare plotly data structure
    choropleth_data = {
        'data': avg_prices.to_dict('records'),
        'locations': avg_prices[location_col].tolist(),
        'z': avg_prices['average_price'].tolist(),
        'location_col': location_col,
        'z_col': 'average_price',
        'color_scale': 'Viridis',
        'title': f'Average {price_col} by {location_col}',
        'colorbar_title': f'Average {price_col}',
        'geojson': geojson_data,
        'statistics': {
            'min_price': float(avg_prices['average_price'].min()),
            'max_price': float(avg_prices['average_price'].max()),
            'median_price': float(avg_prices['average_price'].median()),
            'num_neighbourhoods': len(avg_prices)
        }
    }
    
    return choropleth_data


def prepare_property_type_violin(df: pd.DataFrame,
                                 property_type_col: str = 'property_type',
                                 price_col: str = 'price',
                                 log_scale: bool = True) -> Dict[str, Any]:
    """
    Prepare data for price distribution violin plots by property type
    
    Args:
        df: DataFrame containing property data
        property_type_col: Column name for property type
        price_col: Column name for price data
        log_scale: Whether to use log scale for prices
    
    Returns:
        Dictionary with plotly-ready data structure for violin plot
    """
    # Filter out extreme outliers (top 1%)
    price_threshold = df[price_col].quantile(0.99)
    filtered_df = df[df[price_col] <= price_threshold].copy()
    
    # Apply log transformation if requested
    if log_scale:
        filtered_df['log_price'] = np.log1p(filtered_df[price_col])
        y_col = 'log_price'
        y_title = f'Log({price_col})'
    else:
        y_col = price_col
        y_title = price_col
    
    # Group by property type
    property_types = filtered_df[property_type_col].unique()
    
    # Prepare data for each property type
    violin_data = []
    for prop_type in property_types:
        prop_data = filtered_df[filtered_df[property_type_col] == prop_type]
        if len(prop_data) > 0:
            violin_data.append({
                'property_type': prop_type,
                'prices': prop_data[y_col].tolist(),
                'count': len(prop_data),
                'median_price': float(prop_data[price_col].median()),
                'mean_price': float(prop_data[price_col].mean())
            })
    
    # Sort by median price
    violin_data.sort(key=lambda x: x['median_price'], reverse=True)
    
    # Prepare plotly data structure
    plot_data = {
        'violin_data': violin_data,
        'property_types': [d['property_type'] for d in violin_data],
        'all_prices': filtered_df[y_col].tolist(),
        'y_col': y_col,
        'y_title': y_title,
        'price_col': price_col,
        'property_type_col': property_type_col,
        'log_scale': log_scale,
        'title': f'Price Distribution by {property_type_col}',
        'statistics': {
            'total_properties': len(filtered_df),
            'num_property_types': len(violin_data),
            'price_range': {
                'min': float(filtered_df[price_col].min()),
                'max': float(filtered_df[price_col].max()),
                'median': float(filtered_df[price_col].median())
            }
        }
    }
    
    return plot_data


def prepare_price_vs_reviews_scatter(df: pd.DataFrame,
                                     price_col: str = 'price',
                                     reviews_col: str = 'number_of_reviews',
                                     rating_col: Optional[str] = 'review_scores_rating',
                                     size_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Prepare data for price vs reviews scatter plot
    
    Args:
        df: DataFrame containing property data
        price_col: Column name for price data
        reviews_col: Column name for number of reviews
        rating_col: Optional column name for rating scores
        size_col: Optional column name for point sizing
    
    Returns:
        Dictionary with plotly-ready data structure for scatter plot
    """
    # Filter out properties with no reviews if needed
    scatter_df = df.copy()
    
    # Create hover text
    if rating_col and rating_col in df.columns:
        scatter_df['hover_text'] = scatter_df.apply(
            lambda row: f"Price: ${row[price_col]:.2f}<br>"
                       f"Reviews: {row[reviews_col]}<br>"
                       f"Rating: {row[rating_col]:.1f}/100",
            axis=1
        )
    else:
        scatter_df['hover_text'] = scatter_df.apply(
            lambda row: f"Price: ${row[price_col]:.2f}<br>Reviews: {row[reviews_col]}",
            axis=1
        )
    
    # Prepare size data if available
    if size_col and size_col in df.columns:
        sizes = scatter_df[size_col].fillna(1).tolist()
        size_title = size_col
    else:
        # Default size based on price percentile
        sizes = np.clip(scatter_df[price_col] / scatter_df[price_col].quantile(0.9), 5, 20)
        sizes = sizes.tolist()
        size_title = 'Price Relative'
    
    # Calculate correlation
    correlation = scatter_df[price_col].corr(scatter_df[reviews_col])
    
    # Prepare plotly data structure
    scatter_data = {
        'x': scatter_df[reviews_col].tolist(),
        'y': scatter_df[price_col].tolist(),
        'hover_texts': scatter_df['hover_text'].tolist(),
        'sizes': sizes,
        'x_col': reviews_col,
        'y_col': price_col,
        'size_col': size_title,
        'rating_col': rating_col,
        'title': f'{price_col} vs {reviews_col}',
        'x_title': f'Number of Reviews ({reviews_col})',
        'y_title': f'{price_col} ($)',
        'statistics': {
            'correlation': float(correlation),
            'total_points': len(scatter_df),
            'avg_reviews': float(scatter_df[reviews_col].mean()),
            'avg_price': float(scatter_df[price_col].mean()),
            'price_per_review': float(scatter_df[price_col].mean() / max(1, scatter_df[reviews_col].mean()))
        }
    }
    
    return scatter_data


def prepare_price_vs_distance(df: pd.DataFrame,
                              price_col: str = 'price',
                              lat_col: str = 'latitude',
                              lon_col: str = 'longitude',
                              city_center: Tuple[float, float] = (40.7128, -74.0060),  # NYC default
                              log_scale: bool = True) -> Dict[str, Any]:
    """
    Prepare data for price vs distance from city center analysis
    
    Args:
        df: DataFrame containing property data
        price_col: Column name for price data
        lat_col: Column name for latitude
        lon_col: str = 'longitude',
        city_center: Tuple of (latitude, longitude) for city center
        log_scale: Whether to use log scale for prices
    
    Returns:
        Dictionary with plotly-ready data structure for scatter plot
    """
    # Calculate distances from city center
    distances = []
    for _, row in df.iterrows():
        if pd.notna(row[lat_col]) and pd.notna(row[lon_col]):
            prop_location = (row[lat_col], row[lon_col])
            distance = geodesic(city_center, prop_location).kilometers
            distances.append(distance)
        else:
            distances.append(np.nan)
    
    scatter_df = df.copy()
    scatter_df['distance_from_center_km'] = distances
    
    # Remove rows with NaN distances
    scatter_df = scatter_df.dropna(subset=['distance_from_center_km'])
    
    # Apply log transformation if requested
    if log_scale:
        scatter_df['log_price'] = np.log1p(scatter_df[price_col])
        y_col = 'log_price'
        y_title = f'Log({price_col})'
    else:
        y_col = price_col
        y_title = price_col
    
    # Create hover text
    scatter_df['hover_text'] = scatter_df.apply(
        lambda row: f"Price: ${row[price_col]:.2f}<br>"
                   f"Distance: {row['distance_from_center_km']:.2f} km<br>"
                   f"Coordinates: ({row[lat_col]:.4f}, {row[lon_col]:.4f})",
        axis=1
    )
    
    # Calculate correlation
    correlation = scatter_df[price_col].corr(scatter_df['distance_from_center_km'])
    
    # Prepare plotly data structure
    scatter_data = {
        'x': scatter_df['distance_from_center_km'].tolist(),
        'y': scatter_df[y_col].tolist(),
        'hover_texts': scatter_df['hover_text'].tolist(),
        'prices': scatter_df[price_col].tolist(),
        'latitudes': scatter_df[lat_col].tolist(),
        'longitudes': scatter_df[lon_col].tolist(),
        'x_col': 'distance_from_center_km',
        'y_col': y_col,
        'price_col': price_col,
        'city_center': city_center,
        'log_scale': log_scale,
        'title': f'{price_col} vs Distance from City Center',
        'x_title': 'Distance from Center (km)',
        'y_title': y_title,
        'statistics': {
            'correlation': float(correlation),
            'total_points': len(scatter_df),
            'avg_distance': float(scatter_df['distance_from_center_km'].mean()),
            'avg_price': float(scatter_df[price_col].mean()),
            'min_distance': float(scatter_df['distance_from_center_km'].min()),
            'max_distance': float(scatter_df['distance_from_center_km'].max())
        }
    }
    
    return scatter_data


def identify_top_neighbourhoods(df: pd.DataFrame,
                                location_col: str = 'neighbourhood',
                                price_col: str = 'price',
                                rating_col: Optional[str] = 'review_scores_rating',
                                min_reviews: int = 5,
                                min_properties: int = 10) -> Dict[str, Any]:
    """
    Identify top 10 most expensive and best value neighbourhoods
    
    Args:
        df: DataFrame containing property data
        location_col: Column name for neighbourhood/location
        price_col: Column name for price data
        rating_col: Optional column name for rating scores
        min_reviews: Minimum number of reviews to consider
        min_properties: Minimum number of properties per neighbourhood
    
    Returns:
        Dictionary with top neighbourhoods data for visualization
    """
    # Filter neighbourhoods with sufficient data
    neighbourhood_stats = []
    
    for neighbourhood in df[location_col].unique():
        neighbourhood_df = df[df[location_col] == neighbourhood]
        
        # Apply minimum filters
        if len(neighbourhood_df) < min_properties:
            continue
        
        if rating_col and rating_col in df.columns:
            # Filter for properties with ratings
            rated_df = neighbourhood_df[neighbourhood_df[rating_col].notna()]
            if len(rated_df) < min_properties:
                continue
        else:
            rated_df = neighbourhood_df
        
        # Calculate statistics
        stats = {
            'neighbourhood': neighbourhood,
            'property_count': len(neighbourhood_df),
            'avg_price': float(neighbourhood_df[price_col].mean()),
            'median_price': float(neighbourhood_df[price_col].median()),
            'min_price': float(neighbourhood_df[price_col].min()),
            'max_price': float(neighbourhood_df[price_col].max())
        }
        
        # Add rating statistics if available
        if rating_col and rating_col in df.columns:
            stats.update({
                'avg_rating': float(rated_df[rating_col].mean()),
                'rating_count': len(rated_df),
                'value_score': float(rated_df[rating_col].mean() / max(1, rated_df[price_col].mean()))
            })
        
        neighbourhood_stats.append(stats)
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(neighbourhood_stats)
    
    if len(stats_df) == 0:
        return {
            'most_expensive': [],
            'best_value': [],
            'statistics': {}
        }
    
    # Identify most expensive neighbourhoods
    most_expensive = stats_df.nlargest(10, 'avg_price')[['neighbourhood', 'avg_price', 'median_price', 'property_count']]
    
    # Identify best value neighbourhoods (if ratings available)
    if rating_col and rating_col in df.columns and 'value_score' in stats_df.columns:
        best_value = stats_df.nlargest(10, 'value_score')[['neighbourhood', 'value_score', 'avg_rating', 'avg_price', 'property_count']]
        best_value['price_per_rating'] = best_value['avg_price'] / best_value['avg_rating']
    else:
        # Fallback: cheapest neighbourhoods with sufficient properties
        best_value = stats_df.nsmallest(10, 'avg_price')[['neighbourhood', 'avg_price', 'median_price', 'property_count']]
        best_value['value_score'] = 1 / best_value['avg_price']  # Inverse of price as value proxy
    
    # Prepare plotly data structures
    expensive_data = {
        'neighbourhoods': most_expensive['neighbourhood'].tolist(),
        'avg_prices': most_expensive['avg_price'].tolist(),
        'median_prices': most_expensive['median_price'].tolist(),
        'property_counts': most_expensive['property_count'].tolist(),
        'title': 'Top 10 Most Expensive Neighbourhoods',
        'color_scale': 'Reds'
    }
    
    value_data = {
        'neighbourhoods': best_value['neighbourhood'].tolist(),
        'value_scores': best_value['value_score'].tolist(),
        'avg_prices': best_value['avg_price'].tolist() if 'avg_price' in best_value.columns else [],
        'avg_ratings': best_value['avg_rating'].tolist() if 'avg_rating' in best_value.columns else [],
        'property_counts': best_value['property_count'].tolist(),
        'title': 'Top 10 Best Value Neighbourhoods',
        'color_scale': 'Greens'
    }
    
    # Overall statistics
    overall_stats = {
        'total_neighbourhoods_analyzed': len(stats_df),
        'avg_price_across_neighbourhoods': float(stats_df['avg_price'].mean()),
        'price_range': {
            'min': float(stats_df['avg_price'].min()),
            'max': float(stats_df['avg_price'].max()),
            'std': float(stats_df['avg_price'].std())
        }
    }
    
    return {
        'most_expensive': expensive_data,
        'best_value': value_data,
        'statistics': overall_stats
    }


# Example usage and helper function
def create_all_analyses(df: pd.DataFrame, 
                        city_center: Tuple[float, float] = (40.7128, -74.0060)) -> Dict[str, Any]:
    """
    Create all analyses for a given DataFrame
    
    Args:
        df: DataFrame containing property data
        city_center: Tuple of (latitude, longitude) for city center
    
    Returns:
        Dictionary containing all analysis results
    """
    analyses = {
        'choropleth': prepare_neighbourhood_choropleth(df),
        'violin_plot': prepare_property_type_violin(df),
        'price_vs_reviews': prepare_price_vs_reviews_scatter(df),
        'price_vs_distance': prepare_price_vs_distance(df, city_center=city_center),
        'top_neighbourhoods': identify_top_neighbourhoods(df)
    }
    
    return analyses


if __name__ == "__main__":
    # Example usage with sample data
    print("Price Analysis Module")
    print("Available functions:")
    print("1. prepare_neighbourhood_choropleth() - Choropleth map data")
    print("2. prepare_property_type_violin() - Violin plot data")
    print("3. prepare_price_vs_reviews_scatter() - Scatter plot data")
    print("4. prepare_price_vs_distance() - Distance analysis data")
    print("5. identify_top_neighbourhoods() - Top neighbourhoods data")
    print("6. create_all_analyses() - Generate all analyses")

"""
config.py - Project Configuration for Greek Airbnb Analysis
===========================================================
Central configuration file for all project settings.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = PROJECT_ROOT / ".cache"

# Create directories if they don't exist
for dir_path in [DATA_DIR, OUTPUT_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Inside Airbnb Data URLs
# Format: http://data.insideairbnb.com/greece/{region}/{city}/{date}/data/{file}.csv.gz
INSIDE_AIRBNB_URLS = {
    "athens": {
        "region": "attica",
        "city": "athens",
        "date": "2023-12-26",  # Latest available data
        "center": (37.9838, 23.7275),  # City center coordinates
    },
    "thessaloniki": {
        "region": "central-macedonia",
        "city": "thessaloniki",
        "date": "2023-12-29",
        "center": (40.6401, 22.9444),
    },
    "crete": {
        "region": "crete",
        "city": "crete",
        "date": "2023-12-28",
        "center": (35.2401, 24.8093),  # Center of Crete
    },
    "mykonos": {
        "region": "south-aegean",
        "city": "mykonos",
        "date": "2023-12-29",
        "center": (37.4467, 25.3289),
    },
    "santorini": {
        "region": "south-aegean",
        "city": "santorini",
        "date": "2023-12-29",
        "center": (36.3932, 25.4615),
    },
}

# File names from Inside Airbnb
FILE_NAMES = {
    "listings": "listings.csv.gz",
    "reviews": "reviews.csv.gz",
    "calendar": "calendar.csv.gz",
}

# Dashboard settings
DASHBOARD_CONFIG = {
    "host": "localhost",
    "port": 8050,
    "debug": False,
}

# Analysis parameters
ANALYSIS_CONFIG = {
    # Price analysis
    "price_outlier_percentile": 0.99,
    "min_properties_per_neighbourhood": 5,
    "top_n_neighbourhoods": 10,
    
    # Seasonal analysis
    "summer_months": [6, 7, 8],
    "winter_months": [12, 1, 2],
    
    # Host analysis
    "multi_listing_threshold": 2,  # Hosts with >2 properties are multi-listing
    
    # Review analysis
    "min_reviews_for_analysis": 5,
    "sentiment_sample_size": 10000,  # Sample size for sentiment analysis
}

# Column mappings (standardized column names)
COLUMN_MAPPINGS = {
    "id": "listing_id",
    "host_id": "host_id",
    "host_is_superhost": "host_is_superhost",
    "neighbourhood": "neighbourhood",
    "neighbourhood_cleansed": "neighbourhood",
    "property_type": "property_type",
    "room_type": "room_type",
    "price": "price",
    "minimum_nights": "minimum_nights",
    "maximum_nights": "maximum_nights",
    "availability_365": "availability_365",
    "number_of_reviews": "number_of_reviews",
    "review_scores_rating": "review_scores_rating",
    "review_scores_accuracy": "review_scores_accuracy",
    "review_scores_cleanliness": "review_scores_cleanliness",
    "review_scores_checkin": "review_scores_checkin",
    "review_scores_communication": "review_scores_communication",
    "review_scores_location": "review_scores_location",
    "review_scores_value": "review_scores_value",
    "latitude": "latitude",
    "longitude": "longitude",
    "square_feet": "square_feet",
    "accommodates": "accommodates",
    "bedrooms": "bedrooms",
    "beds": "beds",
}

# Colors for visualizations
COLOR_SCHEME = {
    "quartiles": {
        1: "#2ecc71",  # Green - Lowest
        2: "#f1c40f",  # Yellow
        3: "#e67e22",  # Orange
        4: "#e74c3c",  # Red - Highest
    },
    "cities": {
        "athens": "#3498db",
        "thessaloniki": "#9b59b6",
        "crete": "#1abc9c",
        "mykonos": "#e74c3c",
        "santorini": "#f39c12",
    },
    "host_types": {
        "superhost": "#27ae60",
        "regular": "#95a5a6",
    },
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": OUTPUT_DIR / "analysis.log",
}


def get_city_url(city: str, file_type: str) -> str:
    """
    Construct the URL for a specific city and file type.
    
    Args:
        city: City name (must be in INSIDE_AIRBNB_URLS)
        file_type: Type of file ('listings', 'reviews', or 'calendar')
    
    Returns:
        Full URL to the CSV.gz file
    """
    if city not in INSIDE_AIRBNB_URLS:
        raise ValueError(f"Unknown city: {city}. Available: {list(INSIDE_AIRBNB_URLS.keys())}")
    
    if file_type not in FILE_NAMES:
        raise ValueError(f"Unknown file type: {file_type}. Available: {list(FILE_NAMES.keys())}")
    
    config = INSIDE_AIRBNB_URLS[city]
    file_name = FILE_NAMES[file_type]
    
    url = (
        f"http://data.insideairbnb.com/greece/"
        f"{config['region']}/{config['city']}/"
        f"{config['date']}/data/{file_name}"
    )
    
    return url


def get_city_center(city: str) -> tuple:
    """Get the center coordinates for a city."""
    if city not in INSIDE_AIRBNB_URLS:
        raise ValueError(f"Unknown city: {city}")
    return INSIDE_AIRBNB_URLS[city]["center"]

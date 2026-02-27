"""
data_loader.py - Airbnb Data Loader for Greek Cities
Downloads and processes Airbnb datasets from insideairbnb.com
"""

import pandas as pd
import numpy as np
import requests
import zipfile
import io
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AirbnbDataLoader:
    """Downloads and processes Airbnb datasets for Greek cities."""
    
    # Base URL for insideairbnb.com
    BASE_URL = "http://data.insideairbnb.com/greece"
    
    # City configurations with their specific paths
    CITIES = {
        'athens': {
            'path': 'attica/athens',
            'listings_url': None,  # Will be constructed dynamically
            'reviews_url': None,
            'calendar_url': None
        },
        'thessaloniki': {
            'path': 'central-macedonia/thessaloniki',
            'listings_url': None,
            'reviews_url': None,
            'calendar_url': None
        },
        'crete': {
            'path': 'crete',
            'listings_url': None,
            'reviews_url': None,
            'calendar_url': None
        },
        'mykonos': {
            'path': 'south-aegean/mykonos',
            'listings_url': None,
            'reviews_url': None,
            'calendar_url': None
        },
        'santorini': {
            'path': 'south-aegean/santorini',
            'listings_url': None,
            'reviews_url': None,
            'calendar_url': None
        }
    }
    
    # File names to download
    FILE_NAMES = ['listings.csv', 'reviews.csv', 'calendar.csv']
    
    def __init__(self, data_dir: str = './airbnb_data'):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        self.listings_data = {}
        self.reviews_data = {}
        self.calendar_data = {}
        
        # Build URLs for each city
        self._build_urls()
        
    def _build_urls(self):
        """Build download URLs for each city and file type."""
        for city, config in self.CITIES.items():
            # Note: insideairbnb.com organizes data by date
            # We'll try to get the most recent data
            # In practice, you might want to specify a specific date
            base_path = f"{self.BASE_URL}/{config['path']}"
            
            # Try to find the most recent data folder
            # This is a simplified approach - in production you might want
            # to implement date detection or use a fixed known date
            config['listings_url'] = f"{base_path}/listings.csv.gz"
            config['reviews_url'] = f"{base_path}/reviews.csv.gz"
            config['calendar_url'] = f"{base_path}/calendar.csv.gz"
            
            logger.info(f"Built URLs for {city}:")
            logger.info(f"  Listings: {config['listings_url']}")
            logger.info(f"  Reviews: {config['reviews_url']}")
            logger.info(f"  Calendar: {config['calendar_url']}")
    
    def _download_file(self, url: str, city: str, file_type: str) -> Optional[pd.DataFrame]:
        """
        Download and load a single file.
        
        Args:
            url: URL to download from
            city: City name
            file_type: Type of file ('listings', 'reviews', or 'calendar')
            
        Returns:
            DataFrame with the data or None if download failed
        """
        file_path = self.data_dir / f"{city}_{file_type}.csv"
        
        # Check if file already exists
        if file_path.exists():
            logger.info(f"Loading existing file: {file_path}")
            try:
                return self._load_file(file_path, file_type, city)
            except Exception as e:
                logger.warning(f"Failed to load existing file: {e}")
        
        # Download the file
        logger.info(f"Downloading {file_type} data for {city} from {url}")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # Check if it's a gzipped file
            if url.endswith('.gz'):
                import gzip
                with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz_file:
                    content = gz_file.read()
            else:
                content = response.content
            
            # Save to file
            with open(file_path, 'wb') as f:
                f.write(content)
            
            logger.info(f"Successfully downloaded and saved: {file_path}")
            
            # Load the data
            return self._load_file(file_path, file_type, city)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {file_type} for {city}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing {file_type} for {city}: {e}")
            return None
    
    def _load_file(self, file_path: Path, file_type: str, city: str) -> pd.DataFrame:
        """
        Load a CSV file into a DataFrame with appropriate dtypes.
        
        Args:
            file_path: Path to the CSV file
            file_type: Type of file ('listings', 'reviews', or 'calendar')
            city: City name
            
        Returns:
            DataFrame with the loaded data
        """
        logger.info(f"Loading {file_type} data from {file_path}")
        
        # Define dtype strategies for each file type
        dtype_strategies = {
            'listings': {
                'id': 'int64',
                'host_id': 'int64',
                'latitude': 'float64',
                'longitude': 'float64',
                'price': 'str',  # Will convert later
                'minimum_nights': 'int64',
                'maximum_nights': 'int64',
                'availability_30': 'int64',
                'availability_60': 'int64',
                'availability_90': 'int64',
                'availability_365': 'int64',
                'number_of_reviews': 'int64',
                'review_scores_rating': 'float64'
            },
            'reviews': {
                'listing_id': 'int64',
                'id': 'int64',
                'reviewer_id': 'int64'
            },
            'calendar': {
                'listing_id': 'int64',
                'price': 'str'  # Will convert later
            }
        }
        
        # Define date columns for each file type
        date_columns = {
            'listings': ['last_scraped', 'host_since', 'calendar_last_scraped'],
            'reviews': ['date'],
            'calendar': ['date']
        }
        
        try:
            # Read CSV with appropriate dtypes
            dtypes = dtype_strategies.get(file_type, {})
            date_cols = date_columns.get(file_type, [])
            
            df = pd.read_csv(
                file_path,
                dtype=dtypes,
                parse_dates=date_cols,
                low_memory=False,
                on_bad_lines='warn'  # Skip problematic lines
            )
            
            # Add city column
            df['city'] = city
            
            # Convert price columns
            if 'price' in df.columns:
                df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
            
            logger.info(f"Loaded {len(df)} rows for {city} {file_type}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise
    
    def download_city_data(self, city: str) -> Tuple[Optional[pd.DataFrame], ...]:
        """
        Download all data for a specific city.
        
        Args:
            city: City name
            
        Returns:
            Tuple of (listings_df, reviews_df, calendar_df) - any can be None if download failed
        """
        if city not in self.CITIES:
            logger.error(f"Unknown city: {city}")
            return None, None, None
        
        config = self.CITIES[city]
        
        # Download each file type
        listings_df = self._download_file(config['listings_url'], city, 'listings')
        reviews_df = self._download_file(config['reviews_url'], city, 'reviews')
        calendar_df = self._download_file(config['calendar_url'], city, 'calendar')
        
        # Store in instance variables
        if listings_df is not None:
            self.listings_data[city] = listings_df
        if reviews_df is not None:
            self.reviews_data[city] = reviews_df
        if calendar_df is not None:
            self.calendar_data[city] = calendar_df
        
        return listings_df, reviews_df, calendar_df
    
    def download_all_cities(self) -> Dict[str, Dict[str, Optional[pd.DataFrame]]]:
        """
        Download data for all cities.
        
        Returns:
            Dictionary with download results for each city
        """
        results = {}
        
        for city in self.CITIES.keys():
            logger.info(f"Downloading data for {city}...")
            listings, reviews, calendar = self.download_city_data(city)
            
            results[city] = {
                'listings': listings is not None,
                'reviews': reviews is not None,
                'calendar': calendar is not None
            }
            
            # Add a small delay to be respectful to the server
            time.sleep(1)
        
        # Log summary
        successful = sum(1 for r in results.values() if all(r.values()))
        logger.info(f"Successfully downloaded data for {successful}/{len(self.CITIES)} cities")
        
        return results
    
    def merge_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Merge data from all cities into master DataFrames.
        
        Returns:
            Tuple of (master_listings, master_reviews, master_calendar)
        """
        # Merge listings
        if self.listings_data:
            master_listings = pd.concat(self.listings_data.values(), ignore_index=True)
            logger.info(f"Merged listings: {len(master_listings)} rows from {len(self.listings_data)} cities")
        else:
            master_listings = pd.DataFrame()
            logger.warning("No listings data to merge")
        
        # Merge reviews
        if self.reviews_data:
            master_reviews = pd.concat(self.reviews_data.values(), ignore_index=True)
            logger.info(f"Merged reviews: {len(master_reviews)} rows from {len(self.reviews_data)} cities")
        else:
            master_reviews = pd.DataFrame()
            logger.warning("No reviews data to merge")
        
        # Merge calendar
        if self.calendar_data:
            master_calendar = pd.concat(self.calendar_data.values(), ignore_index=True)
            logger.info(f"Merged calendar: {len(master_calendar)} rows from {len(self.calendar_data)} cities")
        else:
            master_calendar = pd.DataFrame()
            logger.warning("No calendar data to merge")
        
        return master_listings, master_reviews, master_calendar
    
    def get_summary_stats(self) -> pd.DataFrame:
        """
        Get summary statistics for the loaded data.
        
        Returns:
            DataFrame with summary statistics
        """
        summary_data = []
        
        for city in self.CITIES.keys():
            listings = self.listings_data.get(city)
            reviews = self.reviews_data.get(city)
            calendar = self.calendar_data.get(city)
            
            summary_data.append({
                'city': city,
                'listings_count': len(listings) if listings is not None else 0,
                'reviews_count': len(reviews) if reviews is not None else 0,
                'calendar_entries': len(calendar) if calendar is not None else 0,
                'avg_price': listings['price'].mean() if listings is not None and 'price' in listings.columns else None,
                'data_complete': all([listings is not None, reviews is not None, calendar is not None])
            })
        
        return pd.DataFrame(summary_data)


def main():
    """Main function to demonstrate usage."""
    # Initialize the data loader
    loader = AirbnbDataLoader(data_dir='./airbnb_greece_data')
    
    # Download data for all cities
    logger.info("Starting download of Airbnb data for Greek cities...")
    results = loader.download_all_cities()
    
    # Print download results
    print("\nDownload Results:")
    print("-" * 50)
    for city, result in results.items():
        status = "✓" if all(result.values()) else "✗"
        print(f"{status} {city.title()}: "
              f"Listings: {'✓' if result['listings'] else '✗'}, "
              f"Reviews: {'✓' if result['reviews'] else '✗'}, "
              f"Calendar: {'✓' if result['calendar'] else '✗'}")
    
    # Merge all data
    print("\nMerging data from all cities...")
    master_listings, master_reviews, master_calendar = loader.merge_all_data()
    
    # Print summary
    print("\nMaster DataFrames Summary:")
    print("-" * 50)
    print(f"Listings: {len(master_listings):,} rows, {len(master_listings.columns)} columns")
    print(f"Reviews: {len(master_reviews):,} rows, {len(master_reviews.columns)} columns")
    print(f"Calendar: {len(master_calendar):,} rows, {len(master_calendar.columns)} columns")
    
    # Get detailed summary
    summary = loader.get_summary_stats()
    print("\nDetailed Summary by City:")
    print("-" * 50)
    print(summary.to_string(index=False))
    
    # Save merged data to CSV files
    if len(master_listings) > 0:
        master_listings.to_csv('./master_listings.csv', index=False)
        print("\nSaved master_listings.csv")
    
    if len(master_reviews) > 0:
        master_reviews.to_csv('./master_reviews.csv', index=False)
        print("Saved master_reviews.csv")
    
    if len(master_calendar) > 0:
        master_calendar.to_csv('./master_calendar.csv', index=False)
        print("Saved master_calendar.csv")
    
    print("\nData loading complete!")


if __name__ == "__main__":
    main()

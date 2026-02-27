"""
data_cleaning.py - Data Cleaning Module for Greek Airbnb Analysis
=================================================================
Cleans and preprocesses Airbnb data from Inside Airbnb.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List, Tuple
from pathlib import Path

from config import COLUMN_MAPPINGS, ANALYSIS_CONFIG

logger = logging.getLogger(__name__)


class DataCleaner:
    """Cleans and preprocesses Airbnb datasets."""
    
    def __init__(self):
        self.price_multiplier = 1.0  # For currency conversion if needed
    
    def clean_listings(self, df: pd.DataFrame, city: str) -> pd.DataFrame:
        """
        Clean listings DataFrame.
        
        Args:
            df: Raw listings DataFrame
            city: City name
        
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Add city column
        df['city'] = city
        
        # Clean price column
        df = self._clean_price_column(df)
        
        # Clean boolean columns
        df = self._clean_boolean_columns(df)
        
        # Clean numeric columns
        df = self._clean_numeric_columns(df)
        
        # Clean date columns
        df = self._clean_date_columns(df)
        
        # Calculate derived columns
        df = self._calculate_derived_columns(df)
        
        # Remove extreme outliers
        df = self._remove_outliers(df)
        
        logger.info(f"Cleaned listings for {city}: {len(df)} rows")
        return df
    
    def _clean_price_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean price column - remove $, commas, convert to float."""
        if 'price' in df.columns:
            # Handle string prices with $ and commas
            if df['price'].dtype == 'object':
                df['price'] = (
                    df['price']
                    .astype(str)
                    .str.replace('$', '', regex=False)
                    .str.replace(',', '')
                )
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            
            # Remove listings with invalid prices
            invalid_prices = df['price'].isna() | (df['price'] <= 0)
            if invalid_prices.sum() > 0:
                logger.warning(f"Found {invalid_prices.sum()} listings with invalid prices")
                df = df[~invalid_prices]
        
        return df
    
    def _clean_boolean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean boolean columns (host_is_superhost, instant_bookable, etc.)."""
        bool_mappings = {
            't': True, 'T': True, 'true': True, 'True': True, '1': True,
            'f': False, 'F': False, 'false': False, 'False': False, '0': False,
        }
        
        bool_columns = [
            'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified',
            'instant_bookable', 'has_availability'
        ]
        
        for col in bool_columns:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].map(bool_mappings)
                df[col] = df[col].fillna(False).astype(bool)
        
        return df
    
    def _clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert numeric columns."""
        numeric_columns = {
            'host_listings_count': 'Int64',
            'host_total_listings_count': 'Int64',
            'accommodates': 'Int64',
            'bedrooms': 'float64',
            'beds': 'float64',
            'square_feet': 'float64',
            'minimum_nights': 'Int64',
            'maximum_nights': 'Int64',
            'minimum_minimum_nights': 'Int64',
            'maximum_minimum_nights': 'Int64',
            'minimum_maximum_nights': 'Int64',
            'maximum_maximum_nights': 'Int64',
            'minimum_nights_avg_ntm': 'float64',
            'maximum_nights_avg_ntm': 'float64',
            'availability_30': 'Int64',
            'availability_60': 'Int64',
            'availability_90': 'Int64',
            'availability_365': 'Int64',
            'number_of_reviews': 'Int64',
            'number_of_reviews_ltm': 'Int64',
            'number_of_reviews_l30d': 'Int64',
            'review_scores_rating': 'float64',
            'review_scores_accuracy': 'float64',
            'review_scores_cleanliness': 'float64',
            'review_scores_checkin': 'float64',
            'review_scores_communication': 'float64',
            'review_scores_location': 'float64',
            'review_scores_value': 'float64',
            'reviews_per_month': 'float64',
            'latitude': 'float64',
            'longitude': 'float64',
        }
        
        for col, dtype in numeric_columns.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _clean_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse date columns."""
        date_columns = [
            'last_scraped', 'host_since', 'calendar_last_scraped',
            'first_review', 'last_review'
        ]
        
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def _calculate_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived columns."""
        # Price per square meter (where square_feet is available)
        if 'price' in df.columns and 'square_feet' in df.columns:
            df['price_per_m2'] = np.where(
                df['square_feet'].notna() & (df['square_feet'] > 0),
                df['price'] / (df['square_feet'] * 0.092903),  # Convert sq ft to sq m
                np.nan
            )
        
        # Occupancy rate estimate (based on availability_365)
        if 'availability_365' in df.columns:
            df['occupancy_rate'] = 1 - (df['availability_365'] / 365)
            df['occupancy_rate'] = df['occupancy_rate'].clip(0, 1)
        
        # Estimated annual revenue
        if 'price' in df.columns and 'occupancy_rate' in df.columns:
            df['estimated_revenue'] = df['price'] * df['occupancy_rate'] * 365
        
        # Price category
        if 'price' in df.columns:
            price_quantiles = df['price'].quantile([0.25, 0.5, 0.75])
            df['price_category'] = pd.cut(
                df['price'],
                bins=[0, price_quantiles[0.25], price_quantiles[0.5], 
                      price_quantiles[0.75], np.inf],
                labels=['Budget', 'Economy', 'Premium', 'Luxury']
            )
        
        # Host experience (years since host_since)
        if 'host_since' in df.columns:
            df['host_experience_years'] = (
                pd.Timestamp.now() - df['host_since']
            ).dt.days / 365.25
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove extreme outliers."""
        initial_count = len(df)
        
        # Price outliers
        if 'price' in df.columns:
            price_upper = df['price'].quantile(ANALYSIS_CONFIG['price_outlier_percentile'])
            df = df[df['price'] <= price_upper]
        
        # Coordinate validation
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df = df[
                (df['latitude'].between(34, 42)) &  # Greece bounds
                (df['longitude'].between(19, 30))
            ]
        
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} outliers")
        
        return df
    
    def clean_calendar(self, df: pd.DataFrame, city: str) -> pd.DataFrame:
        """
        Clean calendar DataFrame.
        
        Args:
            df: Raw calendar DataFrame
            city: City name
        
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        df['city'] = city
        
        # Parse date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Clean price column
        df = self._clean_price_column(df)
        
        # Clean available column
        if 'available' in df.columns:
            df['available'] = df['available'].map({
                't': True, 'T': True, 'true': True, 'f': False, 'F': False, 'false': False
            }).fillna(False)
        
        # Extract month and year for seasonal analysis
        if 'date' in df.columns:
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
            df['year_month'] = df['date'].dt.to_period('M').astype(str)
        
        logger.info(f"Cleaned calendar for {city}: {len(df)} rows")
        return df
    
    def clean_reviews(self, df: pd.DataFrame, city: str) -> pd.DataFrame:
        """
        Clean reviews DataFrame.
        
        Args:
            df: Raw reviews DataFrame
            city: City name
        
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        df['city'] = city
        
        # Parse date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Clean comments
        if 'comments' in df.columns:
            df['comments'] = df['comments'].astype(str)
            df['comments'] = df['comments'].replace('nan', '')
            df['comments'] = df['comments'].replace('None', '')
        
        logger.info(f"Cleaned reviews for {city}: {len(df)} rows")
        return df


def merge_city_data(
    listings_dict: Dict[str, pd.DataFrame],
    reviews_dict: Dict[str, pd.DataFrame],
    calendar_dict: Dict[str, pd.DataFrame]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Merge data from all cities into master DataFrames.
    
    Args:
        listings_dict: Dictionary of listings DataFrames by city
        reviews_dict: Dictionary of reviews DataFrames by city
        calendar_dict: Dictionary of calendar DataFrames by city
    
    Returns:
        Tuple of (master_listings, master_reviews, master_calendar)
    """
    master_listings = pd.concat(listings_dict.values(), ignore_index=True)
    master_reviews = pd.concat(reviews_dict.values(), ignore_index=True)
    master_calendar = pd.concat(calendar_dict.values(), ignore_index=True)
    
    logger.info(f"Merged data: {len(master_listings)} listings, "
                f"{len(master_reviews)} reviews, {len(master_calendar)} calendar entries")
    
    return master_listings, master_reviews, master_calendar


def save_cleaned_data(
    listings: pd.DataFrame,
    reviews: pd.DataFrame,
    calendar: pd.DataFrame,
    output_dir: Path
):
    """Save cleaned data to CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    listings.to_csv(output_dir / "listings_cleaned.csv", index=False)
    reviews.to_csv(output_dir / "reviews_cleaned.csv", index=False)
    calendar.to_csv(output_dir / "calendar_cleaned.csv", index=False)
    
    logger.info(f"Saved cleaned data to {output_dir}")


if __name__ == "__main__":
    # Example usage
    print("Data Cleaning Module")
    print("=" * 50)
    print("This module cleans and preprocesses Airbnb data.")
    print("\nUsage:")
    print("  from data_cleaning import DataCleaner")
    print("  cleaner = DataCleaner()")
    print("  cleaned_df = cleaner.clean_listings(raw_df, city='athens')")

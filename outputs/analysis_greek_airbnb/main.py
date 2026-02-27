#!/usr/bin/env python3
"""
main.py - Entry Point for Greek Airbnb Analysis Project
======================================================
Main script to run the complete analysis pipeline.

Usage:
    python main.py --download    # Download data from Inside Airbnb
    python main.py --clean       # Clean and preprocess data
    python main.py --analyze     # Run all analyses
    python main.py --dashboard   # Launch the dashboard
    python main.py --all         # Run complete pipeline

Examples:
    python main.py --download --clean --analyze
    python main.py --all
    python main.py --dashboard
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# Import project modules
from config import DATA_DIR, OUTPUT_DIR, INSIDE_AIRBNB_URLS, get_city_url
from data_loader import AirbnbDataLoader
from data_cleaning import DataCleaner, merge_city_data, save_cleaned_data
from price_analysis import create_all_analyses as create_price_analyses
from seasonal_analysis import generate_seasonal_report
from host_analysis import generate_host_analysis_report
from review_analysis import ReviewAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / 'pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


class AirbnbAnalysisPipeline:
    """Main pipeline for Greek Airbnb analysis."""
    
    def __init__(self):
        self.data_loader = AirbnbDataLoader(data_dir=DATA_DIR)
        self.data_cleaner = DataCleaner()
        self.review_analyzer = ReviewAnalyzer()
        self.raw_data = {
            'listings': {},
            'reviews': {},
            'calendar': {}
        }
        self.cleaned_data = {
            'listings': None,
            'reviews': None,
            'calendar': None
        }
    
    def download_data(self, cities: Optional[list] = None):
        """
        Download data for specified cities.
        
        Args:
            cities: List of city names (None = all cities)
        """
        if cities is None:
            cities = list(INSIDE_AIRBNB_URLS.keys())
        
        logger.info("=" * 60)
        logger.info("STEP 1: Downloading Data from Inside Airbnb")
        logger.info("=" * 60)
        
        for city in cities:
            logger.info(f"\n📥 Downloading data for {city.title()}...")
            try:
                listings, reviews, calendar = self.data_loader.download_city_data(city)
                
                if listings is not None:
                    self.raw_data['listings'][city] = listings
                    logger.info(f"  ✓ Listings: {len(listings):,} rows")
                else:
                    logger.warning(f"  ✗ Failed to download listings for {city}")
                
                if reviews is not None:
                    self.raw_data['reviews'][city] = reviews
                    logger.info(f"  ✓ Reviews: {len(reviews):,} rows")
                else:
                    logger.warning(f"  ✗ Failed to download reviews for {city}")
                
                if calendar is not None:
                    self.raw_data['calendar'][city] = calendar
                    logger.info(f"  ✓ Calendar: {len(calendar):,} rows")
                else:
                    logger.warning(f"  ✗ Failed to download calendar for {city}")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"  ✗ Error downloading {city}: {e}")
        
        # Summary
        total_listings = sum(len(df) for df in self.raw_data['listings'].values())
        total_reviews = sum(len(df) for df in self.raw_data['reviews'].values())
        total_calendar = sum(len(df) for df in self.raw_data['calendar'].values())
        
        logger.info("\n" + "=" * 60)
        logger.info("Download Summary:")
        logger.info(f"  Total Listings: {total_listings:,}")
        logger.info(f"  Total Reviews: {total_reviews:,}")
        logger.info(f"  Total Calendar Entries: {total_calendar:,}")
        logger.info("=" * 60)
    
    def clean_data(self):
        """Clean and preprocess all downloaded data."""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Cleaning and Preprocessing Data")
        logger.info("=" * 60)
        
        # Clean listings
        cleaned_listings = {}
        for city, df in self.raw_data['listings'].items():
            logger.info(f"\n🧹 Cleaning listings for {city.title()}...")
            try:
                cleaned = self.data_cleaner.clean_listings(df, city)
                cleaned_listings[city] = cleaned
                logger.info(f"  ✓ Cleaned: {len(cleaned):,} rows")
            except Exception as e:
                logger.error(f"  ✗ Error cleaning {city}: {e}")
        
        # Clean reviews
        cleaned_reviews = {}
        for city, df in self.raw_data['reviews'].items():
            logger.info(f"\n🧹 Cleaning reviews for {city.title()}...")
            try:
                cleaned = self.data_cleaner.clean_reviews(df, city)
                cleaned_reviews[city] = cleaned
                logger.info(f"  ✓ Cleaned: {len(cleaned):,} rows")
            except Exception as e:
                logger.error(f"  ✗ Error cleaning {city}: {e}")
        
        # Clean calendar
        cleaned_calendar = {}
        for city, df in self.raw_data['calendar'].items():
            logger.info(f"\n🧹 Cleaning calendar for {city.title()}...")
            try:
                cleaned = self.data_cleaner.clean_calendar(df, city)
                cleaned_calendar[city] = cleaned
                logger.info(f"  ✓ Cleaned: {len(cleaned):,} rows")
            except Exception as e:
                logger.error(f"  ✗ Error cleaning {city}: {e}")
        
        # Merge data
        logger.info("\n🔄 Merging data from all cities...")
        try:
            self.cleaned_data['listings'], \
            self.cleaned_data['reviews'], \
            self.cleaned_data['calendar'] = merge_city_data(
                cleaned_listings, cleaned_reviews, cleaned_calendar
            )
            
            logger.info("  ✓ Data merged successfully")
        except Exception as e:
            logger.error(f"  ✗ Error merging data: {e}")
        
        # Save cleaned data
        logger.info("\n💾 Saving cleaned data...")
        try:
            save_cleaned_data(
                self.cleaned_data['listings'],
                self.cleaned_data['reviews'],
                self.cleaned_data['calendar'],
                OUTPUT_DIR
            )
            logger.info(f"  ✓ Data saved to {OUTPUT_DIR}")
        except Exception as e:
            logger.error(f"  ✗ Error saving data: {e}")
    
    def run_analyses(self):
        """Run all analyses and generate reports."""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Running Analyses")
        logger.info("=" * 60)
        
        if self.cleaned_data['listings'] is None:
            logger.error("No cleaned data available. Run --clean first.")
            return
        
        listings = self.cleaned_data['listings']
        reviews = self.cleaned_data['reviews']
        calendar = self.cleaned_data['calendar']
        
        # 1. Price Analysis
        logger.info("\n📊 Running Price Analysis...")
        try:
            for city in listings['city'].unique():
                city_data = listings[listings['city'] == city]
                analyses = create_price_analyses(
                    city_data, 
                    city_center=get_city_url(city, 'listings')  # Use config instead
                )
                logger.info(f"  ✓ Price analysis for {city}")
        except Exception as e:
            logger.error(f"  ✗ Error in price analysis: {e}")
        
        # 2. Seasonal Analysis
        if calendar is not None and len(calendar) > 0:
            logger.info("\n📅 Running Seasonal Analysis...")
            try:
                seasonal_report = generate_seasonal_report(
                    calendar,
                    start_date=calendar['date'].min().strftime('%Y-%m-%d'),
                    end_date=calendar['date'].max().strftime('%Y-%m-%d')
                )
                
                # Save report
                pd.json_normalize(seasonal_report).to_json(
                    OUTPUT_DIR / 'seasonal_analysis.json'
                )
                logger.info("  ✓ Seasonal analysis complete")
                
                # Print key findings
                if 'premium_analysis' in seasonal_report:
                    logger.info("\n  📈 Seasonal Insights:")
                    for island, data in seasonal_report['premium_analysis'].items():
                        if data.get('premium_percentage'):
                            logger.info(f"    • {island.title()}: {data['premium_percentage']:.1f}% summer premium")
                
            except Exception as e:
                logger.error(f"  ✗ Error in seasonal analysis: {e}")
        
        # 3. Host Analysis
        logger.info("\n👤 Running Host Analysis...")
        try:
            host_report = generate_host_analysis_report(listings)
            
            # Save report
            with open(OUTPUT_DIR / 'host_analysis.json', 'w') as f:
                import json
                json.dump(host_report, f, indent=2, default=str)
            
            logger.info("  ✓ Host analysis complete")
            
            # Print key findings
            superhost_pct = host_report['superhost_analysis']['summary_statistics']['superhost_percentage']
            multi_host_pct = host_report['multi_listing_analysis']['overall_statistics']['multi_listing_host_percentage']
            
            logger.info(f"\n  👤 Host Insights:")
            logger.info(f"    • {superhost_pct:.1f}% of listings are from Superhosts")
            logger.info(f"    • {multi_host_pct:.1f}% of hosts are multi-listing operators")
            
        except Exception as e:
            logger.error(f"  ✗ Error in host analysis: {e}")
        
        # 4. Review Analysis
        if reviews is not None and len(reviews) > 0:
            logger.info("\n💬 Running Review Analysis...")
            try:
                review_report = self.review_analyzer.generate_review_report(
                    listings, reviews, sample_size=5000
                )
                
                # Save report
                with open(OUTPUT_DIR / 'review_analysis.json', 'w') as f:
                    import json
                    json.dump(review_report, f, indent=2, default=str)
                
                logger.info("  ✓ Review analysis complete")
                
            except Exception as e:
                logger.error(f"  ✗ Error in review analysis: {e}")
        
        logger.info("\n" + "=" * 60)
        logger.info("All analyses complete!")
        logger.info("=" * 60)
    
    def generate_summary_report(self):
        """Generate a summary report with key findings."""
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING SUMMARY REPORT")
        logger.info("=" * 60)
        
        if self.cleaned_data['listings'] is None:
            logger.error("No data available for report")
            return
        
        listings = self.cleaned_data['listings']
        
        report_lines = []
        report_lines.append("# Greek Airbnb Analysis - Key Findings")
        report_lines.append("\n## Overview")
        report_lines.append(f"- **Total Listings Analyzed:** {len(listings):,}")
        report_lines.append(f"- **Cities:** {', '.join(listings['city'].unique())}")
        report_lines.append(f"- **Date Range:** Analysis run on {pd.Timestamp.now().strftime('%Y-%m-%d')}")
        
        # Price insights
        report_lines.append("\n## Price Analysis")
        city_prices = listings.groupby('city')['price'].agg(['mean', 'median', 'min', 'max'])
        for city, row in city_prices.iterrows():
            report_lines.append(f"\n### {city.title()}")
            report_lines.append(f"- Average Price: €{row['mean']:.0f}/night")
            report_lines.append(f"- Median Price: €{row['median']:.0f}/night")
            report_lines.append(f"- Price Range: €{row['min']:.0f} - €{row['max']:.0f}")
        
        # Find most expensive and cheapest cities
        most_expensive = city_prices['mean'].idxmax()
        cheapest = city_prices['mean'].idxmin()
        report_lines.append(f"\n**Key Finding:** {most_expensive.title()} is the most expensive city "
                           f"(€{city_prices.loc[most_expensive, 'mean']:.0f}/night on average)")
        report_lines.append(f"**Key Finding:** {cheapest.title()} offers the best value "
                           f"(€{city_prices.loc[cheapest, 'mean']:.0f}/night on average)")
        
        # Room type distribution
        if 'room_type' in listings.columns:
            report_lines.append("\n## Room Types")
            room_types = listings['room_type'].value_counts()
            for room_type, count in room_types.items():
                pct = count / len(listings) * 100
                report_lines.append(f"- {room_type}: {count:,} ({pct:.1f}%)")
        
        # Superhost stats
        if 'host_is_superhost' in listings.columns:
            superhost_pct = listings['host_is_superhost'].mean() * 100
            report_lines.append(f"\n## Host Analysis")
            report_lines.append(f"- Superhosts: {superhost_pct:.1f}% of all listings")
        
        # Review stats
        if 'number_of_reviews' in listings.columns:
            avg_reviews = listings['number_of_reviews'].mean()
            report_lines.append(f"\n## Reviews")
            report_lines.append(f"- Average Reviews per Listing: {avg_reviews:.1f}")
            
        if 'review_scores_rating' in listings.columns:
            avg_rating = listings['review_scores_rating'].mean()
            report_lines.append(f"- Average Rating: {avg_rating:.1f}/100")
        
        report_text = '\n'.join(report_lines)
        
        # Save report
        report_path = OUTPUT_DIR / 'KEY_FINDINGS.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"\n✓ Summary report saved to: {report_path}")
        
        # Print report
        logger.info("\n" + "=" * 60)
        logger.info("KEY FINDINGS")
        logger.info("=" * 60)
        logger.info(report_text)
        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Greek Airbnb Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --download              # Download all data
  python main.py --download --clean      # Download and clean
  python main.py --clean --analyze       # Clean existing data and analyze
  python main.py --all                   # Run complete pipeline
  python main.py --dashboard             # Launch dashboard only
        """
    )
    
    parser.add_argument('--download', action='store_true',
                        help='Download data from Inside Airbnb')
    parser.add_argument('--clean', action='store_true',
                        help='Clean and preprocess data')
    parser.add_argument('--analyze', action='store_true',
                        help='Run all analyses')
    parser.add_argument('--dashboard', action='store_true',
                        help='Launch the dashboard')
    parser.add_argument('--all', action='store_true',
                        help='Run complete pipeline (download, clean, analyze)')
    parser.add_argument('--cities', nargs='+',
                        default=list(INSIDE_AIRBNB_URLS.keys()),
                        help='Cities to process (default: all)')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any([args.download, args.clean, args.analyze, args.dashboard, args.all]):
        parser.print_help()
        return
    
    # Run dashboard
    if args.dashboard:
        logger.info("Launching dashboard...")
        import subprocess
        subprocess.run([sys.executable, 'dashboard.py'])
        return
    
    # Run pipeline
    pipeline = AirbnbAnalysisPipeline()
    
    if args.all:
        args.download = True
        args.clean = True
        args.analyze = True
    
    if args.download:
        pipeline.download_data(cities=args.cities)
    
    if args.clean:
        if not args.download:
            # Try to load existing data
            logger.info("Loading existing raw data...")
        pipeline.clean_data()
    
    if args.analyze:
        pipeline.run_analyses()
        pipeline.generate_summary_report()
    
    logger.info("\n✅ Pipeline execution complete!")


if __name__ == "__main__":
    main()

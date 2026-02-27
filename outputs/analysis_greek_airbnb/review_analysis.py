"""
review_analysis.py - Review Analysis Module for Greek Airbnb Analysis
=====================================================================
Analyzes reviews including sentiment analysis for Greek and English text.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import re
import logging

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available. Sentiment analysis will be limited.")

logger = logging.getLogger(__name__)


class ReviewAnalyzer:
    """Analyzes Airbnb reviews including sentiment analysis."""
    
    def __init__(self):
        self.sentiment_cache = {}
    
    def analyze_review_scores(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze review score distributions per city.
        
        Args:
            df: Listings DataFrame with review scores
        
        Returns:
            Dictionary with review score analysis
        """
        results = {}
        
        for city in df['city'].unique():
            city_data = df[df['city'] == city]
            
            # Basic review statistics
            score_stats = {
                'total_listings': len(city_data),
                'listings_with_reviews': city_data['number_of_reviews'].gt(0).sum(),
                'avg_number_of_reviews': city_data['number_of_reviews'].mean(),
                'avg_review_score': city_data['review_scores_rating'].mean(),
                'median_review_score': city_data['review_scores_rating'].median(),
                'score_std': city_data['review_scores_rating'].std(),
            }
            
            # Score distribution (bins)
            if 'review_scores_rating' in city_data.columns:
                bins = [0, 60, 70, 80, 90, 95, 100]
                labels = ['<60', '60-70', '70-80', '80-90', '90-95', '95-100']
                city_data['score_bin'] = pd.cut(
                    city_data['review_scores_rating'], 
                    bins=bins, labels=labels
                )
                score_dist = city_data['score_bin'].value_counts().to_dict()
                score_stats['score_distribution'] = {str(k): v for k, v in score_dist.items()}
            
            # Detailed score categories
            score_categories = [
                'review_scores_accuracy',
                'review_scores_cleanliness',
                'review_scores_checkin',
                'review_scores_communication',
                'review_scores_location',
                'review_scores_value'
            ]
            
            for category in score_categories:
                if category in city_data.columns:
                    score_stats[f'avg_{category}'] = city_data[category].mean()
            
            results[city] = score_stats
        
        return results
    
    def analyze_sentiment(
        self, 
        reviews_df: pd.DataFrame,
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform sentiment analysis on review comments.
        
        Args:
            reviews_df: Reviews DataFrame with 'comments' column
            sample_size: Number of reviews to sample (None = all)
        
        Returns:
            Dictionary with sentiment analysis results
        """
        if not TEXTBLOB_AVAILABLE:
            logger.warning("TextBlob not available. Skipping sentiment analysis.")
            return {"error": "TextBlob not installed"}
        
        if 'comments' not in reviews_df.columns:
            return {"error": "No comments column found"}
        
        # Sample if needed
        if sample_size and len(reviews_df) > sample_size:
            reviews_df = reviews_df.sample(n=sample_size, random_state=42)
        
        sentiments = []
        
        for _, row in reviews_df.iterrows():
            comment = str(row['comments'])
            
            # Skip empty comments
            if not comment or comment in ['nan', 'None', '']:
                continue
            
            # Detect language and analyze
            sentiment = self._analyze_comment_sentiment(comment)
            sentiment['listing_id'] = row.get('listing_id', None)
            sentiment['city'] = row.get('city', None)
            sentiment['date'] = row.get('date', None)
            
            sentiments.append(sentiment)
        
        if not sentiments:
            return {"error": "No valid comments to analyze"}
        
        # Aggregate results
        sentiment_df = pd.DataFrame(sentiments)
        
        results = {
            'total_analyzed': len(sentiments),
            'overall_sentiment': {
                'polarity_mean': sentiment_df['polarity'].mean(),
                'polarity_std': sentiment_df['polarity'].std(),
                'subjectivity_mean': sentiment_df['subjectivity'].mean(),
                'positive_pct': (sentiment_df['polarity'] > 0).mean() * 100,
                'neutral_pct': (sentiment_df['polarity'] == 0).mean() * 100,
                'negative_pct': (sentiment_df['polarity'] < 0).mean() * 100,
            },
            'by_city': self._sentiment_by_city(sentiment_df),
            'by_language': sentiment_df['language'].value_counts().to_dict(),
        }
        
        return results
    
    def _analyze_comment_sentiment(self, comment: str) -> Dict[str, Any]:
        """Analyze sentiment of a single comment."""
        # Detect language (simple heuristic)
        language = self._detect_language(comment)
        
        # Clean comment
        clean_comment = self._clean_comment(comment)
        
        # Analyze sentiment with TextBlob
        try:
            blob = TextBlob(clean_comment)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
        except Exception as e:
            logger.debug(f"Sentiment analysis failed: {e}")
            polarity = 0
            subjectivity = 0
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'language': language,
            'comment_length': len(clean_comment),
        }
    
    def _detect_language(self, text: str) -> str:
        """
        Simple language detection based on character patterns.
        
        Returns:
            'greek', 'english', or 'other'
        """
        # Greek Unicode range
        greek_chars = len(re.findall(r'[\u0370-\u03FF\u1F00-\u1FFF]', text))
        total_chars = len(re.findall(r'[a-zA-Z\u0370-\u03FF]', text))
        
        if total_chars == 0:
            return 'unknown'
        
        greek_ratio = greek_chars / total_chars
        
        if greek_ratio > 0.5:
            return 'greek'
        elif greek_ratio > 0.1:
            return 'mixed'
        else:
            return 'english'
    
    def _clean_comment(self, comment: str) -> str:
        """Clean comment text for analysis."""
        # Remove URLs
        comment = re.sub(r'http[s]?://\S+', '', comment)
        # Remove extra whitespace
        comment = re.sub(r'\s+', ' ', comment)
        # Remove special characters but keep letters and basic punctuation
        comment = re.sub(r'[^\w\s.,!?-]', '', comment)
        return comment.strip()
    
    def _sentiment_by_city(self, sentiment_df: pd.DataFrame) -> Dict[str, Any]:
        """Aggregate sentiment by city."""
        results = {}
        
        for city in sentiment_df['city'].unique():
            if pd.isna(city):
                continue
            
            city_data = sentiment_df[sentiment_df['city'] == city]
            
            results[city] = {
                'avg_polarity': city_data['polarity'].mean(),
                'avg_subjectivity': city_data['subjectivity'].mean(),
                'positive_pct': (city_data['polarity'] > 0).mean() * 100,
                'neutral_pct': (city_data['polarity'] == 0).mean() * 100,
                'negative_pct': (city_data['polarity'] < 0).mean() * 100,
                'review_count': len(city_data),
            }
        
        return results
    
    def calculate_correlations(self, listings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlations between review scores, price, and occupancy.
        
        Args:
            listings_df: Listings DataFrame
        
        Returns:
            Correlation matrix as DataFrame
        """
        # Columns to correlate
        corr_columns = [
            'review_scores_rating',
            'review_scores_accuracy',
            'review_scores_cleanliness',
            'review_scores_checkin',
            'review_scores_communication',
            'review_scores_location',
            'review_scores_value',
            'price',
            'number_of_reviews',
        ]
        
        # Add occupancy_rate if available
        if 'occupancy_rate' in listings_df.columns:
            corr_columns.append('occupancy_rate')
        
        # Select available columns
        available_cols = [c for c in corr_columns if c in listings_df.columns]
        
        if len(available_cols) < 2:
            logger.warning("Not enough columns for correlation analysis")
            return pd.DataFrame()
        
        corr_data = listings_df[available_cols].copy()
        
        # Calculate correlation matrix
        correlation_matrix = corr_data.corr()
        
        return correlation_matrix
    
    def get_review_keywords(
        self, 
        reviews_df: pd.DataFrame,
        top_n: int = 20
    ) -> Dict[str, List[str]]:
        """
        Extract common keywords from reviews.
        
        Args:
            reviews_df: Reviews DataFrame
            top_n: Number of top keywords to return
        
        Returns:
            Dictionary with keywords by city
        """
        # Common Greek and English stopwords
        stopwords = set([
            'the', 'and', 'to', 'a', 'of', 'is', 'it', 'in', 'that', 'for',
            'was', 'on', 'with', 'as', 'at', 'this', 'but', 'are', 'be',
            'και', 'το', 'η', 'για', 'την', 'του', 'της', 'απο', 'με', 'σε',
            'που', 'οι', 'τα', 'τον', 'στο', 'στην', 'μου', 'σου', 'μας'
        ])
        
        results = {}
        
        for city in reviews_df['city'].unique():
            if pd.isna(city):
                continue
            
            city_reviews = reviews_df[reviews_df['city'] == city]['comments'].astype(str)
            
            # Combine all reviews
            text = ' '.join(city_reviews.tolist()).lower()
            
            # Extract words
            words = re.findall(r'\b[a-zA-Zα-ωΑ-Ω]+\b', text)
            
            # Filter stopwords and short words
            words = [w for w in words if w not in stopwords and len(w) > 3]
            
            # Count frequency
            word_freq = Counter(words)
            
            results[city] = [word for word, _ in word_freq.most_common(top_n)]
        
        return results
    
    def generate_review_report(
        self,
        listings_df: pd.DataFrame,
        reviews_df: pd.DataFrame,
        sample_size: int = 5000
    ) -> Dict[str, Any]:
        """
        Generate comprehensive review analysis report.
        
        Args:
            listings_df: Listings DataFrame
            reviews_df: Reviews DataFrame
            sample_size: Number of reviews to sample for sentiment analysis
        
        Returns:
            Complete review analysis report
        """
        logger.info("Generating review analysis report...")
        
        report = {
            'review_scores': self.analyze_review_scores(listings_df),
            'sentiment_analysis': self.analyze_sentiment(reviews_df, sample_size),
            'correlations': self.calculate_correlations(listings_df).to_dict(),
            'top_keywords': self.get_review_keywords(reviews_df),
        }
        
        logger.info("Review analysis report complete")
        return report


if __name__ == "__main__":
    print("Review Analysis Module")
    print("=" * 50)
    print("\nAvailable functions:")
    print("  - analyze_review_scores(): Review score distributions per city")
    print("  - analyze_sentiment(): Sentiment analysis of comments")
    print("  - calculate_correlations(): Correlation matrix")
    print("  - generate_review_report(): Complete review analysis")

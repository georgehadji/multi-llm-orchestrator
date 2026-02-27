"""
model_training.py
Trains classification and regression models for restaurant data analysis.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
import joblib
import os
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class RestaurantModelTrainer:
    """Trainer class for restaurant rating and delivery time prediction models."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the model trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.classifier = None
        self.regressor = None
        self.feature_columns = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load processed restaurant data.
        
        Args:
            filepath: Path to processed data CSV file
            
        Returns:
            Loaded DataFrame
        """
        print(f"Loading data from {filepath}...")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def prepare_features_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare features and targets for both classification and regression.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features DataFrame, classification target, regression target)
        """
        print("\nPreparing features and targets...")
        
        # Create classification target: rating binned into low/medium/high
        if 'rating' not in df.columns:
            raise ValueError("DataFrame must contain 'rating' column")
        
        # Bin ratings into 3 categories
        df['rating_tier'] = pd.cut(
            df['rating'],
            bins=[0, 2.5, 3.5, 5],
            labels=['low', 'medium', 'high'],
            include_lowest=True
        )
        
        # Prepare features for classification (predict rating tier)
        features = self._prepare_classification_features(df)
        
        # Prepare regression target (delivery time)
        if 'delivery_time' not in df.columns:
            raise ValueError("DataFrame must contain 'delivery_time' column")
        
        regression_target = df['delivery_time']
        
        return features, df['rating_tier'], regression_target
    
    def _prepare_classification_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for classification model.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Features DataFrame
        """
        features_dict = {}
        
        # 1. Cuisine type encoding (one-hot or label encoding)
        if 'cuisine' in df.columns:
            # Use label encoding for cuisine (could also use one-hot for few categories)
            le_cuisine = LabelEncoder()
            features_dict['cuisine_encoded'] = le_cuisine.fit_transform(df['cuisine'].fillna('Unknown'))
            self.label_encoders['cuisine'] = le_cuisine
        else:
            # Create default cuisine if missing
            features_dict['cuisine_encoded'] = np.zeros(len(df))
        
        # 2. Price tier (extract from price_range or create from avg_price)
        if 'price_range' in df.columns:
            # Map price ranges to numeric tiers
            price_mapping = {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4}
            features_dict['price_tier'] = df['price_range'].map(price_mapping).fillna(2)
        elif 'avg_price' in df.columns:
            # Create price tiers from average price
            features_dict['price_tier'] = pd.cut(
                df['avg_price'],
                bins=[0, 15, 30, 50, float('inf')],
                labels=[1, 2, 3, 4]
            ).astype(float)
        else:
            # Default price tier
            features_dict['price_tier'] = np.ones(len(df)) * 2
        
        # 3. Location encoding (simplified - could use more sophisticated geocoding)
        if 'city' in df.columns:
            le_city = LabelEncoder()
            features_dict['location_encoded'] = le_city.fit_transform(df['city'].fillna('Unknown'))
            self.label_encoders['city'] = le_city
        else:
            features_dict['location_encoded'] = np.zeros(len(df))
        
        # 4. Delivery fee
        if 'delivery_fee' in df.columns:
            features_dict['delivery_fee'] = df['delivery_fee'].fillna(0)
        else:
            features_dict['delivery_fee'] = np.zeros(len(df))
        
        # 5. Additional features that might be predictive
        if 'review_count' in df.columns:
            features_dict['review_count'] = df['review_count'].fillna(0)
        
        if 'distance' in df.columns:
            features_dict['distance'] = df['distance'].fillna(df['distance'].median())
        
        # Create features DataFrame
        features_df = pd.DataFrame(features_dict)
        
        # Scale numerical features
        numerical_cols = ['delivery_fee', 'price_tier']
        if 'review_count' in features_df.columns:
            numerical_cols.append('review_count')
        if 'distance' in features_df.columns:
            numerical_cols.append('distance')
        
        # Only scale if we have numerical columns
        if numerical_cols:
            features_df[numerical_cols] = self.scaler.fit_transform(features_df[numerical_cols])
        
        self.feature_columns = features_df.columns.tolist()
        print(f"Created {len(self.feature_columns)} features: {self.feature_columns}")
        
        return features_df
    
    def prepare_regression_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for delivery time regression model.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Features DataFrame for regression
        """
        features_dict = {}
        
        # 1. Distance (crucial for delivery time)
        if 'distance' in df.columns:
            features_dict['distance'] = df['distance'].fillna(df['distance'].median())
        else:
            # Create synthetic distance if missing
            features_dict['distance'] = np.random.exponential(5, len(df))
        
        # 2. Hour of day (if available, otherwise use default)
        if 'order_hour' in df.columns:
            features_dict['hour'] = df['order_hour']
        else:
            features_dict['hour'] = np.random.randint(0, 24, len(df))
        
        # 3. Day of week (if available)
        if 'order_day' in df.columns:
            # Convert day names to numbers if needed
            if df['order_day'].dtype == 'object':
                day_mapping = {
                    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 
                    'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
                }
                features_dict['day'] = df['order_day'].map(day_mapping).fillna(0)
            else:
                features_dict['day'] = df['order_day']
        else:
            features_dict['day'] = np.random.randint(0, 7, len(df))
        
        # 4. Restaurant type/cuisine
        if 'cuisine' in df.columns:
            if 'cuisine' in self.label_encoders:
                features_dict['restaurant_type'] = self.label_encoders['cuisine'].transform(
                    df['cuisine'].fillna('Unknown')
                )
            else:
                le_cuisine = LabelEncoder()
                features_dict['restaurant_type'] = le_cuisine.fit_transform(
                    df['cuisine'].fillna('Unknown')
                )
                self.label_encoders['cuisine_reg'] = le_cuisine
        else:
            features_dict['restaurant_type'] = np.zeros(len(df))
        
        # 5. Additional potentially relevant features
        if 'price_tier' in df.columns or 'price_range' in df.columns:
            if 'price_range' in df.columns:
                price_mapping = {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4}
                features_dict['price_level'] = df['price_range'].map(price_mapping).fillna(2)
            else:
                features_dict['price_level'] = df.get('price_tier', 2)
        
        # Create DataFrame and scale
        features_df = pd.DataFrame(features_dict)
        
        # Scale numerical features
        numerical_cols = ['distance', 'hour', 'day']
        if 'price_level' in features_df.columns:
            numerical_cols.append('price_level')
        
        # Use a separate scaler for regression or reuse
        features_df[numerical_cols] = self.scaler.fit_transform(features_df[numerical_cols])
        
        return features_df
    
    def train_classifier(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, Any]:
        """Train Random Forest classifier for rating tier prediction.
        
        Args:
            X: Features DataFrame
            y: Target Series (rating tiers)
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training results
        """
        print("\n" + "="*50)
        print("Training Random Forest Classifier")
        print("="*50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"Class distribution in training: {y_train.value_counts().to_dict()}")
        
        # Train Random Forest classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nClassifier Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Feature Importances:")
        print(feature_importance.head(10).to_string(index=False))
        
        # Check if accuracy meets target
        target_accuracy = 0.65
        if accuracy >= target_accuracy:
            print(f"\n✅ Target accuracy of {target_accuracy} achieved!")
        else:
            print(f"\n⚠️  Accuracy below target of {target_accuracy}")
            print("Consider: feature engineering, hyperparameter tuning, or more data")
        
        return {
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'classifier': self.classifier,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def train_regressor(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, Any]:
        """Train Random Forest regressor for delivery time prediction.
        
        Args:
            X: Features DataFrame
            y: Target Series (delivery time)
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training results
        """
        print("\n" + "="*50)
        print("Training Random Forest Regressor")
        print("="*50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"Delivery time stats - Mean: {y_train.mean():.1f} min, Std: {y_train.std():.1f} min")
        
        # Train Random Forest regressor
        self.regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.regressor.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.regressor.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nRegression Performance:")
        print(f"Mean Absolute Error: {mae:.2f} minutes")
        print(f"R² Score: {r2:.4f}")
        
        # Feature importance for regression
        reg_feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.regressor.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop Feature Importances for Delivery Time Prediction:")
        print(reg_feature_importance.to_string(index=False))
        
        return {
            'mae': mae,
            'r2': r2,
            'regressor': self.regressor,
            'feature_importance': reg_feature_importance
        }
    
    def save_models(self, output_dir: str = 'models'):
        """Save trained models and preprocessing objects.
        
        Args:
            output_dir: Directory to save models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.classifier is not None:
            classifier_path = os.path.join(output_dir, 'rating_classifier.pkl')
            joblib.dump(self.classifier, classifier_path)
            print(f"Saved classifier to {classifier_path}")
        
        if self.regressor is not None:
            regressor_path = os.path.join(output_dir, 'delivery_time_regressor.pkl')
            joblib.dump(self.regressor, regressor_path)
            print(f"Saved regressor to {regressor_path}")
        
        # Save preprocessing objects
        preprocessing_path = os.path.join(output_dir, 'preprocessing.pkl')
        preprocessing_objects = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        joblib.dump(preprocessing_objects, preprocessing_path)
        print(f"Saved preprocessing objects to {preprocessing_path}")
        
        # Save feature importance reports
        if hasattr(self, 'classifier_results'):
            importance_path = os.path.join(output_dir, 'feature_importance_classifier.csv')
            self.classifier_results['feature_importance'].to_csv(importance_path, index=False)
            print(f"Saved classifier feature importance to {importance_path}")
    
    def run_full_pipeline(self, data_path: str = 'processed_data/processed_restaurant_data.csv'):
        """Run complete training pipeline.
        
        Args:
            data_path: Path to processed data file
        """
        try:
            # 1. Load data
            df = self.load_data(data_path)
            
            # 2. Prepare features and targets
            features, class_target, reg_target = self.prepare_features_targets(df)
            
            # 3. Train classifier
            self.classifier_results = self.train_classifier(features, class_target)
            
            # 4. Prepare regression features and train regressor
            reg_features = self.prepare_regression_features(df)
            self.regressor_results = self.train_regressor(reg_features, reg_target)
            
            # 5. Save models
            self.save_models()
            
            print("\n" + "="*50)
            print("Training Pipeline Complete!")
            print("="*50)
            
            # Summary
            print("\nSummary:")
            print(f"Classifier Accuracy: {self.classifier_results['accuracy']:.4f}")
            print(f"Regressor MAE: {self.regressor_results['mae']:.2f} minutes")
            print(f"Regressor R²: {self.regressor_results['r2']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"\nError in training pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main execution function."""
    print("Restaurant Model Training Pipeline")
    print("="*50)
    
    # Initialize trainer
    trainer = RestaurantModelTrainer(random_state=42)
    
    # Define data path (adjust as needed based on your preprocessing output)
    data_path = 'processed_data/processed_restaurant_data.csv'
    
    # Check if data exists, if not try alternative paths
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}, checking alternative locations...")
        alternative_paths = [
            'data/processed_restaurant_data.csv',
            '../processed_data/processed_restaurant_data.csv',
            'processed_restaurant_data.csv'
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                data_path = alt_path
                print(f"Found data at: {data_path}")
                break
        else:
            print("Could not find processed data file.")
            print("Please ensure data_preprocessing.py has been run first.")
            return
    
    # Run training pipeline
    success = trainer.run_full_pipeline(data_path)
    
    if success:
        print("\n✅ Models trained and saved successfully!")
        print("Models saved in 'models/' directory:")
        print("  - rating_classifier.pkl (predicts rating tier)")
        print("  - delivery_time_regressor.pkl (predicts delivery time)")
        print("  - preprocessing.pkl (encoders and scaler)")
    else:
        print("\n❌ Training pipeline failed.")


if __name__ == "__main__":
    main()

# Code Review: data_preprocessing.py

## Overall Assessment
The code is well-structured and implements a comprehensive data preprocessing pipeline. It addresses all the requested requirements with good error handling and documentation. However, there are several areas that need improvement for production readiness.

## Detailed Review

### 1. **Correct Handling of Missing Data** ✅ **GOOD**
- **Strengths**: 
  - Properly drops rows with missing critical values (restaurant_id, name, city)
  - Uses appropriate imputation strategies (median for ratings, 0 for fees, 30 for delivery_time)
  - Maintains data integrity by not over-imputing critical fields
- **Issues**: 
  - Line 134: `df_clean['rating'] = df_clean['rating'].fillna(df_clean['rating'].median())` - This will fail if all ratings are NaN
  - No validation of imputed values (e.g., negative delivery fees)

### 2. **Consistency in Column Standardization** ⚠️ **NEEDS IMPROVEMENT**
- **Strengths**: 
  - Good mapping dictionaries for different data sources
  - Creates missing standard columns with sensible defaults
- **Issues**:
  - Line 103-104: `df_std[col] = [f"{source_name.upper()}_{i}" for i in range(len(df_std))]` - This creates sequential IDs that don't preserve original IDs
  - No deduplication logic in `standardize_columns()` function
  - **Inconsistent handling of cuisine field across sources**: Different sources use different column names ('cuisine_type', 'food_category', 'menu_type') without consistent normalization

### 3. **Proper Use of Random Seed for Synthetic Data** ✅ **GOOD**
- **Strengths**:
  - Sets `np.random.seed(42)` at module level for reproducibility
  - Consistent seed usage throughout synthetic data generation
- **Issues**: None identified

### 4. **File Saving Paths Exist** ✅ **GOOD**
- **Strengths**:
  - Uses `os.makedirs(data_dir, exist_ok=True)` to ensure directory exists
  - Good error handling in file loading with try-except blocks
- **Issues**: None identified

### 5. **Code Follows PEP8 (use ruff)** ⚠️ **NEEDS IMPROVEMENT**
- **Strengths**:
  - Good function and variable naming
  - Reasonable line lengths
- **Issues**:
  - Multiple lines exceed 79 characters (PEP8 limit)
  - Inconsistent spacing around operators
  - Missing type hints
  - Some functions are too long (e.g., `generate_synthetic_order_data()` at 94 lines)

## Critical Issues Requiring Fix

### 1. **Rating Imputation Failure Risk**
```python
# Current (line 134):
df_clean['rating'] = df_clean['rating'].fillna(df_clean['rating'].median())

# Fixed:
if 'rating' in df_clean.columns and not df_clean['rating'].isna().all():
    df_clean['rating'] = df_clean['rating'].fillna(df_clean['rating'].median())
else:
    df_clean['rating'] = df_clean['rating'].fillna(3.5)  # Default reasonable value
```

### 2. **ID Generation Issue**
```python
# Current (line 103):
df_std[col] = [f"{source_name.upper()}_{i}" for i in range(len(df_std))]

# Fixed - preserve original IDs when possible:
if col == 'restaurant_id' and col not in df_std.columns:
    # Try to use existing ID columns
    id_candidates = [c for c in df_std.columns if 'id' in c.lower() or 'code' in c.lower()]
    if id_candidates:
        df_std[col] = df_std[id_candidates[0]]
    else:
        df_std[col] = [f"{source_name.upper()}_{i:04d}" for i in range(len(df_std))]
```

### 3. **Cuisine Field Standardization Issue**
```python
# Current: Inconsistent handling across different source mappings
# Fixed: Add consistent cuisine normalization in standardize_columns() function

def normalize_cuisine(cuisine_str):
    """Normalize cuisine strings to consistent categories."""
    if pd.isna(cuisine_str):
        return "Other"
    
    cuisine_str = str(cuisine_str).lower().strip()
    
    # Map variations to standard categories
    cuisine_map = {
        'italian': ['italian', 'pizza', 'pasta'],
        'asian': ['chinese', 'japanese', 'thai', 'sushi', 'asian'],
        'american': ['american', 'burger', 'bbq', 'steak'],
        'mexican': ['mexican', 'taco', 'burrito'],
        'indian': ['indian', 'curry'],
        'mediterranean': ['mediterranean', 'greek', 'middle eastern'],
        'vegetarian': ['vegetarian', 'vegan', 'healthy']
    }
    
    for standard_cuisine, variations in cuisine_map.items():
        if any(variation in cuisine_str for variation in variations):
            return standard_cuisine.capitalize()
    
    return "Other"

# Apply in standardize_columns():
if 'cuisine' in df_std.columns:
    df_std['cuisine'] = df_std['cuisine'].apply(normalize_cuisine)
```

### 4. **Data Validation Missing**
Add validation functions:
```python
def validate_processed_data(df):
    """Validate processed data for common issues."""
    issues = []
    
    # Check for negative values where inappropriate
    if 'delivery_fee' in df.columns and (df['delivery_fee'] < 0).any():
        issues.append("Negative delivery fees found")
    
    if 'rating' in df.columns:
        invalid_ratings = df['rating'][~df['rating'].between(1, 5)]
        if len(invalid_ratings) > 0:
            issues.append(f"Ratings outside 1-5 range: {len(invalid_ratings)} records")
    
    return issues
```

## PEP8 Compliance Issues - Specific Fixes

Running `ruff check` would flag these specific issues:

1. **Line 48**: `p=[0.4, 0.2, 0.1, 0.1, 0.08, 0.07, 0.05]` - line too long (85 > 79)
   ```python
   # Fixed:
   p = [
       0.4, 0.2, 0.1, 0.1, 
       0.08, 0.07, 0.05
   ]
   ```

2. **Line 195**: Multiple spaces around operators
   ```python
   # Current: df['delivery_time']  =  df['delivery_time'].fillna(30)
   # Fixed: df['delivery_time'] = df['delivery_time'].fillna(30)
   ```

3. **Line 241**: `restaurant_names_dict = dict(zip(restaurant_ids, restaurant_names))` - line too long
   ```python
   # Fixed:
   restaurant_names_dict = dict(
       zip(restaurant_ids, restaurant_names)
   )
   ```

4. **Missing type hints throughout** - Add to all function definitions:
   ```python
   # Current: def clean_data(df):
   # Fixed: def clean_data(df: pd.DataFrame) -> pd.DataFrame:
   ```

## Recommendations with Context and Examples

### 1. **Add Comprehensive Data Validation**
**Why**: Ensures data quality and catches issues early in the pipeline
```python
def validate_dataframe(df: pd.DataFrame, df_name: str) -> dict:
    """Perform comprehensive validation on DataFrame.
    
    Returns a dictionary with validation results for easy monitoring.
    """
    results = {
        'name': df_name,
        'issues': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check for duplicates
    if 'restaurant_id' in df.columns:
        duplicates = df['restaurant_id'].duplicated().sum()
        if duplicates > 0:
            warning_msg = f"{duplicates} duplicate restaurant IDs found"
            results['warnings'].append(warning_msg)
        results['stats']['unique_ids'] = df['restaurant_id'].nunique()
    
    # Check data ranges
    numeric_checks = [
        ('rating', 1, 5),
        ('delivery_fee', 0, 50),
        ('delivery_time', 5, 120)
    ]
    
    for col, min_val, max_val in numeric_checks:
        if col in df.columns:
            out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
            if out_of_range > 0:
                warning_msg = f"{out_of_range} {col} values outside range [{min_val}, {max_val}]"
                results['warnings'].append(warning_msg)
            results['stats'][f'{col}_range'] = (df[col].min(), df[col].max())
    
    # Check for missing values in critical columns
    critical_cols = ['restaurant_id', 'name', 'city']
    for col in critical_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                issue_msg = f"{missing} missing values in critical column '{col}'"
                results['issues'].append(issue_msg)
    
    return results
```

### 2. **Improve Error Handling in Main Pipeline**
**Why**: Provides better debugging information and graceful failure handling
```python
import traceback

def main() -> int:
    """Main preprocessing pipeline with comprehensive error handling.
    
    Returns:
        0 for success, 1 for failure
    """
    try:
        # Existing preprocessing code...
        
        # Add validation step
        validation_results = validate_dataframe(processed_df, "final_processed")
        
        if validation_results['issues']:
            print("\nCRITICAL ISSUES FOUND:")
            for issue in validation_results['issues']:
                print(f"  - {issue}")
            return 1
        
        if validation_results['warnings']:
            print("\nWARNINGS:")
            for warning in validation_results['warnings']:
                print(f"  - {warning}")
        
        print("\nValidation Statistics:")
        for stat, value in validation_results['stats'].items():
            print(f"  {stat}: {value}")
            
        return 0
        
    except FileNotFoundError as e:
        print(f"\nERROR: Required data file not found: {e}")
        return 1
    except ValueError as e:
        print(f"\nERROR: Data validation error: {e}")
        return 1
    except Exception as e:
        print(f"\nUNEXPECTED ERROR in preprocessing pipeline: {e}")
        print("Traceback:")
        print(traceback.format_exc())
        return 1
```

### 3. **Add Configuration Management**
**Why**: Improves maintainability, makes the codebase more adaptable to changes, and centralizes settings
```python
# Configuration management - centralizes all tunable parameters
class PreprocessingConfig:
    """Centralized configuration for preprocessing pipeline."""
    
    # Data parameters
    MIN_RESTAURANTS = 200
    DATA_DIR = 'data'
    OUTPUT_DIR = 'processed_data'
    
    # Imputation defaults
    DEFAULT_RATING = 3.5
    DEFAULT_DELIVERY_TIME = 30
    DEFAULT_DELIVERY_FEE = 0
    
    # Validation thresholds
    MIN_RATING = 1
    MAX_RATING = 5
    MAX_DELIVERY_FEE = 50
    MAX_DELIVERY_TIME = 120
    
    # Randomness
    RANDOM_SEED = 42
    
    # File paths
    @classmethod
    def get_source_path(cls, source_name: str) -> str:
        return os.path.join(cls.DATA_DIR, f"{source_name}.csv")
    
    @classmethod
    def get_output_path(cls, filename: str) -> str:
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        return os.path.join(cls.OUTPUT_DIR, filename)

# Usage example in code:
# Instead of hardcoded values:
# df['rating'] = df['rating'].fillna(3.5)
# Use:
# df['rating'] = df['rating'].fillna(PreprocessingConfig.DEFAULT_RATING)
```

### 4. **Add Deduplication Logic**
**Why**: Ensures data integrity by preventing duplicate restaurant entries
```python
def deduplicate_restaurants(df: pd.DataFrame, id_col: str = 'restaurant_id') -> pd.DataFrame:
    """Remove duplicate restaurants, keeping the most complete record.
    
    Strategy:
    1. Group by restaurant_id
    2. For each group, select the row with fewest missing values
    3. If tie, select the highest rating
    4. If still tie, select randomly (with seed for reproducibility)
    """
    if id_col not in df.columns:
        return df
    
    # Calculate completeness score for each row
    df = df.copy()
    df['_completeness'] = df.notna().sum(axis=1)
    
    # Sort by completeness, then rating, then random for tie-breaking
    np.random.seed(PreprocessingConfig.RANDOM_SEED)
    df['_random'] = np.random.rand(len(df))
    
    sort_cols = ['_completeness']
    if 'rating' in df.columns:
        sort_cols.append('rating')
    sort_cols.append('_random')
    
    # Keep first row after sorting (highest completeness, then highest rating, then random)
    df_deduped = df.sort_values(sort_cols, ascending=[False, False, True]) \
                   .groupby(id_col) \
                   .first() \
                   .reset_index()
    
    # Clean up temporary columns
    df_deduped = df_deduped.drop(columns=['_completeness', '_random'])
    
    duplicates_removed = len(df) - len(df_deduped)
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate restaurant records")
    
    return df_deduped
```

## Final Verdict: **CONDITIONAL APPROVAL**

**Approval Conditions**:
1. **Critical Fixes Required**:
   - Fix the rating imputation bug (line 134) to handle all-NaN columns
   - Improve ID preservation logic to maintain original identifiers when possible
   - Add cuisine field normalization for consistent categorization
   - Implement basic data validation to catch invalid values

2. **Quality Improvements Recommended**:
   - Run `ruff format` to fix all PEP8 violations
   - Add type hints to all function definitions
   - Refactor long lines (>79 characters) for better readability
   - Consider adding configuration management for maintainability

**Scoring Criteria Clarification**:
- **Current Score: 0.85/1.00** - Code meets functional requirements but has quality and robustness issues
- **To achieve 0.90+**: Fix all critical issues listed above
- **To achieve 0.95+**: Fix critical issues + implement configuration management + add comprehensive validation
- **To achieve 1.00**: All of the above + full PEP8 compliance + add comprehensive unit tests

**Minor vs. Major Fixes**:
- **Minor fixes**: PEP8 violations, adding type hints, improving comments
- **Major fixes**: Data validation logic, error handling improvements, algorithm changes (like ID preservation)

The code is fundamentally sound and addresses all requirements. With the fixes above, it will be production-ready. The architecture is good, error handling is adequate, and the synthetic data generation is realistic and reproducible.
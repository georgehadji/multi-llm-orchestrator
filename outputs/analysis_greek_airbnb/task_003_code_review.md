# Revised Code Review: `data_loader.py` and `data_cleaning.py`

## Overall Assessment
Both scripts are well-structured with good modular design and comprehensive functionality. However, there are several areas for improvement in error handling, memory efficiency, data type handling, and edge case management. This review provides specific, actionable improvements with clear rationales.

## 1. Proper Error Handling in Downloads

### Issues Found:
1. **No retry mechanism** for failed downloads
2. **No validation of response content** before processing
3. **Incomplete error handling** for gzip decompression
4. **Missing fallback strategies** when primary URLs fail

### Specific Improvements for `data_loader.py`:

```python
# Add retry decorator/function with clear rationale
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests.exceptions

class AirbnbDataLoader:
    # ... existing code ...
    
    @retry(
        stop=stop_after_attempt(3),  # 3 attempts balances success rate with network load
        wait=wait_exponential(multiplier=1, min=4, max=10),  # Exponential backoff: 4s, 8s, 10s
        retry=retry_if_exception_type((
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError
        ))
    )
    def _download_file(self, url: str, city: str, file_type: str) -> Optional[pd.DataFrame]:
        """
        Download file with robust error handling and fallback strategies.
        
        Rationale for retry parameters:
        - 3 attempts: Provides reasonable chance of success without excessive network load
        - Exponential backoff: Reduces server load and handles temporary network issues
        - Specific exceptions: Only retry on recoverable errors, not client errors (e.g., 404)
        """
        # ... existing code ...
        
        try:
            headers = {
                'User-Agent': 'AirbnbDataLoader/1.0 (https://github.com/your-repo)'
            }
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # Validate content before processing
            if not self._validate_response_content(response, city, file_type):
                return None
            
            # Process content based on encoding
            content = self._extract_response_content(response)
            if content is None:
                return None
                
            # ... rest of existing code ...
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"URL not found (404): {url}")
                return self._try_fallback_urls(url, city, file_type)
            else:
                logger.error(f"HTTP error {e.response.status_code} for {city} {file_type}: {e}")
                raise  # Will trigger retry for 5xx errors
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logger.warning(f"Network error for {city} {file_type}: {e}")
            raise  # Will trigger retry
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {file_type} for {city}: {e}")
            return None
    
    def _validate_response_content(self, response, city: str, file_type: str) -> bool:
        """Validate response headers and content size."""
        content_type = response.headers.get('Content-Type', '')
        content_length = response.headers.get('Content-Length')
        
        # Size validation (500MB limit)
        if content_length and int(content_length) > 500_000_000:
            logger.error(f"File too large ({int(content_length)/1_000_000:.1f}MB) for {city} {file_type}")
            return False
        
        # Content type validation
        expected_types = ['text/csv', 'application/gzip', 'application/x-gzip']
        if not any(expected in content_type for expected in expected_types):
            logger.warning(f"Unexpected content type '{content_type}' for {city} {file_type}")
            # Continue anyway as some servers may not set proper headers
        
        return True
    
    def _extract_response_content(self, response) -> Optional[bytes]:
        """Extract content handling gzip and other encodings."""
        try:
            if ('gzip' in response.headers.get('Content-Encoding', '') or 
                response.url.endswith('.gz')):
                import gzip
                return gzip.decompress(response.content)
            else:
                return response.content
        except (gzip.BadGzipFile, EOFError) as e:
            logger.error(f"Failed to decompress content: {e}")
            # Try reading as plain text as fallback
            try:
                return response.content
            except Exception as e2:
                logger.error(f"Failed to read content: {e2}")
                return None
    
    def _try_fallback_urls(self, original_url: str, city: str, file_type: str) -> Optional[pd.DataFrame]:
        """
        Implement fallback strategy for missing data.
        
        Strategy:
        1. Try previous month's data
        2. Try alternative URL patterns
        3. Try different data sources
        """
        fallback_urls = self._generate_fallback_urls(original_url, city, file_type)
        
        for fallback_url in fallback_urls:
            logger.info(f"Trying fallback URL: {fallback_url}")
            try:
                # Use a shorter timeout for fallbacks
                response = requests.get(fallback_url, timeout=15)
                if response.status_code == 200:
                    return self._process_downloaded_content(response.content, city, file_type)
            except requests.exceptions.RequestException:
                continue
        
        logger.error(f"No fallback URLs succeeded for {city} {file_type}")
        return None
    
    def _generate_fallback_urls(self, original_url: str, city: str, file_type: str) -> List[str]:
        """Generate alternative URLs based on common patterns."""
        import datetime
        from urllib.parse import urlparse, urlunparse
        
        fallbacks = []
        parsed = urlparse(original_url)
        
        # Pattern 1: Previous months (common in time-series data)
        path_parts = parsed.path.split('/')
        for i, part in enumerate(path_parts):
            if len(part) == 6 and part.isdigit():  # YYYYMM format
                year_month = datetime.datetime.strptime(part, "%Y%m")
                for months_back in range(1, 4):  # Try up to 3 months back
                    prev_month = year_month - datetime.timedelta(days=30*months_back)
                    prev_part = prev_month.strftime("%Y%m")
                    new_path = '/'.join(path_parts[:i] + [prev_part] + path_parts[i+1:])
                    fallbacks.append(urlunparse(parsed._replace(path=new_path)))
        
        # Pattern 2: Alternative domain (e.g., mirror sites)
        if 'insideairbnb.com' in parsed.netloc:
            alt_domains = ['data.insideairbnb.com', 'archive.insideairbnb.com']
            for domain in alt_domains:
                fallbacks.append(urlunparse(parsed._replace(netloc=domain)))
        
        return fallbacks[:5]  # Limit to 5 fallback attempts
```

## 2. Efficient Memory Usage with Large Datasets

### Issues Found:
1. **Loading entire datasets into memory** before processing
2. **No chunking or streaming** for very large files
3. **Potential memory duplication** when storing data in multiple structures
4. **No memory monitoring** or cleanup

### Specific Improvements with Trade-off Analysis:

```python
import psutil
import tracemalloc
from contextlib import contextmanager

class AirbnbDataLoader:
    # ... existing code ...
    
    @contextmanager
    def memory_monitor(self, operation_name: str):
        """
        Context manager for memory monitoring.
        
        Usage:
            with loader.memory_monitor("load_large_file"):
                df = loader.load_data()
        """
        tracemalloc.start()
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            current_memory = process.memory_info().rss / 1024 / 1024
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:5]
            
            logger.info(f"{operation_name} - Memory usage: {current_memory - start_memory:.1f}MB change")
            logger.info(f"{operation_name} - Top memory allocations:")
            for stat in top_stats:
                logger.info(f"  {stat}")
            
            tracemalloc.stop()
    
    def _load_file(self, file_path: Path, file_type: str, city: str) -> pd.DataFrame:
        """
        Load a CSV file into a DataFrame with memory-efficient strategies.
        
        Trade-off Analysis:
        - Chunking: Reduces memory but increases I/O and processing time
        - Downcasting: Saves memory but may lose precision
        - Column filtering: Saves memory but requires knowing column importance
        """
        
        file_size = file_path.stat().st_size
        
        # Decision matrix for loading strategy
        if file_size > 1_000_000_000:  # > 1GB
            logger.info("File >1GB detected, using streaming processing")
            return self._process_file_streaming(file_path, file_type, city)
        elif file_size > 500_000_000:  # 500MB - 1GB
            logger.info("Large file (500MB-1GB) detected, using chunked loading")
            return self._load_file_chunked(file_path, file_type, city)
        else:
            logger.info("Moderate file size, using optimized single-pass loading")
            return self._load_file_optimized(file_path, file_type, city)
    
    def _load_file_optimized(self, file_path: Path, file_type: str, city: str) -> pd.DataFrame:
        """Load file with optimized memory usage in single pass."""
        
        # Determine which columns to load based on file type
        essential_columns = self._get_essential_columns(file_type)
        
        # Read with optimized settings
        df = pd.read_csv(
            file_path,
            usecols=essential_columns,
            dtype=self._get_optimal_dtypes(file_path, essential_columns),
            parse_dates=self._get_date_columns(file_type),
            low_memory=False,
            on_bad_lines='warn',
            true_values=['t', 'true', 'yes', 'y'],
            false_values=['f', 'false', 'no', 'n']
        )
        
        # Post-processing optimizations
        df = self._optimize_dataframe_memory(df, file_type, city)
        
        return df
    
    def _load_file_chunked(self, file_path: Path, file_type: str, city: str, 
                          chunksize: int = 50000) -> pd.DataFrame:
        """
        Load very large files in chunks.
        
        Trade-offs:
        - Larger chunksize: More memory usage, faster processing
        - Smaller chunksize: Less memory, slower processing
        - 50,000 rows is a reasonable default for most systems
        """
        
        chunks = []
        essential_columns = self._get_essential_columns(file_type)
        date_columns = self._get_date_columns(file_type)
        
        for i, chunk in enumerate(pd.read_csv(
            file_path,
            chunksize=chunksize,
            usecols=essential_columns,
            parse_dates=date_columns,
            low_memory=False
        )):
            # Process chunk
            chunk = self._process_chunk(chunk, file_type, city)
            chunks.append(chunk)
            
            # Memory management every 10 chunks
            if i % 10 == 0 and i > 0:
                import gc
                gc.collect()
                
                # Log progress and memory usage
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                logger.debug(f"Processed {i * chunksize} rows, memory: {memory_mb:.1f}MB")
        
        # Combine chunks efficiently
        if chunks:
            return pd.concat(chunks, ignore_index=True, copy=False)
        else:
            return pd.DataFrame()
    
    def _process_file_streaming(self, file_path: Path, file_type: str, city: str) -> pd.DataFrame:
        """
        Process extremely large files using true streaming.
        
        This method writes intermediate results to disk to avoid memory issues.
        """
        import tempfile
        import pickle
        
        temp_dir = tempfile.mkdtemp()
        chunk_files = []
        
        try:
            # Process in chunks and save to disk
            for i, chunk in enumerate(pd.read_csv(file_path, chunksize=100000)):
                chunk = self._process_chunk(chunk, file_type, city)
                
                # Save chunk to disk
                chunk_file = Path(temp_dir) / f"chunk_{i:04d}.pkl"
                with open(chunk_file, 'wb') as f:
                    pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
                chunk_files.append(chunk_file)
                
                # Clear memory
                del chunk
            
            # Load and combine chunks efficiently
            combined = self._combine_chunk_files(chunk_files)
            return combined
            
        finally:
            # Cleanup temporary files
            for chunk_file in chunk_files:
                try:
                    chunk_file.unlink()
                except:
                    pass
            try:
                Path(temp_dir).rmdir()
            except:
                pass
    
    def _get_optimal_dtypes(self, file_path: Path, columns: List[str]) -> Dict[str, str]:
        """
        Determine optimal dtypes by sampling the file.
        
        Samples first 10,000 rows to infer types, balancing accuracy with overhead.
        """
        sample = pd.read_csv(file_path, nrows=10000, usecols=columns)
        
        dtype_map = {}
        for col in columns:
            if col in sample.columns:
                dtype_map[col] = self._infer_optimal_dtype(sample[col])
        
        return dtype_map
    
    def _infer_optimal_dtype(self, series: pd.Series) -> str:
        """Infer the most memory-efficient dtype for a series."""
        if pd.api.types.is_integer_dtype(series.dtype):
            return self._downcast_integer(series)
        elif pd.api.types.is_float_dtype(series.dtype):
            return self._downcast_float(series)
        elif pd.api.types.is_object_dtype(series.dtype):
            # Check if it's actually categorical or datetime
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.3:  # Low cardinality
                return 'category'
            elif self._could_be_datetime(series):
                return 'str'  # Parse dates separately
        return 'str'
```

## 3. Correct Data Type Conversions

### Issues Found:
1. **Price conversion assumes USD** but Greek data might use EUR
2. **No handling for non-numeric price values** (e.g., "Contact for price")
3. **Date parsing doesn't handle multiple formats**
4. **Categorical columns not optimized**

### Specific Improvements:

```python
class AirbnbDataLoader:
    # ... existing code ...
    
    def _parse_price_column(self, price_series: pd.Series, city: str) -> pd.Series:
        """
        Robust parsing of price columns with currency detection.
        
        Uses multiple strategies:
        1. Detect currency from symbols and location
        2. Handle various formats and special cases
        3. Convert to base currency (USD) for consistency
        """
        prices = price_series.copy().astype(str)
        
        # Detect currency
        currency = self._detect_currency(prices, city)
        logger.info(f"Detected currency for {city}: {currency}")
        
        # Extract numeric values and convert to USD
        numeric_prices = self._extract_numeric_values(prices)
        usd_prices = self._convert_to_usd(numeric_prices, currency)
        
        return usd_prices
    
    def _detect_currency(self, prices: pd.Series, city: str) -> str:
        """Detect currency from symbols and location context."""
        # Check for currency symbols
        symbol_patterns = {
            r'[\$]': 'USD',
            r'[€]': 'EUR',
            r'[£]': 'GBP',
            r'[¥]': 'JPY',
            r'[₹]': 'INR'
        }
        
        sample = prices.head(1000)  # Check first 1000 rows
        for pattern, currency in symbol_patterns.items():
            if sample.str.contains(pattern, regex=True).any():
                return currency
        
        # Fallback to location-based detection
        location_currency = {
            'athens': 'EUR',
            'berlin': 'EUR',
            'london': 'GBP',
            'new-york': 'USD',
            'san-francisco': 'USD',
            'tokyo': 'JPY'
        }
        
        return location_currency.get(city.lower(), 'USD')  # Default to USD
    
    def _extract_numeric_values(self, prices: pd.Series) -> pd.Series:
        """Extract numeric values from various price formats."""
        # Remove all non-numeric characters except decimal point
        cleaned = prices.str.replace(r'[^\d\.]', '', regex=True)
        
        # Handle empty strings and special cases
        cleaned = cleaned.replace({'': np.nan, '.': np.nan})
        
        # Convert to numeric, coercing errors to NaN
        numeric = pd.to_numeric(cleaned, errors='coerce')
        
        return numeric
    
    def _convert_to_usd(self, prices: pd.Series, from_currency: str) -> pd.Series:
        """Convert prices to USD using current exchange rates."""
        if from_currency == 'USD':
            return prices
        
        try:
            # Use a reliable exchange rate API or local cache
            exchange_rate = self._get_exchange_rate(from_currency, 'USD')
            if exchange_rate:
                return prices * exchange_rate
            else:
                logger.warning(f"Could not get exchange rate for {from_currency}, using original values")
                return prices
        except Exception as e:
            logger.error(f"Currency conversion failed: {e}")
            return prices
    

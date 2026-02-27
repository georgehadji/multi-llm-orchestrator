# map_generator.py
import folium
from folium.plugins import MarkerCluster
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_airbnb_map(
    data_path: str = 'airbnb_listings.csv',
    output_path: str = 'airbnb_map.html',
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
    zoom_start: int = 6
) -> Optional[folium.Map]:
    """
    Generate an interactive folium map with color-coded markers by price quartile.
    
    Args:
        data_path (str): Path to the Airbnb listings CSV file
        output_path (str): Path to save the generated HTML map
        center_lat (float, optional): Latitude for map center. Defaults to Greece center.
        center_lon (float, optional): Longitude for map center. Defaults to Greece center.
        zoom_start (int): Initial zoom level for the map (default: 6)
    
    Returns:
        folium.Map: The generated map object, or None if generation failed
    """
    
    # Step 1: Load and validate data
    logger.info(f"Loading data from {data_path}")
    df = _load_and_validate_data(data_path)
    if df is None or len(df) == 0:
        return None
    
    # Step 2: Calculate price quartiles per city
    logger.info("Calculating price quartiles per city...")
    city_quartiles = _calculate_city_quartiles(df)
    df = _assign_quartiles(df, city_quartiles)
    
    # Step 3: Create folium map
    logger.info("Creating interactive map...")
    m = _create_folium_map(
        center_lat=center_lat,
        center_lon=center_lon,
        zoom_start=zoom_start
    )
    
    # Step 4: Add markers with popups
    logger.info("Adding color-coded markers...")
    _add_markers_to_map(m, df)
    
    # Step 5: Add UI elements
    _add_ui_elements(m, df)
    
    # Step 6: Save map
    logger.info(f"Saving map to {output_path}")
    m.save(output_path)
    
    # Step 7: Print summary
    _print_summary_statistics(df, output_path)
    
    return m


def _load_and_validate_data(data_path: str) -> Optional[pd.DataFrame]:
    """
    Load CSV data and perform comprehensive validation.
    
    Args:
        data_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Cleaned and validated DataFrame, or None if validation fails
    """
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} listings from {data_path}")
    except FileNotFoundError:
        logger.error(f"File {data_path} not found.")
        return None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None
    
    # Check required columns
    required_columns = ['city', 'price', 'latitude', 'longitude', 'name', 'room_type']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return None
    
    # Clean data
    initial_count = len(df)
    
    # Remove rows with missing values in critical columns
    df = df.dropna(subset=['latitude', 'longitude', 'price', 'city', 'name'])
    
    # Remove rows with invalid prices
    df = df[df['price'] > 0]
    
    # Validate coordinate ranges
    df = df[
        (df['latitude'].between(-90, 90)) & 
        (df['longitude'].between(-180, 180))
    ]
    
    # Remove duplicates based on coordinates and name
    df = df.drop_duplicates(subset=['latitude', 'longitude', 'name'])
    
    cleaned_count = len(df)
    if cleaned_count == 0:
        logger.error("No valid data after cleaning.")
        return None
    
    logger.info(f"Cleaned data: {initial_count} -> {cleaned_count} valid listings")
    return df


def _calculate_city_quartiles(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate price quartiles for each city.
    
    Args:
        df (pd.DataFrame): DataFrame containing listing data
        
    Returns:
        Dict: Dictionary mapping city names to their quartile values
    """
    city_quartiles = {}
    
    # Calculate overall quartiles as fallback
    overall_quartiles = df['price'].quantile([0.25, 0.5, 0.75]).to_dict()
    
    for city in df['city'].unique():
        city_prices = df[df['city'] == city]['price']
        
        if len(city_prices) >= 4:
            # Use pandas quantile for consistency and better edge case handling
            quartiles = city_prices.quantile([0.25, 0.5, 0.75]).to_dict()
            city_quartiles[city] = {
                'q1': quartiles[0.25],
                'q2': quartiles[0.5],
                'q3': quartiles[0.75]
            }
        else:
            # For cities with fewer than 4 listings, use overall quartiles
            logger.debug(f"City '{city}' has only {len(city_prices)} listings. Using overall quartiles.")
            city_quartiles[city] = {
                'q1': overall_quartiles[0.25],
                'q2': overall_quartiles[0.5],
                'q3': overall_quartiles[0.75]
            }
    
    return city_quartiles


def _assign_quartiles(df: pd.DataFrame, city_quartiles: Dict) -> pd.DataFrame:
    """
    Assign price quartile to each listing based on city quartiles.
    
    Args:
        df (pd.DataFrame): DataFrame containing listing data
        city_quartiles (Dict): Dictionary of city quartile values
        
    Returns:
        pd.DataFrame: DataFrame with added 'price_quartile' column
    """
    def get_quartile(row: pd.Series) -> int:
        """Determine quartile for a single listing."""
        city = row['city']
        price = row['price']
        
        if city not in city_quartiles:
            return 0  # Unknown
        
        q = city_quartiles[city]
        
        if price <= q['q1']:
            return 1  # Lowest quartile
        elif price <= q['q2']:
            return 2  # Second quartile
        elif price <= q['q3']:
            return 3  # Third quartile
        else:
            return 4  # Highest quartile
    
    df['price_quartile'] = df.apply(get_quartile, axis=1)
    return df


def _create_folium_map(
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
    zoom_start: int = 6
) -> folium.Map:
    """
    Create and configure the base folium map.
    
    Args:
        center_lat (float, optional): Latitude for map center
        center_lon (float, optional): Longitude for map center
        zoom_start (int): Initial zoom level
        
    Returns:
        folium.Map: Configured folium map
    """
    # Default to Greece center if not specified
    if center_lat is None or center_lon is None:
        center_lat, center_lon = 39.0742, 21.8243  # Greece center
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles='OpenStreetMap',
        control_scale=True
    )
    
    # Add alternative tile layers
    folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
    
    return m


def _create_popup_html(row: pd.Series) -> str:
    """
    Generate HTML content for marker popup.
    
    Args:
        row (pd.Series): Row containing listing data
        
    Returns:
        str: HTML string for popup
    """
    # Color scheme for quartiles
    color_scheme = {
        1: 'green',
        2: 'lightgreen',
        3: 'orange',
        4: 'red'
    }
    
    # Quartile descriptions
    quartile_descriptions = {
        1: 'Lowest 25% of prices in this city',
        2: '25-50% of prices in this city',
        3: '50-75% of prices in this city',
        4: 'Highest 25% of prices in this city'
    }
    
    quartile = row['price_quartile']
    color = color_scheme.get(quartile, 'gray')
    description = quartile_descriptions.get(quartile, 'Unknown price range')
    
    # Generate HTML using f-string template
    html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 250px;">
        <h4 style="margin: 0 0 10px 0; color: #333; font-size: 14px;">
            {row['name']}
        </h4>
        <hr style="margin: 5px 0; border-color: #eee;">
        <p style="margin: 5px 0; font-size: 12px;">
            <strong>City:</strong> {row['city']}
        </p>
        <p style="margin: 5px 0; font-size: 12px;">
            <strong>Price:</strong> €{row['price']:.2f}
        </p>
        <p style="margin: 5px 0; font-size: 12px;">
            <strong>Room Type:</strong> {row['room_type']}
        </p>
        <p style="margin: 5px 0; font-size: 12px; color: {color};">
            <strong>Price Range:</strong> {description}
        </p>
    </div>
    """
    
    return html


def _add_markers_to_map(m: folium.Map, df: pd.DataFrame) -> None:
    """
    Add color-coded markers to the map.
    
    Args:
        m (folium.Map): The map to add markers to
        df (pd.DataFrame): DataFrame containing listing data
    """
    # Create marker cluster for better performance
    marker_cluster = MarkerCluster(
        name='Airbnb Listings',
        overlay=True,
        control=True
    ).add_to(m)
    
    # Color scheme for marker icons
    color_scheme = {
        1: 'green',
        2: 'lightgreen',
        3: 'orange',
        4: 'red'
    }
    
    # Add markers for each listing
    for _, row in df.iterrows():
        popup_html = _create_popup_html(row)
        
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row['name']} - €{row['price']:.2f}",
            icon=folium.Icon(
                color=color_scheme.get(row['price_quartile'], 'gray'),
                icon='home',
                prefix='fa'
            )
        ).add_to(marker_cluster)


def _add_ui_elements(m: folium.Map, df: pd.DataFrame) -> None:
    """
    Add UI elements (legend, title, layer control) to the map.
    
    Args:
        m (folium.Map): The map to add UI elements to
        df (pd.DataFrame): DataFrame for statistics
    """
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add title
    title_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50px; z-index: 9999; 
                background-color: white; padding: 10px; 
                border-radius: 5px; border: 2px solid grey; 
                font-family: Arial, sans-serif; font-size: 14px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
        <strong>Airbnb Listings in Greece</strong><br>
        <span style="font-size: 12px;">Color coded by price quartile per city</span>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; z-index: 9999;
                background-color: white; padding: 10px;
                border-radius: 5px; border: 2px solid grey;
                font-family: Arial, sans-serif; font-size: 12px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
        <h4 style="margin: 0 0 8px 0; font-size: 13px;">Price Quartiles</h4>
        <div style="display: flex; align-items: center; margin-bottom: 4px;">
            <div style="width: 15px; height: 15px; background-color: green;
                        margin-right: 8px; border-radius: 50%;"></div>
            <span>Lowest 25%</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 4px;">
            <div style="width: 15px; height: 15px; background-color: lightgreen;
                        margin-right: 8px; border-radius: 50%;"></div>
            <span>25-50%</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 4px;">
            <div style="width: 15px; height: 15px; background-color: orange;
                        margin-right: 8px; border-radius: 50%;"></div>
            <span>50-75%</span>
        </div>
        <div style="display: flex; align-items: center;">
            <div style="width: 15px; height: 15px; background-color: red;
                        margin-right: 8px; border-radius: 50%;"></div>
            <span>Highest 25%</span>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))


def _print_summary_statistics(df: pd.DataFrame, output_path: str) -> None:
    """
    Print summary statistics after map generation.
    
    Args:
        df (pd.DataFrame): DataFrame containing listing data
        output_path (str): Path where map was saved
    """
    logger.info("=" * 50)
    logger.info("MAP GENERATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Total listings processed: {len(df)}")
    logger.info(f"Number of cities: {len(df['city'].unique())}")
    
    # Quartile distribution
    quartile_counts = df['price_quartile'].value_counts().sort_index()
    for quartile, count in quartile_counts.items():
        percentage = (count / len(df)) * 100
        logger.info(f"Quartile {quartile}: {count} listings ({percentage:.1f}%)")
    
    logger.info(f"\nMap saved to: {output_path}")
    logger.info("Open the HTML file in a web browser to view the interactive map.")


if __name__ == "__main__":
    # Generate the map when script is run directly
    # Example usage with custom parameters:
    # generate_airbnb_map(
    #     data_path='my_listings.csv',
    #     output_path='custom_map.html',
    #     center_lat=38.0,
    #     center_lon=23.5,
    #     zoom_start=7
    # )
    
    generate_airbnb_map()

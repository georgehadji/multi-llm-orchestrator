"""
dashboard.py - Interactive Plotly Dash Dashboard for Greek Airbnb Analysis
==========================================================================
Main dashboard application with filters, visualizations, and data tables.
Run with: python dashboard.py
Access at: http://localhost:8050
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table, Input, Output, State
from dash.dependencies import ALL
import dash_bootstrap_components as dbc
import folium
from folium.plugins import MarkerCluster
import json
import logging
from pathlib import Path

# Import project modules
from config import (
    DATA_DIR, OUTPUT_DIR, COLOR_SCHEME, 
    INSIDE_AIRBNB_URLS, get_city_center
)
from price_analysis import (
    prepare_neighbourhood_choropleth,
    prepare_property_type_violin,
    prepare_price_vs_reviews_scatter,
    prepare_price_vs_distance,
    identify_top_neighbourhoods
)
from seasonal_analysis import (
    monthly_average_price_per_city,
    occupancy_rate_heatmap_data,
    summer_winter_price_premium
)
from host_analysis import (
    compare_superhosts,
    analyze_multi_listing_hosts,
    generate_host_analysis_report
)
from review_analysis import ReviewAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
app.title = "Greek Airbnb Analytics Dashboard"

# Global data storage
data_cache = {
    'listings': None,
    'reviews': None,
    'calendar': None,
    'loaded': False
}


def load_data():
    """Load cleaned data from CSV files."""
    global data_cache
    
    try:
        listings_path = OUTPUT_DIR / "listings_cleaned.csv"
        reviews_path = OUTPUT_DIR / "reviews_cleaned.csv"
        calendar_path = OUTPUT_DIR / "calendar_cleaned.csv"
        
        if listings_path.exists():
            data_cache['listings'] = pd.read_csv(listings_path)
            # Convert date columns
            date_cols = ['last_scraped', 'host_since', 'first_review', 'last_review']
            for col in date_cols:
                if col in data_cache['listings'].columns:
                    data_cache['listings'][col] = pd.to_datetime(
                        data_cache['listings'][col], errors='coerce'
                    )
            logger.info(f"Loaded {len(data_cache['listings'])} listings")
        
        if reviews_path.exists():
            data_cache['reviews'] = pd.read_csv(reviews_path)
            if 'date' in data_cache['reviews'].columns:
                data_cache['reviews']['date'] = pd.to_datetime(
                    data_cache['reviews']['date'], errors='coerce'
                )
            logger.info(f"Loaded {len(data_cache['reviews'])} reviews")
        
        if calendar_path.exists():
            data_cache['calendar'] = pd.read_csv(calendar_path)
            if 'date' in data_cache['calendar'].columns:
                data_cache['calendar']['date'] = pd.to_datetime(
                    data_cache['calendar']['date'], errors='coerce'
                )
            logger.info(f"Loaded {len(data_cache['calendar'])} calendar entries")
        
        data_cache['loaded'] = data_cache['listings'] is not None
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        data_cache['loaded'] = False


def create_filter_card():
    """Create the filter controls card."""
    cities = list(INSIDE_AIRBNB_URLS.keys()) if not data_cache['loaded'] else \
             data_cache['listings']['city'].unique().tolist()
    
    return dbc.Card([
        dbc.CardHeader(html.H5("Filters", className="mb-0")),
        dbc.CardBody([
            # City selector
            dbc.Row([
                dbc.Col([
                    html.Label("Select Cities:", className="fw-bold"),
                    dcc.Dropdown(
                        id='city-filter',
                        options=[{'label': c.title(), 'value': c} for c in cities],
                        value=cities[:3] if len(cities) >= 3 else cities,
                        multi=True,
                        placeholder="Select cities..."
                    ),
                ], width=12, className="mb-3"),
            ]),
            
            # Price range slider
            dbc.Row([
                dbc.Col([
                    html.Label("Price Range (€):", className="fw-bold"),
                    dcc.RangeSlider(
                        id='price-slider',
                        min=0,
                        max=1000,
                        step=10,
                        value=[0, 500],
                        marks={0: '€0', 250: '€250', 500: '€500', 750: '€750', 1000: '€1000+'},
                    ),
                ], width=12, className="mb-3"),
            ]),
            
            # Room type filter
            dbc.Row([
                dbc.Col([
                    html.Label("Room Type:", className="fw-bold"),
                    dcc.Dropdown(
                        id='room-type-filter',
                        options=[
                            {'label': 'All', 'value': 'all'},
                            {'label': 'Entire Home/Apt', 'value': 'Entire home/apt'},
                            {'label': 'Private Room', 'value': 'Private room'},
                            {'label': 'Shared Room', 'value': 'Shared room'},
                        ],
                        value='all',
                        clearable=False
                    ),
                ], width=6),
                
                dbc.Col([
                    html.Label("Minimum Reviews:", className="fw-bold"),
                    dcc.Slider(
                        id='min-reviews-slider',
                        min=0,
                        max=50,
                        step=5,
                        value=0,
                        marks={0: '0', 10: '10', 25: '25', 50: '50+'},
                    ),
                ], width=6),
            ]),
            
            # Apply button
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "Apply Filters",
                        id="apply-filters",
                        color="primary",
                        className="mt-3 w-100"
                    ),
                ], width=12),
            ]),
        ]),
    ], className="mb-4")


def create_stats_cards():
    """Create summary statistics cards."""
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id="total-listings", className="card-title text-center"),
                    html.P("Total Listings", className="card-text text-center text-muted"),
                ])
            ], color="primary", outline=True)
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id="avg-price", className="card-title text-center"),
                    html.P("Average Price", className="card-text text-center text-muted"),
                ])
            ], color="success", outline=True)
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id="avg-rating", className="card-title text-center"),
                    html.P("Average Rating", className="card-text text-center text-muted"),
                ])
            ], color="info", outline=True)
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id="total-reviews", className="card-title text-center"),
                    html.P("Total Reviews", className="card-text text-center text-muted"),
                ])
            ], color="warning", outline=True)
        ], width=3),
    ], className="mb-4")


def create_tabs():
    """Create main content tabs."""
    return dbc.Tabs([
        # Overview Tab
        dbc.Tab(label="Overview", tab_id="tab-overview", children=[
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="price-distribution-chart")
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="city-comparison-chart")
                ], width=6),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="room-type-chart")
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="superhost-chart")
                ], width=6),
            ]),
        ]),
        
        # Price Analysis Tab
        dbc.Tab(label="Price Analysis", tab_id="tab-price", children=[
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="neighbourhood-price-chart")
                ], width=12, className="mt-3"),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="price-violin-chart")
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="price-vs-reviews-chart")
                ], width=6),
            ]),
        ]),
        
        # Seasonal Analysis Tab
        dbc.Tab(label="Seasonal", tab_id="tab-seasonal", children=[
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="seasonal-price-chart")
                ], width=12, className="mt-3"),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="occupancy-heatmap")
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="summer-winter-premium")
                ], width=6),
            ]),
        ]),
        
        # Map Tab
        dbc.Tab(label="Map", tab_id="tab-map", children=[
            dbc.Row([
                dbc.Col([
                    html.Iframe(
                        id="folium-map",
                        srcDoc="",
                        style={"width": "100%", "height": "700px", "border": "none"}
                    )
                ], width=12, className="mt-3"),
            ]),
        ]),
        
        # Host Analysis Tab
        dbc.Tab(label="Hosts", tab_id="tab-hosts", children=[
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="superhost-comparison-chart")
                ], width=6, className="mt-3"),
                dbc.Col([
                    dcc.Graph(id="multi-listing-chart")
                ], width=6, className="mt-3"),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id="host-analysis-table")
                ], width=12),
            ]),
        ]),
        
        # Data Table Tab
        dbc.Tab(label="Data", tab_id="tab-data", children=[
            dbc.Row([
                dbc.Col([
                    dash_table.DataTable(
                        id='listings-table',
                        columns=[],
                        data=[],
                        filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                        page_action="native",
                        page_current=0,
                        page_size=25,
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '5px'},
                        style_header={'fontWeight': 'bold', 'backgroundColor': 'lightgrey'},
                    )
                ], width=12, className="mt-3"),
            ]),
        ]),
        
    ], id="main-tabs", active_tab="tab-overview")


# App Layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🇬🇷 Greek Airbnb Analytics Dashboard", className="text-center my-4"),
            html.Hr(),
        ], width=12),
    ]),
    
    # Filters and Stats
    dbc.Row([
        dbc.Col([
            create_filter_card(),
            create_stats_cards(),
        ], width=12),
    ]),
    
    # Main Content Tabs
    dbc.Row([
        dbc.Col([
            create_tabs()
        ], width=12),
    ]),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P(
                "Data source: Inside Airbnb | Built with Dash & Plotly",
                className="text-center text-muted"
            ),
        ], width=12),
    ]),
    
    # Store for filtered data
    dcc.Store(id='filtered-data-store'),
    
], fluid=True)


# Callbacks
@app.callback(
    [Output("filtered-data-store", "data"),
     Output("total-listings", "children"),
     Output("avg-price", "children"),
     Output("avg-rating", "children"),
     Output("total-reviews", "children")],
    [Input("apply-filters", "n_clicks")],
    [State("city-filter", "value"),
     State("price-slider", "value"),
     State("room-type-filter", "value"),
     State("min-reviews-slider", "value")]
)
def apply_filters(n_clicks, cities, price_range, room_type, min_reviews):
    """Apply filters and update statistics."""
    if not data_cache['loaded'] or data_cache['listings'] is None:
        return None, "N/A", "N/A", "N/A", "N/A"
    
    df = data_cache['listings'].copy()
    
    # Apply city filter
    if cities:
        df = df[df['city'].isin(cities)]
    
    # Apply price filter
    df = df[
        (df['price'] >= price_range[0]) & 
        (df['price'] <= price_range[1])
    ]
    
    # Apply room type filter
    if room_type != 'all' and 'room_type' in df.columns:
        df = df[df['room_type'] == room_type]
    
    # Apply minimum reviews filter
    if 'number_of_reviews' in df.columns:
        df = df[df['number_of_reviews'] >= min_reviews]
    
    # Calculate statistics
    total = len(df)
    avg_price = f"€{df['price'].mean():.0f}" if 'price' in df.columns else "N/A"
    avg_rating = f"{df['review_scores_rating'].mean():.1f}★" if 'review_scores_rating' in df.columns else "N/A"
    total_reviews = f"{df['number_of_reviews'].sum():,}" if 'number_of_reviews' in df.columns else "N/A"
    
    # Serialize data for store
    data_json = df.to_dict('records')
    
    return data_json, f"{total:,}", avg_price, avg_rating, total_reviews


@app.callback(
    Output("price-distribution-chart", "figure"),
    [Input("filtered-data-store", "data")]
)
def update_price_distribution(data):
    """Update price distribution histogram."""
    if data is None:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    fig = px.histogram(
        df, x='price', color='city',
        nbins=50,
        title="Price Distribution by City",
        labels={'price': 'Price (€)', 'count': 'Number of Listings'},
        color_discrete_map=COLOR_SCHEME['cities']
    )
    fig.update_layout(showlegend=True, bargap=0.1)
    
    return fig


@app.callback(
    Output("city-comparison-chart", "figure"),
    [Input("filtered-data-store", "data")]
)
def update_city_comparison(data):
    """Update city comparison bar chart."""
    if data is None:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    city_stats = df.groupby('city').agg({
        'price': 'mean',
        'id': 'count',
        'review_scores_rating': 'mean'
    }).reset_index()
    city_stats.columns = ['city', 'avg_price', 'listings', 'avg_rating']
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=city_stats['city'],
            y=city_stats['avg_price'],
            name='Avg Price (€)',
            marker_color='#3498db'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=city_stats['city'],
            y=city_stats['avg_rating'],
            name='Avg Rating',
            mode='lines+markers',
            line=dict(color='#e74c3c')
        ),
        secondary_y=True
    )
    
    fig.update_layout(title="City Comparison: Price vs Rating")
    fig.update_yaxes(title_text="Average Price (€)", secondary_y=False)
    fig.update_yaxes(title_text="Average Rating", secondary_y=True)
    
    return fig


@app.callback(
    Output("room-type-chart", "figure"),
    [Input("filtered-data-store", "data")]
)
def update_room_type_chart(data):
    """Update room type pie chart."""
    if data is None or 'room_type' not in pd.DataFrame(data).columns:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    room_counts = df['room_type'].value_counts()
    
    fig = px.pie(
        values=room_counts.values,
        names=room_counts.index,
        title="Distribution by Room Type",
        hole=0.4
    )
    
    return fig


@app.callback(
    Output("superhost-chart", "figure"),
    [Input("filtered-data-store", "data")]
)
def update_superhost_chart(data):
    """Update superhost comparison chart."""
    if data is None or 'host_is_superhost' not in pd.DataFrame(data).columns:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    superhost_counts = df['host_is_superhost'].value_counts()
    labels = ['Regular Host', 'Superhost']
    values = [
        superhost_counts.get(False, 0),
        superhost_counts.get(True, 0)
    ]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=['#95a5a6', '#27ae60'],
        hole=0.4
    )])
    fig.update_layout(title="Superhost Distribution")
    
    return fig


@app.callback(
    Output("neighbourhood-price-chart", "figure"),
    [Input("filtered-data-store", "data")]
)
def update_neighbourhood_chart(data):
    """Update neighbourhood price bar chart."""
    if data is None:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    if 'neighbourhood' not in df.columns:
        return go.Figure()
    
    # Get top neighbourhoods by average price
    neighbourhood_stats = df.groupby('neighbourhood').agg({
        'price': ['mean', 'count']
    }).reset_index()
    neighbourhood_stats.columns = ['neighbourhood', 'avg_price', 'count']
    neighbourhood_stats = neighbourhood_stats[neighbourhood_stats['count'] >= 5]
    neighbourhood_stats = neighbourhood_stats.nlargest(15, 'avg_price')
    
    fig = px.bar(
        neighbourhood_stats,
        x='avg_price',
        y='neighbourhood',
        orientation='h',
        title="Top 15 Neighbourhoods by Average Price",
        labels={'avg_price': 'Average Price (€)', 'neighbourhood': ''},
        color='avg_price',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    return fig


@app.callback(
    Output("price-violin-chart", "figure"),
    [Input("filtered-data-store", "data")]
)
def update_violin_chart(data):
    """Update price violin chart."""
    if data is None:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    if 'property_type' not in df.columns:
        return go.Figure()
    
    # Get top property types
    top_types = df['property_type'].value_counts().head(6).index
    df_filtered = df[df['property_type'].isin(top_types)]
    
    fig = px.violin(
        df_filtered,
        x='property_type',
        y='price',
        box=True,
        title="Price Distribution by Property Type",
        labels={'property_type': 'Property Type', 'price': 'Price (€)'}
    )
    fig.update_layout(xaxis_tickangle=-45)
    
    return fig


@app.callback(
    Output("price-vs-reviews-chart", "figure"),
    [Input("filtered-data-store", "data")]
)
def update_price_reviews_scatter(data):
    """Update price vs reviews scatter plot."""
    if data is None:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    if 'number_of_reviews' not in df.columns:
        return go.Figure()
    
    fig = px.scatter(
        df,
        x='number_of_reviews',
        y='price',
        color='city',
        title="Price vs Number of Reviews",
        labels={'number_of_reviews': 'Number of Reviews', 'price': 'Price (€)'},
        color_discrete_map=COLOR_SCHEME['cities'],
        opacity=0.6
    )
    
    return fig


@app.callback(
    Output("seasonal-price-chart", "figure"),
    [Input("filtered-data-store", "data")]
)
def update_seasonal_chart(data):
    """Update seasonal price line chart."""
    if data_cache['calendar'] is None or not data_cache['loaded']:
        return go.Figure()
    
    df = data_cache['calendar'].copy()
    
    if data:
        listings_df = pd.DataFrame(data)
        cities = listings_df['city'].unique()
        df = df[df['city'].isin(cities)]
    
    if 'date' not in df.columns or 'price' not in df.columns:
        return go.Figure()
    
    # Monthly average prices
    df['month'] = pd.to_datetime(df['date']).dt.month
    monthly_prices = df.groupby(['city', 'month'])['price'].mean().reset_index()
    
    fig = px.line(
        monthly_prices,
        x='month',
        y='price',
        color='city',
        title="Seasonal Price Patterns by City",
        labels={'month': 'Month', 'price': 'Average Price (€)'},
        color_discrete_map=COLOR_SCHEME['cities'],
        markers=True
    )
    fig.update_xaxes(tickvals=list(range(1, 13)))
    
    return fig


@app.callback(
    Output("occupancy-heatmap", "figure"),
    [Input("filtered-data-store", "data")]
)
def update_occupancy_heatmap(data):
    """Update occupancy heatmap."""
    if data_cache['calendar'] is None or not data_cache['loaded']:
        return go.Figure()
    
    df = data_cache['calendar'].copy()
    
    if data:
        listings_df = pd.DataFrame(data)
        cities = listings_df['city'].unique()
        df = df[df['city'].isin(cities)]
    
    if 'available' not in df.columns:
        return go.Figure()
    
    # Calculate occupancy by city and month
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['occupied'] = ~df['available']
    
    occupancy = df.groupby(['city', 'month'])['occupied'].mean().reset_index()
    occupancy_pivot = occupancy.pivot(index='city', columns='month', values='occupied')
    
    fig = px.imshow(
        occupancy_pivot,
        title="Occupancy Rate Heatmap (City × Month)",
        labels={'x': 'Month', 'y': 'City', 'color': 'Occupancy Rate'},
        color_continuous_scale='RdYlGn',
        aspect='auto'
    )
    
    return fig


@app.callback(
    Output("summer-winter-premium", "figure"),
    [Input("filtered-data-store", "data")]
)
def update_seasonal_premium(data):
    """Update summer vs winter premium chart."""
    if data_cache['calendar'] is None or not data_cache['loaded']:
        return go.Figure()
    
    df = data_cache['calendar'].copy()
    
    if data:
        listings_df = pd.DataFrame(data)
        cities = listings_df['city'].unique()
        df = df[df['city'].isin(cities)]
    
    if 'date' not in df.columns or 'price' not in df.columns:
        return go.Figure()
    
    # Calculate summer/winter premiums
    df['month'] = pd.to_datetime(df['date']).dt.month
    summer_months = [6, 7, 8]
    winter_months = [12, 1, 2]
    
    summer_prices = df[df['month'].isin(summer_months)].groupby('city')['price'].mean()
    winter_prices = df[df['month'].isin(winter_months)].groupby('city')['price'].mean()
    
    premium = ((summer_prices - winter_prices) / winter_prices * 100).reset_index()
    premium.columns = ['city', 'premium_pct']
    
    fig = px.bar(
        premium,
        x='city',
        y='premium_pct',
        title="Summer vs Winter Price Premium",
        labels={'city': 'City', 'premium_pct': 'Premium (%)'},
        color='premium_pct',
        color_continuous_scale='Reds'
    )
    
    return fig


@app.callback(
    Output("folium-map", "srcDoc"),
    [Input("filtered-data-store", "data")]
)
def update_map(data):
    """Generate and update Folium map."""
    if data is None:
        return ""
    
    df = pd.DataFrame(data)
    
    # Required columns
    required_cols = ['latitude', 'longitude', 'price', 'city', 'name']
    if not all(col in df.columns for col in required_cols):
        return "<h3>Map data not available</h3>"
    
    # Remove rows with missing coordinates
    df = df.dropna(subset=['latitude', 'longitude'])
    
    if len(df) == 0:
        return "<h3>No valid coordinates</h3>"
    
    # Calculate price quartiles per city
    df['price_quartile'] = df.groupby('city')['price'].transform(
        lambda x: pd.qcut(x, 4, labels=[1, 2, 3, 4], duplicates='drop')
    ).astype(int)
    
    # Create map centered on Greece
    m = folium.Map(location=[39.0742, 21.8243], zoom_start=6)
    
    # Add tile layers
    folium.TileLayer('CartoDB positron', name='Light').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='Dark').add_to(m)
    
    # Create marker cluster
    marker_cluster = MarkerCluster(name='Listings').add_to(m)
    
    # Color scheme for quartiles
    colors = {1: 'green', 2: 'lightgreen', 3: 'orange', 4: 'red'}
    
    # Add markers (sample if too many)
    sample_size = min(2000, len(df))
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    for _, row in df.iterrows():
        popup_html = f"""
        <div style="min-width: 200px;">
            <h5>{row['name'][:50]}{'...' if len(str(row['name'])) > 50 else ''}</h5>
            <p><b>City:</b> {row['city']}</p>
            <p><b>Price:</b> €{row['price']:.0f}</p>
            <p><b>Room Type:</b> {row.get('room_type', 'N/A')}</p>
        </div>
        """
        
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"€{row['price']:.0f}",
            icon=folium.Icon(
                color=colors.get(row['price_quartile'], 'gray'),
                icon='home',
                prefix='fa'
            )
        ).add_to(marker_cluster)
    
    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; 
                background-color: white; padding: 10px; 
                border: 2px solid grey; border-radius: 5px; z-index: 1000;">
        <h5>Price Quartiles</h5>
        <div><span style="color: green;">●</span> Lowest 25%</div>
        <div><span style="color: lightgreen;">●</span> 25-50%</div>
        <div><span style="color: orange;">●</span> 50-75%</div>
        <div><span style="color: red;">●</span> Highest 25%</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    folium.LayerControl().add_to(m)
    
    return m._repr_html_()


@app.callback(
    Output("superhost-comparison-chart", "figure"),
    [Input("filtered-data-store", "data")]
)
def update_superhost_comparison(data):
    """Update superhost comparison chart."""
    if data is None:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    required_cols = ['host_is_superhost', 'price', 'review_scores_rating']
    if not all(col in df.columns for col in required_cols):
        return go.Figure()
    
    # Group by superhost status
    comparison = df.groupby('host_is_superhost').agg({
        'price': 'mean',
        'review_scores_rating': 'mean',
        'id': 'count'
    }).reset_index()
    comparison['host_type'] = comparison['host_is_superhost'].map({
        True: 'Superhost', False: 'Regular Host'
    })
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Avg Price', 'Avg Rating'))
    
    fig.add_trace(
        go.Bar(
            x=comparison['host_type'],
            y=comparison['price'],
            name='Price',
            marker_color=['#95a5a6', '#27ae60']
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=comparison['host_type'],
            y=comparison['review_scores_rating'],
            name='Rating',
            marker_color=['#95a5a6', '#27ae60']
        ),
        row=1, col=2
    )
    
    fig.update_layout(title_text="Superhost vs Regular Host Comparison")
    
    return fig


@app.callback(
    Output("multi-listing-chart", "figure"),
    [Input("filtered-data-store", "data")]
)
def update_multi_listing_chart(data):
    """Update multi-listing hosts chart."""
    if data is None:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    if 'host_id' not in df.columns:
        return go.Figure()
    
    # Count listings per host
    host_counts = df.groupby('host_id').size().reset_index(name='listing_count')
    host_counts['host_type'] = host_counts['listing_count'].apply(
        lambda x: 'Multi-listing (3+)' if x > 2 else ('Double' if x == 2 else 'Single')
    )
    
    host_type_counts = host_counts['host_type'].value_counts()
    
    fig = px.pie(
        values=host_type_counts.values,
        names=host_type_counts.index,
        title="Host Type Distribution",
        color_discrete_sequence=['#3498db', '#e74c3c', '#f39c12'],
        hole=0.4
    )
    
    return fig


@app.callback(
    Output("listings-table", "columns"),
    Output("listings-table", "data"),
    [Input("filtered-data-store", "data")]
)
def update_data_table(data):
    """Update data table."""
    if data is None:
        return [], []
    
    df = pd.DataFrame(data)
    
    # Select key columns for display
    display_cols = [
        'id', 'name', 'city', 'neighbourhood', 'room_type', 
        'price', 'number_of_reviews', 'review_scores_rating'
    ]
    
    available_cols = [c for c in display_cols if c in df.columns]
    
    columns = [{"name": c.replace('_', ' ').title(), "id": c} for c in available_cols]
    data = df[available_cols].head(1000).to_dict('records')
    
    return columns, data


# Run the app
if __name__ == "__main__":
    print("=" * 60)
    print("Greek Airbnb Analytics Dashboard")
    print("=" * 60)
    
    # Load data
    load_data()
    
    if not data_cache['loaded']:
        print("\n⚠️  Warning: No data found!")
        print("Run the following first:")
        print("  python main.py --download")
        print("  python main.py --clean")
    
    print("\nStarting server...")
    print("Open http://localhost:8050 in your browser")
    print("=" * 60)
    
    app.run_server(debug=True, host='localhost', port=8050)

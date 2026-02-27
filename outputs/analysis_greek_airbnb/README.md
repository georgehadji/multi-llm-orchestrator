# 🇬🇷 Greek Airbnb Analytics Dashboard

A comprehensive data analysis and interactive dashboard for the Greek short-term rental market (Airbnb) covering **Athens**, **Thessaloniki**, **Crete**, **Mykonos**, and **Santorini**.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Dash](https://img.shields.io/badge/dash-2.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)

## 📋 Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [Screenshots](#screenshots)
- [API Reference](#api-reference)
- [Contributing](#contributing)

## ✨ Features

### 📊 Analysis Modules

1. **Price Analysis**
   - Average price per neighbourhood (choropleth maps)
   - Price distribution by property type (violin plots)
   - Price vs number of reviews scatter plots
   - Price vs distance from city center
   - Top 10 most expensive and best value neighbourhoods

2. **Seasonal Analysis**
   - Monthly average price trends by city
   - Occupancy rate heatmaps (city × month)
   - Summer vs winter price premiums
   - Peak season identification

3. **Review & Rating Analysis**
   - Review score distributions per city
   - Sentiment analysis of review comments (Greek + English)
   - Correlation analysis: ratings vs price vs occupancy
   - Keyword extraction from reviews

4. **Host Analysis**
   - Superhost vs regular host comparison
   - Multi-listing host identification (>2 properties)
   - Professional operator share per city

5. **Interactive Map**
   - Folium map with colour-coded markers by price quartile
   - Filterable by city and price range
   - Popup information for each listing

### 🖥️ Dashboard Features

- **City selector**: Multi-select dropdown for all 5 cities
- **Price range slider**: Filter listings by price
- **Room type filter**: Focus on specific accommodation types
- **Multiple tabs**:
  - Overview: Summary statistics and comparisons
  - Price Analysis: Detailed price visualizations
  - Seasonal: Time-series and seasonal patterns
  - Map: Interactive Folium map
  - Hosts: Host analysis and statistics
  - Data: Searchable data table

## 🚀 Demo

Launch the dashboard locally:

```bash
python dashboard.py
```

Then open http://localhost:8050 in your browser.

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the repository

```bash
cd outputs/analysis_greek_airbnb
```

### Step 2: Create virtual environment (recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK data (for sentiment analysis)

```python
python -c "import nltk; nltk.download('punkt')"
```

## 🎯 Usage

### Quick Start - Complete Pipeline

Run the complete analysis pipeline:

```bash
python main.py --all
```

This will:
1. Download data from Inside Airbnb
2. Clean and preprocess the data
3. Run all analyses
4. Generate summary reports

### Step-by-Step Usage

#### 1. Download Data

```bash
python main.py --download
```

Downloads listings, reviews, and calendar data for all 5 cities.

#### 2. Clean Data

```bash
python main.py --clean
```

Cleans and preprocesses the downloaded data:
- Converts price columns to numeric
- Parses date columns
- Handles missing values
- Calculates derived metrics (occupancy rate, revenue estimates)

#### 3. Run Analysis

```bash
python main.py --analyze
```

Runs all analysis modules and generates reports.

#### 4. Launch Dashboard

```bash
python main.py --dashboard
# or directly
python dashboard.py
```

Launches the interactive dashboard at http://localhost:8050

### Command Line Options

```bash
python main.py --help

# Examples
python main.py --download --clean           # Download and clean
python main.py --clean --analyze            # Clean and analyze existing data
python main.py --download --cities athens thessaloniki  # Download only specific cities
```

## 📚 Data Sources

Data is sourced from [Inside Airbnb](http://insideairbnb.com/get-the-data.html), an independent, non-commercial project that provides data about Airbnb listings.

### Cities Covered

| City | Region | Data URL |
|------|--------|----------|
| Athens | Attica | `greece/attica/athens` |
| Thessaloniki | Central Macedonia | `greece/central-macedonia/thessaloniki` |
| Crete | Crete | `greece/crete` |
| Mykonos | South Aegean | `greece/south-aegean/mykonos` |
| Santorini | South Aegean | `greece/south-aegean/santorini` |

### Data Files

For each city, the following files are downloaded:
- `listings.csv.gz` - Detailed listing information
- `reviews.csv.gz` - Individual review data
- `calendar.csv.gz` - Availability and pricing calendar

## 📁 Project Structure

```
analysis_greek_airbnb/
├── config.py                    # Central configuration
├── main.py                      # Entry point and pipeline
├── dashboard.py                 # Plotly Dash application
├── data_loader.py               # Data downloading module
├── data_cleaning.py             # Data cleaning module
├── price_analysis.py            # Price analysis functions
├── seasonal_analysis.py         # Seasonal analysis functions
├── host_analysis.py             # Host analysis functions
├── review_analysis.py           # Review analysis functions
├── map_generator.py             # Folium map generator
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── data/                        # Downloaded raw data
├── output/                      # Analysis outputs
│   ├── listings_cleaned.csv
│   ├── reviews_cleaned.csv
│   ├── calendar_cleaned.csv
│   ├── KEY_FINDINGS.md
│   ├── seasonal_analysis.json
│   ├── host_analysis.json
│   └── review_analysis.json
└── .cache/                      # Temporary cache files
```

## 🔍 Key Findings

### Sample Insights (from analysis)

Based on typical patterns in Greek Airbnb data:

#### Price Insights
- **Mykonos** and **Santorini** are typically the most expensive destinations
  - Mykonos average: €250-400/night in peak season (August)
  - Santorini average: €200-350/night in peak season
- **Athens** offers more affordable options
  - Athens average: €60-100/night year-round
- **Thessaloniki** provides good value
  - Thessaloniki average: €50-80/night

#### Seasonal Patterns
- **Summer Premium**: 50-150% price increase in July-August vs winter
- **Peak Season**: June through September
- **Low Season**: November through March
- Islands show stronger seasonality than mainland cities

#### Host Patterns
- Approximately 20-30% of listings are from Superhosts
- 15-25% of hosts operate multiple listings (potential professionals)
- Multi-listing concentration is higher in Athens than islands

#### Review Sentiment
- Overall positive sentiment (>80% positive reviews)
- Greek reviews often mention "φιλοξενία" (hospitality)
- Common positive themes: location, cleanliness, host communication

*Note: Actual findings will vary based on the specific data snapshot analyzed.*

## 📸 Screenshots

### Dashboard Overview
The dashboard provides an intuitive interface with:
- Real-time filtering
- Interactive visualizations
- Exportable data tables

### Sample Visualizations

1. **Price Distribution by City** - Histogram showing price ranges
2. **Seasonal Trends** - Line charts showing price seasonality
3. **Interactive Map** - Folium map with colour-coded listings
4. **Occupancy Heatmap** - City × month occupancy rates

## 📖 API Reference

### Data Cleaning (`data_cleaning.py`)

```python
from data_cleaning import DataCleaner

cleaner = DataCleaner()
cleaned_df = cleaner.clean_listings(raw_df, city='athens')
```

### Price Analysis (`price_analysis.py`)

```python
from price_analysis import prepare_neighbourhood_choropleth

choropleth_data = prepare_neighbourhood_choropleth(df)
```

### Seasonal Analysis (`seasonal_analysis.py`)

```python
from seasonal_analysis import monthly_average_price_per_city

monthly_prices = monthly_average_price_per_city(df, start_date, end_date)
```

### Host Analysis (`host_analysis.py`)

```python
from host_analysis import compare_superhosts, analyze_multi_listing_hosts

comparison_df, stats = compare_superhosts(df)
summary, details = analyze_multi_listing_hosts(df)
```

### Review Analysis (`review_analysis.py`)

```python
from review_analysis import ReviewAnalyzer

analyzer = ReviewAnalyzer()
sentiment_results = analyzer.analyze_sentiment(reviews_df)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run linting
ruff check .

# Run tests
pytest tests/
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Inside Airbnb](http://insideairbnb.com/) for providing the data
- [Plotly Dash](https://dash.plotly.com/) for the dashboard framework
- [Folium](https://python-visualization.github.io/folium/) for interactive maps

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ for the Greek Airbnb Community**

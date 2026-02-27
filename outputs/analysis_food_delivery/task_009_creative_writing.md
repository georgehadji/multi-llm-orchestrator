# Athens Food Delivery Analysis

## Project Overview

This project aims to analyze food delivery trends in Athens, focusing on understanding customer preferences, peak delivery times, and predicting restaurant ratings. By leveraging machine learning models, we provide insights into the most popular cuisines, optimal delivery times, and factors influencing delivery fees. The project also includes a dashboard for interactive data exploration.

## Data Sources

1. **Kaggle Datasets**: The primary dataset was sourced from Kaggle, containing information on restaurant orders, customer reviews, delivery times, and fees. This dataset provides a comprehensive view of the food delivery landscape in Athens.

2. **Web Scraping**: Additional data was collected via web scraping to supplement the Kaggle dataset. This includes real-time restaurant ratings and menu updates from popular food delivery platforms in Athens. Scraping was conducted using Python libraries such as BeautifulSoup and Scrapy, adhering to the respective websites' terms of service.

## Installation Instructions

To set up the project environment, follow these steps:

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/athens-food-delivery-analysis.git
   cd athens-food-delivery-analysis
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the interactive dashboard and explore the data, execute the following command:

```bash
python dashboard.py
```

This will launch a web-based dashboard where you can visualize various aspects of the food delivery data, including popular cuisines, delivery times, and fee trends.

## Key Findings

1. **Most Popular Cuisine in Athens**: The analysis revealed that Mediterranean cuisine is the most ordered type in Athens, followed closely by Italian and Greek cuisines.

2. **Peak Delivery Hour**: The peak hour for food deliveries in Athens is between 7 PM and 8 PM, aligning with typical dinner times.

3. **Rating Prediction Accuracy**: Our Random Forest Classifier achieved an accuracy of 0.68, surpassing the target of 0.65. This indicates a reliable model for predicting restaurant ratings based on customer reviews and order data.

4. **Delivery Fee Trends**: The analysis showed a trend of increasing delivery fees during peak hours and weekends, likely due to higher demand and traffic conditions.

## Limitations and Future Work

### Limitations

- **Data Completeness**: Some datasets had missing values, which were handled through imputation. However, this may introduce bias in the analysis.
- **Generalizability**: The findings are specific to Athens and may not be applicable to other cities or regions without further analysis.
- **Real-time Data**: The data used is historical and may not reflect current trends or sudden market changes.

### Future Work

- **Enhanced Predictive Models**: Future iterations could explore more advanced models, such as deep learning techniques, to improve prediction accuracy.
- **Broader Data Collection**: Expanding data collection to include more cities and diverse sources could enhance the robustness of the analysis.
- **Real-time Dashboard Updates**: Implementing real-time data feeds into the dashboard would provide up-to-date insights and improve decision-making capabilities.

By addressing these limitations and pursuing future work, the project can provide even more valuable insights into the food delivery industry in Athens and beyond.
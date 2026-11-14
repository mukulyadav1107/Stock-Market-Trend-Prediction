# Machine Learning-Based Stock Market Trend Prediction

A Python-based Machine Learning project that predicts the next trading day's stock market direction (UP or DOWN) using historical stock data.

## Project Overview

This project uses historical data from Reliance Industries Limited (RELIANCE.NS) to train and compare five machine learning classification algorithms.

The objective is to predict stock trend direction rather than the exact future stock price.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit
- Yahoo Finance (yfinance)
- Joblib

## Dataset

- Company: Reliance Industries Limited
- Ticker: RELIANCE.NS
- Period: Approximately 5 years
- Frequency: Daily
- Features: Date, Open, High, Low, Close and Volume

## Machine Learning Algorithms

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)

## Feature Engineering

The project calculates:

- Daily Returns
- 5-day, 10-day and 20-day Moving Averages
- Volatility
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- MACD Signal Line

The target variable is:
- 1 = UP
- 0 = DOWN

## Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Experimental Results

| Algorithm | Accuracy |
|---|---:|
| KNN | 57.38% |
| SVM | 54.10% |
| Logistic Regression | 51.23% |
| Decision Tree | 50.00% |
| Random Forest | 48.77% |

These are experimental results from the current chronological test split, not guaranteed future performance.

## Project Structure

- `download_stock_data.py` - Downloads historical stock data.
- `preprocessing.py` - Cleans the dataset.
- `feature_engineering.py` - Creates technical indicators and target labels.
- `train_models.py` - Trains and compares five ML algorithms.
- `visualization.py` - Generates performance comparison graphs.
- `confusion_matrix_visualization.py` - Generates confusion matrices.
- `save_models.py` - Saves trained models.
- `app.py` - Runs the Streamlit prediction application.
- `saved_models/` - Contains trained model files.

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/mukulyadav1107/Stock-Market-Trend-Prediction.git
```

### 2. Enter the project directory

```bash
cd Stock-Market-Trend-Prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## Limitations

- Stock market prices are uncertain and influenced by external events.
- Historical patterns may not continue in the future.
- The application uses downloaded historical data, not live market prices.
- Model predictions are not guaranteed trading signals.

## Future Scope

- Support multiple stocks and market indices.
- Incorporate additional market indicators.
- Improve model validation and predictive performance.
- Explore advanced time-series forecasting techniques.
- Integrate live market data.

## Disclaimer

This project is developed for educational and academic purposes only. It does not provide financial advice or guarantee stock market predictions.

## Author

**Mukul Yadav**

BTech CSE – Artificial Intelligence and Machine Learning

GitHub: https://github.com/mukulyadav1107
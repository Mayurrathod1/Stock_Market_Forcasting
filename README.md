
# Stock Market Prediction

This project explores various methods for predicting stock prices using historical data. By employing five different approaches—Average, Linear Regression, Random Forest, K-Nearest Neighbors (KNN), and Long Short-Term Memory (LSTM) neural networks—the goal is to develop and compare models to understand their strengths and weaknesses in forecasting stock prices.

## Table of Contents

- [Project Overview](#project-overview)
- [Models](#models)
  - [Average](#average)
  - [Linear Regression](#linear-regression)
  - [Random Forest](#random-forest)
  - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
  - [Long Short-Term Memory (LSTM)](#long-short-term-memory-lstm)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)

## Project Overview

The stock market prediction project aims to forecast stock prices based on historical data using five distinct techniques. The dataset includes historical prices, and each model attempts to predict future stock prices based on past trends and patterns.

## Models

### 1. Average

**Description:** This model calculates the average of past prices and uses this as the predicted price for the next time step. While simplistic, it provides a baseline for evaluating the performance of more complex models.

### 2. Linear Regression

**Description:** A statistical method that models the relationship between a dependent variable (stock price) and an independent variable (time) by fitting a line to the data. Linear regression assumes that stock prices have a linear relationship with time, which may work well for short-term predictions.

### 3. Random Forest Regression

**Description:** Random Forest is an ensemble learning method based on decision trees. By constructing multiple trees and averaging their predictions, the Random Forest model can capture complex, non-linear relationships in stock price data.

### 4. K-Nearest Neighbors (KNN)

**Description:** The K-Nearest Neighbors algorithm predicts the stock price by finding the K most similar historical data points (neighbors) and averaging their prices. KNN is effective when there are strong patterns in past prices that can inform future prices.

### 5. Long Short-Term Memory (LSTM)

**Description:** LSTM is a type of Recurrent Neural Network (RNN) that is especially useful for time-series prediction, as it can retain memory of previous states. This model is useful for capturing long-term dependencies in stock price data.

## Setup

### Prerequisites

- Python 3.x
- Required libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tensorflow` or `keras` (for LSTM)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/stock-market-prediction.git
   cd stock-market-prediction
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the Data**: Ensure you have historical stock price data in CSV format. The dataset should include at least two columns: `Date` and `Close` (or `Adjusted Close`).

2. **Run Each Model**: Execute the scripts for each model to see their individual predictions. For example:
   ```bash
   python average_model.py
   python linear_regression.py
   python random_forest.py
   python knn.py
   python lstm.py
   ```

3. **Compare Results**: Each script will output predictions and save plots comparing actual vs. predicted prices.

## Results

Each model's performance can be evaluated based on metrics such as Mean Squared Error (MSE) or Mean Absolute Error (MAE). LSTM may perform better on datasets with long-term dependencies, while Random Forest or Linear Regression could work well on simpler, short-term datasets.

| Model            | Mean Squared Error (MSE) | R-squared Eroor           |
|------------------|--------------------------|---------------------------|
| Average          | 12.7570                  | 0.9170                    |
| Linear Regression| 1.0646                   | 1.0000                    |
| Random Forest Regression  | 1.1764          | 1.0000                    |
| K-Nearest Neighbors (KNN) | 32.6488         | 0.5949                    |
| LSTM             | 285.0629                 | -                         |

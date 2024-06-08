# House Price Prediction Dashboard

This project is a Streamlit-based interactive dashboard for predicting house prices. It allows users to upload a CSV file containing house data, perform exploratory data analysis (EDA), train a machine learning model, and visualize the results.

## Features

- **Data Upload:** Upload a CSV file containing house data.
- **Exploratory Data Analysis (EDA):** Visualize relationships between different features and house prices.
- **Feature Engineering:** Create new features to enhance model performance.
- **Model Training:** Train a linear regression model to predict house prices.
- **Model Evaluation:** Evaluate model performance using Mean Squared Error (MSE) and R^2 score.
- **Visualization:** Plot actual vs. predicted house prices.

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/house-price-prediction-dashboard.git
    cd house-price-prediction-dashboard
    ```

2. **Create a virtual environment:**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app:**

    ```sh
    streamlit run house_price_dashboard.py
    ```

## Usage

1. **Upload a CSV file:** Use the file uploader in the Streamlit app to upload your house data CSV file.
2. **Explore the Data:** View the raw data and scatter plots to understand the relationships between features and house prices.
3. **Train the Model:** The app will preprocess the data, create features, and train a linear regression model.
4. **Evaluate the Model:** View the model's performance metrics and visualizations of actual vs. predicted prices.

## Example Data Format

The CSV file should have the following columns:

- `SquareFootage`: The total square footage of the house.
- `Bedrooms`: The number of bedrooms in the house.
- `Bathrooms`: The number of bathrooms in the house.
- `YearBuilt`: The year the house was built.
- `Price`: The price of the house.

## Screenshots

### Price vs. Square Footage

![Price vs. Square Footage](./images/price_vs_square_footage.png)

### Actual vs. Predicted Prices

![Actual vs. Predicted Prices](./images/actual_vs_predicted_prices.png)

## Code

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit configuration
st.title('House Price Prediction Dashboard')

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Display raw data
    st.subheader('Raw Data')
    st.write(df.head())

    # Data preprocessing
    df.dropna(inplace=True)  # Drop rows with missing values for simplicity

    # Exploratory Data Analysis (EDA)
    st.subheader('Exploratory Data Analysis')
    st.write('Scatter plot: Price vs. Square Footage')
    fig, ax = plt.subplots()
    ax.scatter(df['SquareFootage'], df['Price'])
    ax.set_xlabel('Square Footage')
    ax.set_ylabel('Price')
    st.pyplot(fig)

    # Feature Engineering
    df['Age'] = 2024 - df['YearBuilt']  # Assuming current year is 2024
    features = ['SquareFootage', 'Bedrooms', 'Bathrooms', 'Age']
    target = 'Price'

    # Prepare data for modeling
    X = df[features]
    y = df[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader('Model Performance')
    st.write(f'Mean Squared Error: {mse}')
    st.write(f'R^2 Score: {r2}')

    # Visualization of predictions vs actual values
    st.subheader('Actual vs Predicted Prices')
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    st.pyplot(fig)

else:
    st.write("Please upload a CSV file.")

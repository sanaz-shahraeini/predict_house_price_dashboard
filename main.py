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


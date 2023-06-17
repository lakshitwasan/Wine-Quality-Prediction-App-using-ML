import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# Load the saved model
model = pickle.load(open('wine.pickle', 'rb'))

# Define the app


def main():
    # Set the app title
    st.title('Wine Quality Prediction')

    # Add a brief description
    st.write('This app predicts the quality of wine based on various features.')

    # Add inputs for the features
    fixed_acidity = st.slider(
        'Fixed Acidity', min_value=4.60, max_value=15.90, value=10.00)
    volatile_acidity = st.slider(
        'Volatile Acidity', min_value=0.12, max_value=1.58, value=0.50)
    citric_acid = st.slider(
        'Citric Acid', min_value=0.00, max_value=1.00, value=0.50)
    residual_sugar = st.slider(
        'Residual Sugar', min_value=0.90, max_value=15.5, value=5.00)
    chlorides = st.slider(
        'Chlorides', min_value=0.01, max_value=0.62, value=0.50)
    free_sulfur_dioxide = st.slider(
        'Free Sulfur Dioxide', min_value=1.00, max_value=68.00, value=30.00)
    total_sulfur_dioxide = st.slider(
        'Total Sulfur Dioxide', min_value=6.00, max_value=289.00, value=100.00)
    density = st.slider(
        'Density', min_value=0.9900, max_value=1.0036, value=1.000)
    pH = st.slider('pH', min_value=2.74, max_value=10.00, value=4.01)
    sulphates = st.slider(
        'Sulphates', min_value=0.33, max_value=2.00, value=1.00)
    alcohol = st.slider(
        'Alcohol', min_value=8.40, max_value=14.90, value=12.00)

    # Create a feature vector from the inputs
    features = pd.DataFrame({
        'fixed acidity': [fixed_acidity],
        'volatile acidity': [volatile_acidity],
        'citric acid': [citric_acid],
        'residual sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free sulfur dioxide': [free_sulfur_dioxide],
        'total sulfur dioxide': [total_sulfur_dioxide],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol]
    })

    # Predict the wine quality
    prediction = model.predict(features)[0]

    if st.button("Predict"):

        if (prediction == 1):
            info = 'Good wine'
            st.success('Prediction: {}'.format(info))

        else:
            info = 'Bad wine'
            st.error('Prediction: {}'.format(info))


# Run the app
if __name__ == '__main__':
    main()

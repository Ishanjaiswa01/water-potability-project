import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model and the scaler
model = joblib.load('water_potability_model.joblib')
scaler = joblib.load('scaler.joblib')

# Set up the title and a brief description for the web app
st.title('💧 Water Potability Prediction App')
st.write("""
This app predicts whether water is safe for drinking based on 9 water quality parameters.
""")

# Create sliders in the sidebar for user input
st.sidebar.header('Input Water Quality Parameters:')

def user_inputs():
    """Function to get user input from sliders."""
    ph = st.sidebar.slider('pH', 0.0, 14.0, 7.0, 0.1)
    hardness = st.sidebar.slider('Hardness (mg/L)', 40.0, 330.0, 170.0, 1.0)
    solids = st.sidebar.slider('Solids (ppm)', 300.0, 62000.0, 21000.0, 100.0)
    chloramines = st.sidebar.slider('Chloramines (ppm)', 0.3, 13.2, 7.0, 0.1)
    sulfate = st.sidebar.slider('Sulfate (mg/L)', 120.0, 490.0, 330.0, 1.0)
    conductivity = st.sidebar.slider('Conductivity (μS/cm)', 180.0, 760.0, 420.0, 1.0)
    organic_carbon = st.sidebar.slider('Organic Carbon (ppm)', 2.0, 29.0, 14.0, 0.1)
    trihalomethanes = st.sidebar.slider('Trihalomethanes (μg/L)', 0.7, 124.0, 66.0, 0.1)
    turbidity = st.sidebar.slider('Turbidity (NTU)', 1.4, 6.8, 4.0, 0.1)
    
    # Store the inputs in a dictionary
    data = {
        'ph': ph,
        'Hardness': hardness,
        'Solids': solids,
        'Chloramines': chloramines,
        'Sulfate': sulfate,
        'Conductivity': conductivity,
        'Organic_carbon': organic_carbon,
        'Trihalomethanes': trihalomethanes,
        'Turbidity': turbidity
    }
    # Convert the dictionary to a pandas DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_inputs()

# Display the user's input parameters
st.subheader('Your Input Parameters:')
st.write(input_df)

# Create a button to make the prediction
if st.button('Predict Potability'):
    # Scale the user input using the saved scaler
    input_scaled = scaler.transform(input_df)
    
    # Make a prediction using the loaded model
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # Display the prediction result
    st.subheader('Prediction:')
    if prediction[0] == 1:
        st.success(f"**The water is Drinkable!** 👍")
        st.write(f"Confidence: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.error(f"**The water is Not Drinkable.** 👎")
        st.write(f"Confidence: {prediction_proba[0][0]*100:.2f}%")
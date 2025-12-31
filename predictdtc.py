import streamlit as st
import pickle
import numpy as np
import pandas as pd

def load_model():
    with open('saved_steps_DTS.pkl','rb') as file:
        data = pickle.load(file)
    return data
    
data = load_model()

model = data["model"]
numeric_cols = data["numeric_cols"]
encoded_cols = data["encoded_cols"]
categorical_cols = data["categorical_cols"]
imputer = data["imputer"]
scaler = data["scaler"]
encoder = data["encoder"]

def show_predict_page():
    st.title("Rain Predictor")
    st.write("""### We need some information to predict tomorrow's rain""")

    Locations = ('Adelaide', 'Albany', 'Albury', 'AliceSprings', 'BadgerysCreek',
        'Ballarat', 'Bendigo', 'Brisbane', 'Cairns', 'Canberra', 'Cobar',
        'CoffsHarbour', 'Dartmoor', 'Darwin', 'GoldCoast', 'Hobart',
        'Katherine', 'Launceston', 'Melbourne', 'MelbourneAirport',
        'Mildura', 'Moree', 'MountGambier', 'MountGinini', 'Newcastle',
        'Nhil', 'NorahHead', 'NorfolkIsland', 'Nuriootpa', 'PearceRAAF',
        'Penrith', 'Perth', 'PerthAirport', 'Portland', 'Richmond', 'Sale',
        'SalmonGums', 'Sydney', 'SydneyAirport', 'Townsville',
        'Tuggeranong', 'Uluru', 'WaggaWagga', 'Walpole', 'Watsonia',
        'Williamtown', 'Witchcliffe', 'Wollongong', 'Woomera')
    WindGustDirections = ('E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
        'SSW', 'SW', 'W', 'WNW', 'WSW')
    WindDirections9am = ('E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
        'SSW', 'SW', 'W', 'WNW', 'WSW')
    WindDirections3pm = ('E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
        'SSW', 'SW', 'W', 'WNW', 'WSW')
    RainToday1 = ('No', 'Yes')

    Location = st.selectbox("Location",Locations)
    MinTemp = st.slider("Minimum Temperature",-10,45,3)
    MaxTemp = st.slider("Maximum Temperature",-10,50,3)
    Rainfall = st.slider("Rain fall",0.0,400.0,3.0)
    Evaporation = st.slider("Evaporation",0.0,150.0,3.0)
    Sunshine = st.slider("Sunshine",0.0,20.0,3.0)
    WindGustDir = st.selectbox("Wind Gust Directions",WindGustDirections)
    WindGustSpeed = st.slider("Wind Gust Speed",0.0,150.0,3.0)
    WindDir9am = st.selectbox("Wind Direction 9am",WindDirections9am)
    WindDir3pm = st.selectbox("Wind Directions 3pm",WindDirections3pm)
    WindSpeed9am = st.slider("Wind Speed 9am",0.0,130.0,3.0)
    WindSpeed3pm = st.slider("Wind Speed 3pm",0.0,130.0,3.0)
    Humidity9am = st.slider("Humidity 9am",0.0,100.0,3.0)
    Humidity3pm = st.slider("Humidity 3pm",0.0,100.0,3.0)
    Pressure9am = st.slider("Pressure 9am",900.0,1050.0,3.0)
    Pressure3pm = st.slider("Pressure 3pm",900.0,1050.0,3.0)
    Cloud9am = st.slider("Cloud 9am",0.0,9.0,3.0)
    Cloud3pm = st.slider("Cloud 3pm",0.0,9.0,3.0)
    Temp9am = st.slider("Temperature 9am",-10.0,50.0,3.0)
    Temp3pm = st.slider("Temperature 3pm",-10.0,50.0,3.0)
    RainToday = st.selectbox("Did it rain today",RainToday1)

    ok = st.button("Predict")
    if ok:
        x = np.array([[Location,MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday]])

        input_df = pd.DataFrame(x,columns=["Location","MinTemp","MaxTemp","Rainfall","Evaporation","Sunshine","WindGustDir","WindGustSpeed","WindDir9am","WindDir3pm","WindSpeed9am","WindSpeed3pm","Humidity9am","Humidity3pm","Pressure9am","Pressure3pm","Cloud9am","Cloud3pm","Temp9am","Temp3pm","RainToday"])
        input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
        input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])

        x_input = input_df[numeric_cols+encoded_cols]
        pred = model.predict(x_input)[0]
        prob = model.predict_proba(x_input)[0][list(model.classes_).index(pred)]
        st.subheader(f"Will Tomorrow Rain: {pred[0]}")

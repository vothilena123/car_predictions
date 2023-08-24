import streamlit as st
import pandas as pd
from scipy.stats import f_oneway
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import joblib
import os
import datetime
import numpy as np
import mplcursors
import plotly.express as px
import plotly.graph_objs as go
import uuid
from streamlit_lottie import st_lottie
import requests
import altair as alt

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.sidebar.header('Dashboard `version 1`')


st.sidebar.markdown('''
---
Created with ❤️ by Vo Thi Le Na.
''')


def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_animation_1 = "https://assets9.lottiefiles.com/packages/lf20_verkcp5o.json"

lottie_anime_json = load_lottie_url(lottie_animation_1)

st_lottie(lottie_anime_json, key = "car", width=300, height=300)

st.title('Ford Used Car Fuel Consumption Prediction')
car = pd.read_csv("ford.csv")
rf_model_mpg = joblib.load(os.path.join('ImportanceRFModelMPG.pkl'))
X_for_mpg = joblib.load(os.path.join('XImportanceMPG.pkl'))
model_mpg = joblib.load(os.path.join('MPGPredictRandomForest.pkl'))
X_var_of_mpg_test = joblib.load(os.path.join('XVarOfMPGTest.pkl'))
y_mpg_test = joblib.load(os.path.join('YMPGTest.pkl'))
X_var_of_mpg = car.drop(columns='mpg')
y_mpg = car['mpg']
count = 0


if st.checkbox("Show the Relationships between Car Fuel Consumption and Input Variables", False):
    with st.expander("Chart show Importance of the input variables"):
        # Get the importance of the input variables
        importances_mpg = rf_model_mpg.feature_importances_
        indices_mpg = np.argsort(importances_mpg)[::-1]
        
        # Create DataFrame from input data
        df = pd.DataFrame({
            'Input variable': X_for_mpg.columns[indices_mpg],
            'Importance': importances_mpg[indices_mpg]
        })

        # Creating charts with Altair
        chart = alt.Chart(df).mark_bar().encode(
            x='Importance:Q',
            y=alt.Y('Input variable:N', sort=alt.EncodingSortField(field='Importance', order='descending')),
            tooltip=['Importance', 'Input variable']
        ).properties(
            width=500,
            height=300,
            title='Importance Plot'
        )

        # Show chart on Streamlit
        st.altair_chart(chart, use_container_width=True)

    with st.expander("Relationship between Car Fuel Consumption and Car Model"):
        if st.checkbox("Show Mean mpg of each model", key=count+1):
            mean_mpg_model = car.groupby('model')['mpg'].mean()
            st.dataframe(mean_mpg_model, width = 500, height = 700)

        if st.checkbox("Show the relationship by using ADOVA", key=count + 2):
            grouped_model_mpg = [car[car['model'] == model]['mpg'] for model in car['model'].unique()]
            f_statistic, p_value = f_oneway(*grouped_model_mpg)
            col1, col2= st.columns(2)
            col1.metric("F-statistic",f_statistic)
            col2.metric("P-value:", p_value)
           

        if st.checkbox("Show the relationship by scatterplot", key=count + 3):
            fig = px.scatter(car, x='mpg', y='model', color='model', hover_data=['mpg'])

            fig.update_layout(
                title='Relationship between Car Fuel Consumption and Car Model',
                xaxis_title='Fuel Consumption',
                yaxis_title='Car Model'
            )

            st.plotly_chart(fig)


    with st.expander("Show the Relationship between Car Fuel Consumption and Transmission"):
        if st.checkbox("Show Mean mpg of each transmission", key=count + 4):
            mean_mpg_model = car.groupby('transmission')['mpg'].mean()
            st.dataframe(mean_mpg_model, width = 500)

        if st.checkbox("Show the relationship by using ADOVA", key=count + 5):
            grouped_transmission_mpg = [car[car['transmission'] == model]['mpg'] for model in car['transmission'].unique()]
            f_statistic, p_value = f_oneway(*grouped_transmission_mpg)
            col1, col2= st.columns(2)
            col1.metric("F-statistic",f_statistic)
            col2.metric("P-value:", p_value)

        if st.checkbox("Show the relationship by violinplot", key=count + 6):
            fig = px.violin(car, x='mpg', y='transmission', color='transmission', hover_data=['mpg'])

            fig.update_layout(
                title='Relationship between Car Fuel Consumption and Car Transmission',
                xaxis_title='Fuel Consumption',
                yaxis_title='Car Transmission'
            )

            st.plotly_chart(fig)

    with st.expander("Show the Relationship between Car Fuel Consumption and Fuel Type"):
        if st.checkbox("Show Mean mpg of each fuelType", key=count + 7):
            mean_mpg_model = car.groupby('fuelType')['mpg'].mean()
            st.dataframe(mean_mpg_model, width = 500)

        if st.checkbox("Show the relationship using ADOVA", key=count + 8):
            grouped_fuelType_mpg = [car[car['fuelType'] == model]['mpg'] for model in car['fuelType'].unique()]
            f_statistic, p_value = f_oneway(*grouped_fuelType_mpg)
            col1, col2= st.columns(2)
            col1.metric("F-statistic",f_statistic)
            col2.metric("P-value:", p_value)

        if st.checkbox("Show the relationship by boxplot", key=count + 9):

            fig = px.box(car, x='fuelType', y='mpg', color='fuelType', hover_data=['mpg'])

            fig.update_layout(
                title='Relationship between Car Fuel Consumption and Car Fuel Type',
                xaxis_title='Fuel Consumption',
                yaxis_title='Car Fuel Type'
            )

            st.plotly_chart(fig)
    
    with st.expander("Heat Map show the Relationship between Fuel Consumption and the rest Attributes"):
        
        # Get a dataframe containing only numeric columns
        numeric_cols = car.select_dtypes(include=['int64', 'float64']).columns
        car_numeric = car[numeric_cols]

        # Draw heatmap
        fig = go.Figure(data=go.Heatmap(
                z=car_numeric.corr(),
                x=car_numeric.columns,
                y=car_numeric.columns,
                colorscale = 'Greens'
        ))

        fig.update_layout(
            title='Correlation Heatmap',
            xaxis_title='Features',
            yaxis_title='Features'
        )

        st.plotly_chart(fig)


if st.checkbox("Performance coefficients for model accuracy", False):
    y_mpg_predict = model_mpg.predict(X_var_of_mpg_test)
    r2Price = r2_score(y_mpg_test, y_mpg_predict)
    maePrice = mean_absolute_error(y_mpg_test, y_mpg_predict)
       
    col1, col2= st.columns(2)
    col1.metric("Coefficient of Determination (R2 Score)",r2Price)
    col2.metric("Mean absolute error:", maePrice)

if st.checkbox("Compare real and predicted car fuel comsumption chart", False):
    y_mpg_predict = model_mpg.predict(X_var_of_mpg_test)
    engine_size_test = X_var_of_mpg_test['engineSize']
    compare = pd.DataFrame({'y':y_mpg_test, 'y_predict':y_mpg_predict, 'engine_size_test': engine_size_test})
    compare_sorted_by_engine_size = compare.sort_values('engine_size_test')
    

    fig = go.Figure()
    # Add trace representing data points
    fig.add_trace(go.Scatter(
        x=compare_sorted_by_engine_size['engine_size_test'],
        y=compare_sorted_by_engine_size['y'],
        mode='markers',
        name='MPG Test',
        marker=dict(size=8, color='violet', opacity=0.8)
    ))
    fig.add_trace(go.Scatter(
        x=compare_sorted_by_engine_size['engine_size_test'],
        y=compare_sorted_by_engine_size['y_predict'],
        mode='markers',
        name='MPG Predict',
        marker=dict(size=8, color='orange', opacity=0.8)
    ))

    # Add a trace showing the line connecting the points y and y_predict
    fig.add_trace(go.Scatter(
        x=compare_sorted_by_engine_size['engine_size_test'],
        y=compare_sorted_by_engine_size['y'],
        mode='lines',
        name='MPG Test Line',
        line=dict(color='blue', width=2, dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=compare_sorted_by_engine_size['engine_size_test'],
        y=compare_sorted_by_engine_size['y_predict'],
        mode='lines',
        name='MPG Predict Line',
        line=dict(color='orange', width=2)
    ))

    # Set title and parameters on axis
    fig.update_layout(
        title='Comparison of MPG Test and MPG Predict',
        xaxis_title='MPG Test',
        yaxis_title='MPG'
    )

    # Show chart
    st.plotly_chart(fig)

car_model=sorted(car['model'].unique())
car_engine_size = sorted(car['engineSize'].unique())
car_engine_size.insert(0, "")
car_transmission = sorted(car['transmission'].unique())
car_transmission.insert(0, "")
car_fuel_type= sorted(car['fuelType'].unique())
car_fuel_type.insert(0, "")
car_engine_size= sorted(car['engineSize'].unique())
car_engine_size.insert(0, "")


current_year = datetime.datetime.now().year

c_model = st.selectbox("Choose the car's model", car_model)
c_year = st.slider("Choose the year of manufacturer", 1980, current_year, current_year)
c_transmission = st.selectbox("Choose the type of transmission",car_transmission)
c_mileage = st.text_input("Input the number of miles the car has traveled")
c_fuel_type= st.selectbox("Choose the type of fuel",car_fuel_type)
c_tax = st.text_input("Input car's tax")
c_price = st.text_input("Input car's price")
c_engine_size = st.selectbox("Choose the engine size",car_engine_size)


if c_transmission == "":
    c_transmission ="Automatic"
if c_mileage == "":
    c_mileage =X_var_of_mpg_test['mileage'].mean()
if c_fuel_type == "":
    c_fuel_type ="Diesel"
if c_tax == "":
    c_tax =X_var_of_mpg_test['tax'].mean()
if c_price == "":
    c_price =X_var_of_mpg_test['price'].mean()
if c_engine_size == "":
    c_engine_size ="1.0"


if st.button('Predict'):
    
    dataFrame = pd.DataFrame({'model': c_model, 'year': c_year, 'transmission': c_transmission, 'mileage': c_mileage, 'fuelType': c_fuel_type, 'tax': c_tax, 'price': c_price, 'engineSize': c_engine_size}, index=[0])
    pricePrediction = model_mpg.predict(dataFrame)
    price = str(np.round(pricePrediction[0],2))
    col1 = st.columns(1)
    col1[0].metric("Car's Fuel Consumption prediction (Mile per Gallon)", price)
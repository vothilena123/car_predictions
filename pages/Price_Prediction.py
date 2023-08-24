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
from bokeh.plotting import figure
from bokeh.models import HoverTool

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


st.title('Ford Used Car Price Prediction')
car = pd.read_csv("ford.csv")
car_20 = car.head(20)
rf_model=joblib.load(os.path.join('ImportanceRFModel.pkl'))
X=joblib.load(os.path.join('XImportance.pkl'))
modelPrice=joblib.load(os.path.join('PriceLinearRegression.pkl'))
X_test=joblib.load(os.path.join('TestVariablesForPrice.pkl'))
y_test=joblib.load(os.path.join('TestPrice.pkl'))
X_model = car.drop(columns='price')
y_model = car['price']

st.write('')
st.write('')

count = 0

if st.checkbox("Show the Relationships between Car Price and Input Variables", False):
    with st.expander("Chart show Importance of the input variables"):
        # Get the importance of the input variables
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Create DataFrame from input data
        df = pd.DataFrame({
            'Input variable': X.columns[indices],
            'Importance': importances[indices]
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


    with st.expander("Relationship between Car Price and Car Model"):
        if st.checkbox("Show Histogram of Model and Mean Price", key=count+1):
            # Create a DataFrame with the average value of each model
            mean_prices_model = car.groupby('model')['price'].mean().reset_index()

            # Add location column to DataFrame
            mean_prices_model['position'] = abs(mean_prices_model['price'] - mean_prices_model['price'].max())

            # Sort DataFrame by position descending and value ascending
            mean_prices_model = mean_prices_model.sort_values(by=['position', 'price'], ascending=[True, False])

            # Draw a chart and change the color of the columns
            fig = px.histogram(mean_prices_model, x='price', y='model', orientation='h', color_discrete_sequence=['#00b3e6', '#91e6c7', '#e6c791', '#e66300'])

            # Add hover text to display the corresponding value when hovering the mouse on each column
            fig.update_traces(hovertemplate='<b>Model:</b> %{y} <br> <b>Mean price:</b> %{x:.2f}')

            # Set the axis name and title for the chart
            fig.update_layout(xaxis_title='Mean price', yaxis_title='Model', title='Mean Prices by Model')

            st.plotly_chart(fig)
           

        if st.checkbox("Show the relationship by using ANOVA", key=count + 2):
            grouped_model_price = [car[car['model'] == model]['price'] for model in car['model'].unique()]
            f_statistic, p_value = f_oneway(*grouped_model_price)
            col1, col2= st.columns(2)
            col1.metric("F-statistic",f_statistic)
            col2.metric("P-value:", p_value)
           

        if st.checkbox("Show Price range that a Model can have by scatterplot", key=count + 3):
            fig = px.scatter(car, x='price', y='model', color='model', hover_data=['price'])

            fig.update_layout(
                title='Show Price range that a Model can have',
                xaxis_title='Price',
                yaxis_title='Car Model'
            )

            st.plotly_chart(fig)


    with st.expander("Show the Relationship between Car Price and Transmission"):
        if st.checkbox("Show Histogram of Transmission and Mean Price", key=count + 4):
            
            mean_prices_transmission = car.groupby('transmission')['price'].mean().reset_index()

            fig = px.bar(mean_prices_transmission, x='transmission', y='price', color='price', color_continuous_scale='Peach', labels={'price':'Mean Prices'})

            fig.update_traces(marker_line_color='white', marker_line_width=1, hovertemplate='<b>%{x}</b><br>Mean Price: %{y:.2f}')

            fig.update_layout(title='Mean Prices by Transmission Type', xaxis_title='Transmission Type', yaxis_title='Mean Prices')

            st.plotly_chart(fig, use_container_width=True)

        if st.checkbox("Show the relationship by using ANOVA", key=count + 5):
            grouped_transmission_price = [car[car['transmission'] == model]['price'] for model in car['transmission'].unique()]
            f_statistic, p_value = f_oneway(*grouped_transmission_price)
            col1, col2= st.columns(2)
            col1.metric("F-statistic",f_statistic)
            col2.metric("P-value:", p_value)

        if st.checkbox("Show the relationship by violinplot", key=count + 6):
            fig = px.violin(car, x='price', y='transmission', color='transmission', hover_data=['price'])

            fig.update_layout(
                title='Show Price range that a Transmission can have ',
                xaxis_title='Price',
                yaxis_title='Car Transmission'
            )

            st.plotly_chart(fig)

    with st.expander("Show the Relationship between Car Price and Fuel Type"):
        if st.checkbox("Show Histogram of Fuel Type and Mean Price", key=count + 7):
            mean_prices_fuelType = car.groupby('fuelType')['price'].mean().reset_index()
            fig = px.bar(mean_prices_fuelType, x='fuelType', y='price', color='price', color_continuous_scale='Blues', labels={'price':'Mean Prices'})
            fig.update_traces(marker_line_color='white', marker_line_width=1, hovertemplate='<b>%{x}</b><br>Mean Price: %{y:.2f}')

            fig.update_layout(title='Mean Prices by Fuel Type', xaxis_title='Transmission Type', yaxis_title='Mean Prices')

            st.plotly_chart(fig, use_container_width=True)

        if st.checkbox("Show the relationship using ANOVA", key=count + 8):
            grouped_fuelType_price = [car[car['fuelType'] == model]['price'] for model in car['fuelType'].unique()]
            f_statistic, p_value = f_oneway(*grouped_fuelType_price)
            col1, col2= st.columns(2)
            col1.metric("F-statistic",f_statistic)
            col2.metric("P-value:", p_value)

        if st.checkbox("Show the relationship by boxplot", key=count + 9):

            fig = px.box(car, x='fuelType', y='price', color='fuelType', hover_data=['price'])

            fig.update_layout(
                title='Show Price range that a Fuel Type can have',
                xaxis_title='Car Fuel Type',
                yaxis_title='Price'
            )

            st.plotly_chart(fig)
        

    with st.expander("Heat Map show the Relationship between Price and Numeric Input Variables"):
        
        # Get a dataframe containing only numeric columns
        numeric_cols = car.select_dtypes(include=['int64', 'float64']).columns
        car_numeric = car[numeric_cols]

        # Draw heatmap
        fig = go.Figure(data=go.Heatmap(
                z=car_numeric.corr(),
                x=car_numeric.columns,
                y=car_numeric.columns,
                colorscale = 'Blues'
        ))

        fig.update_layout(
            title='Correlation Heatmap',
            xaxis_title='Features',
            yaxis_title='Features'
        )

        st.plotly_chart(fig)


if st.checkbox("Performance coefficients for model accuracy", False):
    y_pred = modelPrice.predict(X_test)
    r2Price = r2_score(y_test, y_pred)
    maePrice = mean_absolute_error(y_test, y_pred)
       
    col1, col2= st.columns(2)
    col1.metric("Coefficient of Determination (R2 Score)",r2Price)
    col2.metric("Mean absolute error:", maePrice)

if st.checkbox("Compare real and predicted car prices chart", False):
    y_pred = modelPrice.predict(X_test)
    year_test = X_test['year']
    compare = pd.DataFrame({'y':y_test, 'y_predict':y_pred, 'year_test': year_test}).head(100)
    compare_sorted_by_year = compare.sort_values('year_test')
    

    fig = go.Figure()
    # Add trace representing data points
    fig.add_trace(go.Scatter(
        x=compare_sorted_by_year['year_test'],
        y=compare_sorted_by_year['y'],
        mode='markers',
        name='Price Test',
        marker=dict(size=8, color='violet', opacity=0.8)
    ))
    fig.add_trace(go.Scatter(
        x=compare_sorted_by_year['year_test'],
        y=compare_sorted_by_year['y_predict'],
        mode='markers',
        name='Price Predict',
        marker=dict(size=8, color='orange', opacity=0.8)
    ))

    # Add a trace showing the line connecting the points y and y_predict
    fig.add_trace(go.Scatter(
        x=compare_sorted_by_year['year_test'],
        y=compare_sorted_by_year['y'],
        mode='lines',
        name='Price Test Line',
        line=dict(color='blue', width=2, dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=compare_sorted_by_year['year_test'],
        y=compare_sorted_by_year['y_predict'],
        mode='lines',
        name='Price Predict Line',
        line=dict(color='orange', width=2)
    ))

    # Set title and parameters on axis
    fig.update_layout(
        title='Comparison of Price Test and Price Predict',
        xaxis_title='Price Test',
        yaxis_title='Price'
    )

    st.plotly_chart(fig)
    

st.write('')
st.write('')

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
c_mpg = st.text_input("Input fuel consumption (Mile per Gallon)")
c_engine_size = st.selectbox("Choose the engine size",car_engine_size)


if c_transmission == "":
    c_transmission ="Automatic"
if c_mileage == "":
    c_mileage =X_test['mileage'].mean()
if c_fuel_type == "":
    c_fuel_type ="Diesel"
if c_tax == "":
    c_tax =X_test['tax'].mean()
if c_mpg == "":
    c_mpg =X_test['mpg'].mean()
if c_engine_size == "":
    c_engine_size ="1.0"


if st.button('Predict'):
    
    dataFrame = pd.DataFrame({'model': c_model, 'year': c_year, 'transmission': c_transmission, 'mileage': c_mileage, 'fuelType': c_fuel_type, 'tax': c_tax, 'mpg': c_mpg, 'engineSize': c_engine_size}, index=[0])
    pricePrediction = modelPrice.predict(dataFrame)
    price = str(np.round(pricePrediction[0],2))
    col1 = st.columns(1)
    col1[0].metric("Car's Price prediction ($)", price)


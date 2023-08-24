import streamlit as st
import pandas as pd
from scipy.stats import f_oneway
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import seaborn as sns
import joblib
import os
import datetime
import numpy as np
from numerize.numerize import numerize
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests



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

lottie_animation_1 = "https://assets6.lottiefiles.com/packages/lf20_MTmNU2.json"

lottie_anime_json = load_lottie_url(lottie_animation_1)

st_lottie(lottie_anime_json, key = "car", width=300, height=300)

st.title("Car's Attributes Prediction base on Price")
car = pd.read_csv("ford.csv")



if st.checkbox("Heat Map show the Correlation coefficients of Car Price and all Attribute", False):
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


X_test_price = joblib.load(os.path.join('XTestPrice.pkl'))

model_year = joblib.load(os.path.join('YearGradientBoostingRegressor.pkl'))
y_test_year = joblib.load(os.path.join('yTestYear.pkl'))

model_mileage = joblib.load(os.path.join('MileageGradientBoostingRegressor.pkl'))
y_test_mileage = joblib.load(os.path.join('yTestMileage.pkl'))

model_tax = joblib.load(os.path.join('TaxGradientBoostingRegressor.pkl'))
y_test_tax = joblib.load(os.path.join('yTestTax.pkl'))

model_mpg = joblib.load(os.path.join('MPGGradientBoostingRegressor.pkl'))
y_test_mpg = joblib.load(os.path.join('yTestMPG.pkl'))

model_engineSize = joblib.load(os.path.join('EngineSizeGradientBoostingRegressor.pkl'))
y_test_engineSize = joblib.load(os.path.join('yTestengineSize.pkl'))


y_year_pred = model_year.predict(X_test_price) # Prediction on test set
yearMse = mean_squared_error(y_test_year, y_year_pred) # Calculate mean squared error
yearRmse = np.sqrt(yearMse)

y_mileage_pred = model_mileage.predict(X_test_price) # Prediction on test set
mileageMse = mean_squared_error(y_test_mileage, y_mileage_pred) # Calculate mean squared error
mileageRmse = np.sqrt(mileageMse)

y_tax_pred = model_tax.predict(X_test_price) # Prediction on test set
taxMse = mean_squared_error(y_test_tax, y_tax_pred) # Calculate mean squared error
taxRmse = np.sqrt(taxMse)

y_mpg_pred = model_mpg.predict(X_test_price) # Prediction on test set
mpgMse = mean_squared_error(y_test_mpg, y_mpg_pred) # Calculate mean squared error
mpgRmse = np.sqrt(mpgMse)


y_engineSize_pred = model_engineSize.predict(X_test_price) # Prediction on test set
engineMse = mean_squared_error(y_test_engineSize, y_engineSize_pred) # Calculate mean squared error
engineRmse = np.sqrt(engineMse)

if st.checkbox('Root mean squared error bewteen price and all attributes'):
    
    total1,total2,total3,total4,total5 = st.columns(5,gap='large')

    with total1:
        st.image('images/calendar.webp', width=100,use_column_width='Auto')
        st.metric(label = 'RMSE of Year', value= numerize(yearRmse))
        
    with total2:
        st.image('images/mileage.png', width=100,use_column_width='Auto')
        st.metric(label='RMSE of Mileage', value=numerize(mileageRmse))

    with total3:
        st.image('images/tax.webp', width=100,use_column_width='Auto')
        st.metric(label= 'RMSE of Tax',value=numerize(taxRmse))

    with total4:
        st.image('images/R.jfif', width=100,use_column_width='Auto')
        st.metric(label='RMSE of MPG',value=numerize(mpgRmse))

    with total5:
        st.image('images/engine.gif', width=100,use_column_width='Auto')
        st.metric(label='RMSE of Engine Size',value=numerize(engineRmse))


yearR2 = r2_score(y_test_year,y_year_pred)
mileageR2 = r2_score(y_test_mileage,y_mileage_pred)
taxR2 = r2_score(y_test_tax,y_tax_pred)
mpgR2 = r2_score(y_test_mpg,y_mpg_pred)
engineR2 = r2_score(y_test_engineSize,y_engineSize_pred)

if st.checkbox('Coefficient of Determination between price and all attributes'):
    
    total1,total2,total3,total4,total5 = st.columns(5,gap='large')

    with total1:
        st.image('images/calendar.webp', width=100,use_column_width='Auto')
        st.metric(label = 'R2 of Year', value= numerize(yearR2))
        
    with total2:
        st.image('images/mileage.png', width=100,use_column_width='Auto')
        st.metric(label='R2 of Mileage', value=numerize(mileageR2))

    with total3:
        st.image('images/tax.webp', width=100,use_column_width='Auto')
        st.metric(label= 'R2 of Tax',value=numerize(taxR2))

    with total4:
        st.image('images/R.jfif', width=100,use_column_width='Auto')
        st.metric(label='R2 of MPG',value=numerize(mpgR2))

    with total5:
        st.image('images/engine.gif', width=100,use_column_width='Auto')
        st.metric(label='R2 of Engine Size',value=numerize(engineR2))


X_test_price_1d = X_test_price.flatten()


if st.checkbox('Charts show the real and prediction value of all attributes'):

    with st.expander("Charts show the real and prediction value of Year"):
        compare_year = pd.DataFrame({'year_test':y_test_year, 'year_predict':y_year_pred, 'price_test': X_test_price_1d}).head(100)
        compare_year_sorted_by_price = compare_year.sort_values('price_test')
        fig = go.Figure()
        # Add trace representing data points
        fig.add_trace(go.Scatter(
            x=compare_year_sorted_by_price['price_test'],
            y=compare_year_sorted_by_price['year_test'],
            mode='markers',
            name='Year Test',
            marker=dict(size=8, color='violet', opacity=0.8)
        ))
        fig.add_trace(go.Scatter(
            x=compare_year_sorted_by_price['price_test'],
            y=compare_year_sorted_by_price['year_predict'],
            mode='markers',
            name='Year Predict',
            marker=dict(size=8, color='gold', opacity=0.8)
        ))

        # Add trace showing the line connecting the points year_test and year_predict
        fig.add_trace(go.Scatter(
            x=compare_year_sorted_by_price['price_test'],
            y=compare_year_sorted_by_price['year_test'],
            mode='lines',
            name='Year Test Line',
            line=dict(color='blue', width=2, dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=compare_year_sorted_by_price['price_test'],
            y=compare_year_sorted_by_price['year_predict'],
            mode='lines',
            name='Year Predict Line',
            line=dict(color='orange', width=2)
        ))

        # Set title and parameters on axis
        fig.update_layout(
            title='Comparison of Year Test and Year Predict',
            xaxis_title='Price Test',
            yaxis_title='Year'
        )

        st.plotly_chart(fig)

    with st.expander("Charts show the real and prediction value of Mileage"):
        compare_mileage = pd.DataFrame({'mileage_test':y_test_mileage, 'mileage_predict':y_mileage_pred, 'price_test': X_test_price_1d}).head(100)
        compare_mileage_sorted_by_price = compare_mileage.sort_values('price_test')
        fig = go.Figure()
        # Add trace representing data points
        fig.add_trace(go.Scatter(
            x=compare_mileage_sorted_by_price['price_test'],
            y=compare_mileage_sorted_by_price['mileage_test'],
            mode='markers',
            name='Mileage Test',
            marker=dict(size=8, color='violet', opacity=0.8)
        ))
        fig.add_trace(go.Scatter(
            x=compare_mileage_sorted_by_price['price_test'],
            y=compare_mileage_sorted_by_price['mileage_predict'],
            mode='markers',
            name='Mileage Predict',
            marker=dict(size=8, color='gold', opacity=0.8)
        ))

        # Add a trace showing the line connecting the mileage_test and mileage_predict points
        fig.add_trace(go.Scatter(
            x=compare_mileage_sorted_by_price['price_test'],
            y=compare_mileage_sorted_by_price['mileage_test'],
            mode='lines',
            name='Mileage Test Line',
            line=dict(color='pink', width=2, dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=compare_mileage_sorted_by_price['price_test'],
            y=compare_mileage_sorted_by_price['mileage_predict'],
            mode='lines',
            name='Mileage Predict Line',
            line=dict(color='orange', width=2)
        ))

        # Set title and parameters on axis
        fig.update_layout(
            title='Comparison of mileage Test and Mileage Predict',
            xaxis_title='Price Test',
            yaxis_title='Mileage'
        )

        st.plotly_chart(fig)
    
    
    with st.expander("Charts show the real and prediction value of Tax"):

        compare_tax = pd.DataFrame({'tax_test':y_test_tax, 'tax_predict':y_tax_pred, 'price_test': X_test_price_1d}).head(100)
        compare_tax_sorted_by_price = compare_tax.sort_values('price_test')
        fig = go.Figure()
        # Add trace representing data points
        fig.add_trace(go.Scatter(
            x=compare_tax_sorted_by_price['price_test'],
            y=compare_tax_sorted_by_price['tax_test'],
            mode='markers',
            name='Tax Test',
            marker=dict(size=8, color='violet', opacity=0.8)
        ))
        fig.add_trace(go.Scatter(
            x=compare_tax_sorted_by_price['price_test'],
            y=compare_tax_sorted_by_price['tax_predict'],
            mode='markers',
            name='Tax Predict',
            marker=dict(size=8, color='orange', opacity=0.8)
        ))

        # Add a trace showing the line connecting the points tax_test and tax_predict
        fig.add_trace(go.Scatter(
            x=compare_tax_sorted_by_price['price_test'],
            y=compare_tax_sorted_by_price['tax_test'],
            mode='lines',
            name='Tax Test Line',
            line=dict(color='violet', width=2, dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=compare_tax_sorted_by_price['price_test'],
            y=compare_tax_sorted_by_price['tax_predict'],
            mode='lines',
            name='Tax Predict Line',
            line=dict(color='gold', width=2)
        ))

        # Set title and parameters on axis
        fig.update_layout(
            title='Comparison of Tax Test and Tax Predict',
            xaxis_title='Price Test',
            yaxis_title='Tax'
        )

        st.plotly_chart(fig)


    with st.expander("Charts show the real and prediction value of MPG"):
        compare_mpg = pd.DataFrame({'mpg_test':y_test_mpg, 'mpg_predict':y_mpg_pred, 'price_test': X_test_price_1d}).head(100)
        compare_mpg_sorted_by_price = compare_mpg.sort_values('price_test')
        fig = go.Figure()
        # Add trace representing data points
        fig.add_trace(go.Scatter(
            x=compare_mpg_sorted_by_price['price_test'],
            y=compare_mpg_sorted_by_price['mpg_test'],
            mode='markers',
            name='MPG Test',
            marker=dict(size=8, color='violet', opacity=0.8)
        ))
        fig.add_trace(go.Scatter(
            x=compare_mpg_sorted_by_price['price_test'],
            y=compare_mpg_sorted_by_price['mpg_predict'],
            mode='markers',
            name='MPG Predict',
            marker=dict(size=8, color='orange', opacity=0.8)
        ))

        # Added trace showing the line connecting the points mpg_test and mpg_predict
        fig.add_trace(go.Scatter(
            x=compare_mpg_sorted_by_price['price_test'],
            y=compare_mpg_sorted_by_price['mpg_test'],
            mode='lines',
            name='MPG Test Line',
            line=dict(color='blue', width=2, dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=compare_mpg_sorted_by_price['price_test'],
            y=compare_mpg_sorted_by_price['mpg_predict'],
            mode='lines',
            name='MPG Predict Line',
            line=dict(color='green', width=2)
        ))

        # Set title and parameters on axis
        fig.update_layout(
            title='Comparison of MPG Test and MPG Predict',
            xaxis_title='Price Test',
            yaxis_title='MPG'
        )

        st.plotly_chart(fig)

    with st.expander("Charts show the real and prediction value of Engine Size"):

        compare_engineSize = pd.DataFrame({'engineSize_test':y_test_engineSize, 'engineSize_predict':y_engineSize_pred, 'price_test': X_test_price_1d}).head(100)
        compare_engineSize_sorted_by_price = compare_engineSize.sort_values('price_test')
        fig = go.Figure()
        # Add trace representing data points
        fig.add_trace(go.Scatter(
            x=compare_engineSize_sorted_by_price['price_test'],
            y=compare_engineSize_sorted_by_price['engineSize_test'],
            mode='markers',
            name='Engine Size Test',
            marker=dict(size=8, color='violet', opacity=0.8)
        ))
        fig.add_trace(go.Scatter(
            x=compare_engineSize_sorted_by_price['price_test'],
            y=compare_engineSize_sorted_by_price['engineSize_predict'],
            mode='markers',
            name='Engine Size Predict',
            marker=dict(size=8, color='orange', opacity=0.8)
        ))

        # Added trace showing the line connecting the points engineSize_test and engineSize_predict
        fig.add_trace(go.Scatter(
            x=compare_engineSize_sorted_by_price['price_test'],
            y=compare_engineSize_sorted_by_price['engineSize_test'],
            mode='lines',
            name='Engine Size Test Line',
            line=dict(color='green', width=2, dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=compare_engineSize_sorted_by_price['price_test'],
            y=compare_engineSize_sorted_by_price['engineSize_predict'],
            mode='lines',
            name='Engine Size Predict Line',
            line=dict(color='orange', width=2)
        ))

        # Set title and parameters on axis
        fig.update_layout(
            title='Comparison of Engine Size Test and Engine Size Predict',
            xaxis_title='Price Test',
            yaxis_title='Engine Size'
        )

        st.plotly_chart(fig)


c_price = st.text_input("Input The Price Of Car ($)")

if st.button('Predict'):

    price_input = pd.DataFrame({'price': c_price}, index=[0])
        
    y_year_predict = model_year.predict(price_input) 
    y_mileage_predict = model_mileage.predict(price_input) 
    y_tax_predict = model_tax.predict(price_input) 
    y_mpg_predict = model_mpg.predict(price_input) 
    y_engineSize_predict = model_engineSize.predict(price_input) 

    year = str(np.round(y_year_predict[0],2))
    mileage = str(np.round(y_mileage_predict[0],2))
    tax = str(np.round(y_tax_predict[0],2))
    mpg = str(np.round(y_mpg_predict[0],2))
    engineSize = str(np.round(y_engineSize_predict[0],2))


    col1, col2, col3 = st.columns(3)
    col1.metric(" Year Prediction", year, "year")
    col2.metric("Mileage Prediction", mileage, "mileage")
    col3.metric("Tax Prediction", tax, "$")

    col1, col2= st.columns(2)
    col1.metric("MPG Prediction", mpg, "mpg")
    col2.metric("Engine Size", engineSize, "cc")





   

import streamlit as st
import pandas as pd
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

lottie_animation_1 = "https://assets8.lottiefiles.com/packages/lf20_iilTGisiuG.json"

lottie_anime_json = load_lottie_url(lottie_animation_1)

st_lottie(lottie_anime_json, key = "car", width=300, height=300)
    
st.title('Ford Used Car Dataset')

car = pd.read_csv("ford.csv")


if st.checkbox("Show 50 Head Row", False):
    st.dataframe(car.head(50), width=1000)

if st.checkbox("Show 50 Tail Row", False):
    st.dataframe(car.tail(50), width=1000)

st.dataframe(car, width=1000, height=700)


import streamlit as st
import pandas as pd
import plost

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('Dashboard `version 1`')


st.sidebar.markdown('''
---
Created with ❤️ by Vo Thi Le Na.
''')


video_html = """
		<style>

		#myVideo {
		  position: fixed;
		  right: 0;
		  bottom: 0;
		  min-width: 100%; 
		  min-height: 100%;
		}

		.content {
		  position: fixed;
		  bottom: 0;
		  background: rgba(0, 0, 0, 0.5);
		  color: #f1f1f1;
		  width: 100%;
		  padding: 20px;
		}

		</style>	
		<video autoplay muted loop id="myVideo">
		  <source src="https://static.streamlit.io/examples/star.mp4")>
		  Your browser does not support HTML5 video.
		</video>
        """

st.markdown(video_html, unsafe_allow_html=True)

st.markdown(
    "<div style='color: white; font-size: 40px;'>Welcome to Ford Used Car Prediction</div>",
    unsafe_allow_html=True
)



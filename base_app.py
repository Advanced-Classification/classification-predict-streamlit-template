"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
from ctypes import alignment

from scipy.misc import central_diff_weights
import streamlit as st
import numpy as np
import altair as alt
import time
import hydralit_components as hc
import matplotlib.pyplot as plt
import joblib,os

# Data dependencies
import pandas as pd
from PIL import Image
# Vectorizer
news_vectorizer = open("resources/Vectorizer2.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app

def main():

	prediction =[]
	models= ['Logistic Regression', 'Random Forest', 'Support Vector', 'Ada Boost',
			 'K-Neighbors']
	results= {'-1': 'Anti' , '0': 'Neutral', '1': 'Pro' , '2' : 'News'}	 
	menu_data = [
        {'id':'predict','icon':"🐙",'label':"Predict"},
		{'id':'rawData', 'icon': "far fa-clone", 'label':"Raw Data"},
        {'id':'visualize', 'icon': "far fa-chart-bar", 'label':"Visualize"},#no tooltip message
		{'id': 'info' , 'icon': "far fa-copy", 'label':"Info"},
		{'id': 'aboutUs',  'icon': "far fa-address-book ", 'label':"About Us"},
       
]
	over_theme = {'txc_inactive': '#FFFFFF'}
	menu_id = hc.nav_bar(menu_definition=menu_data,home_name='Home',override_theme=over_theme)


	if menu_id == 'predict':
		
			# Creating a text box for user input
		st.image("resources/imgs/tweet1.jpg", width =400)
		m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(255, 49, 49);
}
</style>""", unsafe_allow_html=True)

		s = st.markdown("""
<style>
div.stText_Area > button:first-child {
    background-color: rgb(255, 49, 49);
}
</style>""", unsafe_allow_html=True)
				
				
		tweet_text = st.text_area("Enter text you would like to classify",key = 1)
		
		model_choice= st.selectbox("Choose a model", models)

		if model_choice == 'Logistic Regression':
			if st.button("Classify"):
				with hc.HyLoader(f'Classifying  with {model_choice}..',hc.Loaders.standard_loaders,index=[2,2]):
					time.sleep(3)
				# Transforming user input with vectorizer
					vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
					predictor = joblib.load(open(os.path.join("resources/lr_regression_base_main.pkl"),"rb"))
			
					prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
		
					st.metric("Text Categorized as: " , prediction)
					
		if model_choice == 'Random Forest':
			# Creating a text box for user input
			if st.button("Classify"):
				with hc.HyLoader(f'Classifying  with {model_choice}..', hc.Loaders.standard_loaders,index=[2,2]):
					time.sleep(3)
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
		
				predictor = joblib.load(open(os.path.join("resources/rf_classifier_base.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				st.metric("Text Categorized as: " , prediction)

		if model_choice == 'Support Vector':
	
			if st.button("Classify"):
				with hc.HyLoader('Classifying with Support Vector',hc.Loaders.standard_loaders,index=[2,2]):
					time.sleep(3)
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				predictor = joblib.load(open(os.path.join("resources/Support_Vector.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				st.metric("Text Categorized as: " , prediction)

		if model_choice == 'Ada Boost':

			if st.button("Classify"):
				with hc.HyLoader('Classifying with Ada Boost..',hc.Loaders.standard_loaders,index=[2,2]):
					time.sleep(3)
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				predictor = joblib.load(open(os.path.join("resources/AdaBoost.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				st.metric("Text Categorized as: " , prediction)
		
		if model_choice == 'K-Neighbors':
	
			if st.button("Classify"):
				with hc.HyLoader('Classifying with K-Neighbors..',hc.Loaders.standard_loaders,index=[2,2]):
					time.sleep(3)
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				predictor = joblib.load(open(os.path.join("resources/K-Neighbors.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				st.metric("Text Categorized as: " , prediction)


	if menu_id == 'rawData':
		st.markdown("<h2 style='text-align: center;'>Tweets Dataframe</h2>",
			unsafe_allow_html=True)
		st.write(raw[['sentiment', 'message']])
	if menu_id == 'visualize':
		labels = 'Anti', 'Neutral', 'Pro', 'News'
		sizes = [12, 25, 85, 35]
		explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

		fig1, ax1 = plt.subplots()
		ax1.pie(sizes, explode=explode, labels=labels, autopct='%0.1f%%',
				shadow=True, startangle=90)
		ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

		st.pyplot(fig1)
		with st.expander("See explanation"):
			st.write("""
				This pie chart shows the distribution of all the sentiments towards 
				climate change being man made or not.
			""")
	# if menu_id == 'Home':
	# 	st.markdown("<h2 style='text-align: center;'>Climate Change Tweet Classifier</h2>",
	# 	 unsafe_allow_html=True)
	# 	st.image("resources/imgs/Advanced Classification 2022 EDSA.jpg" ,  width = 650 )
	if menu_id == 'aboutUs':
		st.image("resources/imgs/team.jpg")
		st.markdown("<h2 style='text-align: center;'>Who are we?</h2>",
			unsafe_allow_html=True)
		st.write("We are a group of Data Scientists "
		+"who are committed to bringing change to the world through Data Analytics. "
		+"The team is comprised of")
		st.write("- Siyanda Madlopha")
		st.write("- Theo Sdinani")
		st.write("- Angela Morris")
		st.write("- Marcus Mohlake")
		st.write("- Morglin Olivier")
		st.write("Website: [Data Science Magicians](https://github.com/Advanced-Classification)")
		st.markdown("<h2 style='text-align: center;'>Contact Us</h2>",
		 unsafe_allow_html=True)
		st.write("Email : data.analytics@magicians.co.za")
		st.write("Telephone : 011 345 7787")
	if menu_id=='info':
		st.markdown("<h2 style='text-align: center;'>Sentiment catagories explained.</h2>",
			unsafe_allow_html=True)
		st.write('- `1 Pro:` the tweet supports the belief of man-made climate change')
		st.write('- `2 News:` the tweet links to factual news about climate change')
		st.write('- `0 Neutral:` the tweet neither supports nor refutes the belief of man-made climate change')
		st.write('- `-1 Anti:` the tweet does not believe in man-made climate change')
	if menu_id == 'Home':
		st.markdown("<h2 style='text-align: center;'>Climate Change Tweet Classifier</h2>",
		 unsafe_allow_html=True)
		st.image("resources/imgs/Advanced Classification 2022 EDSA.jpg" ,  width = 650 )		

		

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()

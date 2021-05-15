"""
Description
This is a NLP proof of concept app to showcase models. 
"""

# Core Pkgs

import os
from pathlib import Path 
import json

import requests
import pandas as pd
import numpy as np # linear algebra

import streamlit as st 

from utils import radar_chart_plot, bar_chart_plot
import backend

st.set_option('deprecation.showPyplotGlobalUse', False)

# load demo configs
demo_config_file = open('demo_config.json')
demos = json.load(demo_config_file)
demo_config_file.close()

# Lowest score value to consider
MIN_SCORE_THRESHOLD = 0.6

# Title
st.sidebar.subheader("Text classification labs for FSDL")
st.sidebar.markdown("This is app showcases experiments on text classification.")

selected_demo = st.sidebar.selectbox("Select demo", list(demos.keys()), 0)

def main():
	
	if(selected_demo=="DBpedia14"):
		st.header(demos[selected_demo]["title"])

		st.sidebar.write(demos[selected_demo]["description"])	

		selected_input_sample = st.selectbox("Predefined examples", demos[selected_demo]["samples"])
		input_text = st.text_area("Type your message here.", selected_input_sample, height=250)
		selected_model = st.selectbox("Select model", demos[selected_demo]["available_models"], 0)
		
		if st.button("Classify text"):
			st.subheader("Results")
			if(selected_model == "Zero-shot classification" ):

				loading = st.info(f"Running prediction request ...")
				prediction_result = backend.remote_zeroshot_inference_request(input_text, demos[selected_demo]["labels"], multi_class=False)
				loading.empty()

				# drawing bar chart
				bar_chart_plot(prediction_result)
				st.pyplot()				

				with st.beta_expander("See detailed scores", expanded=True):
					st.write(prediction_result)

			else:
				loading = st.info(f"Running prediction request ...")
				prediction_result = backend.remote_inference_request(input_text, selected_model)
				loading.empty()
		
				st.markdown(f"**{prediction_result['prediction'][0]['class']}** is the most likely category.")
				with st.beta_expander("See detailed scores", expanded=True):
					st.write(prediction_result)		
			
				backend.import_to_labelstudio(input_text, 
										demos[selected_demo]["labelstudio_project_id"], 
										predicted_labels = [prediction_result["prediction"][0]["class"]], 
										predicted_scores = prediction_result["prediction"][0]["confidence"],
										model_name=selected_model)
				
		
	elif selected_demo == "Toxicity":

		st.header(demos[selected_demo]["title"])
		st.sidebar.write(demos[selected_demo]["description"])	

		selected_input_sample = st.selectbox("Select example", demos[selected_demo]["samples"]	, index=0)
		input_text = st.text_area("Or edit here your comment",value=selected_input_sample)

		selected_model = st.selectbox("Select model", demos[selected_demo]["available_models"], 0)
		
		if st.button("Classify comment"):
			st.subheader("Results")

			if(selected_model == "Zero-shot classification" ):

				loading = st.info(f"Running prediction request ...")
				prediction_result = backend.remote_zeroshot_inference_request(input_text, demos[selected_demo]["labels"], multi_class=True)
				loading.empty()

				predicted_scores = np.array(prediction_result["scores"])
				predicted_labels = prediction_result["labels"]

				if(len(predicted_scores[predicted_scores >= MIN_SCORE_THRESHOLD]) == 0):
					
					st.markdown(f"Model didn't detect any toxic content :)")
				else:
					st.markdown(f"Model detected potential toxic content.")

					df = pd.DataFrame.from_dict(dict(scores = predicted_scores, 
													labels = predicted_labels))\
													.sort_values(by="scores", axis=0, ascending=False)

					st.write(radar_chart_plot(df))

					with st.beta_expander("See detailed scores", expanded=True):
						st.write(prediction_result)					
			else:
				
				prediction_result = backend.remote_inference_request(input_text, selected_model)

				predicted_scores = prediction_result["prediction"][0]["confidence"]
				predicted_labels = prediction_result["prediction"][0]["class"] 

				if(len(predicted_labels) == 0):
					st.markdown(f"Model didn't detect any toxic content :)")
				else:
					st.markdown(f"Model detected potential toxic content.")

					result_table = {}
					for label in demos[selected_demo]["labels"]:
						result_table[label] = 0

					for idx, predicted_label in enumerate(predicted_labels):
						result_table[predicted_label] = predicted_scores[idx]

					
					df = pd.DataFrame.from_dict(dict(scores = result_table.values(), 
													labels = result_table.keys()))\
													.sort_values(by="scores", axis=0, ascending=False)
					
					st.write(radar_chart_plot(df))

					with st.beta_expander("See detailed scores", expanded=True):
						st.write(result_table)

					backend.import_to_labelstudio(input_text, 
										demos[selected_demo]["labelstudio_project_id"], 
										predicted_labels = prediction_result["prediction"][0]["class"], 
										predicted_scores = prediction_result["prediction"][0]["confidence"],
										model_name=selected_model)

if __name__ == '__main__':
	main()
	

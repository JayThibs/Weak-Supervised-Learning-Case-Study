import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

def radar_chart_plot(df):  
    
    fig = px.line_polar(df, r='scores', theta='labels', line_close=True)
    
    return fig

def bar_chart_plot(prediction_result):
	plt.plot()

	height = prediction_result["scores"]
	bars = prediction_result["labels"]
	y_pos = np.arange(len(bars))
	
	# Create bars
	plt.bar(y_pos, height)
	
	# Create names on the x-axis
	plt.xticks(y_pos, bars, rotation='vertical')	
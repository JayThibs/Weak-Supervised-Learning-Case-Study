import json
import requests
import os

# make sure you specify .env
ML_INFERENCE_SERVER_ENDPOINT = os.environ['ML_INFERENCE_SERVER_ENDPOINT']
ML_ZEROSHOT_INFERENCE_SERVER_ENDPOINT = os.environ['ML_ZEROSHOT_INFERENCE_SERVER_ENDPOINT']
LABELSTUDIO_ENDPOINT = os.environ['LABELSTUDIO_ENDPOINT']
LABELSTUDIO_API_TOKEN = os.environ['LABELSTUDIO_API_TOKEN']

def remote_zeroshot_inference_request(input_text, candidate_labels, multi_class=False):
	
	# loading = st.info(f"Running prediction request ...")

	payload = {
		"inputs": [input_text],
		"parameters": {
			"candidate_labels": candidate_labels,
			"multi_class": multi_class
		}
	}
	
	print(payload)
	response = requests.request("POST", ML_ZEROSHOT_INFERENCE_SERVER_ENDPOINT, json=payload, 
					headers={"Content-Type": "application/json"})
                    
	return json.loads(response.text)

def remote_inference_request(input_text, model_name):
	payload = {
		"text": input_text,
		"model_name":model_name
		}

	headers = {"Content-Type": "application/json"}

	response = requests.request("POST", ML_INFERENCE_SERVER_ENDPOINT, json=payload, headers=headers)

	return json.loads(response.text)

def import_to_labelstudio(input_text, project_id, predicted_labels, predicted_scores, model_name="", multilabel=False):
	
	url = f"{LABELSTUDIO_ENDPOINT}/api/projects/{project_id}/tasks/bulk/"

	auth_token = f"Token {LABELSTUDIO_API_TOKEN}"

	payload = [
			{
			"data":{
				"text": input_text,
				"meta_info": {
					"model_name":model_name
					} 
			},		
			"annotations": [
			{
				
				"result": [
					{
						"from_name": "category",
						"to_name": "content",
						"type": "choices",
						'value': {'choices': predicted_labels}				
					}
				],
			}
			]
			}
		]
	
	print(payload)
	headers = {
		"Content-Type": "application/json",
		"Authorization": auth_token
	}

	response = requests.request("POST", url, json=payload, headers=headers)

	print(response.text)
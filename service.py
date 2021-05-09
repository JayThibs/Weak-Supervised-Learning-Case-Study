import argparse
import datetime
import json
import logging
import os
import sys
import time

import numpy as np
import torch
from flask import Flask, abort, jsonify, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)

device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")

tokenizer, network = None, None
service_name = None

num_classes_dict = {
    "dbpedia": 14,
    "toxic_comments": 6,
}
label_encoder_dict = {
    "dbpedia": [
        "Company",
        "EducationalInstitution",
        "Artist",
        "Athlete",
        "OfficeHolder",
        "MeanOfTransportation",
        "Building",
        "NaturalPlace",
        "Village",
        "Animal",
        "Plant",
        "Album",
        "Film",
        "WrittenWork",
    ],
    "toxic_comments": ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
}


def configure_logger(level=logging.INFO):
    logger_format = "[%(asctime)s][%(levelname)s]\t%(message)s"
    logging.basicConfig(level=level, format=logger_format, handlers=[logging.StreamHandler(sys.stdout)])


def process_prediction(logits):
    if service_name == "toxic_comments":
        Y_predict = (logits >= 0.5).float()
        Y_probas = torch.sigmoid(logits).cpu().data.numpy().tolist()[0]
    else:
        _, Y_predict = torch.max(logits, dim=1)
        Y_probas = torch.max(torch.softmax(logits, dim=1)).cpu().data.numpy().tolist()
    Y_predict = Y_predict.cpu().data.numpy().tolist()[0]
    return Y_predict, Y_probas


def prepare_for_inference(text):
    # No need to pad sequence since we only perform inference on 1 example
    encoded_dict = tokenizer(
        text,
        add_special_tokens=True,
        return_attention_mask=True,
        padding=False,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_ids = encoded_dict["input_ids"].to(device)
    mask = encoded_dict["attention_mask"].to(device)
    return input_ids, mask


@torch.no_grad()
def do_inference(input_ids, mask):
    outputs = network(input_ids, token_type_ids=None, attention_mask=mask)
    predicted_class, predicted_class_probas = process_prediction(outputs["logits"])
    # Converted predicted class int to string
    if service_name == "toxic_comments":
        mask = list(map(bool, predicted_class))
        predicted_class = np.array(label_encoder_dict[service_name])[mask].tolist()
        predicted_class_probas = np.array(predicted_class_probas)[mask].tolist()
    else:
        predicted_class = label_encoder_dict[service_name][predicted_class]
    return predicted_class, predicted_class_probas


def init(args):
    global tokenizer
    global network
    global service_name
    model_type = args.model_type
    model_name = args.model_name
    service_name = args.service_name
    num_classes = num_classes_dict[service_name]
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    network = AutoModelForSequenceClassification.from_pretrained(
        model_type,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    # Replace head manually before loading state dict
    network.classifier = torch.nn.Linear(network.config.hidden_size, num_classes)
    network.config.num_labels = num_classes
    network.num_labels = num_classes

    # Load state dict
    model_path = os.path.join(os.getcwd(), "models", model_name)
    network_state_dict = torch.load(model_path, map_location=device)
    network.load_state_dict(network_state_dict, strict=False)
    network = network.to(device)


@app.after_request
def apply_caching(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Max-Age"] = 1000
    # note that '*' is not valid for Access-Control-Allow-Headers
    response.headers["Access-Control-Allow-Headers"] = "origin, x-csrftoken, content-type, accept"
    return response


@app.route("/", methods=["POST"])
def service():
    if not request.is_json:
        abort(400, "Invalid request payload in JSON format")
    try:
        request_json = request.get_json()
        logging.info("Processing prediction request...")
        telemetry = {
            "request_start": str(datetime.datetime.now()),
            "content_length": len(json.dumps(request_json)),
            "model_inference": None,
            "service_name": service_name,
        }
        request_start_time = time.monotonic()
        # interpret incoming request based on API VERSION
        response = {"prediction": []}
        request_text = request_json.get("text", "")
        if len(request_text) != 0:
            logging.info("Running model inference ...")
            model_inference_start_time = time.monotonic()
            input_ids, mask = prepare_for_inference(request_text)
            predicted_class, predicted_probas = do_inference(input_ids, mask)
            telemetry["model_inference"] = time.monotonic() - model_inference_start_time
            response["prediction"].append({"class": predicted_class, "confidence": predicted_probas})
        telemetry["time_delta"] = time.monotonic() - request_start_time
        telemetry["request_end"] = str(datetime.datetime.now())
        return json.dumps(response).encode("utf-8"), telemetry

    except Exception as e:
        logging.exception("Traceback: ")
        abort(e)


if __name__ == "__main__":
    configure_logger()
    parser = argparse.ArgumentParser(
        description="""Start service""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to serve on",
        default=8895,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name to load",
        default="network.p",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="Model Architecture Name",
        default="bert-base-cased",
    )
    parser.add_argument(
        "--service_name",
        type=str,
        help="Name of Service",
        choices=["dbpedia", "toxic_comments"],
    )
    args = parser.parse_args()
    init(args=args)
    app.run(host="0.0.0.0", port=args.port)

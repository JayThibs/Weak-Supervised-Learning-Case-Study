import argparse
import json
import logging
import os
import sys

from flask import Flask, abort, jsonify, request

app = Flask(__name__)


def configure_logger(level=logging.INFO):
    logger_format = "[%(asctime)s][%(levelname)s]\t%(message)s"
    logging.basicConfig(
        level=level, format=logger_format, handlers=[logging.StreamHandler(sys.stdout)]
    )


# TODO - Implement init
def init(args):
    pass


@app.after_request
def apply_caching(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Max-Age"] = 1000
    # note that '*' is not valid for Access-Control-Allow-Headers
    response.headers[
        "Access-Control-Allow-Headers"
    ] = "origin, x-csrftoken, content-type, accept"
    return response


@app.route("/", methods=["POST"])
def service():
    if not request.is_json:
        abort(400, "Invalid request payload in JSON format")
    try:
        request_data = request.get_json()
        response = {}
        return json.dumps(response).encode("utf-8")
    except Exception as e:
        logging.exception("Traceback: ")
        abort(e.code, e.msg)


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
        choices=8895,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to model to load",
        default=os.path.join(os.getcwd(), "network.p"),
    )
    args = parser.parse_args()
    init(args=args)
    app.run(host="0.0.0.0", port=args.port)

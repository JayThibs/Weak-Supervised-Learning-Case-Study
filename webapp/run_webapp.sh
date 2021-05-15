#!/bin/bash

set -a
source .env
set +a

streamlit run app.py --server.port=8501
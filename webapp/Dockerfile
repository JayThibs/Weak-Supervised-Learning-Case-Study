FROM python:3.7-stretch


ENV PYTHONPATH="/app"

EXPOSE 8501

# Set the working directory
WORKDIR /app

# Copy the specific directories into the container at /app
COPY . /app/

#  Installing dependencies
RUN pip install -r requirements.txt

# Run app.py when the container launches
ENTRYPOINT ["/app/run_webapp.sh"]
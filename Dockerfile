FROM anibali/pytorch:1.8.1-cuda11.1

ENV PYTHONPATH="/app"

# Set the working directory
WORKDIR /app

# Copy the specific directories into the container at /app
COPY ./* /app

# Run app.py when the container launches
ENTRYPOINT ["python", "service.py"]
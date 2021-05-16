# Webapp prototype

This is a simple UI to test endpoints and let users play with the models.

Here are the steps to recreate the webapp:

1. Clone the repo and cd into /webapp.

2. Create a `.env` file with the same structure found in `.env-structure`.

3. Build the image
```
docker build -t weaklabeling-ui .
```

4. Running it
```
docker run -it -p 8501:8501 weaklabeling-ui
```  

Open http://localhost:8501

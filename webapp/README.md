# Webapp prototype

This is a simple UI to test endpoints and let users play with the models.

Build the image
```
docker build -t weaklabeling-ui .
```

Running it
```
docker run -it -p 8501:8501 weaklabeling-ui
```  

Open http://localhost:8501
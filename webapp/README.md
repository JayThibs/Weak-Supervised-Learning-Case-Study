# Streamlit Web App for Text Classification

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

5. Open http://localhost:8501 and test it out!

![Screen Shot 2021-05-16 at 3 09 12 AM](https://user-images.githubusercontent.com/19174440/118387287-18f09600-b5f4-11eb-891c-28e0915218b4.png)

![Screen Shot 2021-05-16 at 3 09 58 AM](https://user-images.githubusercontent.com/19174440/118387310-332a7400-b5f4-11eb-93bd-bc7744e621b1.png)

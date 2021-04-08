# Train with GPU and deploy a model endpoint and Streamlit app

This example demonstrates how to train with GPU and deploy two HTTP endpoints:
- a model endpoint
- a Streamlit app for testing the model endpoint

## Goals
At the end of the tutorial, the user will be able to
- train a model with GPU on Bedrock
- deploy a model endpoint
- deploy a Streamlit app

## Model
This demo is using [yolov5](https://github.com/ultralytics/yolov5) framework.
Most of the codes are copied from there in their original form and directory structure.
Two additional files are added as required to train a model.
- `models/custom_yolov5s.yaml`
- `data.yaml`

## Data
Data can be downloaded from [here](https://public.roboflow.com/object-detection/shellfish-openimages). We have already uploaded the data to S3.

### Additional files created for deploying on Bedrock
#### Training
- `bedrock.hcl`: Set Bedrock configuration
- `requirements-train.txt`: Requirements for training
- `task_train.py`: Training script entry point
- `utils/s3.py`: S3 utility functions

#### Serving
- `requirements-serve.txt`: Requirements for serving
- `serve_http.py`: Serving script
- `utils/serve.py`: Image utility functions

## Testing endpoint
A Streamlit app `app.py` is provided to test deployed endpoints with sample images from `test_images/`. Deploy the app as a HTTP endpoint using `bedrock_streamlit.hcl`, setting **Authentication method as 'None'**. 

In the app, you will need to enter both API URL and token. They can be retrieved from API docs of your deployed endpoint.

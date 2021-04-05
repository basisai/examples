# End-to-end Bedrock tutorial in Google Cloud

This example demonstrates how you can use Bedrock to preprocess data, train a model and create a HTTPS endpoint for serving. This example complements [Bedrock documentation](https://docs.basis-ai.com/guides/quickstart).

This example will use sample customer churn data. The code shows how to
- read data from GCS bucket
- preprocess the data and generate features
- train a LightGBM classifier model
- serve the model as an endpoint

### Data exploration and Model prototyping
See [notebook](./doc/churn_prediction.ipynb)

### Data processing and model training flowchart
![flowchart](./doc/flow.png)

## Goals
At the of the tutorial, the user will be able to
- set up pipelines in Bedrock
- monitor the training
- deploy a model endpoint in HTTPS
- query the endpoint
- monitor the endpoint

## Run on Bedrock
Follow the [documentation](https://docs.basis-ai.com/guides/quickstart).

## Test your endpoint
```
curl -X POST \
  <MODEL_ENDPOINT_URL> \
  -H 'Content-Type: application/json' \
  -H 'X-Bedrock-Api-Token: <MODEL_ENDPOINT_TOKEN>' \
  -d '{"State": "ME", "Area_Code": 408, "Intl_Plan": 1, "VMail_Plan": 1, "VMail_Message": 21, "CustServ_Calls": 4, "Day_Mins": 156.5, "Day_Calls": 122, "Eve_Mins": 209.2, "Eve_Calls": 125, "Night_Mins": 158.7, "Night_Calls": 81, "Intl_Mins": 11.1, "Intl_Calls": 3}'
```

# bedrock-examples
A repository to host examples for notebooks, pipelines, batch scoring and deployment on Bedrock

This repository is home to the following types of examples:
* [End-to-end](#end-to-end)
* [Component](#component)


A copy of the hcl file template is also provided as `bedrock.hcl.tmpl`.

## End-to-end

### [Binary class](./binary_class)
This example covers the following concepts:
1. Set up a Bedrock training pipeline, either on Google Cloud or AWS
2. Monitor the training
3. Deploy a model endpoint in HTTPS
4. Query the endpoint
5. Monitor the endpoint API metrics

### [Credit risk](./credit_risk)
This example covers the following concepts:
1. Set up a Bedrock training pipeline
2. Log training-time feature and inference distributions
3. Log model explainability and fairness metrics
4. Check model explainability and fairness from Bedrock web UI
5. Deploy a model endpoint in HTTPS with logging feature and inference distributions
6. Monitor the endpoint by simulating a query stream

### [Multiclass](./multiclass)
This example covers the following concepts:
1. Train a multiclass classification machine learning model
2. Log metrics for multiclass classfication
3. Log ROC, PR, and confusion matrices by micro-averaging on all classes
4. Customise logging of feature distributions
5. Log predictions for multiclass classification
6. Log model explainabilty and fairness for multiclass
7. Serving code for multiclass classification using bedrock express


## Component

### [Miniconda](./miniconda)
This example covers the following concepts:
1. Install conda packaages
2. Activate a conda environment on bedrock

### [PySpark step](./pyspark_turnstile)
This example covers the following concepts:
1. Write PySpark step in hcl file

### [Batch scoring pipeline](./batch_score)
This example covecrs the following concepts:
1. Set up a Bedrock batch scoring pipeline on Google Cloud
2. Save the output in Google BigQuery

### [Serve with gRPC](./grpc_serve)
This example covers the following concepts:
1. Deploy a gRPC endpoint on Google Cloud

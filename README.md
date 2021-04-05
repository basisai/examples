# bedrock-examples
A repository to host examples for notebooks, pipelines, batch scoring and deployment on Bedrock

This repository is home to the following types of examples:
* [End-to-end](#end-to-end)
* [Component](#component)

## End-to-end

### [Binary class GCP](./binary_class_gcp)

This example covers the following concepts:
1. Set up a Bedrock training pipeline in Google Cloud
2. Monitor the training
3. Deploy a model endpoint in HTTPS
4. Query the endpoint
5. Monitor the endpoint

### [Binary class AWS](./binary_class_aws)

This example covers the following concepts:
1. Set up a Bedrock training pipeline in AWS
2. Monitor the training
3. Deploy a model endpoint in HTTPS
4. Query the endpoint
5. Monitor the endpoint

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

### [PySpark step](./pyspark_turnstile)

This example covers the following concepts:
1. Write PySpark step in `bedrock.hcl`

### [Batch scoring pipeline](./batch_score)

This example covecrs the following concepts:
1. Set up a Bedrock batch scoring pipeline in Google Cloud
2. Save the output in Google BigQuery

### [Serve with GRPC](./grpc_serve)

This example covecrs the following concepts:
1. Deploy a model endpoint in GRPC
2. Query the endpoint
3. Monitor the endpoint

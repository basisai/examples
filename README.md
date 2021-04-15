# bedrock-examples
A repository to host examples for notebooks, pipelines, batch scoring and deployment on Bedrock

This repository is home to the following types of examples:
* [End-to-end](#end-to-end)
* [Component](#component)

### hcl template
A copy of the [hcl file](./bedrock.hcl.tmpl) is provided as a template.


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

### [yolov5](./yolov5_shellfish)
This example covers the following concepts:
1. Train a model with GPU on Bedrock
2. Deploy a model endpoint
3. Deploy a Streamlit app


## Component

### [Miniconda](./miniconda)
This example covers the following concepts:
1. Install conda packaages
2. Activate a conda environment on bedrock

### [PySpark step](./pyspark_step)
This example covers the following concepts:
1. Write hcl with Spark configuration

### [Use GPU](./gpu)
This example covers the following concepts:
1. Deploy a GPU instance

### [Batch scoring pipeline](./batch_score)
This example covecrs the following concepts:
1. Set up a Bedrock batch scoring pipeline on Google Cloud
2. Save the output in Google BigQuery

### [Serve with gRPC](./grpc_serve)
This example covers the following concepts:
1. Deploy a gRPC endpoint on Google Cloud


## Documentation
Refer to the [documentation](https://docs.basis-ai.com/) for more details.

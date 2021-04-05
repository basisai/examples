# bedrock-examples
A repository to host examples for notebooks, pipelines, batch scoring and deployment on Bedrock

This repository is home to the following types of examples:
* [End-to-end](#end-to-end)
* [Component](#component)

## End-to-end

### [Binary class](./binary_class)

This example covers the following concepts:
1. Set up pipelines in Bedrock
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

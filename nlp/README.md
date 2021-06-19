# NLP demo

In this example, we will demonstrate how to apply an explainability tool on a trained classifcation model.

## Goals
At the end of the tutorial, the user will be able to
- fine-tune `transformers.DistilBertForSequenceClassification` model
- use `captum.attr.LayerIntegratedGradients` for explainability

## Visualize results
After deploying the model as an endpoint, you can use `test_endpoint.ipynb` to test it and visualize the results.

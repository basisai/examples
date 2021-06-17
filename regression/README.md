# Regression on Bedrock

In this walkthrough, you will learn how to use bedrock to train a regression model and then deploy the model on bedrock

## Goals

* Demonstrate how to train a regression model on bedrock
* Demonstrate how to log metrics for regression using the bedrock client
* Demonstrate how to customise logging of feature distributions
* Demonstrate how to log model explanabilty and XAI for regression on bedrock
* Demonstrate how to serve a trained regression model using bedrock express

## Sample json data
```yaml
{
    "LongestShell": "0.35", 
    "Diameter": "0.195",
    "Height": "0.06", 
    "WholeWeight": "0.95",
    "ShuckedWeight": "0.0445",
    "VisceraWeight": "0.0245", 
    "Rings": "9",
    "large_rings": "0"
}
```
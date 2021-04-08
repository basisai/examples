# PySpark Notebooks on Bedrock

You will learn how to run PySpark on Bedrock managed notebooks.

## Goals

* Demonstrate how to start a PySpark cluster within a notebook `pyspark.ipynb`
* Demonstrate how to read a CSV file using Pandas and convert it into a PySpark dataframe
* Demonstrate how to read a CSV file from AWS S3 directly into a PySpark dataframe
* Demonstrate how to read a CSV file from GCP GCS directly into a PySpark dataframe

## Additional Notes

* These notebooks are designed to run as a Bedrock managed notebook.
* The notebook is tested on a PySpark 3.0 image (with cloud storage packages) on a small notebook instance (3 CPU, 11.3 GiB Memory) acting as the Spark driver
* For the AWS example to work, ensure that your managed notebook is running on an AWS environment on Bedrock
* For the GCP example to work, ensure that your managed notebook is running on a GCP environment on Bedrock



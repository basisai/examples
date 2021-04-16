# Create a PySpark step in Bedrock pipeline

In this example, we will demonstrate how to write hcl to run a Bedrock pipeline step in PySpark.

## Goals
At the end of the tutorial, the user will be able to
- write `bedrock.hcl` with Spark configuration.

## HCL
You can add your Spark configuration for a pipeline step in the hcl file if you wish to use Spark.

In order to read and write data from the buckets, it is required to include additional lines in `conf`. Refer to [`bedrock_gcp.hcl`](./bedrock_gcp.hcl) for Google Cloud, and [`bedrock_aws.hcl`](./bedrock_aws.hcl) for AWS.

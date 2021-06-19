version = "1.0"

train {
  step process {
    image = "quay.io/basisai/workload-standard:v0.3.1"
    install = [
      "pip3 install --upgrade pip",
      "pip3 install -r requirements-aws.txt",
    ]
    script = [
      {
        spark-submit = {
          script = "preprocess.py"
          conf = {
            "spark.kubernetes.container.image"       = "quay.io/basisai/workload-standard:v0.3.1"
            "spark.kubernetes.pyspark.pythonVersion" = "3"
            "spark.driver.memory"                    = "4g"
            "spark.driver.cores"                     = "2"
            "spark.executor.instances"               = "2"
            "spark.executor.memory"                  = "4g"
            "spark.executor.cores"                   = "2"
            "spark.memory.fraction"                  = "0.5"
            "spark.sql.parquet.compression.codec"    = "gzip"
            "spark.hadoop.fs.s3a.impl"               = "org.apache.hadoop.fs.s3a.S3AFileSystem"
            "spark.hadoop.fs.s3a.endpoint"           = "s3.ap-southeast-1.amazonaws.com"
          }
        }
      }
    ]
    resources {
      cpu    = "0.5"
      memory = "1G"
    }
  }

  parameters {
    RAW_SUBSCRIBERS_DATA = "s3a://bedrock-sample/churn_data/subscribers.gz.parquet"
    RAW_CALLS_DATA       = "s3a://bedrock-sample/churn_data/all_calls.gz.parquet"
    PROCESSED_DATA       = "s3a://bdrk-sandbox-aws-data/churn_data/processed"
  }
}

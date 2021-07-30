version = "1.0"

train {
  step "process" {
    image = "quay.io/basisai/workload-standard:v0.3.1"
    install = []
    script = [
      {
        spark-submit {
          script = "preprocess.py"
          conf = {
            "spark.executor.instances"               = "2"
            "spark.executor.memory"                  = "4g"
            "spark.executor.cores"                   = "2"
            "spark.sql.parquet.compression.codec"    = "gzip"
          }
        }
      }
    ]
    resources {
      cpu = "0.5"
      memory = "1G"
    }
  }

  parameters {
    RAW_SUBSCRIBERS_DATA = "gs://bedrock-sample/churn_data/subscribers.gz.parquet"
    RAW_CALLS_DATA       = "gs://bedrock-sample/churn_data/all_calls.gz.parquet"
    PROCESSED_DATA       = "gs://span-temp-production/churn_data/processed"
  }
}

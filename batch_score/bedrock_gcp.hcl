version = "1.0"

batch_score {
  step "preprocess" {
    image = "quay.io/basisai/workload-standard:v0.3.1"
    install = []
    script = [
      {
        spark-submit = {
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
      cpu    = "0.5"
      memory = "1G"
    }
    retry {
      limit = "2"
    }
  }

  step "generate_features" {
    image = "quay.io/basisai/workload-standard:v0.3.1"
    install = [
      "pip3 install --upgrade pip",
      "pip3 install -r requirements-gcp.txt",
    ]
    script = [
      {
        spark-submit = {
          script = "generate_features.py"
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
      cpu    = "0.5"
      memory = "1G"
    }
    retry {
      limit = "2"
    }
    depends_on = ["preprocess"]
  }

  step "batch_score" {
    image = "python:3.7"
    install = [
      "pip3 install --upgrade pip",
      "pip3 install -r requirements-gcp.txt",
    ]
    script = [{ sh = ["python3 batch_score.py"] }]
    resources {
      cpu    = "0.5"
      memory = "1G"
    }
    retry {
      limit = "2"
    }
    depends_on = ["generate_features"]
  }

  parameters {
    RAW_SUBSCRIBERS_DATA  = "gs://bedrock-sample/churn_data/subscribers.gz.parquet"
    RAW_CALLS_DATA        = "gs://bedrock-sample/churn_data/all_calls.gz.parquet"
    TEMP_DATA_BUCKET      = "gs://span-temp-production/"
    PREPROCESSED_DATA     = "churn_data/preprocessed"
    FEATURES_DATA         = "churn_data/features.csv"
    SUBSCRIBER_SCORE_DATA = "churn_data/subscriber_score.csv"
    OUTPUT_MODEL_NAME     = "lgb_model.pkl"
  }
}

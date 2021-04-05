version = "1.0"

train {
    step process {
        image = "basisai/workload-standard:v0.2.2"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements.txt",
        ]
        script = [
            {spark-submit {
                script = "process.py"
                // to be passed in as --conf key=value
                conf {
                    spark.kubernetes.container.image = "basisai/workload-standard:v0.2.2"
                    spark.kubernetes.pyspark.pythonVersion = "3"
                    spark.driver.memory = "8g"
                    spark.driver.cores = "2"
                    spark.executor.instances = "3"
                    spark.executor.memory = "8g"
                    spark.executor.cores = "2"
                    spark.memory.fraction = "0.5"
                    spark.sql.parquet.compression.codec = "gzip"
                    spark.hadoop.fs.AbstractFileSystem.gs.impl = "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS"
                    spark.hadoop.google.cloud.auth.service.account.enable = "true"
                }
                // to be passed in as --key=value
                settings {
                }
            }}
        ]
        resources {
            cpu = "0.5"
            memory = "1G"
        }
    }

    parameters {
        RAW_INPUT_DATA = "gs://bedrock-sample/mta-turnstile"
        OUTPUT_DATA = ""gs://span-temp-production/ts_daily_traffic.csv"
    }
}

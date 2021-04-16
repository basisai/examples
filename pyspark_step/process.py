"""
Script to process subscribers.
"""
from os import getenv
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

RAW_SUBSCRIBERS_DATA = getenv("RAW_SUBSCRIBERS_DATA")
RAW_CALLS_DATA = getenv("RAW_CALLS_DATA")
PROCESSED_DATA = getenv("PROCESSED_DATA")


def preprocess_subscriber(spark):
    """Preprocess subscriber data."""
    # Load subscribers
    subscribers_df = (
        spark.read.parquet(RAW_SUBSCRIBERS_DATA)
        .withColumn("Intl_Plan", F.when(F.col("Intl_Plan") == "yes", 1).otherwise(0))
        .withColumn("VMail_Plan", F.when(F.col("VMail_Plan") == "yes", 1).otherwise(0))
        .withColumn("Churn", F.when(F.col("Churn") == "yes", 1).otherwise(0))
    )

    # Load raw calls
    calls_df = (
        spark.read.parquet(RAW_CALLS_DATA)
        .groupBy("User_id")
        .pivot("Call_type", ["Day", "Eve", "Night", "Intl"])
        .agg(F.sum("Duration").alias("Mins"), F.count("Duration").alias("Calls"))
    )

    # Join subscribers with calls
    joined_df = subscribers_df.join(calls_df, on="User_id", how="left")
    joined_df = joined_df.fillna(0)
    return joined_df


def main():
    """Preprocess data"""
    with SparkSession.builder.appName("Preprocessing").getOrCreate() as spark:
        spark.sparkContext.setLogLevel("FATAL")

        print("\nProcessing")
        preprocessed_df = preprocess_subscriber(spark)
        preprocessed_df.cache()
        print(f"\n\tNumber of rows = {preprocessed_df.count()}")

        print("\nSaving")
        preprocessed_df.repartition(1).write.mode("overwrite").parquet(PROCESSED_DATA)


if __name__ == "__main__":
    main()

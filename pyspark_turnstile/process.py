"""
Script to aggregate time series.
"""
from os import getenv
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType, TimestampType
from pyspark.sql.window import Window

RAW_INPUT_DATA = getenv("RAW_INPUT_DATA")
OUTPUT_DATA = getenv("OUTPUT_DATA")


def preprocess_data(input_df):
    """Preprocess data."""
    tmp_df = (
        input_df
        .drop("linename", "division", "desc")
        .withColumn("time", F.unix_timestamp(F.col("time").cast(TimestampType())))
        .withColumn("entries", F.col("entries").cast(FloatType()))
        .withColumn("exits", F.col("exits").cast(FloatType()))
        .dropna()
    )

    # Floor time to 4H
    tmp_df = tmp_df.withColumn(
        "time_rounded", ((F.col("time") / (4 * 3600)).cast(IntegerType()) * 4 * 3600)
        .cast(TimestampType()))

    # Compute incremental values for entries and exits
    win = Window.partitionBy("ca", "unit", "scp", "station").orderBy("time")
    tmp_df = (
        tmp_df
        .withColumn("entries_count", F.lag("entries", count=-1).over(win) - F.col("entries"))
        .withColumn("exits_count", F.lag("exits", count=-1).over(win) - F.col("exits"))
        .withColumn("time_diff", (F.lag("time", count=-1).over(win) - F.col("time")) / 3600)
    )
    # Sum entries_count and exits_count in the same 4-hour bin for each turnstile
    output_df = (
        tmp_df
        .groupBy('time_rounded', 'station', 'ca', 'unit', 'scp')
        .agg(
            F.sum("entries_count").alias("entries_count"),
            F.sum("exits_count").alias("exits_count"),
            F.sum("time_diff").alias("time_diff"),
        )
    )
    return output_df


def impute_nans(df, col_name, group_bys=["station", "ca", "unit", "scp", "hour", "wkdy"]):
    """Impute nans."""
    avg_df = df.groupBy(group_bys).agg(F.mean(col_name).alias("avg_val"))
    df = (
        df
        .join(avg_df, on=group_bys, how="left")
        .withColumn(col_name, F.when(F.isnull(col_name), F.col("avg_val")).otherwise(F.col(col_name)))
        .drop("avg_val")
    )
    return df


def clean_data(input_df, threshold=15000):
    """Clean data"""
    # Set counts that correspond to time_diff >= 8h to NaNs
    # Take absolute values of entries_count and exits_count
    tmp_df = (
        input_df
        .withColumn("entries_count", F.when(F.col("time_diff") >= 8, None).otherwise(
            F.abs(F.col("entries_count"))))
        .withColumn("exits_count", F.when(F.col("time_diff") >= 8, None).otherwise(
            F.abs(F.col("exits_count"))))
    )

    # Replace absolute values of entries_count and exits_count > 15,000 with NaNs
    tmp_df = (
        tmp_df
        .withColumn("entries_count", F.when(F.col("entries_count") > threshold, None).otherwise(
            F.col("entries_count")))
        .withColumn("exits_count", F.when(F.col("exits_count") > threshold, None).otherwise(
            F.col("exits_count")))
    )

    # Impute NaNs with average counts of the same turnstile, hour & day of week
    tmp_df = (
        tmp_df
        .withColumn("hour", F.hour("time_rounded"))
        .withColumn("wkdy", F.dayofweek("time_rounded"))
    )
    tmp_df = impute_nans(tmp_df, "entries_count")
    tmp_df = impute_nans(tmp_df, "exits_count")

    # Compute traffic
    output_df = tmp_df.withColumn("traffic", F.col("entries_count") + F.col("exits_count"))
    return output_df


def agg_timeseries(input_df, group_bys):
    """Aggregate traffic by date."""
    output_df = (
        input_df
        .withColumn("date", F.to_date(F.col("time_rounded")).cast("date"))
        .groupBy(group_bys)
        .agg(F.sum("traffic").alias("traffic"))
    )
    return output_df


def main():
    """Train pipeline"""
    with SparkSession.builder.appName("Preprocessing").getOrCreate() as spark:
        spark.sparkContext.setLogLevel("FATAL")

        print("\tLoading raw data")
        raw_df = spark.read.csv(RAW_INPUT_DATA, header=True)

        print("\tPreprocessing time series")
        proc_df = clean_data(preprocess_data(raw_df))

        print("\tAggregating time series by date")
        start = time.time()
        output_df = agg_timeseries(proc_df, "date").toPandas().sort_values("date")
        print("\t\tNumber of rows = {}".format(output_df.shape[0]))
        print("\t\tTime taken = {:.2f} mins".format((time.time() - start) / 60))

        print("\tAggregating time series by date and station")
        start = time.time()
        sample_df1 = agg_timeseries(proc_df, ["date", "station"])
        sample_df1.cache()
        print("\t\tNumber of rows = {}".format(sample_df1.count()))
        print("\t\tTime taken = {:.2f} mins".format((time.time() - start) / 60))
        sample_df1.show(3)

        print("\tAggregating time series by date, station and unit")
        start = time.time()
        sample_df2 = agg_timeseries(proc_df, ["date", "station", "ca", "unit", "scp"])
        sample_df2.cache()
        print("\t\tNumber of rows = {}".format(sample_df2.count()))
        print("\t\tTime taken = {:.2f} mins".format((time.time() - start) / 60))
        sample_df2.show(3)

    print("\tSaving time series")
    output_df.to_csv(OUTPUT_DATA, index=False)


if __name__ == "__main__":
    main()

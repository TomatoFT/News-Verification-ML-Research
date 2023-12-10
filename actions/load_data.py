from pyspark.sql.functions import col, lit, when

from features_engineering.entity import (get_entities_of_news,
                                         get_news_categories,
                                         get_sentiment_of_the_news)
from features_engineering.relevency import get_the_similarity_score
from features_engineering.summarization import get_summarization_of_the_news


def load_csv_into_spark(file_path, spark_session):
    """
    Load a CSV file into a PySpark DataFrame.

    Parameters:
    - file_path (str): Path to the CSV file.
    - spark_session (SparkSession): PySpark SparkSession object.

    Returns:
    - list: List of dictionaries containing the data from the CSV file.
    """
    # Read CSV into DataFrame
    df = spark_session.read.csv(file_path, header=True, inferSchema=True)

    df = df.withColumn("summarization", get_summarization_of_the_news(col("content")))

    df = df.withColumn("entities", get_entities_of_news(col("summarization")))

    df = df.withColumn("category", get_news_categories(col("summarization")))

    # Add a new column 'numeric' filled with zeros
    df = df.withColumn("numeric", lit(0))

    df = df.withColumn("sentiment", get_sentiment_of_the_news(col("summarization")))

    # Add a new column 'target' based on the condition
    df = df.withColumn("target", when(col("name") == "vtv", 1).otherwise(0))

    # # Transform DataFrame to the specified format
    # data = [
    #     {'text': row['title'], 'numeric': row['numeric'], 'target': row['target']}
    #     for row in df.collect()
    # ]

    return df

from pyspark.sql.functions import col, lit, when


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

    # Add a new column 'numeric' filled with zeros
    df = df.withColumn("numeric", lit(0))

    # Add a new column 'target' based on the condition
    df = df.withColumn("target", when(col("name") == "vtv", 1).otherwise(0))

    # Transform DataFrame to the specified format
    data = [
        {'text': row['title'], 'numeric': row['numeric'], 'target': row['target']}
        for row in df.collect()
    ]

    return data

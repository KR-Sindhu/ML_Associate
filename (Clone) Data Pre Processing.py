# Databricks notebook source
file_path="/FileStore/tables/listings-3.csv"
raw_df = spark.read.csv(file_path, header=True, inferSchema=True, multiLine=True, escape='"')
display(raw_df)

# COMMAND ----------

raw_df.columns


# COMMAND ----------

columns_to_keep = [
    "host_total_listings_count",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "minimum_nights",
    "number_of_reviews",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "price"
]

base_df = raw_df.select(columns_to_keep)
base_df.cache().count()
display(base_df)

# COMMAND ----------

from pyspark.sql.functions import col, translate

fixed_price_df = base_df.withColumn("price", translate(col("price"), "$,", "").cast("double"))

display(fixed_price_df)

# COMMAND ----------

display(fixed_price_df.describe())

# COMMAND ----------

display(fixed_price_df.select("price").describe())

# COMMAND ----------

fixed_price_df.filter(col("price") == 0).count()

# COMMAND ----------

pos_prices_df = fixed_price_df.filter(col("price") > 0)

# COMMAND ----------

display(pos_prices_df
        .groupBy("minimum_nights").count()
        .orderBy(col("count").desc(), col("minimum_nights"))
       )

# COMMAND ----------

min_nights_df = pos_prices_df.filter(col("minimum_nights") <= 365)

display(min_nights_df)

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType

integer_columns = [x.name for x in min_nights_df.schema.fields if x.dataType == IntegerType()]
doubles_df = min_nights_df

for c in integer_columns:
    doubles_df = doubles_df.withColumn(c, col(c).cast("double"))

columns = "\n - ".join(integer_columns)
print(f"Columns converted from Integer to Double:\n - {columns}")

# COMMAND ----------

from pyspark.sql.functions import when

impute_cols = [
    "bedrooms",
    "bathrooms",
    "beds", 
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value"
]

for c in impute_cols:
    doubles_df = doubles_df.withColumn(c + "_na", when(col(c).isNull(), 1.0).otherwise(0.0))

# COMMAND ----------

display(doubles_df.describe())

# COMMAND ----------

from pyspark.ml.feature import Imputer

imputer = Imputer(strategy="median", inputCols=impute_cols, outputCols=impute_cols)

imputer_model = imputer.fit(doubles_df)
imputed_df = imputer_model.transform(doubles_df)

# COMMAND ----------

imputed_df.write.format("delta").mode("overwrite").save("/dbfs/FileStore/tables/tmp/imputed_df")

# COMMAND ----------


data = spark.read.format("delta").load("dbfs:/dbfs/FileStore/tables/tmp/imputed_df")
display(data)
  


# COMMAND ----------



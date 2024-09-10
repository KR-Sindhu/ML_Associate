# Databricks notebook source
# MAGIC %md
# MAGIC Load Dataset

# COMMAND ----------

data = spark.read.format("delta").load("dbfs:/dbfs/FileStore/tables/tmp/imputed_df")
train_df, test_df = data.randomSplit([.8, .2], seed=42)

# COMMAND ----------

print(train_df.cache().count())

# COMMAND ----------

display(train_df.select("price"))

# COMMAND ----------

display(train_df)

# COMMAND ----------

display(train_df.groupBy("room_type").count())

# COMMAND ----------

train_repartition_df, test_repartition_df = (data
                                             .repartition(24)
                                             .randomSplit([.8, .2], seed=42))

print(train_repartition_df.count())

# COMMAND ----------

display(train_df.select("price", "bedrooms"))

# COMMAND ----------

display(train_df.select("price", "bedrooms").summary())

# COMMAND ----------

display(train_df)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
assembler = VectorAssembler(
    inputCols=["bedrooms"],
    outputCol="features")

# Transform the train_df to have a features column
train_df_transformed = assembler.transform(train_df)

# Now, you can use the transformed dataframe with the LinearRegression model
lr = LinearRegression(featuresCol="features", labelCol="price")
lr_model = lr.fit(train_df_transformed)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

vec_assembler = VectorAssembler(inputCols=["bedrooms"], outputCol="features")

vec_train_df = vec_assembler.transform(train_df)

# COMMAND ----------

lr = LinearRegression(featuresCol="features", labelCol="price")
lr_model = lr.fit(vec_train_df)

# COMMAND ----------

m = lr_model.coefficients[0]
b = lr_model.intercept

print(f"The formula for the linear regression line is y = {m:.2f}x + {b:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

vec_test_df = vec_assembler.transform(test_df)

pred_df = lr_model.transform(vec_test_df)

pred_df.select("bedrooms", "features", "price", "prediction").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the Model

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")

rmse = regression_evaluator.evaluate(pred_df)
print(f"RMSE is {rmse}")

# COMMAND ----------



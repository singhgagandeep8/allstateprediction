#predicting cost
#training model
from pyspark.ml.feature import RFormula
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
from pyspark.ml import Pipeline

#loading the data and randomly splitting
df = sqlContext.read.load('mnt/allstate-gjs/train_data.csv', format='com.databricks.spark.csv', header='true', inferSchema='true')
(trainingData, validData) = df.randomSplit([0.7, 0.3], seed=12456)

#training the model using the trainingData
formula=RFormula(formula="cost~A+B+C+D+E+F+G+shopping_pt+group_size+homeowner+car_age+car_value+risk_factor+age_oldest+age_youngest+married_couple+C_previous+duration_previous")
estimator = LinearRegression()
model = Pipeline(stages=[formula,estimator]).fit(trainingData)
#making predictions on training data only
predictions = model.transform(trainingData)
predictions.select("customer_ID",'A','B','C','D','E','F','G','cost','prediction').show(10)

#using the trained model to make predictions on validation data to check RMSE and MAE values
prediction1 = model.transform(validData)
pred = prediction1.select('prediction')
evaluator_linear_mae = RegressionEvaluator(metricName = "mae")
mae = evaluator_linear_mae.evaluate(prediction1)
print("Regression:MAE = " +str(mae))
evaluator_linear_rmse = RegressionEvaluator(metricName = "rmse")
rmse = evaluator_linear_rmse.evaluate(prediction1)
print("Regression:RMSE = " +str(rmse))

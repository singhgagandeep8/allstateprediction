#randomforest and cost prediction on testdataset

#importing all the required files
from pyspark.sql import SQLContext
from pyspark.sql.types import *
sqlContext = SQLContext(sc)
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.feature import RFormula
from pyspark.ml.regression import LinearRegression
#repeating the same process on test data
testdata = sqlContext.read.load('mnt/allstate-gjs/test_cleaned.csv', format='com.databricks.spark.csv', header='true', inferSchema='true')
stringIndexerTest = StringIndexer(inputCol= "state", outputCol="stateIndex").fit(testdata)
# Use OneHotEncoder to convert categorical variables into binary SparseVectors
encoder = OneHotEncoder(inputCol="stateIndex", outputCol="stateVec") 
features_original=["shopping_pt","group_size","homeowner","car_age","car_value","risk_factor","age_oldest","age_youngest","married_couple","C_previous","duration_previous","A_0","A_1","A_2","B_0","B_1","C_1","C_2","C_3","C_4","D_1","D_2","D_3","E_0","E_1","F_0","F_1","F_2","F_3","G_1","G_2","G_3","G_4","stateVec"]
assembler = VectorAssembler(inputCols=[x for x in features_original], outputCol="features")

labels = ['A','B','C','D','E','F','G']
list_test = []
finalid=testdata.select("customer_ID")
list_test.append(finalid)
for x in labels:
  labelIndexer = StringIndexer(inputCol=x, outputCol="label").fit(testdata)
  rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=200)
  labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=labelIndexer.labels)
  pipeline = Pipeline(stages=[stringIndexerTest, encoder, labelIndexer, assembler, rf, labelConverter])
  fit = pipeline.fit(testdata)
  predictions = fit.transform(testdata)
  pred = predictions.select("customer_ID","predictedLabel")
  list_test.append(pred)
  
#we got a list of dataframes. converting it into a dataframe
combined_list=list_test[0].join(list_test[1],"customer_ID","outer").join(list_test[2],"customer_ID","outer").join(list_test[3],"customer_ID","outer").join(list_test[4],"customer_ID","outer").join(list_test[5],"customer_ID","outer").join(list_test[6],"customer_ID","outer").join(list_test[7],"customer_ID","outer")
#combined_df.show(5)
#renaming the columns
testoutput=combined_list.toDF('customer_ID','A','B','C','D','E','F','G')
testoutput.show(5)
#saving the final predictions
#testoutput.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("mnt/allstate-gjs/test_output_rf.csv")

#using the trained model to predict the cost. Features include customer information and earlier predicted labels.
#dropping the given unconfirmed labels from testdata as we will be using the predicted labels to predict cost.
testdata1 = testdata.drop('A','B','C','D','E','F','G')
#creating a dataframe including the required features
ftest=testoutput.join(testdata1, "customer_ID","outer")
#linear model
formula=RFormula(formula="cost~A+B+C+D+E+F+G+shopping_pt+group_size+homeowner+car_age+car_value+risk_factor+age_oldest+age_youngest+married_couple+C_previous+duration_previous")
estimator = LinearRegression()
#fitting the model and making predictions
model = Pipeline(stages=[formula,estimator]).fit(ftest)
predictions = model.transform(ftest)
predictions.select("customer_ID",'A','B','C','D','E','F','G','prediction').show(10)
#now preparing the final dataframe including all the predictions
finalpred = predictions.select("customer_ID",'A','B','C','D','E','F','G','prediction')
finalpred = finalpred.toDF("customer_ID",'A','B','C','D','E','F','G','cost')
finalpred.show(5)
#saving the final predictions(both quotation combination and cost)
finalpred.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("mnt/allstate-gjs/test_output_pred.csv")

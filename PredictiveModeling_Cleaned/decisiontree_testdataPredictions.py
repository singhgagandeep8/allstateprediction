#using decision tree to predicit the labels in testdata
from pyspark.sql import SQLContext
from pyspark.sql.types import *
sqlContext = SQLContext(sc)
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.feature import IndexToString, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.fpm import FPGrowth
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.ml.classification import DecisionTreeClassifier

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
  dt = DecisionTreeClassifier(labelCol="label",featuresCol="features")
  labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=labelIndexer.labels)
  pipeline = Pipeline(stages=[stringIndexerTest, encoder, labelIndexer, assembler, dt, labelConverter])
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
#testoutput.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("mnt/allstate-gjs/test_output_dt.csv")

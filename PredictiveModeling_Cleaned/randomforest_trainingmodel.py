#randomforest_training

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

#loading data
df = sqlContext.read.load('mnt/allstate-gjs/train_data.csv', format='com.databricks.spark.csv', header='true', inferSchema='true')
#splitting into training and validation datasets
(trainingData, validData) = df.randomSplit([0.7, 0.3], seed=12456)
#indexing the categorical variable
stringIndexerTrain = StringIndexer(inputCol= "state", outputCol="stateIndex").fit(trainingData)
stringIndexerValid = StringIndexer(inputCol= "state", outputCol="stateIndex").fit(validData)
# Use OneHotEncoder to convert categorical variables into binary SparseVectors
encoder = OneHotEncoder(inputCol="stateIndex", outputCol="stateVec") 
#storing features in a list
features_original=["shopping_pt","group_size","homeowner","car_age","car_value","risk_factor","age_oldest","age_youngest","married_couple","C_previous","duration_previous","A_0","A_1","A_2","B_0","B_1","C_1","C_2","C_3","C_4","D_1","D_2","D_3","E_0","E_1","F_0","F_1","F_2","F_3","G_1","G_2","G_3","G_4","stateVec"]
#iterating through features list and assembling them as vector
assembler = VectorAssembler(inputCols=[x for x in features_original], outputCol="features")
#output labels stored as list in labels
labels = ['A','B','C','D','E','F','G']
#list_train would be the final list in which outputs will be stored
list_train = []
finalid=trainingData.select("customer_ID")
list_train.append(finalid)
#iterating through each label(A through G)
for x in labels:
  #indexing label  
  labelIndexer = StringIndexer(inputCol=x, outputCol="label").fit(trainingData)
  #making randomforest model
  rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=200)
  #converting the indexed label back to normal form
  labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=labelIndexer.labels)
  #defining the stages using pipeline
  pipeline = Pipeline(stages=[stringIndexerTrain, encoder, labelIndexer, assembler, rf, labelConverter])
  #setting up pipeline on trainingData
  fit = pipeline.fit(trainingData)
  #making predictions on trainingData
  predictions = fit.transform(trainingData)
  pred = predictions.select("customer_ID","predictedLabel")
  list_train.append(pred)

#we got a list of dataframes. Now joining the list of dataframes and converting it to a dataframe.
combine_df=list_train[0].join(list_train[1],"customer_ID","outer").join(list_train[2],"customer_ID","outer").join(list_train[3],"customer_ID","outer").join(list_train[4],"customer_ID","outer").join(list_train[5],"customer_ID","outer").join(list_train[6],"customer_ID","outer").join(list_train[7],"customer_ID","outer")
#renaming the columns
trainoutput=combine_df.toDF('customer_ID','A','B','C','D','E','F','G')
#trainoutput.show(5)


#now repeating the same process for validation dataset and checking accuracy and test error
labels = ['A','B','C','D','E','F','G']
list_valid = []
finalid=validData.select("customer_ID")
list_valid.append(finalid)
testerrorlabel=[]
accuracylabel = []
for x in labels:
  labelIndexer = StringIndexer(inputCol=x, outputCol="label").fit(validData)
  rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=200)
  labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=labelIndexer.labels)
  evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
  pipeline = Pipeline(stages=[stringIndexerValid, encoder, labelIndexer, assembler, rf, labelConverter])
  fit = pipeline.fit(validData)
  predictions = fit.transform(validData)
  pred = predictions.select("customer_ID","predictedLabel")
  list_valid.append(pred)
  accuracy = evaluator.evaluate(predictions)
  accuracylabel.append(accuracy)
  testerror = 1.0 - accuracy
  testerrorlabel.append(testerror)
  
#randomforest accuracy for all the labels
print accuracylabel

table2 = sqlContext.createDataFrame([('A', 0.9166), ('B', 0.9198), ('C', 0.9169), ('D', 0.9391), ('E', 0.9226), ('F', 0.9158), ('G', 0.8584)], ["label", "accuracy"])
table2.show()

#we got a list of dataframes.
combined_df=list_valid[0].join(list_valid[1],"customer_ID","outer").join(list_valid[2],"customer_ID","outer").join(list_valid[3],"customer_ID","outer").join(list_valid[4],"customer_ID","outer").join(list_valid[5],"customer_ID","outer").join(list_valid[6],"customer_ID","outer").join(list_valid[7],"customer_ID","outer")
#combined_df.show(5)

#renaming the columns
validoutput=combined_df.toDF('customer_ID','A','B','C','D','E','F','G')
validoutput.show(5)

#can store the final results in csv file on AWS or HDFS by uncommenting the following line
#trainoutput.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("mnt/allstate-gjs/train_output_rf.csv")

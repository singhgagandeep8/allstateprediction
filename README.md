# allstateprediction
Predicting policy combination to be quoted to customer and its cost based on customer's characterstics and transaction history 
This document contains instructions in two sets:
1. for data preprocessing 
2. for predictive modeling 
 
For uncleaned datasets (Using Excel, RStudio and Python) 
1. ‘train_original.csv’ and ‘test_original.csv’ are the uncleaned initial datasets.  
2. Replace missing values in C_previous and duration_previous column with 0 in both.  
3. Convert car_value from string values of a-i to a numerical index from 1-9 respectively. 
4. Impute missing values of car_value with the mode = 5 in both. 
5. For risk_factor missing values, imputation is done in RStudio using multinomial regression. Model will be built on available observations of train data- train_rf_train.csv. Put these in a separate csv. Missing observations will be in train_rf_test.csv. Similarly, for test dataset- test_rf_train will have all available rows and Missing rows will be in test_rf_test.csv. Read all these files into Rstudio. 
6. Run the ‘imputation_riskfactor.R’ file and install ‘nnet’ package first if not available.  
7. The missing values will be imputed in train_complete_rf.csv and test_complete_rf.csv for train and test data respectively. Combine these with train_rf_train and test_rf_train csvs. This will give us the entire datasets with missing values imputed.  
8. Remove Location, Time, Day columns since they are not very significant.  
9. Run ‘customer_aggregation.py’ to aggregate rows for customers and creating additional features for different versions of each policy option. Please run this file after imputing the missing values by following the above steps. 
 
 
For cleaned datasets(using PySpark):  
1. Upload the cleaned training and test datasets on AWS(or HDFS). Use HDFS put command to load the data to HDFS. In order to upload on AWS, create S3 bucket and put the datafiles in it. 
2. Update the path for stored train and test files in all python code files for successful execution.  
3. Install and load the required libraries if you are running the code on local machine. There’s no need to install libraries and packages if you are using databricks. You need to sign in to databricks community cloud to do so. 
4. Now, you can run the python code files in local terminal or databricks. On local machine, you can submit .py files by using spark submit command.  
5. Run the ‘randomforest_trainingmodel.py’ to train randomforest model for predicting policy combination and check the accuracy of the built model.  
6. Run the ‘decision_trainingmodel.py’ to train decisiontree model for predicting policy combination and check the accuracy of the built model. 
7. Run the ‘regression_trainingmodel_cost.py’ to train linear regression model and check the error of the built model. 
8. Run ‘decisiontree_testdataPredictions.py’ to make policy combination predictions on test dataset using decision tree model. 
9. Run ‘final(Cost+policyquotation)onTEST.py’ to make policy combination and cost predictions on test dataset using random forest model. 

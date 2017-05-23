import pandas as pd
import numpy as np

#creating new columns for each version of all the options
train = pd.read_csv('~/Desktop/Semester 2/Big Data/project_bigdata/training_count.csv', index_col=0)
train = pd.get_dummies(train, columns = ['A','B','C','D','E','F','G'])
train_sum = train.groupby(level=0).sum()
train_sum.to_csv('~/Desktop/Semester 2/Big Data/project_bigdata/aggregated_train.csv')

#creating new columns for each version of all the options
test = pd.read_csv('~/Desktop/Semester 2/Big Data/project_bigdata/test_agg.csv', index_col=0)
test = pd.get_dummies(test, columns = ['A','B','C','D','E','F','G'])
test_sum = test.groupby(level=0).sum()
test_sum.to_csv('~/Desktop/Semester 2/Big Data/project_bigdata/aggregated_test.csv')

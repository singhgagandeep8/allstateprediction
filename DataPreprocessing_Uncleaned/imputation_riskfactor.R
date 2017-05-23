#read training data rows with no missing values of risk_factor in dftrain
dftrain <-read.csv ("C:/Users/janhv/Documents/Sem 2/Big Data/Project/Train/rf/train_rf_train.csv")
attach(dftrain)
library(nnet)

#multinomial regression using customer characteristics
fit <- multinom(risk_factor~homeowner+married_couple+age_oldest+group_size,data=dftrain)

#Read rows with risk_factor missing values of train and test csv into dftest1 and dftest2
dftest1 <- read.csv("C:/Users/janhv/Documents/Sem 2/Big Data/Project/Train/rf/train_rf_test.csv")
dftest2<-read.csv("C:/Users/janhv/Documents/Sem 2/Big Data/Project/test_v2.csv/rf/test_rf_test.csv")

#predict values and put them in risk_factor column inplace of NA
dftest1$risk_factor <- predict(fit,newdata=dftest1) 
dftest2$risk_factor <- predict(fit,newdata=dftest2) 

#write these to a new csv which contains all imputed values for missing risk_factors
write.csv(dftest1, file="train_complete_rf.csv")
write.csv(dftest2, file="test_complete_rf.csv")

read.csv("train_complete_rf.csv")
read.csv("test_complete_rf.csv")

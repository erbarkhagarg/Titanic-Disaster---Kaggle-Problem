import warnings as wrg
wrg.filterwarnings('ignore')

import pandas as pd
import numpy as np

#import regression model Library
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


titanic_train_data = pd.read_csv('train.csv')
print(len(titanic_train_data))
titanic_train_data.head()


titanic_train_data.info()


#Get the result matrix
result = titanic_train_data['Survived']

#Passenger Name, his gender, ticket, cabin and from where he embarked may not help us predict something useful. Hence picking the rest
#Already in appropriate datatypes, so no conversion needed
titanic_train_selected_fields = titanic_train_data[['PassengerId','Pclass','Age','SibSp','Parch','Fare']]
titanic_train_selected_fields.head()


#Check for null values
titanic_train_selected_fields.isnull().sum()

#Let impute men age for missing values
titanic_train_selected_fields.loc[pd.isnull(titanic_train_selected_fields['Age']),'Age'] = titanic_train_selected_fields.Age.mean()
titanic_train_selected_fields.isnull().sum()

#help(RandomForestClassifier)
test_model = RandomForestClassifier(n_estimators=200)


#Fit the inputs
test_model.fit(titanic_train_selected_fields,result)


accuracy_score(result,test_model.predict(titanic_train_selected_fields))


#Get the Test Data
titanic_test_data = pd.read_csv('test.csv')
titanic_test_data.head()


test_data_selected_variables = titanic_test_data[['PassengerId','Pclass','Age','SibSp','Parch','Fare']]

test_data_selected_variables.isnull().sum()


#Imputing mean to null values
test_data_selected_variables.loc[pd.isnull(test_data_selected_variables['Age']),'Age'] = test_data_selected_variables.Age.mean()
test_data_selected_variables.loc[pd.isnull(test_data_selected_variables['Fare']),'Fare'] = test_data_selected_variables.Fare.mean()
test_data_selected_variables.isnull().sum()


prediction_result = test_model.predict(test_data_selected_variables)
prediction_result




final_dataset = pd.DataFrame({"PassengerId" : test_data_selected_variables.PassengerId, "Survived" : prediction_result})
final_dataset.head()



final_dataset.to_csv("gender_submission.csv", index=False)


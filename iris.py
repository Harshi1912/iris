import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating the SVC model and storing the accuracy score in a variable 'score'.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
score = svc_model.score(X_train, y_train)
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
score1 = rf_model.score(X_train, y_train)
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
score2 = lr_model.score(X_train, y_train)
st.header('IRIS FLOWER PREDICTION APP')
pl=st.sidebar.slider('PETAL LENGTH',float(iris_df['PetalLengthCm'].min()),float(iris_df['PetalLengthCm'].max()))
sw=st.sidebar.slider('SEPAL WIDTH',float(iris_df['SepalWidthCm'].min()),float(iris_df['SepalWidthCm'].max()))
sl=st.sidebar.slider('SEPAL LENGTH',float(iris_df['SepalLengthCm'].min()),float(iris_df['SepalLengthCm'].max()))
pw=st.sidebar.slider('PETAL WIDTH',float(iris_df['PetalWidthCm'].min()),float(iris_df['PetalWidthCm'].max()))
select=st.sidebar.selectbox('CLASSIFIER',['RANDOM FOREST','LOGISTIC REG','SVC'])
@st.cache()
def prediction(model,sl,sw,pl,pw):
	pr=model.predict([[sl,sw,pl,pw]])
	if pr[0]==0:
		return 'Iris-setosa'
	elif pr[0]==1:
		return 'Iris-virginica'
	else:
		return 'Iris-versicolor'
if st.sidebar.button('PREDICT'):
	if select=='RANDOM FOREST':
		rf=prediction(rf_model,sl,sw,pl,pw)
		st.write('the predicted iris flower is:',rf,'-----The aaccuracy of this model is:',score1)
	elif select=='LOGISTIC REGRESSION':
		lr=prediction(lr_model,sl,sw,pl,pw)
		st.write('the predicted iris flower is:',lr,'-----The aaccuracy of this model is:',score2)
	else:
		svc=prediction(svc_model,sl,sw,pl,pw)
		st.write('the predicted iris flower is:',svc,'-----The aaccuracy of this model is:',score)




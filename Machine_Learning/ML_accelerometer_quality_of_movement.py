# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Ali Beheshti
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing#for scaling before PCA
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.externals.six import StringIO  
import pydotplus
import urllib
import urllib.request
url1 = 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
urllib.request.urlretrieve(url1, 'dataset_train_8.csv')
url1 = 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
urllib.request.urlretrieve(url1, 'dataset_test_8.csv')
data_train = pd.read_csv('dataset_train_8.csv')
data_test = pd.read_csv('dataset_test_8.csv')
#explore the data
data_train.info()#Equivalent of str in R
data_train.classe.value_counts()
#remove features with more than 10000 null entries
a=np.asarray(np.where(data_train.isnull().sum()>10000))#dropna is another option
na_indices=a[0][:]
na_indices=np.append(0,na_indices)
data_train=data_train.drop(data_train.columns[na_indices], axis=1)
data_test=data_test.drop(data_test.columns[na_indices], axis=1)
#check if there remains any null value
data_train.isnull().sum().sum()
data_train=data_train.drop(data_train.columns[0:5], axis=1)#contain name, and time stamp
data_test=data_test.drop(data_test.columns[0:5], axis=1)
output_ind= list(data_train).index('classe')
classe_col_train=data_train['classe']
data_train=data_train.drop(data_train.columns[output_ind], axis=1)
data_test=data_test.drop(data_test.columns[output_ind], axis=1)
columns = data_train.columns
selector = VarianceThreshold(0.5)
selector.fit_transform(data_train)#should remove classe column first but keep it somewhere
labels = [columns[x] for x in selector.get_support(indices=True)]
data_train_nzv=pd.DataFrame(selector.fit_transform(data_train), columns=labels)
data_test_nzv=data_test.iloc[:,selector.get_support(indices=True)]
#train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))]) if test and train data are not separate
#do the PCA first
input_train, input_validate,output_train,output_validate = train_test_split(data_train_nzv,
         classe_col_train, test_size=0.3, random_state=0)
std_scale = preprocessing.StandardScaler().fit(input_train)
input_train_std = std_scale.transform(input_train)
input_valid_std = std_scale.transform(input_validate)
input_test_std = std_scale.transform(data_test_nzv)
pca_std = PCA(n_components=0.9).fit(input_train_std)
input_train_std = pca_std.transform(input_train_std)
input_valid_std = pca_std.transform(input_valid_std)
data_test_nzv_std=pca_std.transform(data_test_nzv)
dt = tree.DecisionTreeClassifier()#can provide depth value in ()
model=dt.fit(X=input_train_std,y=output_train)
predictions = cross_val_predict(model, X=input_valid_std, y=output_validate, cv=10)#njobs for number of CPUs
print('Decision Tree Score:',metrics.accuracy_score(output_validate,predictions))
#print “Score:”, model.score(X_valid, y_valid)
#Visualize Tree
tree.export_graphviz(model,
     out_file='tree.dot') 
dot_data = StringIO() 
tree.export_graphviz(model, out_file=dot_data) 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
graph.write_png("decision_tree.png") 
#fit random forest
rf = RandomForestClassifier()
model_rf=rf.fit(X=input_train_std,y=output_train)
predictions_rf = cross_val_predict(model_rf, X=input_valid_std, y=output_validate, cv=10)
print('Random Forest Score:',metrics.accuracy_score(output_validate,predictions_rf))
#digitRecognizer using k-nearest neighbour algorithm fom shogun
import numpy as np
import pandas as pd
import csv as csv
from modshogun import MulticlassLabels, RealFeatures
from modshogun import KNN, EuclideanDistance

#header=0 makes zeroth row in csv as coulumn names
train_db=pd.read_csv("train.csv",header=0)
test_db=pd.read_csv("test.csv",header=0)

#use k-nn of shogun to train this data
print 'Training...'

#converting train data to numpy array
train_data=train_db.values
#mat  = loadmat('../../../data/multiclass/usps.mat')
x_train=np.array(train_data[0::,1::],dtype=np.double)
#squeeze function is used to convert a matrix to array and MultiClassLabels requires a vector,not a matrix
y_train=np.array(train_data[0::,0].squeeze(),dtype=np.double)

#required for shogun's algorithm
labels=MulticlassLabels(y_train)
feats=RealFeatures(x_train.T)
k=3

#training the data using knn algorithm
dist=EuclideanDistance()
knn=KNN(k,dist,labels)
knn.train(feats)


print 'Predicting...'

#converting test data to numpy array
test_data=test_db.values
#converting x_test to numpy array
x_test=np.array(test_data[0::,0::],dtype=np.double)

#applysing the knn shogun algorithm on to predict values in the test file
feats_test=RealFeatures(x_test.T)
output = knn.apply_multiclass(feats_test)

#size of input
count=x_test.shape[0]

#creating the ids array
ids=[]
for i in range(1,count+1):
	ids.append(i)

#getting the integer values of the output
int_out=[]
for i in range(0,count):
	int_out.append(int(output[i]))
	
#put the output in a file
predictions_file = open("knn_shogun.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["ImageId","Label"])
open_file_object.writerows(zip(ids,int_out))
predictions_file.close()
print 'Done.'




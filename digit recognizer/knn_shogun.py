#digitRecognizer using k-nearest neighbour algorithm fom shogun
import numpy as np
import pandas as pd
import csv as csv
#from sklearn.neighbors import KNeighborsClassifier
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

#y_train=train_data[0::,0]
labels=MulticlassLabels(y_train)
feats=RealFeatures(x_train.T)
k=3

dist=EuclideanDistance()
knn=KNN(k,dist,labels)
knn.train(feats)


print 'Predicting...'

#converting train data to numpy array
test_data=test_db.values

x_test=np.array(test_data[0::,0::],dtype=np.double)


feats_test=RealFeatures(x_test.T)
output = knn.apply_multiclass(feats_test)


count=x_test.shape[0]
ids=[]
for i in range(1,count+1):
	ids.append(i)
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




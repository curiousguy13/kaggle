#digitRecognizer using k-nearest neighbour algorithm fom shogun
import numpy as np
import pandas as pd
import csv as csv
from sklearn.neighbors import KNeighborsClassifier
from modshogun import MulticlassLabels, RealFeatures
from modshogun import KNN, EuclideanDistance

#header=0 makes zeroth row in csv as coulumn names
train_db=pd.read_csv("train.csv",header=0)
test_db=pd.read_csv("test.csv",header=0)

#use k-nn of shogun to train this data
print 'Training...'

#converting train data to numpy array

train_data=train_db.values
x_train=train_data[0::,1::]
y_train=train_data[0::,0]
labels=MulticlassLabels(y_train)
feats=RealFeatures(x_train)
k=3

dist=EuclideanDistance()
knn=KNN(k,dist,labels)
knn.train(feats)


print 'Predicting...'

#converting train data to numpy array
test_data=test_db.values
x_test=test_data[0::,1::]
y_test=test_data[0::,0]

output = knn.apply_multiclass(test_data).astype(int)


#put the output in a file
predictions_file = open("knn_shogun.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["ImageId","Label"])
open_file_object.writerows(zip(ids,output))
predictions_file.close()
print 'Done.'




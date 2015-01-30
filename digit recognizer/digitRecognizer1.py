#digitRecognizer using k-nearest neighbour algorithm fom scikit-learn
import numpy as np
import pandas as pd
import csv as csv
from sklearn.neighbors import KNeighborsClassifier

#header=0 makes zeroth row in csv as coulumn names
train_db=pd.read_csv("train.csv",header=0)
test_db=pd.read_csv("test.csv",header=0)

#use k-nn of scikit-learn to train this data
print 'Training...'

#converting train data to numpy array
train_data=train_db.values
neigh=KNeighborsClassifier(n_neighbors=3)

#passing the data to the classifier
neigh=neigh.fit(train_data[0::,1::],train_data[0::,0])

print 'Predicting...'

#converting train data to numpy array
test_data=test_db.values

output = neigh.predict(test_data).astype(int)


#put the output in a file
predictions_file = open("knn_sklearn.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["label"])
open_file_object.writerows(zip(output))
predictions_file.close()
print 'Done.'




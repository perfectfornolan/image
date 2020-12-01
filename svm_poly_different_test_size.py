# import the essential library
import numpy as np
import sklearn.metrics as metrics 
from sklearn import svm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# gather the original data from the sklearn datasets
mnist = fetch_openml('mnist_784')

# get the x data  with suitable format
x_origin = mnist.data.astype('float32')
# show corresponding range
print("Image Array")
print(x_origin)
print("Image Array Range")
print(x_origin.min(), x_origin.max())

# get the y label  with suitable format
y_origin = mnist.target.astype('int64')
# show corresponding range
print("Label Array")
print(y_origin)
print("Label Array Range")
print(y_origin.min(), y_origin.max())

# normalize the data to be in range (0,1)
x_origin /= x_origin.max()
print("New Image Array Range")
print(x_origin.min(), x_origin.max())
print("Label Array Range (not change)")
print(y_origin.min(), y_origin.max())

# set kernel to be Poly
# test different test size by using for loops
# test size = 10 %, 20 %, ... , 90%
# example
# set test size equals to 90%
# Meaning: 10% training data, 90% training

for t_size in range(1, 10):
  x_train,x_test,y_train,y_test = train_test_split(x_origin,y_origin,test_size= t_size /10)
  print("Case:", t_size)
  print("Test size = ", t_size * 10, "%")
  model_poly = svm.SVC(kernel='poly')
  model_poly.fit(x_train,y_train)
  model_poly_score = model_poly.score(x_test,y_test)
  print("Score for kernel='Poly'",model_poly_score)
  y_predicted=model_poly.predict(x_test)
  print(metrics.classification_report(y_test, y_predicted))

# dataset is mnist_784
# Reference: https://www.openml.org/d/554
# The MNIST database of handwritten digits with 784 features, raw data available at: http://yann.lecun.com/exdb/mnist/. 
# It can be split in a training set of the first 60,000 examples, and a test set of 10,000 examples
# It is a subset of a larger set available from NIST.

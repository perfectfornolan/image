# import the essential library
import numpy as np
import sklearn.metrics as metrics 
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# gather the original data from the sklearn datasets
mnist = fetch_openml('mnist_784')

# get the x data  with suitable format
x_origin = mnist.data.astype('float32')
print("Original Image Shape")
print(x_origin.shape)
print("Original Image Array")
print(x_origin)
print("Original Image Array Range")
print(x_origin.min(), x_origin.max())

# get the y label  with suitable format
y_origin = mnist.target.astype('int64')
print("Original Label Shape")
print(y_origin.shape)
print("Original Label Array")
print(y_origin)
print("Original Label Array Range")
print(y_origin.min(), y_origin.max())

# normalize the data to be in range (0,1)
x_origin /= x_origin.max()
print("New Image Array Range")
print(x_origin.min(), x_origin.max())
print("Label Array Range (maintain the same)")
print(y_origin.min(), y_origin.max())

# separate the data into two groups: 10% for training, 90% for testing
# the test size can be changed
x_train,x_test,y_train,y_test = train_test_split(x_origin,y_origin, test_size = 0.9)
print("New Image Array for Training")
print(x_train.shape)
print("New Image Array for Testing")   
print(x_test.shape)
print("New Label Train Array for Training")     
print(y_train.shape)
print("New Label Test Array for Testing")     
print(y_test.shape)

# plot the cumulative explained variance and the three Explained Variance
pca = PCA().fit(x_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of reuqired components")
plt.ylabel("Cumulative explained variance")
plt.title("The required amount of Components to acheive certain variance")
plt.axhline(y = 0.95, color='c', linestyle='-', label = '95% Explained Variance')
plt.axhline(y = 0.85, color='m', linestyle='-', label = '85% Explained Variance')
plt.axhline(y = 0.75, color='y', linestyle='-', label = '75% Explained Variance')
plt.legend(loc='best')
plt.show

# find the extact value to achive the required variance
variance_pca = [0.75, 0.85, 0.95]
for x in variance_pca:
  pca = PCA(x)
  pca.fit(x_train)
# index start from zero, +1 is required
  num_component = pca.n_components_ + 1   
  print("To achieve", x*100,"% variance,")
  print("The number of component can decrease from the original",x_train.shape[1] , "to", num_component)

# plot two dimentional graph with the labels although the variance is not high
# set number of components to be 2
pca = PCA(n_components=2)
pca_x_train = pca.fit_transform(x_train)
print("Original training data shape:   ", x_train.shape)
print("Transformed  training data shape:", pca_x_train.shape)

# create dataframe to include two key compoents and the corresponding labels

pca_train_Df = pd.DataFrame(data = pca_x_train
             , columns = ['Key Component:1', 'Key Component:2'])
pca_train_Df['Label'] = y_train
pca_train_Df.tail()

# show the two dimentional graph with the label

plt.figure(figsize=(20,20))
sns.scatterplot(
    x="Key Component:1", y="Key Component:2",
    hue="Label",
    palette=sns.color_palette("hls", 10),
    data=pca_train_Df,
    legend="full",
    alpha=0.5
)

# dataset is mnist_784
# Reference: https://www.openml.org/d/554
# The MNIST database of handwritten digits with 784 features, raw data available at: http://yann.lecun.com/exdb/mnist/. 
# It can be split in a training set of the first 60,000 examples, and a test set of 10,000 examples
# It is a subset of a larger set available from NIST.

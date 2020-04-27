# Making Case Study - Make a hybrid Deep Learning Model
"""
Hybrid Neural NEtwork using an ANN and a SOM
The SOM does not require a dependant variable and allows us to get a list of the customers that are outliners,
or the customers that outline the rules of the application and that we need to keep an eye because there is 
a high change that they cheated or lie, independently if their application was approved yet or not.
The SOM identifies possible frauds, then an array that is the dependent variable for the ANN, will tell us
which customers are likely to commit fraud given that their applications are outliners. We would never be able
to tell if a customer application is fraudulent if it wasn't for the SOM. 
The ANN will be trained using this dependent variable and the features of the customers serving as
independent variables.
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




# Part 1 - Indentify the frauds with the self-organizing maps

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values 

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1)) #Normalization from 0 to 1
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom  #Using Minisom class from Minisom.py, a library designed for SOM
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 2.5, learning_rate = 0.5)
som.train_random(data = X, num_iteration = 1000)

# Visualising the results
from pylab import bone, pcolor, colorbar, plot, show
bone() # initialising the window that will contain the map
pcolor(som.distance_map().T) #distance_map will return all the MID in one matrix, but we need the Transpose of this matrix
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
# Red circles are the markers of customer who didn't get approval. Green squares are the ones who did get approval
for i, x in enumerate(X): # 'i' is the index for the rows and x will correspond to the vector containing all the information of each row
    W = som.winner(x) #Winner method gets us the winning node for a customer (each customer is one row)
    plot(W[0] + 0.5, # Moving the marker to the center of the x coordinate of the winning node
         W[1] + 0.5, # Moving the marker to the center of the y coordinate of the winning node
         markers[y[i]], # Acoording to the information in y, for each specific index, the marker square will be put on the winning node when the customer got an approval (number 1) or a circle when it got declined
         markeredgecolor = colors[y[i]], #green if customer got approval, red if it didn't
         markerfacecolor = 'None', # Color of the face of the shape
         markersize = 15,
         markeredgewidth = 3)
show()

# Finding the frauds
mappings = som.win_map(X) # Gives a dictionary of all the winning nodes for our data. 
frauds = np.concatenate((mappings[(5,6)], mappings[(5,7)]), axis = 0) # Axis 0 is vertically
# We visualize in the map which nodes are the whitest, and we obtain the information from those nodes
frauds = sc.inverse_transform(frauds) # Method by the sc object to inverse the feature scaling


# Part 2 - Going from unsupervised to supervised Deep Learning

# Creating the matrix of features
customers = dataset.iloc[:, 1:].values # we exclude the customer ID and include the information of wether the customer's application was approved or not

# Creating the dependent variable
is_fraud = np.zeros(len(dataset)) #Dependent variable, initialized in 0. 
for i in range (len(dataset)):
    if dataset.iloc[i,0] in frauds: #This will look if the customer ID is in the list of frauds
        is_fraud[i] = 1 # if the customer ID in an specific index matches with the one in the list of frauds, then it will populate that index with a  1 in the array

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)
 
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))
# classifier.add(Dropout(p = 0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 3, epochs = 10) 
#Due to the simplicity of the model, it does not require too many neurons or epochs to converge 

# Predicting the probabilities of fraud results
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:,0:1].values, y_pred), axis = 1)
# this last step give us an array with the customer ID and their probability that they will cheat
y_pred = y_pred[y_pred[:,1].argsort()] # This step sorts the probabilities with and keeping the customer ID to which they belong to




TASK 1 - EMAIL SPAM DETECTION WITH MACHINE LEARNING 
 
INTRODUCTION

In today’s world, email has become a crucial way for people to communicate. Spam is those annoying emails we didn’t ask for, like ads or scams, that fill up our email inboxes and that is sent to a massive number of users at one time. In this Project,use Python to build an email spam detector. Then, use machine learning to train the spam detector to recognize and classify emails into spam and non-spam. Let’s get started!

FEATURES
 
Utilizes Python for building the email spam detector. Applies machine learning algorithms for training and classification. Implements label encoding to transform categorical data. Uses Logistic Regression for the classification task. Allows for custom prediction inputs.

INSTALLATION
 
To run the email spam detector, please ensure you have the following dependencies installed:

•Python 3.x
•numpy
•pandas
•scikit-learn

********************************************************************

TASK -2  IRIS FLOWER CLASSIFICATION

INTRODUCTION

This repository contains the code for training a machine learning model to classify Iris flowers based on their measurements. The Iris flower dataset consists of three species: Setosa, Versicolor, and Virginica, each having distinct measurement characteristics. The goal is to develop a model that can accurately classify Iris flowers based on their measurements.

DATASET

The dataset used for this project is the famous Iris flower dataset, which is commonly used for classification tasks. It includes measurements of sepal length, sepal width, petal length, and petal width for 150 Iris flowers, with 50 samples for each species. The dataset is available in the repository as iris.csv.

DEPENDENCIES

The following Python libraries are used in this project:

*NumPy
*Pandas
*Seaborn
*Matplotlib
*Model Training

Three different models are trained:

*Support Vector Machine (SVM)
*Logistic Regression
*Decision Tree Classifier
*Each model is trained using the Iris flower dataset and evaluated for its accuracy

TESTING

After training the models, a new test dataset is used to assess their performance. The test dataset contains measurements of Iris flowers with unknown species. The trained models predict the species of these flowers, and their accuracy is evaluated.


******************************************************************


TASK  3 -  SALES PREDICTION WITH PYTHON 


OBJECTIVE

The aim of this project is to predict sales based on advertising expenditure using the given dataset. The dataset contains information about advertising spending on different platforms (TV, Radio, and Newspaper) and the corresponding sales amount.


LIBRARIES USED

The following important libraries were used for this project:

-numpy
-pandas
-matplotlib.pyplot
-seaborn
-sklearn.model_selection.train_test_split
-sklearn.linear_model.LinearRegression

DATASET

The dataset was loaded using pandas as a DataFrame from the file "sales.csv".

DATA PROCESSING

The shape and descriptive statistics for the dataset were displayed using df.shape and df.describe().
A pair plot was created to visualize the relationship between advertising expenditure on TV, Radio, Newspaper, and sales using seaborn.pairplot.
Histograms were plotted to observe the distribution of advertising expenditure on TV, Radio, and Newspaper using matplotlib.pyplot.hist.

ANALYSIS

A correlation matrix heatmap was plotted to observe the correlation between advertising expenditure on TV, Radio, Newspaper, and sales using seaborn.heatmap.

MODEL TRAINING

The data was split into training and testing sets using train_test_split.
Linear regression model was trained on the training data using sklearn.linear_model.LinearRegression.

MODEL PREDICTION 

The model was used to predict sales based on advertising expenditure on TV for the test set using model.predict(X_test) and the corresponding advertising expenditure on TV in the test set.
The model coefficients and intercept were obtained using model.coef_ and model.intercept_.
The predictions and actual sales values were plotted using matplotlib.pyplot.plot and matplotlib.pyplot.scatter

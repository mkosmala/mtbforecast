#!/usr/bin/env python

import sys
import csv

#from datetime import date, timedelta

from math import sqrt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
#from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


import pickle

#from sklearn import datasets

import matplotlib.pyplot as plt
from matplotlib import rcParams
#from mlxtend.plotting import plot_decision_regions

# -*- coding: utf-8 -*-
"""
Create logistic regression model to analyze data 

Created on Wed Jan 17 10:59:55 2018

@author: mkosmala
"""


def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    
    rcParams.update({'font.size': 14})

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);
    
    ax.tick_params(axis='x', rotation=70)    
    
    #plt.show()
    
    plt.savefig("correlations_before.png",bbox_inches='tight')
    
    
def prepare_data(data):
    
    # features only; remove the other coloumns
    feature_data = data.drop(['rider','date','rode','possible_riders','actual_riders',
                              'fraction','distance','speed'],
                              axis=1)
    
    # we're going to drop min and max temp
    feature_data = feature_data.drop(['min_temp','max_temp',
                                      'prev_ave_temp','prev_sunshine'],axis=1)
    
    # retained columns are:
    # year, doy, weekend, precip, snow_depth, ave_temp, sunshine, peak_wind,
    # prev_peak_wind, prev_precip, prev_temp_diff
    
    
    # feature names
    feature_labels = feature_data.columns
    
    # we lose a ton of data if we delete rows with missing data
    # so, instead, impute missing data as column means
    # The sklearn Imputer doesn't play well with pandas. So I'll do this by hand.
    col_ave = feature_data.mean(axis=0,skipna=True,numeric_only=True)
    imputed_data = feature_data.fillna(value=col_ave,axis=0)
    
    # now standardize the data, so each feature column falls between 0 and 1
    # except for the first
    scaler = MinMaxScaler()
    transformed = scaler.fit_transform(imputed_data)
    scaled_data = pd.DataFrame(transformed,
                               columns=imputed_data.columns)
    
    #scaler = StandardScaler()
    #transformed = scaler.fit_transform(imputed_data)
    #scaled_data = pd.DataFrame(transformed,columns=imputed_data.columns)
    
    # show correlations
    plot_corr(feature_data)    
    
    return [scaled_data,feature_data,feature_labels,scaler]


def split_into_testing_and_training(splitfilename,all_x,all_y):

    # split into testing and training datasets
    testtrain = {}
    with open(splitfilename,'r') as sfile:
        sreader = csv.reader(sfile)
        for row in sreader:
            #testtrain.append(1-int(row[1])) # assumes already sorted
            testtrain[row[0]] = 1-int(row[1])
    
    
    #y_train = np.extract(testtrain,all_y)
    # for some reason extract doesn't work on multi-dim data  :-P
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    act_fract = []
    train_act_fract = []
    for i in range(0,len(all_x)):
        d = data["date"][i]
        tt = testtrain[d]
        if tt==1:
            x_train.append(all_x[i])
            y_train.append(all_y[i])
            train_act_fract.append(data["fraction"][i])
        else:
            x_test.append(all_x[i])
            y_test.append(all_y[i])
            act_fract.append(data["fraction"][i])
    
    return [x_train,y_train,x_test,y_test,train_act_fract,act_fract]    


def print_accuracies(act_fract,pred_fract):
        
    rmse = sqrt(mean_squared_error(act_fract,pred_fract))
    print("RMSE")
    print(rmse)
    
    #train_predictions = logreg.predict_proba(x_train)
    #train_pred_fract = train_predictions[:,1]
    #rmse = sqrt(mean_squared_error(train_act_fract,train_pred_fract))
    #print(rmse)
    
    correct = 0
    close = 0
    for a,p in zip(act_fract,pred_fract):
        if a<0.1 and p<0.1:
            correct += 1
        elif a>=0.1 and a<0.16 and p>=0.1 and p<0.16:
            correct += 1
        elif a>=0.16 and a<0.24 and p>=0.16 and p<0.24:
            correct += 1
        elif a>=0.24 and p>=0.24:
            correct += 1
    
        if a<0.16 and p<0.16:
            close += 1
        elif a>=0.1 and a<0.24 and p>=0.1 and p<0.24:
            close += 1
        elif a>=0.16 and p>=0.16:
            close += 1        
            
    accuracy = correct*1.0/len(act_fract)
    print("Group accuracy")
    print(accuracy)
    accuracy_close = close*1.0/len(act_fract)
    print("Within-1-bin accuracy")
    print(accuracy_close)
    
    
def print_coefficients(feature_labels,logreg):
    
    print("Coefficients")
    print(feature_labels)
    print(logreg.coef_)


def plot_prediction_distribution(pred_fract):

    rcParams.update({'font.size': 16})

    # distributions of predictions
    histbins = np.arange(0,0.4,0.01).tolist()
    plt.hist(pred_fract,bins=histbins)
    plt.xlabel("Predicted fraction of riders")
    plt.ylabel("Number of days")
    #plt.show()
    plt.savefig("pred_dist.png",bbox_inches='tight')
    
def plot_fit_of_predictions(act_fract,pred_fract):

    plt.plot([-1,0.1,0.1],[0.1,0.1,-1],color="red",linestyle='dashed')
    plt.plot([0.1,0.16,0.16,0.1,0.1],[0.1,0.1,0.16,0.16,0.1],color="red",linestyle='dashed')
    plt.plot([0.16,0.24,0.24,0.16,0.16],[0.16,0.16,0.24,0.24,0.16],color="red",linestyle='dashed')
    plt.plot([0.24,0.24,0.6],[0.6,0.24,0.24],color="red",linestyle='dashed')
    
    plt.plot([-1,1],[-1,1])
    plt.scatter(act_fract,pred_fract)
    plt.xlim(-0.01,0.6)
    plt.ylim(-0.01,0.5)
    plt.xlabel("Actual fraction of riders")
    plt.ylabel("Predicted fraction of riders")
    #plt.show()
    plt.savefig("rmse_binned.png",bbox_inches='tight')
    
def save_model_as_pickle(feature_data,logreg,scaler):

    # save the model as a pickle
    # tutorial here: https://xcitech.github.io/tutorials/heroku_tutorial/
    index_dict = dict(zip(feature_data,range(feature_data.shape[1])))
    with open('pickle/logistic_regression_trials_model.pkl','wb') as fileid:
        pickle.dump(logreg,fileid)
    with open('pickle/categories_trials','wb') as fileid:
        pickle.dump(index_dict,fileid)
    with open('pickle/transformations_trials','wb') as fileid:
        pickle.dump(scaler,fileid)
    


    
# combined file = file that has data about riders and data about weather
# split file = file that tells how to split into testing and training data
if len(sys.argv) < 3 :
    print ("format: logistic_regression_multi.py <combined file> <split file>")
    exit(1)

infilename = sys.argv[1]
splitfilename = sys.argv[2]
#outfilename = sys.argv[3]

# read in the data file and put it in the right format
data = pd.read_csv(infilename, header = 0)
#original_headers = list(data.columns.values)
#print(original_headers)

# do data transformations as needed
scaled_data, feature_data, feature_labels, scaler = prepare_data(data)

# X's are features
all_x = scaled_data.as_matrix()

# Y's are rode or didn't ride
all_y = np.transpose(data.as_matrix(['rode']))[0]

# split data in to training and testing subsets
x_train, y_train, x_test, y_test, train_act_fract, act_fract = split_into_testing_and_training(splitfilename,all_x,all_y)

# try multi-class
# the larger the C, the weaker the regularization
# (see pp.73-76 in Python Machine Learning)
# takes numpy arrays
logreg = LogisticRegression(C=1)
logreg.fit(x_train,y_train)

# what we want to compare is fraction of folks who rode vs. the 
# expected fraction of folks who rode (i.e. the predicted probability of a ride))
predictions = logreg.predict_proba(x_test)
pred_fract = predictions[:,1]

# print some stuff
print_accuracies(act_fract,pred_fract)
print_coefficients(feature_labels,logreg)

# do some plotting
plot_prediction_distribution(pred_fract)
plt.clf() # clear
plot_fit_of_predictions(act_fract,pred_fract)


# save the model as a pickle for predictions
save_model_as_pickle(feature_data,logreg,scaler)


#pip install -U numpy==1.18.5

from flask import Flask
from flask import render_template,request
from datetime import time
import pandas as pd1

from datetime import datetime
from datetime import timedelta
import requests
# from bs4 import BeautifulSoup
import pandas as pd
import datetime
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from bs4 import BeautifulSoup 
import csv 
import re

import numpy
#import tflearn
import tensorflow
import tweepy as tw
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

from flask_wtf import Form
from wtforms.fields import DateField
#importing required libraries
import numpy as np

#to plot within notebook
import matplotlib.pyplot as plt
from flask import jsonify 
from tensorflow.keras.layers import Dropout, LSTM
import requests_html 
import tensorflow as tf

from sklearn.metrics import confusion_matrix
import seaborn as sns

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB * 2 of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 2)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

global dt4 
global dst
global ttf4
ttf4=[]
dt4=[]

# function to calculate percentage difference considering baseValue as 100%
def percentageChange(baseValue, currentValue):
    return((float(currentValue)-baseValue) / abs(baseValue)) *100.00

# function to get the actual value using baseValue and percentage
def reversePercentageChange(baseValue, percentage):
    return float(baseValue) + float(baseValue * percentage / 100.00)

# function to transform a list of values into the list of percentages. For calculating percentages for each element in the list
# the base is always the previous element in the list.
def transformToPercentageChange(x):
    baseValue = x[0]
    x[0] = 0
    for i in range(1,len(x)):
        pChange = percentageChange(baseValue,x[i])
        baseValue = x[i]
        x[i] = pChange

# function to transform a list of percentages to the list of actual values. For calculating actual values for each element in the list
# the base is always the previous calculated element in the list.

dictionaryofdateandprice={}

def reverseTransformToPercentageChange(baseValue, x):
    x_transform = []
    for i in range(0,len(x)):
        value = reversePercentageChange(baseValue,x[i])
        baseValue = value
        x_transform.append(value)
    return x_transform

def ScrpLiveData(stock):
    
    from datetime import datetime
    response = requests.get('https://api.coingecko.com/api/v3/coins/'+stock+'/market_chart',
                            params={'vs_currency': 'usd', 'days': '360'})

    if response.status_code == 200:
        data = response.json()
        prices = data['prices']
    else:
        print('Error occurred while fetching data:', response.status_code)
        exit()

    with open('livedata/'+stock+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Date', 'Start', 'Close'])
        
        for entry in prices:
            timestamp = entry[0] / 1000  # Convert milliseconds to seconds
            date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')  # Convert timestamp to date
            start_price = entry[1]
            close_price = entry[1]
            writer.writerow([date, start_price, close_price])

dictofdateandprice={}
def predictpriceofdata(stockname):
    global dictionaryofdateandprice
    global date_index
    global train_transform
    global future_date_index
    global future_closing_price_transform
    global label1
    global label2
    global dt4 
    global ttf4 
    global predictresult 

    df = pd.read_csv('livedata\\'+stockname+'.csv')
# store the first element in the series as the base value for future use.
    baseValue = df['Close'][0]

# create a new dataframe which is then transformed into relative percentages
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
    for i in range(0,len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Close'][i] = data['Close'][i]

# transform the 'Close' series into relative percentages
    transformToPercentageChange(new_data['Close'])

# set Dat column as the index
    new_data.index = new_data.Date
    new_data.drop('Date', axis=1, inplace=True)

# create train and test sets
    dataset = new_data[0:1500].values
    # print("====dataset====")
    # print(dataset)
    print("====LSTM Prediction Results====")
    print(len(dataset))
    train, valid = train_test_split(dataset, train_size=0.99, test_size=0.01, shuffle=False)

# convert dataset into x_train and y_train.
# prediction_window_size is the size of days windows which will be considered for predicting a future value.

    prediction_window_size = 60
    x_train, y_train = [], []
    for i in range(prediction_window_size,len(train)):
        x_train.append(dataset[i-prediction_window_size:i,0])
        y_train.append(dataset[i,0])
    x_train, y_train = np.array(x_train).astype(np.float32), np.array(y_train).astype(np.float32)
       
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


    # X_train = []
    # y_train = []
    # for i in range(60, 1500):
    #     X_train.append(np.array(dataset[60:1600].astype(np.float32))[i-60:i])
    #     y_train.append(np.array(dataset[60:1600].astype(np.float32))[i])
    # X_train, y_train = np.array(X_train), np.array(y_train)
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    x_valid, y_valid = [], []
    for i in range(60,120):
        x_valid.append(dataset[i-prediction_window_size:i,0])
        y_valid.append(dataset[i,0])
        
    X_test = np.asarray(x_valid).astype('float32')
    y_test = np.asarray(y_valid).astype('float32')

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))


##################################################################################################
# create and fit the LSTM network
# Initialising the RNN
    model = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    # mode=8
# Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))

# Adding the output layer
    model.add(Dense(units = 1))
# Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
   
    
    # final_acuracy=accu*mode
    # print(final_acuracy)
# Fitting the RNN to the Training set
    model.fit(x_train, y_train, epochs = 10, batch_size = 16)
    # def predict_prob(number):
    #   return [number[0],1-number[0]]
    
    # y_prob = np.array(list(map(predict_prob, model_cEXT.predict(X_test))))
##################################################################################################
#predicting future values, using past 60 from the train data
# for next 10 yrs total_prediction_days is set to 3650 days
    total_prediction_days = 3650
    inputs = new_data[-total_prediction_days:].values
    # print("======len(new_data)==========")
    # print(len(new_data))
    # print("======len(inputs)==========")
    # print(len(inputs))
    
    inputs = inputs.reshape(-1,1)

# create future predict list which is a two dimensional list of values.
# the first dimension is the total number of future days
# the second dimension is the list of values of prediction_window_size size
    X_predict = []
    for i in range(prediction_window_size,inputs.shape[0]):
        X_predict.append(inputs[i-prediction_window_size:i,0])
    X_predict = np.array(X_predict).astype(np.float32)
   
# predict the future
    X_predict = np.reshape(X_predict, (X_predict.shape[0],X_predict.shape[1],1))
    future_closing_price = model.predict(X_predict)   
    
    
    print("-----------------------------------------")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    accu = model.evaluate(X_test,y_test)
    print(f"Mean Squared Error Loss: {accu:.4f}")
    # Calculate Root Mean Squared Error (RMSE)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    threshold = 0.0 
    binary_predictions = (predictions > threshold).astype(int)
    binary_y_test = (y_test > threshold).astype(int)
    
    accuracy = accuracy_score(binary_y_test, binary_predictions)
    accuracy1 =accuracy*100
    print(accuracy1)
    precision = precision_score(binary_y_test, binary_predictions)
    recall = recall_score(binary_y_test, binary_predictions)
    f1 = f1_score(binary_y_test, binary_predictions)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    predictresult={"Accuracy":accuracy1,"Precision":precision,"Recall":recall,"F1-score":f1,"RMSE":rmse,"MAE":accu}
    print(predictresult)
    
    print("-----------------------------------------")
    
   
    train, valid = train_test_split(new_data, train_size=0.99, test_size=0.01, shuffle=False)
    date_index = pd.to_datetime(train.index)

#converting dates into number of days as dates cannot be passed directly to any regression model
    x_days = (date_index - pd.to_datetime('1970-01-01')).days

# we are doing prediction for next 5 years hence prediction_for_days is set to 1500 days.
    prediction_for_days = 1000
    future_closing_price = future_closing_price[:prediction_for_days]

# create a data index for future dates
    x_predict_future_dates = np.asarray(pd.RangeIndex(start=x_days[-1] + 1, stop=x_days[-1] + 1 + (len(future_closing_price))))
    future_date_index = pd.to_datetime(x_predict_future_dates, origin='1970-01-01', unit='D')

# transform a list of relative percentages to the actual values
    train_transform = reverseTransformToPercentageChange(baseValue, train['Close'])

# for future dates the base value the the value of last element from the training set.
    baseValue = train_transform[-1]
    valid_transform = reverseTransformToPercentageChange(baseValue, valid['Close'])
    future_closing_price_transform = reverseTransformToPercentageChange(baseValue, future_closing_price)

# recession peak date is the date on which the index is at the bottom most position.
    recessionPeakDate =  future_date_index[future_closing_price_transform.index(min(future_closing_price_transform))]
    minCloseInFuture = min(future_closing_price_transform)
    print("The stock market will reach to its lowest bottom on", recessionPeakDate)
    print("The lowest index the stock market will fall to is ", minCloseInFuture)

    
# plot the graphs
    label1='Close Price History of'+ stockname +'company'
    label2='Predicted Close of'+ stockname +'company'
    plt.figure(figsize=(16,8))
    df_x = pd.to_datetime(new_data.index)
    plt.plot(date_index,train_transform, label=label1)
    plt.plot(future_date_index,future_closing_price_transform, label=label2)

# set the title of the graph
    plt.suptitle('Stock Market Predictions', fontsize=16)

# set the title of the graph window
    # fig = plt.gcf()
    # fig.canvas.set_window_title('Stock Market Predictions')

#display the legends
    plt.legend()

#display the graph
    #plt.show()
    plt.savefig('static/futuregraph/abc.png')
    
    dictofdateandprice={}
    try:
        for i in range(38,875):
            #date_object = datetime.datetime.strptime(str(future_date_index[i]), '%y/%m/%d')
            datetimeobt=str(future_date_index[i]).split(" ")
            # print("-------------datetimeobt-------")
            # print(datetimeobt)
            dictionaryofdateandprice[datetimeobt[0]]=future_closing_price_transform[i]
            # print("-------------dict-------")
            # print(dictionaryofdateandprice)
            # print('date obtained',str(datetimeobt[0]))
            dictofdateandprice[str(future_date_index[i])]=future_closing_price_transform[i]
            # print("------------dictofdateandprice-------")
            # print(dictofdateandprice)
    except:
        for i in range(38,215):
            #date_object = datetime.datetime.strptime(str(future_date_index[i]), '%y/%m/%d')
            datetimeobt=str(future_date_index[i]).split(" ")
            # print("-------------datetimeobt-------")
            # print(datetimeobt)
            dictionaryofdateandprice[datetimeobt[0]]=future_closing_price_transform[i]
            # print("-------------dict-------")
            # print(dictionaryofdateandprice)
            # print('date obtained',str(datetimeobt[0]))
            dictofdateandprice[str(future_date_index[i])]=future_closing_price_transform[i]
            # print("------------dictofdateandprice-------")
            # print(dictofdateandprice)
        
    dt4 = date_index.append(future_date_index)

    ttf4 = train_transform + future_closing_price_transform
    return jsonify(dictofdateandprice), ttf4, dt4,predictresult



dictofdateandprice2={}
from tensorflow.keras.layers import Bidirectional
def BiDirectionalLSTM(stockname):
    global dictionaryofdateandprice
    global date_index
    global train_transform
    global future_date_index
    global future_closing_price_transform
    global label1
    global label2
    global BDLSTMttf 
    global BDLSTMdt
    global predictresult2

    df = pd.read_csv('livedata\\'+stockname+'.csv')
# store the first element in the series as the base value for future use.
    baseValue = df['Close'][0]

# create a new dataframe which is then transformed into relative percentages
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
    for i in range(0,len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Close'][i] = data['Close'][i]

# transform the 'Close' series into relative percentages
    transformToPercentageChange(new_data['Close'])

# set Dat column as the index
    new_data.index = new_data.Date
    new_data.drop('Date', axis=1, inplace=True)

# create train and test sets
    dataset = new_data[0:1500].values
    # print("====dataset====")
    # print(dataset)
    print("====Bidirectional LSTM Prediction====")
    print(len(dataset))
    train, valid = train_test_split(dataset, train_size=0.99, test_size=0.01, shuffle=False)

# convert dataset into x_train and y_train.
# prediction_window_size is the size of days windows which will be considered for predicting a future value.

    prediction_window_size = 60
    x_train, y_train = [], []
    for i in range(prediction_window_size,len(train)):
        x_train.append(dataset[i-prediction_window_size:i,0])
        y_train.append(dataset[i,0])
    x_train, y_train = np.array(x_train).astype(np.float32), np.array(y_train).astype(np.float32)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


    # X_train = []
    # y_train = []
    # for i in range(60, 1500):
    #     X_train.append(np.array(dataset[60:1600].astype(np.float32))[i-60:i])
    #     y_train.append(np.array(dataset[60:1600].astype(np.float32))[i])
    # X_train, y_train = np.array(X_train), np.array(y_train)
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



    x_valid, y_valid = [], []
    for i in range(60,120):
        x_valid.append(dataset[i-prediction_window_size:i,0])
        y_valid.append(dataset[i,0])
        
    X_test = np.asarray(x_valid).astype('float32')
    y_test = np.asarray(y_valid).astype('float32')

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))


##################################################################################################
# create and fit the LSTM network
# Initialising the RNN
    model = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
    model.add(Bidirectional(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1))))
    model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))

# Adding the output layer
    model.add(Dense(units = 1))
# Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    # accu = model.evaluate(x_test,y_test)
    # print("accuracy is")
    # print(acu)
# Fitting the RNN to the Training set
    model.fit(x_train, y_train, epochs = 10, batch_size = 16)
##################################################################################################
#predicting future values, using past 60 from the train data
# for next 10 yrs total_prediction_days is set to 3650 days
    total_prediction_days = 3650
    inputs = new_data[-total_prediction_days:].values
    # print("======len(new_data)==========")
    # print(len(new_data))
    # print("======len(inputs)==========")
    # print(len(inputs))
    inputs = inputs.reshape(-1,1)

# create future predict list which is a two dimensional list of values.
# the first dimension is the total number of future days
# the second dimension is the list of values of prediction_window_size size
    X_predict = []
    for i in range(prediction_window_size,inputs.shape[0]):
        X_predict.append(inputs[i-prediction_window_size:i,0])
    X_predict = np.array(X_predict).astype(np.float32)
   
# predict the future
    X_predict = np.reshape(X_predict, (X_predict.shape[0],X_predict.shape[1],1))
    future_closing_price = model.predict(X_predict)

    train, valid = train_test_split(new_data, train_size=0.99, test_size=0.01, shuffle=False)
    date_index = pd.to_datetime(train.index)

#converting dates into number of days as dates cannot be passed directly to any regression model
    x_days = (date_index - pd.to_datetime('1970-01-01')).days

# we are doing prediction for next 5 years hence prediction_for_days is set to 1500 days.
    prediction_for_days = 1000
    future_closing_price = future_closing_price[:prediction_for_days]

# create a data index for future dates
    x_predict_future_dates = np.asarray(pd.RangeIndex(start=x_days[-1] + 1, stop=x_days[-1] + 1 + (len(future_closing_price))))
    future_date_index = pd.to_datetime(x_predict_future_dates, origin='1970-01-01', unit='D')

# transform a list of relative percentages to the actual values
    train_transform = reverseTransformToPercentageChange(baseValue, train['Close'])

# for future dates the base value the the value of last element from the training set.
    baseValue = train_transform[-1]
    valid_transform = reverseTransformToPercentageChange(baseValue, valid['Close'])
    future_closing_price_transform = reverseTransformToPercentageChange(baseValue, future_closing_price)

# recession peak date is the date on which the index is at the bottom most position.
    recessionPeakDate =  future_date_index[future_closing_price_transform.index(min(future_closing_price_transform))]
    minCloseInFuture = min(future_closing_price_transform)
    print("The stock market will reach to its lowest bottom on", recessionPeakDate)
    print("The lowest index the stock market will fall to is ", minCloseInFuture)
    
    print("-----------------------------------------")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    accu = model.evaluate(X_test,y_test)
    print(f"Mean Squared Error Loss: {accu:.4f}")
    # Calculate Root Mean Squared Error (RMSE)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    threshold = 0.0 
    binary_predictions = (predictions > threshold).astype(int)
    binary_y_test = (y_test > threshold).astype(int)
    
    accuracy = accuracy_score(binary_y_test, binary_predictions)
    accuracy1 =accuracy*100
    print(accuracy1)
    precision = precision_score(binary_y_test, binary_predictions)
    recall = recall_score(binary_y_test, binary_predictions)
    f1 = f1_score(binary_y_test, binary_predictions)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    predictresult2={"Accuracy":accuracy1,"Precision":precision,"Recall":recall,"F1-score":f1,"RMSE":rmse,"MAE":accu}
    print(predictresult2)
    
    print("-----------------------------------------")

    
# plot the graphs
    label1='Close Price History of'+ stockname +'company'
    label2='Predicted Close of'+ stockname +'company'
    plt.figure(figsize=(16,8))
    df_x = pd.to_datetime(new_data.index)
    plt.plot(date_index,train_transform, label=label1)
    plt.plot(future_date_index,future_closing_price_transform, label=label2)

# set the title of the graph
    plt.suptitle('Stock Market Predictions', fontsize=16)

# set the title of the graph window
    # fig = plt.gcf()
    # fig.canvas.set_window_title('Stock Market Predictions')

#display the legends
    plt.legend()

#display the graph
    #plt.show()
    plt.savefig('static/futuregraph/abc.png')
    
    dictofdateandprice2={}
    try:
        for i in range(38,875):
            #date_object = datetime.datetime.strptime(str(future_date_index[i]), '%y/%m/%d')
            datetimeobt=str(future_date_index[i]).split(" ")
            # print("-------------datetimeobt-------")
            # print(datetimeobt)
            dictionaryofdateandprice[datetimeobt[0]]=future_closing_price_transform[i]
            # print("-------------dict-------")
            # print(dictionaryofdateandprice)
            # print('date obtained',str(datetimeobt[0]))
            dictofdateandprice2[str(future_date_index[i])]=future_closing_price_transform[i]
            # print("------------dictofdateandprice-------")
            # print(dictofdateandprice)
    except:
        for i in range(38,215):
            #date_object = datetime.datetime.strptime(str(future_date_index[i]), '%y/%m/%d')
            datetimeobt=str(future_date_index[i]).split(" ")
            # print("-------------datetimeobt-------")
            # print(datetimeobt)
            dictionaryofdateandprice[datetimeobt[0]]=future_closing_price_transform[i]
            # print("-------------dict-------")
            # print(dictionaryofdateandprice)
            # print('date obtained',str(datetimeobt[0]))
            dictofdateandprice2[str(future_date_index[i])]=future_closing_price_transform[i]
            # print("------------dictofdateandprice-------")
            # print(dictofdateandprice)
        
    BDLSTMdt = date_index.append(future_date_index)

    BDLSTMttf = train_transform + future_closing_price_transform
    return jsonify(dictofdateandprice2), BDLSTMttf, BDLSTMdt,predictresult2



from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D ,Conv1D ,RepeatVector
from tensorflow.keras.layers import MaxPooling1D , MaxPooling2D
from tensorflow.keras.layers import LSTM,GRU,Bidirectional
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import Flatten 
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Reshape


dictofdateandprice3={}
def GRUpredictprice(stockname):
    global dictionaryofdateandprice
    global date_index
    global train_transform
    global future_date_index
    global future_closing_price_transform
    global label1
    global label2
    global GRUdt
    global GRUttf
    global predictresult3
    

    df = pd.read_csv('livedata\\'+stockname+'.csv')
# store the first element in the series as the base value for future use.
    baseValue = df['Close'][0]

# create a new dataframe which is then transformed into relative percentages
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
    for i in range(0,len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Close'][i] = data['Close'][i]

# transform the 'Close' series into relative percentages
    transformToPercentageChange(new_data['Close'])

# set Dat column as the index
    new_data.index = new_data.Date
    new_data.drop('Date', axis=1, inplace=True)

# create train and test sets
    dataset = new_data[0:1500].values
    # print("====dataset====")
    # print(dataset)
    print("====Bidirectional GRU Prediction====")
    print(len(dataset))
    train, valid = train_test_split(dataset, train_size=0.99, test_size=0.01, shuffle=False)

# convert dataset into x_train and y_train.
# prediction_window_size is the size of days windows which will be considered for predicting a future value.

    prediction_window_size = 60
    x_train, y_train = [], []
    for i in range(prediction_window_size,len(train)):
        x_train.append(dataset[i-prediction_window_size:i,0])
        y_train.append(dataset[i,0])
    x_train, y_train = np.array(x_train).astype(np.float32), np.array(y_train).astype(np.float32)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


    # X_train = []
    # y_train = []
    # for i in range(60, 1500):
    #     X_train.append(np.array(dataset[60:1600].astype(np.float32))[i-60:i])
    #     y_train.append(np.array(dataset[60:1600].astype(np.float32))[i])
    # X_train, y_train = np.array(X_train), np.array(y_train)
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



    x_valid, y_valid = [], []
    for i in range(60,120):
        x_valid.append(dataset[i-prediction_window_size:i,0])
        y_valid.append(dataset[i,0])
        
    X_test = np.asarray(x_valid).astype('float32')
    y_test = np.asarray(y_valid).astype('float32')

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))


##################################################################################################

    model = Sequential()
    model.add(GRU(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(GRU(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(GRU(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(GRU(units = 50))
    model.add(Dropout(0.2))

    model.add(Dense(units = 1))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    model.fit(x_train, y_train, epochs = 10, batch_size = 16)

##################################################################################################
#predicting future values, using past 60 from the train data
# for next 10 yrs total_prediction_days is set to 3650 days
    total_prediction_days = 3650
    inputs = new_data[-total_prediction_days:].values
    # print("======len(new_data)==========")
    # print(len(new_data))
    # print("======len(inputs)==========")
    # print(len(inputs))
    inputs = inputs.reshape(-1,1)

# create future predict list which is a two dimensional list of values.
# the first dimension is the total number of future days
# the second dimension is the list of values of prediction_window_size size
    X_predict = []
    for i in range(prediction_window_size,inputs.shape[0]):
        X_predict.append(inputs[i-prediction_window_size:i,0])
    X_predict = np.array(X_predict).astype(np.float32)
   
# predict the future
    X_predict = np.reshape(X_predict, (X_predict.shape[0],X_predict.shape[1],1))
    future_closing_price = model.predict(X_predict)

    train, valid = train_test_split(new_data, train_size=0.99, test_size=0.01, shuffle=False)
    date_index = pd.to_datetime(train.index)

#converting dates into number of days as dates cannot be passed directly to any regression model
    x_days = (date_index - pd.to_datetime('1970-01-01')).days

# we are doing prediction for next 5 years hence prediction_for_days is set to 1500 days.
    prediction_for_days = 1000
    future_closing_price = future_closing_price[:prediction_for_days]

# create a data index for future dates
    x_predict_future_dates = np.asarray(pd.RangeIndex(start=x_days[-1] + 1, stop=x_days[-1] + 1 + (len(future_closing_price))))
    future_date_index = pd.to_datetime(x_predict_future_dates, origin='1970-01-01', unit='D')

# transform a list of relative percentages to the actual values
    train_transform = reverseTransformToPercentageChange(baseValue, train['Close'])

# for future dates the base value the the value of last element from the training set.
    baseValue = train_transform[-1]
    valid_transform = reverseTransformToPercentageChange(baseValue, valid['Close'])
    future_closing_price_transform = reverseTransformToPercentageChange(baseValue, future_closing_price)

# recession peak date is the date on which the index is at the bottom most position.
    recessionPeakDate =  future_date_index[future_closing_price_transform.index(min(future_closing_price_transform))]
    minCloseInFuture = min(future_closing_price_transform)
    print("The stock market will reach to its lowest bottom on", recessionPeakDate)
    print("The lowest index the stock market will fall to is ", minCloseInFuture)

    print("-----------------------------------------")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    accu = model.evaluate(X_test,y_test)
    print(f"Mean Squared Error Loss: {accu:.4f}")
    # Calculate Root Mean Squared Error (RMSE)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    threshold = 0.0 
    binary_predictions = (predictions > threshold).astype(int)
    binary_y_test = (y_test > threshold).astype(int)
    
    accuracy = accuracy_score(binary_y_test, binary_predictions)
    accuracy1 =accuracy*100
    print(accuracy1)
    precision = precision_score(binary_y_test, binary_predictions)
    recall = recall_score(binary_y_test, binary_predictions)
    f1 = f1_score(binary_y_test, binary_predictions)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    predictresult3={"Accuracy":accuracy1,"Precision":precision,"Recall":recall,"F1-score":f1,"RMSE":rmse,"MAE":accu}
    print(predictresult3)
    
    print("-----------------------------------------")

# plot the graphs
    label1='Close Price History of'+ stockname +'company'
    label2='Predicted Close of'+ stockname +'company'
    plt.figure(figsize=(16,8))
    df_x = pd.to_datetime(new_data.index)
    plt.plot(date_index,train_transform, label=label1)
    plt.plot(future_date_index,future_closing_price_transform, label=label2)

# set the title of the graph
    plt.suptitle('Stock Market Predictions', fontsize=16)

# set the title of the graph window
    # fig = plt.gcf()
    # fig.canvas.set_window_title('Stock Market Predictions')

#display the legends
    plt.legend()

#display the graph
    #plt.show()
    plt.savefig('static/futuregraph/abc.png')
    
    dictofdateandprice3={}
    try:
        for i in range(38,875):
            #date_object = datetime.datetime.strptime(str(future_date_index[i]), '%y/%m/%d')
            datetimeobt=str(future_date_index[i]).split(" ")
            # print("-------------datetimeobt-------")
            # print(datetimeobt)
            dictionaryofdateandprice[datetimeobt[0]]=future_closing_price_transform[i]
            # print("-------------dict-------")
            # print(dictionaryofdateandprice)
            # print('date obtained',str(datetimeobt[0]))
            dictofdateandprice3[str(future_date_index[i])]=future_closing_price_transform[i]
            # print("------------dictofdateandprice-------")
            # print(dictofdateandprice)
    except:
        for i in range(38,215):
            #date_object = datetime.datetime.strptime(str(future_date_index[i]), '%y/%m/%d')
            datetimeobt=str(future_date_index[i]).split(" ")
            # print("-------------datetimeobt-------")
            # print(datetimeobt)
            dictionaryofdateandprice[datetimeobt[0]]=future_closing_price_transform[i]
            # print("-------------dict-------")
            # print(dictionaryofdateandprice)
            # print('date obtained',str(datetimeobt[0]))
            dictofdateandprice3[str(future_date_index[i])]=future_closing_price_transform[i]
            # print("------------dictofdateandprice-------")
            # print(dictofdateandprice)
        
    GRUdt = date_index.append(future_date_index)

    GRUttf = train_transform + future_closing_price_transform
    return jsonify(dictofdateandprice3), GRUttf, GRUdt,predictresult3




dictofdateandprice4={}
from tensorflow.keras.layers import Bidirectional
def BiDirectionalGRU(stockname):
    global dictionaryofdateandprice
    global date_index
    global train_transform
    global future_date_index
    global future_closing_price_transform
    global label1
    global label2
    global BDGRUttf
    global BDGRUdt
    global predictresult4
    

    df = pd.read_csv('livedata\\'+stockname+'.csv')
# store the first element in the series as the base value for future use.
    baseValue = df['Close'][0]

# create a new dataframe which is then transformed into relative percentages
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
    for i in range(0,len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Close'][i] = data['Close'][i]

# transform the 'Close' series into relative percentages
    transformToPercentageChange(new_data['Close'])

# set Dat column as the index
    new_data.index = new_data.Date
    new_data.drop('Date', axis=1, inplace=True)

# create train and test sets
    dataset = new_data[0:1500].values
    # print("====dataset====")
    # print(dataset)
    print("====GRU Prediction Results====")
    print(len(dataset))
    train, valid = train_test_split(dataset, train_size=0.99, test_size=0.01, shuffle=False)

# convert dataset into x_train and y_train.
# prediction_window_size is the size of days windows which will be considered for predicting a future value.

    prediction_window_size = 60
    x_train, y_train = [], []
    for i in range(prediction_window_size,len(train)):
        x_train.append(dataset[i-prediction_window_size:i,0])
        y_train.append(dataset[i,0])
    x_train, y_train = np.array(x_train).astype(np.float32), np.array(y_train).astype(np.float32)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


    # X_train = []
    # y_train = []
    # for i in range(60, 1500):
    #     X_train.append(np.array(dataset[60:1600].astype(np.float32))[i-60:i])
    #     y_train.append(np.array(dataset[60:1600].astype(np.float32))[i])
    # X_train, y_train = np.array(X_train), np.array(y_train)
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



    x_valid, y_valid = [], []
    for i in range(60,120):
        x_valid.append(dataset[i-prediction_window_size:i,0])
        y_valid.append(dataset[i,0])
        
    X_test = np.asarray(x_valid).astype('float32')
    y_test = np.asarray(y_valid).astype('float32')

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))


##################################################################################################
# create and fit the LSTM network
# Initialising the RNN
    model = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
    model.add(Bidirectional(GRU(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1))))
    model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
    model.add(GRU(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
    model.add(GRU(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
    model.add(GRU(units = 50))
    model.add(Dropout(0.2))

# Adding the output layer
    model.add(Dense(units = 1))
# Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    # accu = model.evaluate(x_test,y_test)
    # print("accuracy is")
    # print(acu)
# Fitting the RNN to the Training set
    model.fit(x_train, y_train, epochs = 10, batch_size = 16)
##################################################################################################
#predicting future values, using past 60 from the train data
# for next 10 yrs total_prediction_days is set to 3650 days
    total_prediction_days = 3650
    inputs = new_data[-total_prediction_days:].values
    # print("======len(new_data)==========")
    # print(len(new_data))
    # print("======len(inputs)==========")
    # print(len(inputs))
    inputs = inputs.reshape(-1,1)

# create future predict list which is a two dimensional list of values.
# the first dimension is the total number of future days
# the second dimension is the list of values of prediction_window_size size
    X_predict = []
    for i in range(prediction_window_size,inputs.shape[0]):
        X_predict.append(inputs[i-prediction_window_size:i,0])
    X_predict = np.array(X_predict).astype(np.float32)
   
# predict the future
    X_predict = np.reshape(X_predict, (X_predict.shape[0],X_predict.shape[1],1))
    future_closing_price = model.predict(X_predict)

    train, valid = train_test_split(new_data, train_size=0.99, test_size=0.01, shuffle=False)
    date_index = pd.to_datetime(train.index)

#converting dates into number of days as dates cannot be passed directly to any regression model
    x_days = (date_index - pd.to_datetime('1970-01-01')).days

# we are doing prediction for next 5 years hence prediction_for_days is set to 1500 days.
    prediction_for_days = 1000
    future_closing_price = future_closing_price[:prediction_for_days]

# create a data index for future dates
    x_predict_future_dates = np.asarray(pd.RangeIndex(start=x_days[-1] + 1, stop=x_days[-1] + 1 + (len(future_closing_price))))
    future_date_index = pd.to_datetime(x_predict_future_dates, origin='1970-01-01', unit='D')

# transform a list of relative percentages to the actual values
    train_transform = reverseTransformToPercentageChange(baseValue, train['Close'])

# for future dates the base value the the value of last element from the training set.
    baseValue = train_transform[-1]
    valid_transform = reverseTransformToPercentageChange(baseValue, valid['Close'])
    future_closing_price_transform = reverseTransformToPercentageChange(baseValue, future_closing_price)

# recession peak date is the date on which the index is at the bottom most position.
    recessionPeakDate =  future_date_index[future_closing_price_transform.index(min(future_closing_price_transform))]
    minCloseInFuture = min(future_closing_price_transform)
    print("The stock market will reach to its lowest bottom on", recessionPeakDate)
    print("The lowest index the stock market will fall to is ", minCloseInFuture)

    print("-----------------------------------------")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    accu = model.evaluate(X_test,y_test)
    print(f"Mean Squared Error Loss: {accu:.4f}")
    # Calculate Root Mean Squared Error (RMSE)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    threshold = 0.0 
    binary_predictions = (predictions > threshold).astype(int)
    binary_y_test = (y_test > threshold).astype(int)
    
    accuracy = accuracy_score(binary_y_test, binary_predictions)
    accuracy1 =accuracy*100
    print(accuracy1)
    precision = precision_score(binary_y_test, binary_predictions)
    recall = recall_score(binary_y_test, binary_predictions)
    f1 = f1_score(binary_y_test, binary_predictions)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    predictresult4={"Accuracy":accuracy1,"Precision":precision,"Recall":recall,"F1-score":f1,"RMSE":rmse,"MAE":accu}
    print(predictresult4)
    
    print("-----------------------------------------")
# plot the graphs
    label1='Close Price History of'+ stockname +'company'
    label2='Predicted Close of'+ stockname +'company'
    plt.figure(figsize=(16,8))
    df_x = pd.to_datetime(new_data.index)
    plt.plot(date_index,train_transform, label=label1)
    plt.plot(future_date_index,future_closing_price_transform, label=label2)

# set the title of the graph
    plt.suptitle('Stock Market Predictions', fontsize=16)

# set the title of the graph window
    # fig = plt.gcf()
    # fig.canvas.set_window_title('Stock Market Predictions')

#display the legends
    plt.legend()

#display the graph
    #plt.show()
    plt.savefig('static/futuregraph/abc.png')
    
    dictofdateandprice4={}
    try:
        for i in range(38,875):
            #date_object = datetime.datetime.strptime(str(future_date_index[i]), '%y/%m/%d')
            datetimeobt=str(future_date_index[i]).split(" ")
            # print("-------------datetimeobt-------")
            # print(datetimeobt)
            dictionaryofdateandprice[datetimeobt[0]]=future_closing_price_transform[i]
            # print("-------------dict-------")
            # print(dictionaryofdateandprice)
            # print('date obtained',str(datetimeobt[0]))
            dictofdateandprice4[str(future_date_index[i])]=future_closing_price_transform[i]
            # print("------------dictofdateandprice-------")
            # print(dictofdateandprice)
    except:
        for i in range(38,215):
            #date_object = datetime.datetime.strptime(str(future_date_index[i]), '%y/%m/%d')
            datetimeobt=str(future_date_index[i]).split(" ")
            # print("-------------datetimeobt-------")
            # print(datetimeobt)
            dictionaryofdateandprice[datetimeobt[0]]=future_closing_price_transform[i]
            # print("-------------dict-------")
            # print(dictionaryofdateandprice)
            # print('date obtained',str(datetimeobt[0]))
            dictofdateandprice4[str(future_date_index[i])]=future_closing_price_transform[i]
            # print("------------dictofdateandprice-------")
            # print(dictofdateandprice)
        
    BDGRUdt = date_index.append(future_date_index)

    BDGRUttf = train_transform + future_closing_price_transform
    return jsonify(dictofdateandprice4), BDGRUttf, BDGRUdt,predictresult4


dictofdateandprice5={}
def CNNLSTMpredictprice(stockname):
    global dictionaryofdateandprice
    global date_index
    global train_transform
    global future_date_index
    global future_closing_price_transform
    global label1 
    global label2
    global CLSTMdt
    global CLSTMtf
    global predictresult5
    

    df = pd.read_csv('livedata\\'+stockname+'.csv')
# store the first element in the series as the base value for future use.
    baseValue = df['Close'][0]

# create a new dataframe which is then transformed into relative percentages
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
    for i in range(0,len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Close'][i] = data['Close'][i]

# transform the 'Close' series into relative percentages
    transformToPercentageChange(new_data['Close'])

# set Dat column as the index
    new_data.index = new_data.Date
    new_data.drop('Date', axis=1, inplace=True)

# create train and test sets
    dataset = new_data[0:1500].values
    # print("====dataset====")
    # print(dataset)
    print("====CNN LSTM Prediction Results====")
    print(len(dataset))
    train, valid = train_test_split(dataset, train_size=0.99, test_size=0.01, shuffle=False)

# convert dataset into x_train and y_train.
# prediction_window_size is the size of days windows which will be considered for predicting a future value.

    prediction_window_size = 60
    x_train, y_train = [], []
    for i in range(prediction_window_size,len(train)):
        x_train.append(dataset[i-prediction_window_size:i,0])
        y_train.append(dataset[i,0])
    x_train, y_train = np.array(x_train).astype(np.float32), np.array(y_train).astype(np.float32)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    
    x_valid, y_valid = [], []
    for i in range(60,120):
        x_valid.append(dataset[i-prediction_window_size:i,0])
        y_valid.append(dataset[i,0])
        
    X_test = np.asarray(x_valid).astype('float32')
    y_test = np.asarray(y_valid).astype('float32')

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))


##################################################################################################

    # model = Sequential()                        
    # model.add(Conv1D(filters=16, kernel_size=5, strides=1, padding="same", input_shape=(x_train.shape[1],1), activation='relu'))
    # model.add(MaxPooling1D())
    # model.add(LSTM(64, return_sequences=True))
    # model.add(Flatten())
    # model.add(Dense(1, activation="sigmoid"))     
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model = Sequential()
    # Add the CNN layers                        
    model.add(Conv1D(32, kernel_size=5, input_shape=(x_train.shape[1],1), activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    
    # # Add the LSTM layer
    model.add(LSTM(64, return_sequences=True))
    
    # # Flatten the output from the CNN layers
    model.add(Flatten())
    
    # # Add the output layer
    model.add(Dense(1, activation="sigmoid"))
    
    # # Compile the model
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
    
    model.fit(x_train, y_train, epochs = 10, batch_size = 16)

##################################################################################################
#predicting future values, using past 60 from the train data
# for next 10 yrs total_prediction_days is set to 3650 days
    total_prediction_days = 3650
    inputs = new_data[-total_prediction_days:].values
    # print("======len(new_data)==========")
    # print(len(new_data))
    # print("======len(inputs)==========")
    # print(len(inputs))
    inputs = inputs.reshape(-1,1)

# create future predict list which is a two dimensional list of values.
# the first dimension is the total number of future days
# the second dimension is the list of values of prediction_window_size size
    X_predict = []
    for i in range(prediction_window_size,inputs.shape[0]):
        X_predict.append(inputs[i-prediction_window_size:i,0])
    X_predict = np.array(X_predict).astype(np.float32)
   
# predict the future
    X_predict = np.reshape(X_predict, (X_predict.shape[0],X_predict.shape[1],1))
    future_closing_price = model.predict(X_predict)

    train, valid = train_test_split(new_data, train_size=0.99, test_size=0.01, shuffle=False)
    date_index = pd.to_datetime(train.index)

#converting dates into number of days as dates cannot be passed directly to any regression model
    x_days = (date_index - pd.to_datetime('1970-01-01')).days

# we are doing prediction for next 5 years hence prediction_for_days is set to 1500 days.
    prediction_for_days = 1000
    future_closing_price = future_closing_price[:prediction_for_days]

# create a data index for future dates
    x_predict_future_dates = np.asarray(pd.RangeIndex(start=x_days[-1] + 1, stop=x_days[-1] + 1 + (len(future_closing_price))))
    future_date_index = pd.to_datetime(x_predict_future_dates, origin='1970-01-01', unit='D')

# transform a list of relative percentages to the actual values
    train_transform = reverseTransformToPercentageChange(baseValue, train['Close'])

# for future dates the base value the the value of last element from the training set.
    baseValue = train_transform[-1]
    valid_transform = reverseTransformToPercentageChange(baseValue, valid['Close'])
    future_closing_price_transform = reverseTransformToPercentageChange(baseValue, future_closing_price)

# recession peak date is the date on which the index is at the bottom most position.
    recessionPeakDate =  future_date_index[future_closing_price_transform.index(min(future_closing_price_transform))]
    minCloseInFuture = min(future_closing_price_transform)
    print("The stock market will reach to its lowest bottom on", recessionPeakDate)
    print("The lowest index the stock market will fall to is ", minCloseInFuture)
    print("-----------------------------------------")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    accu = model.evaluate(X_test,y_test)
    print(f"Mean Squared Error Loss: {accu:.4f}")
    # Calculate Root Mean Squared Error (RMSE)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    threshold = 0.0 
    binary_predictions = (predictions > threshold).astype(int)
    binary_y_test = (y_test > threshold).astype(int)
    
    accuracy = accuracy_score(binary_y_test, binary_predictions)
    accuracy1 =accuracy*100
    print(accuracy1)
    precision = precision_score(binary_y_test, binary_predictions)
    recall = recall_score(binary_y_test, binary_predictions)
    f1 = f1_score(binary_y_test, binary_predictions)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    predictresult5={"Accuracy":accuracy1,"Precision":precision,"Recall":recall,"F1-score":f1,"RMSE":rmse,"MAE":accu}
    print(predictresult5)
    
    print("-----------------------------------------")
    
# plot the graphs
    label1='Close Price History of'+ stockname +'company'
    label2='Predicted Close of'+ stockname +'company'
    plt.figure(figsize=(16,8))
    df_x = pd.to_datetime(new_data.index)
    plt.plot(date_index,train_transform, label=label1)
    plt.plot(future_date_index,future_closing_price_transform, label=label2)

# set the title of the graph
    plt.suptitle('Stock Market Predictions', fontsize=16)

# set the title of the graph window
    # fig = plt.gcf()
    # fig.canvas.set_window_title('Stock Market Predictions')

#display the legends
    plt.legend()

#display the graph
    #plt.show()
    plt.savefig('static/futuregraph/abc.png')
    
    dictofdateandprice5={}
    try:
        for i in range(38,875):
            #date_object = datetime.datetime.strptime(str(future_date_index[i]), '%y/%m/%d')
            datetimeobt=str(future_date_index[i]).split(" ")
            # print("-------------datetimeobt-------")
            # print(datetimeobt)
            dictionaryofdateandprice[datetimeobt[0]]=future_closing_price_transform[i]
            # print("-------------dict-------")
            # print(dictionaryofdateandprice)
            # print('date obtained',str(datetimeobt[0]))
            dictofdateandprice5[str(future_date_index[i])]=future_closing_price_transform[i]
            # print("------------dictofdateandprice-------")
            # print(dictofdateandprice)
    except:
        for i in range(38,215):
            #date_object = datetime.datetime.strptime(str(future_date_index[i]), '%y/%m/%d')
            datetimeobt=str(future_date_index[i]).split(" ")
            # print("-------------datetimeobt-------")
            # print(datetimeobt)
            dictionaryofdateandprice[datetimeobt[0]]=future_closing_price_transform[i]
            # print("-------------dict-------")
            # print(dictionaryofdateandprice)
            # print('date obtained',str(datetimeobt[0]))
            dictofdateandprice5[str(future_date_index[i])]=future_closing_price_transform[i]
            # print("------------dictofdateandprice-------")
            # print(dictofdateandprice)
        
    CLSTMdt = date_index.append(future_date_index)

    CLSTMtf = train_transform + future_closing_price_transform
    return jsonify(dictofdateandprice5), CLSTMtf, CLSTMdt,predictresult5




dictofdateandprice6={}
def CNNGRUpredictprice(stockname):
    global dictionaryofdateandprice
    global date_index
    global train_transform
    global future_date_index
    global future_closing_price_transform
    global label1
    global label2
    global CGRUdt
    global CGRUtf 
    global predictresult6
    

    df = pd.read_csv('livedata\\'+stockname+'.csv')
# store the first element in the series as the base value for future use.
    baseValue = df['Close'][0]

# create a new dataframe which is then transformed into relative percentages
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
    for i in range(0,len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Close'][i] = data['Close'][i]

# transform the 'Close' series into relative percentages
    transformToPercentageChange(new_data['Close'])

# set Dat column as the index
    new_data.index = new_data.Date
    new_data.drop('Date', axis=1, inplace=True)

# create train and test sets
    dataset = new_data[0:1500].values
    # print("====dataset====")
    # print(dataset)
    print("====CNN GRU Prediction Results====")
    print(len(dataset))
    train, valid = train_test_split(dataset, train_size=0.99, test_size=0.01, shuffle=False)

# convert dataset into x_train and y_train.
# prediction_window_size is the size of days windows which will be considered for predicting a future value.

    prediction_window_size = 60
    x_train, y_train = [], []
    for i in range(prediction_window_size,len(train)):
        x_train.append(dataset[i-prediction_window_size:i,0])
        y_train.append(dataset[i,0])
    x_train, y_train = np.array(x_train).astype(np.float32), np.array(y_train).astype(np.float32)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    
    x_valid, y_valid = [], []
    for i in range(60,120):
        x_valid.append(dataset[i-prediction_window_size:i,0])
        y_valid.append(dataset[i,0])
        
    X_test = np.asarray(x_valid).astype('float32')
    y_test = np.asarray(y_valid).astype('float32')

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))


##################################################################################################

    # model = Sequential()                        
    # model.add(Conv1D(filters=16, kernel_size=5, strides=1, padding="same", input_shape=(x_train.shape[1],1), activation='relu'))
    # model.add(MaxPooling1D())
    # model.add(LSTM(64, return_sequences=True))
    # model.add(Flatten())
    # model.add(Dense(1, activation="sigmoid"))     
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model = Sequential()                        
    model.add(Conv1D(filters=16, kernel_size=5, strides=1, padding="same", input_shape=(x_train.shape[1],1), activation='relu'))
    model.add(MaxPooling1D())
    model.add(GRU(64, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))     
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    model.fit(x_train, y_train, epochs = 10, batch_size = 16)

##################################################################################################
#predicting future values, using past 60 from the train data
# for next 10 yrs total_prediction_days is set to 3650 days
    total_prediction_days = 3650
    inputs = new_data[-total_prediction_days:].values
    # print("======len(new_data)==========")
    # print(len(new_data))
    # print("======len(inputs)==========")
    # print(len(inputs))
    inputs = inputs.reshape(-1,1)

# create future predict list which is a two dimensional list of values.
# the first dimension is the total number of future days
# the second dimension is the list of values of prediction_window_size size
    X_predict = []
    for i in range(prediction_window_size,inputs.shape[0]):
        X_predict.append(inputs[i-prediction_window_size:i,0])
    X_predict = np.array(X_predict).astype(np.float32)
   
# predict the future
    X_predict = np.reshape(X_predict, (X_predict.shape[0],X_predict.shape[1],1))
    future_closing_price = model.predict(X_predict)

    train, valid = train_test_split(new_data, train_size=0.99, test_size=0.01, shuffle=False)
    date_index = pd.to_datetime(train.index)

#converting dates into number of days as dates cannot be passed directly to any regression model
    x_days = (date_index - pd.to_datetime('1970-01-01')).days

# we are doing prediction for next 5 years hence prediction_for_days is set to 1500 days.
    prediction_for_days = 1000
    future_closing_price = future_closing_price[:prediction_for_days]

# create a data index for future dates
    x_predict_future_dates = np.asarray(pd.RangeIndex(start=x_days[-1] + 1, stop=x_days[-1] + 1 + (len(future_closing_price))))
    future_date_index = pd.to_datetime(x_predict_future_dates, origin='1970-01-01', unit='D')

# transform a list of relative percentages to the actual values
    train_transform = reverseTransformToPercentageChange(baseValue, train['Close'])

# for future dates the base value the the value of last element from the training set.
    baseValue = train_transform[-1]
    valid_transform = reverseTransformToPercentageChange(baseValue, valid['Close'])
    future_closing_price_transform = reverseTransformToPercentageChange(baseValue, future_closing_price)

# recession peak date is the date on which the index is at the bottom most position.
    recessionPeakDate =  future_date_index[future_closing_price_transform.index(min(future_closing_price_transform))]
    minCloseInFuture = min(future_closing_price_transform)
    print("The stock market will reach to its lowest bottom on", recessionPeakDate)
    print("The lowest index the stock market will fall to is ", minCloseInFuture)
    print("-----------------------------------------")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    accu = model.evaluate(X_test,y_test)
    print(f"Mean Squared Error Loss: {accu:.4f}")
    # Calculate Root Mean Squared Error (RMSE)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    threshold = 0.0 
    binary_predictions = (predictions > threshold).astype(int)
    binary_y_test = (y_test > threshold).astype(int)
    
    accuracy = accuracy_score(binary_y_test, binary_predictions)
    accuracy1 =accuracy*100
    print(accuracy1)
    precision = precision_score(binary_y_test, binary_predictions)
    recall = recall_score(binary_y_test, binary_predictions)
    f1 = f1_score(binary_y_test, binary_predictions)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    predictresult6={"Accuracy":accuracy1,"Precision":precision,"Recall":recall,"F1-score":f1,"RMSE":rmse,"MAE":accu}
    print(predictresult6)
    
    print("-----------------------------------------")
    
# plot the graphs
    label1='Close Price History of'+ stockname +'company'
    label2='Predicted Close of'+ stockname +'company'
    plt.figure(figsize=(16,8))
    df_x = pd.to_datetime(new_data.index)
    plt.plot(date_index,train_transform, label=label1)
    plt.plot(future_date_index,future_closing_price_transform, label=label2)

# set the title of the graph
    plt.suptitle('Stock Market Predictions', fontsize=16)

# set the title of the graph window
    # fig = plt.gcf()
    # fig.canvas.set_window_title('Stock Market Predictions')

#display the legends
    plt.legend()

#display the graph
    #plt.show()
    plt.savefig('static/futuregraph/abc.png')
    
    dictofdateandprice6={}
    try:
        for i in range(38,875):
            #date_object = datetime.datetime.strptime(str(future_date_index[i]), '%y/%m/%d')
            datetimeobt=str(future_date_index[i]).split(" ")
            # print("-------------datetimeobt-------")
            # print(datetimeobt)
            dictionaryofdateandprice[datetimeobt[0]]=future_closing_price_transform[i]
            # print("-------------dict-------")
            # print(dictionaryofdateandprice)
            # print('date obtained',str(datetimeobt[0]))
            dictofdateandprice6[str(future_date_index[i])]=future_closing_price_transform[i]
            # print("------------dictofdateandprice-------")
            # print(dictofdateandprice)
    except:
        for i in range(38,215):
            #date_object = datetime.datetime.strptime(str(future_date_index[i]), '%y/%m/%d')
            datetimeobt=str(future_date_index[i]).split(" ")
            # print("-------------datetimeobt-------")
            # print(datetimeobt)
            dictionaryofdateandprice[datetimeobt[0]]=future_closing_price_transform[i]
            # print("-------------dict-------")
            # print(dictionaryofdateandprice)
            # print('date obtained',str(datetimeobt[0]))
            dictofdateandprice6[str(future_date_index[i])]=future_closing_price_transform[i]
            # print("------------dictofdateandprice-------")
            # print(dictofdateandprice)
        
    CGRUdt = date_index.append(future_date_index)

    CGRUtf = train_transform + future_closing_price_transform
    return jsonify(dictofdateandprice6), CGRUtf, CGRUdt,predictresult6




dictofdateandprice={}
def predictpriceofdata2(stockname2):
    global dictionaryofdateandprice2
    global dt1 
    global dt2 
    global ttf1
    global ttf2
    
    df2 = pd.read_csv('data\\'+stockname2+'.csv')
# store the first element in the series as the base value for future use.
    baseValue = df2['Close'][0]

# create a new dataframe which is then transformed into relative percentages
    data2 = df2.sort_index(ascending=True, axis=0)
    new_data2 = pd.DataFrame(index=range(0,len(df2)),columns=['Date', 'Close'])
    for i in range(0,len(data2)):
        new_data2['Date'][i] = data2['Date'][i]
        new_data2['Close'][i] = data2['Close'][i]

# transform the 'Close' series into relative percentages
    transformToPercentageChange(new_data2['Close'])

# set Dat column as the index
    new_data2.index = new_data2.Date
    new_data2.drop('Date', axis=1, inplace=True)

# create train and test sets
    dataset2 = new_data2[0:1500].values
    # print("====dataset 2====")
    # print(dataset2)
    # print("====len dataset2====")
    # print(len(dataset2))
    train2, valid2 = train_test_split(dataset2, train_size=0.99, test_size=0.01, shuffle=False)

# convert dataset into x_train and y_train.
# prediction_window_size is the size of days windows which will be considered for predicting a future value.
    prediction_window_size2 = 60
    x_train2, y_train2 = [], []
    for i in range(prediction_window_size2,len(train2)):
        x_train2.append(dataset2[i-prediction_window_size2:i,0])
        y_train2.append(dataset2[i,0])
    x_train2, y_train2 = np.array(x_train2).astype(np.float32), np.array(y_train2).astype(np.float32)
    x_train2 = np.reshape(x_train2, (x_train2.shape[0],x_train2.shape[1],1))


    # X_train3 = []
    # y_train3 = []
    # for i in range(60, 1500):
    #     X_train3.append(np.array(dataset2[60:1600]).astype(np.float32)[i-60:i])
    #     y_train3.append(np.array(dataset2[60:1600]).astype(np.float32)[i])
    # X_train3, y_train3 = np.array(X_train3), np.array(y_train3)
    # X_train3 = np.reshape(X_train3, (X_train3.shape[0], X_train3.shape[1], 1))



    x_valid2, y_valid2 = [], []
    for i in range(60,120):
        x_valid2.append(dataset2[i-prediction_window_size2:i,0])
        y_valid2.append(dataset2[i,0])
        
    X_test2 = np.asarray(x_valid2).astype('float32')
    #y_test = np.asarray(y_valid).astype('float32')

    X_test2 = np.array(X_test2)
    X_test2 = np.reshape(X_test2, (X_test2.shape[0],X_test2.shape[1],1))


##################################################################################################
# create and fit the LSTM network
# Initialising the RNN
    model = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train2.shape[1], 1)))
    model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))

# Adding the output layer
    model.add(Dense(units = 1))
# Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    # accu = model.evaluate(x_test,y_test)
    # print("accuracy is")
    # print(acu)
# Fitting the RNN to the Training set
    model.fit(x_train2, y_train2, epochs = 10, batch_size = 16)

##################################################################################################
#predicting future values, using past 60 from the train data
# for next 10 yrs total_prediction_days is set to 3650 days
    total_prediction_days2 = 3650
    inputs2 = new_data2[-total_prediction_days2:].values
    inputs2 = inputs2.reshape(-1,1)

# create future predict list which is a two dimensional list of values.
# the first dimension is the total number of future days
# the second dimension is the list of values of prediction_window_size size
    X_predict2 = []
    for i in range(prediction_window_size2,inputs2.shape[0]):
        X_predict2.append(inputs2[i-prediction_window_size2:i,0])
    X_predict2 = np.array(X_predict2).astype(np.float32)
   
# predict the future
    X_predict2 = np.reshape(X_predict2, (X_predict2.shape[0],X_predict2.shape[1],1))
    future_closing_price2 = model.predict(X_predict2)

    train2, valid2 = train_test_split(new_data2, train_size=0.99, test_size=0.01, shuffle=False)
    date_index2 = pd.to_datetime(train2.index)

#converting dates into number of days as dates cannot be passed directly to any regression model
    x_days2 = (date_index2 - pd.to_datetime('1970-01-01')).days

# we are doing prediction for next 5 years hence prediction_for_days is set to 1500 days.
    prediction_for_days2 = 1000
    future_closing_price2 = future_closing_price2[:prediction_for_days2]

# create a data index for future dates
    x_predict_future_dates2 = np.asarray(pd.RangeIndex(start=x_days2[-1] + 1, stop=x_days2[-1] + 1 + (len(future_closing_price2))))
    future_date_index2 = pd.to_datetime(x_predict_future_dates2, origin='1970-01-01', unit='D')

# transform a list of relative percentages to the actual values
    train_transform2 = reverseTransformToPercentageChange(baseValue, train2['Close'])

# for future dates the base value the the value of last element from the training set.
    baseValue2 = train_transform2[-1]
    valid_transform2 = reverseTransformToPercentageChange(baseValue2, valid2['Close'])
    future_closing_price_transform2 = reverseTransformToPercentageChange(baseValue2, future_closing_price2)

# recession peak date is the date on which the index is at the bottom most position.
    recessionPeakDate2 =  future_date_index2[future_closing_price_transform2.index(min(future_closing_price_transform2))]
    minCloseInFuture2 = min(future_closing_price_transform2)
    print("The stock market will reach to its lowest bottom on", recessionPeakDate2)
    print("The lowest index the stock market will fall to is ", minCloseInFuture2)

    # print("==========date_index====")
    # print(type(date_index))
    # print("======train_transform=====")
    # print(len(train_transform))
    # print("=====future_date_index======")
    # print(type(future_date_index))
    # print("========future_closing_price_transform=======")
    # print(len(future_closing_price_transform))

    
# plot the graphs
    plt.figure(figsize=(16,8))
    df_x = pd.to_datetime(new_data2.index)
    plt.plot(date_index,train_transform, label=label1)
    plt.plot(future_date_index,future_closing_price_transform, label=label2)
    plt.plot(date_index2,train_transform2, label='Close Price History of'+ stockname2 + 'company')
    plt.plot(future_date_index2,future_closing_price_transform2, label='Predicted Close of'+ stockname2 + 'company')
    
# set the title of the graph
    plt.suptitle('Stock Market Predictions', fontsize=16)

# set the title of the graph window
    # fig = plt.gcf()
    # fig.canvas.set_window_title('Stock Market Predictions')

#display the legends
    plt.legend()

#display the graph
    #plt.show()
    plt.savefig('static/futuregraph/future.png')

###############################################################################
    dt1 = date_index.append(future_date_index)
    dt2 = date_index2.append(future_date_index2)
    

    ttf1 = train_transform + future_closing_price_transform
    ttf2 = train_transform2 + future_closing_price_transform2
    
    return jsonify(dictofdateandprice), dt1, dt2, ttf1, ttf2

def fetchcurrentmarketprice(stock):
    stock1=stock
    #for ticker in ticker_list1:
    url = 'https://in.finance.yahoo.com/quote/' + stock1
    print(url)
    session = requests_html.HTMLSession()
    r = session.get(url)
    content = BeautifulSoup(r.content, 'html')
    try:
        price = str(content).split('data-reactid="32"')[4].split('</span>')[0].replace('>','')
        #print(str(content).split('data-reactid="47"'))
        openprice = str(content).split('data-reactid="49"')[3].split('</span>')[0].replace('>','')
        rangeobt = str(content).split('data-reactid="67"')[2].split('</span>')[0]
        #price = str(content).split('data-reactid="32"')[4].split('</span>')[0].replace('>','')
        #price = str(content).split('data-reactid="32"')[4].split('</span>')[0].replace('>','')
        #price = str(content).split('data-reactid="32"')[4].split('</span>')[0].replace('>','')
    except IndexError as e:
        price = 0.00
        price = price or "0"
    try:
        price = float(price.replace(',',''))
    except ValueError as e:
        price = 0.00
        time.sleep(1)
   
    print( price)
    print(openprice)
    print(rangeobt)
        #cursor.execute(_SQL, (unidecode.unidecode(ticker[0]), price, unidecode.unidecode(ticker[1]), unidecode.unidecode(ticker[2]), unidecode.unidecode(ticker[3])))
    return price



#urltofetch='https://www.usatoday.com/story/money/2020/04/22/amazon-doing-free-deliveries-food-banks-during-coronavirus-emergency/2997254001/'

#alldata=parsenews(urltofetch)
#print(alldata)

#Python program to scrape website  
#and save quotes from website 
import requests 
from bs4 import BeautifulSoup 
import csv 
import re
from datetime import date, timedelta

def callingnews(query):

    URL = "https://www.usatoday.com/search/?q="+query
    r = requests.get(URL) 
#print(r)
  
    soup = BeautifulSoup(r.content, 'html.parser') 
#print(soup)
    quotes=[]  # a list to store quotes 
  

    table1 = soup.find_all('a', attrs = {'class':'gnt_se_a gnt_se_a__hd gnt_se_a__hi'}) 
    #print(table1)

#table13 = table11.get_text()
#print(table13) 

    table11 = soup.find_all('div', attrs = {'class':'gnt_pr'}) 
    #print(table11)
    datalist=[]
    linksdata=[]
#print(table11)
    for ik in table1:
        datalist.append(ik.get_text())
        print(ik.get_text())

    pos=0
    listtocheck=[]
    for ik in table1:
        links = re.findall("href=[\"\'](.*?)[\"\']", str(ik))
        linksdata.append('https://www.usatoday.com'+links[0])
        if 'story' not in links[0]:
            listtocheck.append(pos)
        pos+=1
        print(links)

    print("list check is ",listtocheck)

    for ij in range(len(listtocheck)):
        print(ij)
        datalist.pop(ij)
        linksdata.pop(ij)
    #print(listtocheck[ij])

    print(len(datalist))
    print(len(linksdata))
    return datalist,linksdata


#df
df1=pd1.read_csv('fortune23.csv')
df=pd.DataFrame()

app = Flask(__name__)
app.secret_key = "super secret key"

class ExampleForm(Form):
    dt = DateField('container', format='%d-%m-%Y')

@app.route("/parsenews")
def parsenews(): 
    newsinfo = request.args.get('msg')
    URL =newsinfo.rstrip().lstrip().strip()# "https://www.hindustantimes.com/delhi-news/protest-at-delhi-s-jama-masjid-against-citizenship-act-4-metro-stations-closed-in-area/story-q7vKj5IUdIKMExw5eGBfxI.html"
    #URL ="https://www.hindustantimes.com/delhi-news/protest-at-delhi-s-jama-masjid-against-citizenship-act-4-metro-stations-closed-in-area/story-q7vKj5IUdIKMExw5eGBfxI.html"
    #print repr(URL)
    r = requests.get(URL) 
    #print(r)
    soup = BeautifulSoup(r.content, 'html.parser') 
  
    quotes=[]  # a list to store quotes 
  
    table = soup.find('div', attrs = {'class':'gnt_ar_b'}) 
    #print(table)
    alltestdata='<a href=\''+URL+'\' target="_blank" >'+URL+'</a>'+'<br>'
    print(alltestdata)
    try:
        table = table.find_all('p')
        
        for row in table.find_all('p'):
            quote = {} 
            quote['data'] = row.text 
            alltestdata=alltestdata+row.text+" "
            quotes.append(quote)
    except:
        alltestdata='<a href=\''+URL+'\' target="_blank" >'+URL+'</a>'+'<br>'
    #print(alltestdata)
    print(alltestdata)
    return alltestdata

import requests 
from bs4 import BeautifulSoup 
import csv 
import re

def moneynews(company):
    dff = df1.loc[(df1['Name'] == company).values]
    dff = list(dff["Url"])
    url_link = dff[0]
    request = requests.get(url_link).text
    
    Soup = BeautifulSoup(request, 'html.parser')
    # print(soup)
    table1 = Soup.find_all('h1', attrs = {'class':'article_title artTitle'})
    heading = table1[0].text.strip()
    lst=[]
    for para in Soup.find_all("p"):
        a = para.get_text()
        lst.append(a)
    
    para = " ".join(lst[64:-6])
    return heading, para



@app.route("/searchforcompany",methods=['GET','POST'])
def searchforcompany():
    if request.method =="POST":
        global df
        global company
        global dfop
        global op1
        global dst
        global stockname
        global user_image
        global dst1
        legend = 'Stock Price data'
        company =request.form.get('company1')
        print("company1")
        print(company)
    #----------------- company 1 -------------------------------------------------------
    
        op1= company.replace(" ","").lower()
        
        ScrpLiveData(op1)
        # print(op1)
        df=pd1.read_csv('livedata//'+op1+'.csv')
        temperatures1 = list(df['Close'])
        # print("temperatures1")
        # print(temperatures1)
        times1 = list(df['Date'])
        # print("times1")
        # print(times1)
        
        from datetime import date
        dtnow = date.today()
        final_dt = str(dtnow).split("-")

        company_name = company
        cmp_name = company_name.replace(" ","%20")
        cmp_name = cmp_name.replace("&","%26")

        url="https://news.google.com/search?q="+cmp_name+"&hl=en-IN&gl=IN&ceid=IN%3Aen"

        html_content = requests.get(url).text

        soup = BeautifulSoup(html_content, 'html.parser')
        # print(soup.prettify())

        table1 = soup.find_all('a', attrs = {'class':'DY5T1d RZIKme'})

        headng = []
        lnk = []
        for i in table1:
            a = i["href"]
            lnk.append(a)
            b = i.text
            headng.append(b)

        headng = headng[:5]
        lnk = lnk[:5]
        
        link_lst = []
        for i in range(len(lnk)):
            a = "https://news.google.com/"+lnk[i][2:]
            link_lst.append(a)
        lnk = link_lst

        flst = zip(headng,lnk)

        a=str((op1).replace('.', '_'))+".png"
        # print("a")
        # print(a)
        dst1=r'static/logo/'+a
        
        global dt4 
        global ttf4
        ft1 = predictpriceofdata(op1)
 
        # global BDLSTMttf 
        # global BDLSTMdt
        dictofdateandprice, BDLSTMttf, BDLSTMdt,predictresult2 = BiDirectionalLSTM(op1)

        # # global BDGRUttf
        # # global BDGRUdt
        dictofdateandprice, GRUttf, GRUdt,predictresult3 = BiDirectionalGRU(op1)

        # # global GRUdt
        # # global GRUttf
        dictofdateandprice, BGRUttf, BGRUdt,predictresult4 = GRUpredictprice(op1)
        
        # # global CLSTMdt
        # # global CLSTMtf
        dictofdateandprice, BGRUttf, BGRUdt,predictresult5 = CNNLSTMpredictprice(op1)

        # # global CGRUdt
        # # global CGRUtf
        dictofdateandprice, BGRUttf, BGRUdt,predictresult6 = CNNGRUpredictprice(op1)

        return render_template('line_chart1.html', flst=flst, user_image=dst1, values=temperatures1, labels=times1, legend=legend,stockname=company,symbolis=op1)
    return render_template('line_chart1.html')

@app.route("/searchsingle",methods=['GET','POST'])
def searchsinglecompany():
    if request.method =="POST":
        global dfc
        global company4
        global dfop
        global dfop
        global dst
        
        legend = 'Stock Price data'
        company4 =request.form.get('company1')
        print("company")
        print(company4)
        dfop="coin_"+company4.replace(" ","")
        print("==dfop==")
        print(dfop)
        # dfop=str(dfop['Symbol'].iloc[0])
        # print(dfop)
        dfc=pd1.read_csv('data//'+dfop+'.csv')
        temperatures = list(dfc['Close'])
        times = list(dfc['Date'])

        from datetime import date
        dtnow = date.today()
        final_dt = str(dtnow).split("-")

        company_name = company4
        cmp_name = company_name.replace(" ","%20")
        cmp_name = cmp_name.replace("&","%26")

        url="https://news.google.com/search?q="+cmp_name+"&hl=en-IN&gl=IN&ceid=IN%3Aen"

        html_content = requests.get(url).text

        soup = BeautifulSoup(html_content, 'html.parser')
        # print(soup.prettify())

        table1 = soup.find_all('a', attrs = {'class':'DY5T1d RZIKme'})

        headng2 = []
        lnk2 = []
        for i in table1:
            a = i["href"]
            lnk2.append(a)
            b = i.text
            headng2.append(b)

        headng2 = headng2[:5]
        lnk2 = lnk2[:5]
        
        link_lst = []
        for i in range(len(lnk2)):
            a = "https://news.google.com/"+lnk2[i][2:]
            link_lst.append(a)
        lnk2 = link_lst

        flst2 = zip(headng2,lnk2)


        urlofsite='https://www.usatoday.com'
        io=0
       
        a=str((dfop).replace('.', '_'))+".png"
        print("a")
        print(a)
        
        dst=r'static/logo/'+a
        print("dst")
        print(dst)
        return render_template('line_chart3.html',flst2=flst2, user_image4=dst, values=temperatures, labels=times, legend=legend,stockname=company4,symbolis=dfop)
    return render_template('line_chart3.html')
    #return op1

import tweepy as tw
import tweepy
  
# Fill the X's with the credentials obtained by  
# following the above mentioned procedure. 
consumer_key = '5L4iF101tBHb0vVUQ7uCph3LR'
consumer_secret = 'kKCDjgvrIO012yCAE8FCsL6kcHDo344i0SJjSlm4FG2YxL7x5f'
access_key = '2842121736-dv73nAcb76ssBtHt0YSimalWRnvOiwnyXeEE9SW'
access_secret = "MgeXZivCLXglBxxAjtPafsveVMQiJLeSTn82zCKm3JnpB"
  
# Function to extract tweets 
def get_tweets(username): 
          
        # Authorization to consumer key and consumer secret 
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 
  
        # Access to user's access key and access secret 
        auth.set_access_token(access_key, access_secret) 
  
        # Calling api 
        api = tweepy.API(auth) 
  
        # 10 tweets to be extracted 
        tweets = tw.Cursor(api.search_tweets, q=username, lang="en").items(5)
  
        # Empty Array 
        tmp=[]  
  
        # create array of tweet information: username,  
        # tweet id, date/time, text 
        tweets_for_csv = [tweet.text for tweet in tweets] # CSV file created  
        for j in tweets_for_csv: 
  
            # Appending tweets to the empty array tmp 
            tmp.append(j)  
  
        # Printing the tweets 
        # print(tmp) 
        return tmp

#-----------------------------------------------------------------------------------------------------------------------
#                                       Tweet Sentiment
#-----------------------------------------------------------------------------------------------------------------------
from textblob import TextBlob
# d = df['tweet_text'].astype(str)
new_list=[]
def get_tweet_sentiment(d):    
    for i in range(len(d)):
        # print(d[i])
        val=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ", d[i]).split())
        analysis = TextBlob(val)  
        if analysis.sentiment.polarity > 0: 
            #print('positive')
            #return 'positive'
            a = 'positive'
            new_list.append(a)
        elif analysis.sentiment.polarity == 0: 
            #print('neutral')
            #return 'neutral'
            b = 'neutral'
            new_list.append(b)
        else: 
            #print('negative')
            #return 'negative'
            c = 'negative'
            new_list.append(c)
    return new_list

@app.route("/futurepriceprediction")
def futurepriceprediction():
    companySymbol = request.args.get('msg')
    dictis=predictpriceofdata(companySymbol)
    #print('price is')
    print(dictis)
    #print(sendingcompaniesinfo)
    return dictis  


@app.route("/")
def searching():
    temperatures = dict(df1['Name'])

    sendingcompaniesinfo={}
    for keys in temperatures: 
        temperatures[keys] = str(temperatures[keys]) 

        alg=str(temperatures[keys])+".png"
        # print("alg")
        # print(alg)
        lg=r'static/logo/'+alg
        # print("lg")
        # print(lg)

        sendingcompaniesinfo[temperatures[keys]]=lg
    # print(sendingcompaniesinfo)
    # print(temperatures[keys])
    return render_template('searching.html', values=sendingcompaniesinfo)

@app.route("/searchsing")
def searchsing():
    temperatures = dict(df1['Name'])

    sendingcompaniesinfo={}
    for keys in temperatures: 
        temperatures[keys] = str(temperatures[keys]) 

        alg=str(temperatures[keys])+".png"
        # print("alg")
        # print(alg)
        lg=r'static/logo/'+alg
        # print("lg")
        # print(lg)

        sendingcompaniesinfo[temperatures[keys]]=lg
    # print(sendingcompaniesinfo)
    # print(temperatures[keys])
    return render_template('searching2.html', values=sendingcompaniesinfo)

@app.route("/pred")
def pred():
    temperatures = dict(df1['Name'])

    sendingcompaniesinfo={}
    for keys in temperatures: 
        temperatures[keys] = str(temperatures[keys]) 

        alg=str(temperatures[keys])+".png"
        # print("alg")
        # print(alg)
        lg=r'static/logo/'+alg
        # print("lg")
        # print(lg)

        sendingcompaniesinfo[temperatures[keys]]=lg
    print(sendingcompaniesinfo)
    # print(temperatures[keys])
    return render_template('dt.html', values=sendingcompaniesinfo)

@app.route("/predictionofprice",methods=['GET','POST'])
def pricepred():
    global dst
    if request.method =="POST":
        global dst
        global ttf4
        
        import datetime as dt
        legend = 'Stock Price data'
        
        company =request.form.get('company')
        datefromui=request.form.get("date1")

        print("company")
        print(company)
        dfop=df1.loc[df1['Name'] == company]
        print
        op1=str(dfop['Symbol'].iloc[0])
        df=pd1.read_csv('data//'+op1+'.csv')
        temperatures = list(df['Close'])
        times = list(df['Date'])
        
        a=str((op1).replace('.', '_'))+".png"
        # print("a")
        # print(a)
        dst=r'static/logo/'+a
        # print("dst")
        # print(dst)
        
        dictofdateandprice, ttf4, dt4 = predictpriceofdata(op1)
        ttf4.append(ttf4)
        dt4.append(dt4)
        
            
        #dt = dt.datetime(int(datefromui))
        print("datefromui")
        print(datefromui)
        #date_object = datetime.datetime.strptime(str(datefromui), '%d/%m/%y')
        priceis=dictionaryofdateandprice[datefromui]
        print(priceis)
        return render_template('predictionobtained.html',user_image=dst, values=temperatures, labels=times, legend=legend,stockname=company,symbolis=op1,dt=datefromui, priceis=priceis)
    return render_template('predictionobtained.html')
        #return op1

@app.route("/index")
def index():
    return render_template('index.html')

@app.route("/contact")
def contact():
    return render_template('contact.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/fgraph", methods=["GET", "POST"])
def fgraph():
    print('hi')
    # dic = 'static/futuregraph/abc.png'    
    global dt4 
    global ttf4
    global predictresult
   
   
    global BDLSTMttf 
    global BDLSTMdt
    global predictresult2
    
  
    global BDGRUttf
    global BDGRUdt
    global predictresult3
  
    global GRUdt
    global GRUttf
    global predictresult4
   
    global CLSTMdt
    global CLSTMtf
    global predictresult5
  
    global CGRUdt
    global CGRUtf
    global predictresult6
    
    return render_template('line_chart2.html',stockname=company,user_image=dst1,
                            values1=ttf4,labels1=dt4,predictresult=predictresult, 
                            values2=BDLSTMttf, labels2=BDLSTMdt,predictresult2=predictresult2,
                            values3=BDGRUttf, labels3=BDGRUdt,predictresult3=predictresult3, 
                            values4=GRUttf, labels4=GRUdt,predictresult4=predictresult4, 
                            values5=CLSTMtf, labels5=CLSTMdt,predictresult5=predictresult5, 
                            values6=CGRUtf, labels6=CGRUdt,predictresult6=predictresult6,
                            symbolis=op1)



@app.route("/fgraph2", methods=["GET", "POST"])
def fgraph2():
    print('hi')
    # dic = 'static/futuregraph/abc.png'
    global dt4 
    global dst
    global ttf4
    

    return render_template('line_chart4.html',user_image4=dst, values4=ttf4, labels4=dt4, symbolis4=dfop, stockname4=company4)

@app.route("/simple_chart")
def chart():
    legend = 'Monthly Data'
    labels = ["January", "February", "March", "April", "May", "June", "July", "August"]
    values = [10, 9, 8, 7, 6, 4, 7, 8]
    return render_template('chart.html', values=values, labels=labels, legend=legend)

@app.route("/line_chart")
def line_chart():
    legend = 'Temperatures'
    temperatures = list(df['Close'])
    times = list(df['Date'])
    return render_template('line_chart.html', values=temperatures, labels=times, legend=legend)

@app.route("/price")
def price():
    global df
    userText = request.args.get('msg')
    print(userText)
    print(df)
    op=dict(df.iloc[int(userText)])#tuple(list(df.iloc[int(userText)]))
    print(op)
    #for dicts in test_list: 
    for keys in op: 
        op[keys] = str(op[keys]) 
    return op

@app.route("/price2")
def price2():
    global dfc
    userText2 = request.args.get('msg')
    print("==userText2==")
    print(userText2)
    print("====dff===")
    print(dfc)
    op5=dict(dfc.iloc[int(userText2)])#tuple(list(df.iloc[int(userText)]))
    print(op5)
    #for dicts in test_list: 
    for keys in op5: 
        op5[keys] = str(op5[keys]) 
    return op5

if __name__ == "__main__":
    app.run('0.0.0.0')
    # app.run(debug=True)
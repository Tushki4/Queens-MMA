# -*- coding: utf-8 -*-
import datetime
import numpy as np
import pandas as pd
import sys
 
from pandas.io.data import DataReader
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn import svm
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn import svm, cross_validation, neighbors

 
from TechInd import MA
from TechInd import RSI
from TechInd import MOM
from TechInd import BBANDS
from TechInd import MACD
from TechInd import MassI
from TechInd import TSI
from TechInd import FORCE
 
 
 
def create_lagged_series(symbol, start_date, end_date, lags=5):
    """This creates a pandas DataFrame that stores the percentage returns of the
    adjusted closing value of a stock obtained from Yahoo Finance, along with
    a number of lagged returns from the prior trading days (lags defaults to 5 days).
    Trading volume, as well as the Direction from the previous day, are also included."""
 
    # Obtain stock information from Yahoo Finance
    ts = DataReader(symbol, "yahoo", start_date-datetime.timedelta(days=365), end_date)
   
    # Create the new lagged DataFrame
    tslag = pd.DataFrame(index=ts.index)
    tslag["Volume"] = ts["Volume"]
    tslag["Today"] = ts["Adj Close"]
 
    # Create the shifted lag series of prior trading period close values
    for i in range(0,lags):
        tslag["Lag%s" % str(i+1)] = ts["Adj Close"].shift(i+1)
   
    # Create the returns DataFrame
    tsret = pd.DataFrame(index=tslag.index)
    tsret["Volume"] = tslag["Volume"]
    tsret["Today"] = tslag["Today"].pct_change()*100.0
    tsret["Adj Close"] = tslag["Today"]
   
    #########Calculate Techicals##############
    tsret["MA5"] = ts["Adj Close"].expanding(min_periods=5).mean()
    tsret["MA10"] = ts["Adj Close"].expanding(min_periods=10).mean()
    tsret["STD05"] = ts["Adj Close"].expanding(min_periods=5).std()
    tsret["STD10"] = ts["Adj Close"].expanding(min_periods=10).std()
    tsret = MOM(tsret,10)
    tsret = MACD(tsret,10,20)
##    tsret = MassI(tsret, 10)
##    tsret = FORCE(tsret, 10)
   
        
    #print (tsret)
    # If any of the values of percentage returns equal zero, set them to
    # a small number (stops issues with QDA model in scikit-learn)
    for i,x in enumerate(tsret["Today"]):
        if (abs(x) < 0.0001):
            tsret["Today"][i] = 0.0001
 
    # Create the lagged percentage returns columns
    for i in range(0,lags):
        tsret["Lag%s" % str(i+1)] = tslag["Lag%s" % str(i+1)].pct_change()*100.0
 
    # Create the "Direction" column (+1 or -1) indicating an up/down day
    tsret["Direction"] = np.sign(tsret["Today"])
    tsret = tsret[tsret.index >= start_date]

    tsret.to_csv('c:\\stocks\\' + symbol + '.csv')
    
    return tsret
 
def fit_model(name, model, X_train, y_train, X_test, pred):
    """Fits a classification model (for our purposes this is LR, LDA and QDA)
    using the training data, then makes a prediction and subsequent "hit rate"
    for the test data."""
 
    # Fit and predict the model on the training, and then test, data
    model.fit(X_train, y_train)
    pred[name] = model.predict(X_test)
   
    # Create a series with 1 being correct direction, 0 being wrong
    # and then calculate the hit rate based on the actual direction
    pred["%s_Correct" % name] = (1.0+pred[name]*pred["Actual"])/2.0
    hit_rate = np.mean(pred["%s_Correct" % name])
    print ("%s: %.3f" % (name, hit_rate))

    

   
       
def decision_maker(pred, symbol):
   
    pred['sumPrediction'] = pred["rfor"] + pred["LDA"] + pred["QDA"] + pred["SVM"] + pred["ADA"]
    pred['decision'] = np.where(pred['sumPrediction'] >= 1, 'BUY', 'SELL' )
    
#    print(pred)
    #hit_rate.to_csv('c:\\stocks\\hitrates\\' + symbol + '_hitrates.csv')   
    pred.to_csv('c:\\stocks\\predictions\\' + symbol + '_prediction.csv')
   
def measure_success(pred):
    return
 
def generate_predictions(snpret, symbol, start_test):   
 
    # Create a lagged series of the S&P500 US stock market index
    #snpret = create_lagged_series("^GSPC", datetime.datetime(2001,1,10), datetime.datetime(2010,12,31), lags=5)
    #snpret = create_lagged_series("AAPL", datetime.datetime(2001,1,10), datetime.datetime(2015,12,31), lags=5)
    #snpret = create_lagged_series(symbol, datetime.datetime(2001,1,10), datetime.datetime(2015,12,31), lags=5)
   
        
    # Use the prior two days of returns as predictor values, with direction as the response
    #X = snpret[["Lag1","Lag2","Lag3","Lag4", "MA5", "Momentum_10", "Cluster", "Volume"]]
    #X = snpret[["Lag1","Lag2","Lag3","Lag4","MA5", "Momentum_10", "MACD_10_20", "MACDsign_10_20", "MACDdiff_10_20", "Volume"]]

    X = snpret[["Lag1","Lag2","Lag3","Lag4","MA5", "Momentum_10", "MACD_10_20", "MACDsign_10_20", "MACDdiff_10_20", "Volume", "STD05", "STD10"]]
    #X = snpret[["MA5", "MA10", "Volume"]]
    y = snpret["Direction"]
    
    # The test data is split into two parts: Before and after 1st Jan 2005.
    #start_test = datetime.datetime(2008,1,1)
    #start_test = datetime.datetime(2008,1,1)
 
    # Create training and test sets
    X_train = preprocessing.scale(X[X.index < start_test])
    X_test = preprocessing.scale(X[X.index >= start_test])
    y_train = y[y.index < start_test]
    y_test = y[y.index >= start_test]

    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.3)
 
    # Create prediction DataFrame
    pred = pd.DataFrame(index=y_test.index)
    pred["Symbol"] = symbol
    pred["Actual"] = y_test
    pred["Adj Close"] = snpret["Adj Close"]
   
        
##    # Create and fit the three models   
    print ("Hit Rates:")
    models = [("rfor", RandomForestClassifier()), ("LDA", LDA()), ("QDA", QDA()), ("SVM", svm.LinearSVC()), ("ADA", GradientBoostingClassifier())]
    
    for m in models:
        fit_model(m[0], m[1], X_train, y_train, X_test, pred)
        
##    forest = ExtraTreesClassifier(n_estimators=250,
##                              random_state=0)
##
##    forest.fit(X_train, y_train)
##    importances = forest.feature_importances_
##    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
##             axis=0)
##    indices = np.argsort(importances)[::-1]
##
##    # Print the feature ranking
##    print("Feature ranking:")
##
##    for f in range(X.shape[1]):
##            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
##
##    # Plot the feature importances of the forest
##    plt.figure()
##    plt.title("Feature importances")
##    plt.bar(range(X.shape[1]), importances[indices],
##       color="r", yerr=std[indices], align="center")
##    plt.xticks(range(X.shape[1]), indices)
##    plt.xlim([-1, X.shape[1]])
##    plt.show()
   
    return pred
     
    #for index, row in pred.iterrows():
    #    if row["sumPred"] >= 1:
    #        print(row)  
        
if __name__ == "__main__":
    start_date = datetime.datetime(2014,1,1)
    end_date = datetime.datetime(2017,2,28)
    start_test = datetime.datetime(2015,7,1)
   
    file = open("C:\\temp\\SP100.csv")
    #with open("C:\\temp\\SP100.csv", "rb") as f:

    counter = 0;
   
    for line in file:

        print (line)

        try:

            symbolArr = line.split(',')
            symbol = symbolArr[0]
            cluster = symbolArr[1].split('\n')
            #ts = DataReader(symbol, "yahoo", start_date-datetime.timedelta(days=365), end_date)
            #snpret = create_lagged_series(symbol, datetime.datetime(2001,1,10), datetime.datetime(2015,12,31), lags=5)
            snpret = create_lagged_series(symbol, start_date-datetime.timedelta(days=365), end_date, lags=5)
            snpret["Cluster"] = cluster[0]
   
#            print(snpret)

            pred = generate_predictions(snpret, symbol, start_test)   
            decision_maker(pred, symbol)

            counter = counter + 1
            print(repr(counter) + ' - ' + symbol)
            
#            break

        except:

            print (sys.exc_info()[0])
            print('cannot retrieve ' + symbol)

            break

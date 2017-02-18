# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import os.path
import datetime as dt

stock_data_path = "C://stocks//"


def get_data(symbol):    
    """Reads stock data (adjusted close) for given symbols from CSV files."""

    load_place = stock_data_path + symbol + ".csv"

    print ("load_place =", load_place)
    df_temp = pd.read_csv(load_place, 
                          index_col='Date',
                          parse_dates=True,  
                          na_values=['nan'])   
    return df_temp
    
def get_all_data():
    """Loading all stock data"""
    
    stock_data = {}    
    
    stock_list = get_stock_list()
    for stock in stock_list:
        stock_data[stock] = get_data(stock)
    return stock_data    
    
def get_stock_list():
    """Reads list of stocks from file"""
    load_place = stock_data_path + "SP100noABBVorNEEorPYPL.csv"

    stocklist = []
    infile = open( load_place,'r')
    for line in infile:
        stock = line.strip()
        if stock != 'BRK.B':
            stocklist.append(stock)
    infile.close()
    return stocklist

def create_portfolio(stock_list, cash):
    """Creates portfolio object"""

    portfolio = {}

    for stock in stock_list:
        portfolio[stock] = {'HOLD':0,'VAL':0}
    
    portfolio['MYCASH'] = {'HOLD':cash,'VAL':cash}
    return portfolio

def get_predictions():
    
    stock_predictions = {}    
    
    stock_list = get_stock_list()
    for stock in stock_list:
        load_place = stock_data_path + "predictions//"+ stock + "_prediction.csv"

        if os.path.isfile(load_place):
            df_temp = pd.read_csv(load_place, 
                          index_col='Date',
                          parse_dates=True,  
                          na_values=['nan'])[['decision']] 
            stock_predictions[stock] = df_temp
    return stock_predictions    

def get_SP100():    
    """Reads S&P 100 from CSV file."""

    load_place = stock_data_path + "SP100_OEX.csv"

    df_temp = pd.read_csv(load_place, 
                          index_col='Date',
                          parse_dates=True,  
                          na_values=['nan'])   
    return df_temp



if __name__=="__main__":
   print ("One does not simply think up a strategy")


   print(get_stock_list())   
 
   #symbol = 
          
   prices = get_data(symbol)

   print(symbol)
   print(prices)


#   prices['2014'][['Adj Close','MA_10']].plot(figsize=(12,8));
#   prices['2014'][['Volume']].plot(figsize=(12,3));
#   prices['2014'][['Momentum_10']].plot(figsize=(12,3));

   
   print(get_predictions())

   print(get_SP100()['Adj Close'])
    
      

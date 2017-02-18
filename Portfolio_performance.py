# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime as dt
import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import util


INITIAL_CASH = 10000000
START_DATE = dt.date(2016,1,1) 
END_DATE = dt.date(2016,12,31)
COST_OF_TRADE = 8


PORTFOLIO_FILE = "C://stocks//portfolio.csv"

def run_simulation():
    """ Run simulation """
    
    print ("Getting list of stocks")
    stock_list = util.get_stock_list()
    
    
    print ("Loading historical data")
    stock_data = util.get_all_data()

    print ("Creating initial portfolio") 
    portfolio = util.create_portfolio(stock_list, INITIAL_CASH)

    print ("Get stock predictions")
    stock_predictions = util.get_predictions()
    #print stock_predictions

    print ("Get S&P 100 historical data")
    sp100 = util.get_SP100()
    #print sp100['Adj Close']

    #print "Showing historical data for MSFT"
    #stock_data['MSFT']['2014'][['Adj Close','MA_10']].plot(figsize=(12,8));

    #print portfolio
    
    # Date range to simulate trading
    dates = pd.date_range(START_DATE, END_DATE)
    print (dates)
    #test_dates = pd.date_range(dt.date(2015,12,1), dt.date(2015,12,31))
    
    #Portfolio Dataframe - is used as a program output
    portfolio_df = pd.DataFrame(columns=['VAL','BENCHMARK'],index=dates)
    
    #is used to figure out first date with "good' data
    first_avaliable_date_in_range = None
    
    for date in dates:
        actions = {}

        print (date)
        #Genereate action/prediction for each stock 
#        for stock in stock_list:
#            if date in stock_data[stock].index:
#                # generate action randonmly for testing
#                action = random.choice(['BUY','SELL','DO_NOTHING'])
#                actions[stock] = action


        #Try to find recommended actions for the date
        for stock in stock_list:
            print (stock)
            if date in stock_data[stock].index:
                print (stock_data[stock].index)
                if stock in stock_predictions:
                    action = stock_predictions[stock].at[date,'decision']
                #else:
#                    print("Empty predictions for: ", stock)
                #    action = 'DO_NOTHING'
                
                actions[stock] = action
                
                # Valuate portfolio for date 
                stock_price = stock_data[stock].at[date,'Adj Close'] 
                portfolio[stock]['VAL'] =  portfolio[stock]['HOLD'] * stock_price
 
       # if date in test_dates :
       #      print "Date:" , date, "Processing Actions: ", len(actions.keys())
 
        
        #Simulate actions directly on portfolio
        stocks_for_actions = actions.keys()
 
       # Calculate position 'buy and hold for index'
        if len(stocks_for_actions) > 0 and first_avaliable_date_in_range == None:
            first_avaliable_date_in_range = date
            sp100_position = INITIAL_CASH / sp100.at[date,'Adj Close']
            print("sp100_position",sp100_position)



        if len(stocks_for_actions) > 0:    
            
           # print "My portfolio before actions"               
           # print portfolio
            
            #Process actions
            
            # SELL based on prediction
            stocks_to_buy = []              
            for stock in stocks_for_actions:
                if actions[stock] == 'SELL':
                    if portfolio[stock]['HOLD'] > 0:
                        stock_price = stock_data[stock].at[date,'Adj Close']                  
                        #sell full position
                        portfolio['MYCASH']['VAL'] = portfolio['MYCASH']['VAL'] + stock_price * portfolio[stock]['HOLD']  - COST_OF_TRADE
                        portfolio['MYCASH']['HOLD'] = portfolio['MYCASH']['VAL']
                        portfolio[stock]['HOLD'] = 0
                        portfolio[stock]['VAL'] = 0
                elif actions[stock] == 'BUY':
                    stocks_to_buy.append(stock)
            
            #if date in test_dates :
            #    print "Date: ", date, "Stocks to buy: ", stocks_to_buy  
            
            #Find portfolio value after recommended sell
            portfolio_value_after_sell = 0
            stocks_to_hold = []            
            for position in portfolio.keys():
                if portfolio[position]['HOLD'] > 0: 
                    portfolio_value_after_sell = portfolio_value_after_sell + portfolio[position]['VAL'] 
                    if not(position=='MYCASH'):
                        stocks_to_hold.append(position)
           
            #print 'Stocks to hold: ', stocks_to_hold
            
            #Find uinion of stocks to hold and buy
            result_list = [stocks_to_hold,  stocks_to_buy]
            stocks_for_next_cycle = list(set().union(*result_list))
            
            #print 'Stocks for next cycle: ', stocks_for_next_cycle

            #Find average for each stock to support rebalancing
            average_val = (portfolio_value_after_sell - len(stocks_for_next_cycle)*COST_OF_TRADE) / len(stocks_for_next_cycle)
            if average_val < 0:
                average_val = 0
            
            print ("Average Val: ", average_val)
            
            # rebalance portfolio
            
            
            #it should be called with buy also, but not changing name now
            stocks_to_sell_for_cost = 0 
            
            # sell some based on rebalancing rule
            for stock in stocks_for_next_cycle:
                stock_price = stock_data[stock].at[date,'Adj Close']
                number_to_hold = int(average_val / stock_price)
                if portfolio[stock]['HOLD'] > number_to_hold:
                    stocks_to_sell_for_cost = stocks_to_sell_for_cost + 1
                    how_many_to_sell =  portfolio[stock]['HOLD']  -  number_to_hold
                    portfolio['MYCASH']['VAL'] = portfolio['MYCASH']['VAL'] + stock_price * how_many_to_sell
                    portfolio['MYCASH']['HOLD'] = portfolio['MYCASH']['VAL']
                    portfolio[stock]['HOLD'] =  number_to_hold
                    portfolio[stock]['VAL'] =  stock_price * number_to_hold
            
            # buy some based on rebalancing rule
            for stock in stocks_for_next_cycle:
                stock_price = stock_data[stock].at[date,'Adj Close']
                number_to_hold = int(average_val / stock_price)
                if portfolio[stock]['HOLD'] < number_to_hold:
                    stocks_to_sell_for_cost = stocks_to_sell_for_cost + 1
                    how_many_to_buy =  number_to_hold - portfolio[stock]['HOLD']
                    portfolio['MYCASH']['VAL'] = portfolio['MYCASH']['VAL'] - stock_price * how_many_to_buy
                    portfolio['MYCASH']['HOLD'] = portfolio['MYCASH']['VAL']
                    portfolio[stock]['HOLD'] =  number_to_hold
                    portfolio[stock]['VAL'] =  stock_price * number_to_hold
            
            #update value for each position based on historical data 

            # update CASH based on trading costs

            #calculate costs
            portfolio['MYCASH']['VAL'] = portfolio['MYCASH']['VAL'] -  stocks_to_sell_for_cost * COST_OF_TRADE

            if portfolio['MYCASH']['VAL'] < 0:
                print ('bad: negative cash: ', portfolio['MYCASH']['VAL']) 
            else:
                #print 'ok: cach poistion: ', portfolio['MYCASH']['VAL']
                pass

#            #Using available MYCASH to buy stocks (one by one)
#            can_buy = True
#            while can_buy:
#               can_buy = False
#               for stock in stocks_to_buy:
#                   stock_price = stock_data[stock].at[date,'Adj Close'] 
#                   if portfolio['MYCASH']['VAL'] > stock_price:
#                       portfolio['MYCASH']['VAL'] = portfolio['MYCASH']['VAL'] - stock_price
#                       portfolio['MYCASH']['HOLD'] = portfolio['MYCASH']['VAL']
#                       portfolio[stock]['HOLD'] =  portfolio[stock]['HOLD'] + 1
#                       portfolio[stock]['VAL'] =  portfolio[stock]['HOLD'] *  stock_price
#                       can_buy = True
                       

#            print "My portfolio after processing actions"               
#            print portfolio

            #calculate portfolio value including stocks and cash
            portfolio_value = 0
            for position in portfolio.keys():
                portfolio_value = portfolio_value + portfolio[position]['VAL']
            
            portfolio_df.at[date,'VAL'] = portfolio_value
            portfolio_df.at[date,'BENCHMARK'] = sp100.at[date,'Adj Close'] *  sp100_position
            #portfolio_df["StdDev"] = portfolio_df["BENCHMARK"].expanding(min_periods=10).std()
            
    portfolio_df.dropna(inplace=True)

    #print(portfolio.head())

    
    #Saving portfolio dataframe to csv file    
    portfolio_df.to_csv(PORTFOLIO_FILE)   
    
    print ("Showing Portfolio performance")
    portfolio_df.plot(figsize=(12,8))
    plt.show()

    covmat = np.cov(portfolio_df['VAL'], portfolio_df['BENCHMARK'])

    beta = covmat[0,0]/covmat[1,1]
    alpha= np.mean(portfolio_df["VAL"])-beta*np.mean(portfolio_df["BENCHMARK"])

# r_squared     = 1. - SS_res/SS_tot
    yprd = alpha + beta * portfolio_df["BENCHMARK"]
    SS_res = np.sum(np.power(yprd-portfolio_df["VAL"],2))
    SS_tot = covmat[0,0]*(len(portfolio_df)-1) # SS_tot is sample_variance*(n-1)
    r_squared = 1. - SS_res/SS_tot
# 5- year volatiity and 1-year momentum
    volatility = np.sqrt(covmat[0,0])
    momentum = np.prod(1+portfolio_df["VAL"].tail(12).values) -1

# annualize the numbers
##    prd = 12. # used monthly returns; 12 periods to annualize
##    alpha = alpha*prd
    #volatility = volatility*np.sqrt(prd)

    print(beta, alpha, r_squared, volatility, momentum)
    
    
    #print portfolio_df['2015-12']
            
    #stock_price = stock_data['MSFT'].loc[dt.date(2015,4,9),'Adj Close']
  
    #print "Checking index"  
    #print  dt.date(2015,4,9) in stock_data['MSFT'].index    
    
    #print "Last actions"
    #print actions


if __name__=="__main__":
   # print "One does not simply think up a strategy"


   print("Let's start simulation!")   
 
   run_simulation()

   print("We are done with simulation")

   

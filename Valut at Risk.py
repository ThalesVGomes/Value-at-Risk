import numpy as np
import pandas as pd
from pandas_datareader import DataReader
from pandas_datareader._utils import RemoteDataError
import scipy.stats as st

def ewma_vol(returns, lamb=0.94):
    
    returns = np.array(returns)
    
    size = np.array(range(len(returns)))[::-1]
    mean = np.mean(returns)
    
    vol = np.sqrt(np.sum((returns-mean) ** 2 * (1-lamb) * (lamb ** size)))
    return vol

def ewma_corr(a, b, lamb=0.94):
    a = np.array(a)
    b = np.array(b)
    
    size = np.array(range(len(a)))[::-1]
    corr = np.sum((1-lamb) * (lamb ** size) * (a - np.mean(a)) * (b - np.mean(b)))
    
    vol1 = ewma_vol(a, lamb)
    vol2 = ewma_vol(b, lamb)
    
    corr = corr / (vol1 * vol2)
    
    return corr

def Value_at_risk(tickers, date, weights=None, EWMA_model=True, confidence=0.99, samples=252, lamb=0.94):
    
    
    """
    Calculates the daily Value at Risk of a
    given stock or portfolio (list of stocks)
    
    The model assumes a parametric T-distribution for the returns.
    
    The returns are modeled as logarithm.

    -------------------------------------------------------
    tickers:
        List of stock names, ex: [GOOG, AAPL] or [GOOG] for just one stock
        Important: You must pass the exact name used in https://finance.yahoo.com/
    date:
        The date of the VaR. 
        Note: The VaR always refers to the maximum loss of the next day of the date.
    weights:
        The weight of each stock in the portfolio composition.
        If None, assumes a equally weighed portfolio.
    EWMA_model:
        To calculate the VaR using the EWMA model of volatility and correlation
        EWMA means Exponentially Weighted Moving Average
    confidence:
        The level of confidence of the VaR.
    samples:
        How many past return days to use to calculate the VaR
    lamb:
        It's the lambda parameter for EWMA model.
        More on: https://www.investopedia.com/articles/07/ewma.asp
    """
    
    # Deals with possible errors
    
    if not isinstance(tickers, list):
        raise Exception("You must pass the argument tickers as a list.")
        
    tickers = [t.upper() for t in tickers]

    if weights is None:
        q = len(tickers)
        weights = np.array([1/q for _ in range(q)]) # Equal weights
    else:
        if len(weights) != len(tickers):
            raise Exception('weights and tickers must have the same size')
        weights = np.array(weights)
        
    # Start processing
    
    volatility = np.array([])
    
    start = pd.to_datetime(date, dayfirst=True)
    adjust_date = ((30 + samples)/252) * 365 # Some extra days to compensate holidays
    start = start - pd.to_timedelta(adjust_date, unit='d') 
    date = pd.to_datetime(date, dayfirst=True)
    
    all_data = pd.DataFrame()
    
    for pos, ticker in enumerate(tickers):
        
        try:
            data = DataReader(ticker, 'yahoo', start, date)
        except Exception as e:
            print(f"Couldn't find {ticker} in Yahoo Finance. Error code: {e}")
            
        size = data.shape[0]
        data = data.iloc[size-(samples+1):]

        data[f'Log Returns {ticker}'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
        data = data[f'Log Returns {ticker}'].to_frame()
        data.dropna(inplace=True)
        
        if EWMA_model:
            vol = ewma_vol(data[f'Log Returns {ticker}'], lamb=lamb)
        else:
            vol = data[f'Log Returns {ticker}'].std(ddof=1)
            
        volatility = np.append(volatility, vol)
        
        if pos == 0:
            all_data = all_data.append(data)
        else:
            all_data = pd.merge(all_data, data, left_index=True, right_index=True)
            
            
    if EWMA_model:
        corr = all_data.corr(ewma_corr)
    else:
        corr = all_data.corr()
        
    factor = st.t.ppf(q=confidence, df=samples)
    weights_vol = weights * volatility

    var = np.sqrt(np.dot(np.dot(weights_vol.T, corr), weights_vol)) * factor
    
    return var

# Ticker name must be exactly the same in https://finance.yahoo.com/
var = Value_at_risk(['GOOG', 'PETR4.SA', 'ITUB4.SA', 'VALE3.SA'], '12/02/2021', weights=[0.3, 0.1, 0.4, 0.2],
                    EWMA_model=True, confidence=0.99, samples=252, lamb=0.94)


print(f'Value at risk = {round(var*100,3)}%')
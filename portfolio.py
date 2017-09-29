import numpy as np
import pandas as pd
from lazy import lazy

class Portfolio (object):

    '''
    A Portfolio Object with

    Attributes:
    asset_return: Asset expected return. 1d np array
    asset_cov: 2d pandas dataframe asset covariance matrix with asset name as index and columns.
    asset_count: asset_cov.shape[0]
    asset_name: asset_cov.index
    asset_weight: the weight of each asset. 1d np array

    fully_invested: True/False. If the portfolio is fully invested (sum of weight is one)
    dollar_neutral: True/False. If the portfolio is dollar neutral (sum of weight is zero)
    expected_return: Portfolio expected return
    volatility: Portfolio volatility annualized

    '''



    def __init__(self, asset_ret, asset_cov, weight, benchmark_portfolio= None ):
        self.asset_return= np.array(asset_ret)
        self.asset_cov= asset_cov
        self.asset_name= asset_cov.index.tolist()
        self.asset_count= asset_cov.shape[0]
        self.weight= np.array(weight)
        self.benchmark_portfolio= benchmark_portfolio

    @lazy
    def full_invested(self):
        return (np.sum(self.weight)==1)

    @lazy
    def dollar_neutral( self):
        return np.sum(self.weight)==0

    @lazy
    def expected_return( self):
        return (np.dot( self.weight, self.asset_return))
    @lazy
    def volatility(self):
        return (np.sqrt( np.dot(np.dot( self.weight, self.asset_cov), self.weight)))


    def SharpeRatio(self, risk_free=0):
        return (self.expected_return-risk_free)/ self.volatility

    @lazy
    def DR(self):
        return np.dot(np.sqrt(np.diag(self.asset_cov)), self.weight)/ self.volatility

    def implied_ExpectedReturn(self, gamma, risk_free= 0, covariance= None):
        '''
        Assume the portfolio is mean-variance optimal
        so the expected return which leads to this portfolio weight can be recovered,
        given asset_cov and risk aversion factor gamma.

        gamma: risk aversion factor, usually 2-4
        covariance (optional): default to be self.asset_cov
        risk_free: risk free rate (default to be 0)

        :return: the implied expected return, 1d np array
        '''

        if covariance is None:
            covariance= self.asset_cov

        return (risk_free+ gamma * np.dot(covariance, self.weight)).tolist()


    def set_benchmark(self, benchmark_portfolio):
        '''
        attribute a benchmark portfolio object to self.benchmark_portfolio

        :param benchmark_portfolio: One portfolio object
        :return:
        '''

        self.benchmark_portfolio= benchmark_portfolio


    @lazy
    def active_weight(self):
        return (self.weight- self.benchmark_portfolio.weight)

    @lazy
    def tracking_error(self):
        return np.sqrt(np.dot(np.dot(self.active_weight, self.asset_cov), self.active_weight))

    @lazy
    def active_expectedReturn(self):
        return np.dot(self.active_weight, self.asset_return)

    @lazy
    def IR(self):
        return self.active_expectedReturn/ self.tracking_error

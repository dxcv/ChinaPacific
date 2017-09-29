
import pandas as pd
import numpy as np
from lazy import lazy


class naive_BlackLitterman(object):
    '''
    naive_BlackLitterman the posterior distribution of expected return, given prior and views.

    Attributes:
    prior_ExpectedReturn: the prior Expected Return, 1d np array
    prior_cov: prior return covariance matrix
    views_weight: the weight of view portfolios (each row represents a portfolio holding weight)
    views_return: the expected return of views portfolios, 1d np array
    prior_uncertainty: the matrix measuring the risk/ uncertainty of prior (can be diagonal or not diagonal)
    view_uncertainty: the matrix measuring the risk/ uncertainty of views (can be diagonal or not diagonal)

    post_ExpectedReturn: the post Expected Return, 1d np array
    post_uncertainty: the matrix measuring the risk/ uncertainty of post expected returns
    post_cov: post return covariance matrix

    '''


    def __init__(self, prior_ExpectedReturn, prior_uncertainty,
                 views_weight, views_return, views_uncertainty,
                 prior_cov):
        self.prior_ExpectedReturn= prior_ExpectedReturn
        self.prior_uncertainty= prior_uncertainty
        self.views_weight= views_weight
        self.views_return= views_return
        self.views_uncertainty= views_uncertainty
        self.prior_cov= prior_cov

    @lazy
    def prior_uncertainty_inv_(self):
        return np.linalg.inv(self.prior_uncertainty)

    @lazy
    def views_uncertainty_inv_(self):
        return np.linalg.inv( self.views_uncertainty)


    @lazy
    def post_ExpectedReturn(self):

        B= np.dot(self.prior_uncertainty_inv_, self.prior_ExpectedReturn)+ np.dot( np.dot(self.views_weight.T, self.views_uncertainty_inv_), self.views_return)
        return np.dot(self.post_uncertainty, B)

    @lazy
    def post_uncertainty(self):
        return np.linalg.inv( self.prior_uncertainty_inv_+ np.dot(np.dot(self.views_weight.T, self.views_uncertainty_inv_), self.views_weight))

    @lazy
    def post_cov(self):
        return (self.post_uncertainty+ self.prior_cov)


    def arith2geo(self):
        '''
        Transfer the arithemetic expected return to geometric if the self.post_ExpectedReturn is arithmetic
        :return:
        Geometric post Expected return

        '''

        return self.post_ExpectedReturn - .5* np.diag(self.post_cov)

    def geo2arith(self):
        '''
        Transfer the geometric expected return to arithmetic if the self.post_ExpectedReturn is geometric

        :return:
        Arithmetic post Expected Return
        '''

        return self.post_ExpectedReturn+ .5* np.diag(self.post_cov)
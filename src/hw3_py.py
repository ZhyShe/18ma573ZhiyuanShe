# -*- coding: utf-8 -*-
"""hw3.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1I7PZA5O8ck_9OWqH8Eu6aX4RhCh1xZDr
"""

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

# the following codes are used to define option class

'''
Otype: the typre of option, 1 for call option; -1 for Put option.
Strike: strike price for the option.
Maturity: expiry time for the option.
Market_Price: the price of underlying stock.

'''
#Code starts here
#----------------------
class Option:
  def __init__(self, Otype=1, Strike=110, Maturity=1, Market_Price=110):
    self.Otype=Otype
    self.Strike=Strike
    self.Maturity=Maturity
    self.Market_Price=Market_Price
  @staticmethod
  def Payoff(option, S):
    results=[]
    Otype=option.Otype
    K=option.Strike
    Maturity=option.Maturity
    for s in S:
      results.append(np.max([0,(s-K)*Otype]))
    return results

class GBM:
  def __init__(self, init_state=100, drift_ratio=0.0475, vol_ratio=.2):
    self.init_state = init_state
    self.drift_ratio = drift_ratio
    self.vol_ratio = vol_ratio
    self.BsmPrice=0
  @staticmethod
  def bsm_price(gbm,option):
    s0 = gbm.init_state
    sigma = gbm.vol_ratio
    r = gbm.drift_ratio
    
    otype = option.Otype
    k = option.Strike
    maturity = option.Maturity
    
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma ** 2) 
          * maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)
    gbm.BsmPrice=(otype * s0 * ss.norm.cdf(otype * d1)- otype * np.exp(-r * maturity) * k * ss.norm.cdf(otype * d2))
    return gbm.BsmPrice


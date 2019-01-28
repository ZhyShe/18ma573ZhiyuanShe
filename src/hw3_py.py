# -*- coding: utf-8 -*-
"""hw3.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1I7PZA5O8ck_9OWqH8Eu6aX4RhCh1xZDr
"""

import numpy as np
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
  def Payoff(self, S):
    results=[]
    Otype=self.Otype
    K=self.Strike
    Maturity=self.Maturity
    for s in S:
      results.append(np.max([0,(s-K)*Otype]))
    return results




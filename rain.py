# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 18:56:21 2019

@author: abzehr
"""

import math
import pandas as pd

def get_rainfall_surface(dd, file="/rainfall_coefficients.csv"):
    return pd.read_csv(dd + file)

def predict_rainfall_generic(coeff, c_p, delta_p, radius, lon, angle):
    
    
    beta_0 = coeff.at[0,'x'] # intercept
    beta_1 = coeff.at[1,'x'] # c_p
    beta_2 = coeff.at[2,'x'] # r_max
    beta_3 = coeff.at[3,'x'] # lon
    beta_4 = coeff.at[4,'x'] # angle == 0
    beta_5 = coeff.at[5,'x'] # angle == 45
    beta_6 = coeff.at[6,'x'] # c_p*r_max
    beta_7 = coeff.at[7,'x'] # lon*(angle==0)
    beta_8 = coeff.at[8,'x'] # lon*(angle==45)
    
    rainfall = beta_0 + (beta_1 * c_p) + \
    (beta_2 * radius) + (beta_3 * lon) + \
    (beta_6 * radius * c_p)
    if (angle == 0):
        rainfall = rainfall + beta_4 + (beta_7 * lon)
    if (angle == 45):
        rainfall = rainfall + beta_5 + (beta_8 * lon)
    rainfall = math.exp(rainfall)
    return rainfall
            
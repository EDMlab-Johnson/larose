# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:52:27 2020

@author: abzehr
"""
from rhodium import *

import pickle

directory = 'D:/Projects/Larose' #find actual file path later
dd = directory + '/Data'

import os
import math
import sys
import json

import numpy as np
from scipy.optimize import brentq as root
import pandas as pd
import time
import csv

import storms as stm
import jpmos as jpm
import cost_model as c_mdl
import metrics_full_interpolation as mtc
import rain as rain
import copy
from numba import jit
import larose_function_variations as lfv

import timeit
start= timeit.default_timer()

#np.random.seed(42)




#Read in needed data from config file, include dd as argument
dmg_data, nms_data, bfe_data, nsc_data = mtc.get_analytica_data(dd)

rain_params = rain.get_rainfall_surface(dd)

storm_params = stm.get_storm_params(dd)
storm_params = storm_params.sort_values('storm_id')
storm_params = storm_params.set_index('storm_id')
storm_params['storm_id'] = storm_params.index

locParams = stm.get_loc_params(dd)
locParams = locParams.sort_values(['storm_id','reach_id'])
locParams = locParams.set_index('storm_id')
locParams['storm_id'] = locParams.index

locParams = stm.get_loc_params(dd)
locParams = locParams.sort_values(['storm_id','reach_id'])
locParams = locParams.set_index(['storm_id','reach_id'])


surgeSurfaceParams = stm.get_surge_surface_params(dd)
surgeSurfaceParams = surgeSurfaceParams.sort_values(['reach_id'])
surgeSurfaceParams = surgeSurfaceParams.set_index('reach_id')
surgeSurfaceParams['reach_id'] = surgeSurfaceParams.index

waveSurfaceParams = stm.get_wave_surface_params(dd)
waveSurfaceParams = waveSurfaceParams.sort_values(['reach_id'])
waveSurfaceParams = waveSurfaceParams.set_index('reach_id')
waveSurfaceParams['reach_id'] = waveSurfaceParams.index


wavePeriodSurfaceParams = stm.get_wave_period_surface_params(dd)
wavePeriodSurfaceParams = wavePeriodSurfaceParams.sort_values(['reach_id'])
wavePeriodSurfaceParams = wavePeriodSurfaceParams.set_index('reach_id')
wavePeriodSurfaceParams['reach_id'] = wavePeriodSurfaceParams.index

sigmaSurfaceParams = stm.sigma_coefs_config(dd)
storms_and_tracks = storm_params.loc[:,['track','storm_id']]
sigmaSurfaceParams = storms_and_tracks.merge(sigmaSurfaceParams, on = 'track')
sigmaSurfaceParams = sigmaSurfaceParams.sort_values(['storm_id','reach_id'])
sigmaSurfaceParams = sigmaSurfaceParams.set_index(['storm_id','reach_id'])



radii, historic_theta_bar,historic_theta_var, historic_x_freq, lon = jpm.get_storm_stats(dd)
polderObject = stm.polder(dd)
 #should be tested in model validation when compared to CLARA
#TODO make argument. will try 
historic_c_p_a0, historic_c_p_a1 = jpm.get_jpmos_coefs(dd)
#reach_objects = stm.construct_reach_objects(dd,file = "/reach_config_file_ipet.csv")
reach_objects = stm.construct_reach_objects(dd,file = "/reach_config_file_new_system_ipet.csv")
#base_frequency = storm_param_desc.loc['index','column'] #fix

unit_price = c_mdl.get_unit_price(dd)
#reach_df = stm.get_reach_df(dd,file = "/reach_config_file_ipet.csv").sort_values('reachID')
reach_df = stm.get_reach_df(dd,file = "/reach_config_file_new_system_ipet.csv").sort_values('reachID')

#rainfall = rain.rainfall_prediction(storm_params, rain_params)
#rainfall = 0
base_crest_heights = reach_objects['reachHeight'].to_numpy() #something in the reach objects
print("parameters set")


partic_rate = 0
sea_lvl = 1.03
rainfall_modifier = 0
intensity = 0
base_frequency = 0.22
pop_mult = 1
frequency_mod = 0
ns_cost_mult = 1
acq_cost_mult = 1
str_cost_mult = 1
NS_std = 1
acq_threshold = 1
res_fp = False
MCIterates = 50


arguments = [reach_objects, dmg_data, nms_data, bfe_data, nsc_data, storm_params,
    locParams, surgeSurfaceParams, waveSurfaceParams, wavePeriodSurfaceParams,
    sigmaSurfaceParams, radii, historic_theta_bar, historic_theta_var,
    historic_x_freq, lon, polderObject, MCIterates, base_frequency,
    unit_price, rainfall_modifier, base_crest_heights, partic_rate, pop_mult,
    sea_lvl, intensity, frequency_mod, ns_cost_mult, acq_cost_mult,
    str_cost_mult, NS_std, acq_threshold, res_fp, rain_params, historic_c_p_a0,
    historic_c_p_a1, reach_df]



# The planning_horizon and discount_rate currently need to be changed in the function definition itself, as Rhodium makes it hard to 
# pass keyword arguments (at least it makes it hard for me to figure out how to do so)
def Larose_problem_pareto(crest_height_upgrade, needed_arguments = arguments, planning_horizon = 50, discount_rate = 0.03):
    np.random.seed(42)
	# levers defined as integers below for Rhodium
    upgrade = copy.deepcopy(crest_height_upgrade)
    upgrade[:] = [x / 10 for x in upgrade]
    EAD, COST = lfv.min_cost_ead_updated(upgrade, needed_arguments, planning_horizon, discount_rate)
    return (EAD, COST)
    

model = Model(Larose_problem_pareto)

# Get rid of non-structural parameters

model.parameters = [Parameter("crest_height_upgrade")]          

# non-structural   

model.responses = [Response("EAD", Response.MINIMIZE),
                  Response("COST", Response.MINIMIZE)]

# 20 to 90 represents crest heights between 2 and 9 meters in the new system
model.levers = [IntegerLever("crest_height_upgrade", 20, 90, length = 12)]

model.uncertainties = []
model.constraints = []


pareto_frontier = optimize(model, "NSGAII",10000)

pareto_frontier.save('optimization_results_new_system_2020-10-14.csv')
#output_file = open('min_cost_ead_updated_test.pkl', 'wb')
#pickle.dump(pareto_frontier, output_file)
#output_file.close()


stop = timeit.default_timer()
print('time:',stop-start)
















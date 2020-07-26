from rhodium import *

import pickle

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

np.random.seed(42)

# comment out one of these two lines depending on whether using command-line
# or a GUI terminal
#dd = sys.argv[1]
dd = 'D:/Projects/Johnson/larose_git/larose/data'

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
# reach_config_file_2017mp_ipet uses the IPET fragility curve and the average 
# actual crest heights by reach that were used in 2017MP
reach_objects = stm.construct_reach_objects(dd,file = "/reach_config_file_2017mp_ipet.csv")

unit_price = c_mdl.get_unit_price(dd)
# TODO: set config file as a command-line input or a control variable specified
#       at top of script to ensure this line uses same file as above
reach_df = stm.get_reach_df(dd,file = "/reach_config_file_2017mp_ipet.csv").sort_values('reachID')

base_crest_heights = reach_objects['reachHeight'].to_numpy() #something in the reach objects
print("parameters set")


partic_rate = 0
sea_lvl = 1.03 # NAVD88 feet, baseline mean eustatic sea level for LA in 2015
rainfall = 0
intensity = 0
frequency_mod = 0
base_frequency = 0.22
pop_mult = 1
ns_cost_mult = 1
acq_cost_mult = 1
str_cost_mult = 1
NS_std = 1
acq_threshold = 1
res_fp = False
MCIterates = 100

arguments = [reach_objects, dmg_data, nms_data, bfe_data, nsc_data, storm_params,
             locParams, surgeSurfaceParams, waveSurfaceParams, wavePeriodSurfaceParams,
             sigmaSurfaceParams, radii, historic_theta_bar, historic_theta_var,
             historic_x_freq, lon, polderObject, MCIterates, base_frequency,
             unit_price, rainfall, base_crest_heights, partic_rate, pop_mult,
             ns_cost_mult, acq_cost_mult, str_cost_mult, NS_std, acq_threshold,
             res_fp, rain_params, historic_c_p_a0, historic_c_p_a1, reach_df]

# Low, Medium, High environmental scenarios from 2017MP
sea_lvl_scenarios = [0.46, 0.63, 0,83] # increase in m by end of planning horizon
int_scenarios = [0.1, 0.125, 0.15] # proportional increase in average intensity
freq_scenarios = [0, -0.14, -0.28] # proportional change in underlying storm frequency

# crest_height_upgrade: upgrade over existing top of levee (in meters), by reach
# default values correspond to 2017MP Medium scenario
def larose_over_time(crest_height_upgrade, time_step = 10, 
                     total_rise = sea_lvl_scenarios[1], intensity_change = int_scenarios[1],
                     frequency_change = freq_scenarios[1], 
                     slr_rate_0 = 0.005, slv_0 = sea_lvl, planning_horizon = 50, 
                     discount_rate = 0.03, needed_arguments = arguments):
    # cheating but it gets the job done for now
    if discount_rate == 0:
        discount_rate = 0.00000001
    
    # slr_rate_0 is the initial rate of SLR in 2015 (m/yr)
    # slr_b is the calculated acceleration of SLR (m/yr^2) leading to the total_rise
    #   by the end of the planning_horizon
    slr_b = (total_rise - (slr_rate_0 * planning_horizon)) / (planning_horizon ** 2)
    
    # num_steps must be equal to an integer
    num_steps = int(planning_horizon / time_step)
    ead_list = []
    years_from_start = [time_step * x for x in list(range(0,num_steps+1))]
    total_cost = 0
    
    # Calculate the EAD for the first time period as well as the cost for the initial upgrades, maintenance costs to be propagated
    
    #crest_height_upgrade, needed_arguments, planning_horizon, discount_rate
    for year in years_from_start:
        # calculate sea level in iterated year, then convert to feet
        sea_lvl = slv_0 + (slr_b * (year ** 2) + (slr_rate_0 * year)) * (3.28084)
        # assumes intensity and frequency of storms changes linearly over time
        intensity = (intensity_change / planning_horizon) * year
        frequency_mod = (frequency_change / planning_horizon) * year
        
        if year == 0:
            # only run the cost model if in the base year to save runtime
            update_cost = 1
        else:
            update_cost = 0
            
        ead, cost = lfv.larose_future_updated(sea_lvl, intensity, frequency_mod, crest_height_upgrade, needed_arguments, update_cost, planning_horizon, discount_rate)
        ead_list.append(ead)
        # note: cost is only non-zero in the first period because of the conditional flag set above
        total_cost = total_cost + cost
        
        
    num_endpoints = len(ead_list)
    
    # initialize present value with EAD in current conditions
    pv_ead = ead_list[0]
    # for each time period of length time_step, we have an annuity and a uniform arithmetic gradient,
    # so apply formulas for present values of these quantities and add them up
    for x in range(0, num_endpoints - 1):
        # the interpolated value of EAD one year after time period x
        gradient = (ead_list[x+1] - ead_list[x])/time_step
        annuity = ead_list[x] + gradient
        pv_annuity = (annuity * ((1+discount_rate) ** time_step - 1)/(discount_rate * (1+discount_rate) ** time_step)) / (1+discount_rate)**(x*time_step)
        pv_gradient = (gradient * (1/discount_rate) * (((1+discount_rate)**time_step-1)/(discount_rate*(1+discount_rate)**time_step) - time_step/(1+discount_rate)**time_step)) / (1+discount_rate)**(x*time_step)
        pv_ead = pv_ead + pv_annuity + pv_gradient
    
    # we ran current conditions, and then the next planning_horizon years, so we need to subtract off the last year run
    pv_ead = pv_ead - ead_list[len(ead_list) - 1]/(1 + discount_rate) ** planning_horizon
        
    return (pv_ead, total_cost) 

# upgrade for 03a.HP.20 project from 2017MP
upgrade2017mp =  [1.33, 2.11, 2.84, 2.77, 1.92, 2.50, 0.78, 0, 1.72, 1.72, 1.41, 0.76]
# don't look at 03a.HP.101 because it changes geometry to add wave berm

# FWOA
upgrade0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


ead2017mp_project, cost2017mp_project = larose_over_time(crest_height_upgrade = upgrade2017mp, discount_rate = 0.001)

ead2017mp_fwoa, cost2017mp_fwoa = larose_over_time(crest_height_upgrade = upgrade0, discount_rate = 0.001)

ead_reduction = ead2017mp_fwoa - ead2017mp_project
incremental_cost = cost2017mp_project - cost2017mp_fwoa


cc_ead, cc_cost = lfv.larose_future_updated(sea_lvl, intensity, frequency_mod, upgrade0, arguments, 1, 0, 1, 0.001)
cc_2017mp_ead, cc_2017mp_cost = lfv.larose_future_updated(sea_lvl, intensity, frequency_mod, upgrade2017mp, arguments, 1, 1, 0.001)


#pareto_frontier = optimize(model, "NSGAII", 5)

#output_file = open('over_time_pareto.pkl', 'wb')
#pickle.dump(pareto_frontier, output_file)
#output_file.close()





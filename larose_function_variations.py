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


def min_cost_ead_updated(crest_height_upgrade, needed_arguments, oldFragFlag = False,\
                   planning_horizon = 50, discount_rate=0.03): 
   
    
    reach_object, dmg_data, nms_data, bfe_data, nsc_data, storm_params, \
    locParams, surgeSurfaceParams, waveSurfaceParams, wavePeriodSurfaceParams, \
    sigmaSurfaceParams, radii, historic_theta_bar, historic_theta_var, \
    historic_x_freq, lon, polderObject, MCIterates, base_frequency, \
    unit_price, rainfall_modifier, base_crest_heights, partic_rate, pop_mult, \
    sea_lvl, intensity, frequency_mod, ns_cost_mult, acq_cost_mult, \
    str_cost_mult, NS_std, acq_threshold, res_fp, rain_params, historic_c_p_a0, historic_c_p_a1, reach_df = copy.deepcopy(needed_arguments)
    
    
    
    #adjust crest heights
    #print("started call")
    #we want to include values rounded up and down to 1 decimal point, and 
    #when we have something with exactly one decimal point we need the surrounding values
    #regardless. And there are some odd numerical precision issues forcing us to go
    #slightly wide
    
    active_dmg_data = dmg_data[(dmg_data.partic <= partic_rate + 0.11) &\
                               (dmg_data.partic >= partic_rate - 0.11) &\
                               (dmg_data.pop_mult <= pop_mult + 0.11)&\
                               (dmg_data.pop_mult >= pop_mult - 0.11)&\
                               (dmg_data.ns_std <= NS_std + 0.11)&\
                               (dmg_data.ns_std >= NS_std - 0.11)]

    
    
    frequency = base_frequency * (1 + frequency_mod)

    
    upgraded_reach = copy.deepcopy(reach_object)
    #upgraded_reach['height'] = reach_object['reachHeight'] + crest_height_upgrade
    upgraded_reach['reachHeight'] = reach_object['reachHeight'] + crest_height_upgrade
    #for overtopping and fragility calculations we must treat sections
    #of reaches with flood walls as separate from sections without floodwalls

    
    #####
    #storms
    
    swe_dist_by_storm = pd.DataFrame(columns = ['depths','depth_probs','storm_id']) #initialize empty dataframe
    storm_ids = storm_params.index
    
    for storm_id in storm_ids:
        this_storm = storm_params.loc[storm_id,:]
        surgeObject = stm.construct_surge_objects(this_storm,\
                                                   locParams.loc[storm_id,:],\
                                                   surgeSurfaceParams, waveSurfaceParams,\
                                                   wavePeriodSurfaceParams,sigmaSurfaceParams.loc[storm_id,:],\
                                                   sea_lvl) #make sure it references id and not row
        #construct_surge_objects is partially written but needs some functions that others have been working on, 
        #please see the storm file - NG
        #also I'm making some assumptions about column names in storm_params, loc_params, need to adjust in the storm file
        #when those are settled - NG
        rainfall = rain.predict_rainfall_generic(rain_params,this_storm['c_p'],this_storm['delta_p'],\
                                                 this_storm['radius'],this_storm['lon'],this_storm['angle'])*(1+rainfall_modifier)
        
        flood_elevs = stm.calcFloodElevs(upgraded_reach,surgeObject,rainfall,polderObject,storm_id,MCIterates,oldFragFlag = oldFragFlag) #add column with storm id [wait, no, we have storm_id already from the loop - NGs]
        
        swe_dist_by_storm = swe_dist_by_storm.append(flood_elevs) 
    
    prob_by_storm = jpm.get_storm_probs(intensity, radii, historic_theta_bar, historic_theta_var,\
                                         historic_x_freq, storm_params, lon, historic_c_p_a0, historic_c_p_a1)

    swe_cdf = jpm.get_swe_cdf(swe_dist_by_storm, prob_by_storm, frequency, min_swe= -13.0)
    
    
    COST = mtc.get_cost(nsc_data, NS_std, pop_mult, partic_rate, ns_cost_mult, acq_cost_mult, crest_height_upgrade, unit_price, reach_df, str_cost_mult, planning_horizon, discount_rate)[0]
    dmg_cdf = mtc.dmg_calc_cdf(active_dmg_data, partic_rate, pop_mult, swe_cdf, NS_std)
    EAD = mtc.ead_calc(dmg_cdf, frequency)[0]
        
    return (EAD,COST)

def larose_future_updated(sea_lvl, intensity, frequency_mod, crest_height_upgrade, needed_arguments, calc_cost, oldFragFlag = False,\
                   planning_horizon = 50, discount_rate=0.03): 
   
    
    reach_object, dmg_data, nms_data, bfe_data, nsc_data, storm_params, \
    locParams, surgeSurfaceParams, waveSurfaceParams, wavePeriodSurfaceParams, \
    sigmaSurfaceParams, radii, historic_theta_bar, historic_theta_var, \
    historic_x_freq, lon, polderObject, MCIterates, base_frequency, \
    unit_price, rainfall_modifier, base_crest_heights, partic_rate, pop_mult, \
    ns_cost_mult, acq_cost_mult, \
    str_cost_mult, NS_std, acq_threshold, res_fp, rain_params, historic_c_p_a0, \
    historic_c_p_a1, reach_df  = copy.deepcopy(needed_arguments)
    
    
    
    #adjust crest heights
   # print("started call")
    #we want to include values rounded up and down to 1 decimal point, and 
    #when we have something with exactly one decimal point we need the surrounding values
    #regardless. And there are some odd numerical precision issues forcing us to go
    #slightly wide
    
    active_dmg_data = dmg_data[(dmg_data.partic <= partic_rate + 0.11) &\
                               (dmg_data.partic >= partic_rate - 0.11) &\
                               (dmg_data.pop_mult <= pop_mult + 0.11)&\
                               (dmg_data.pop_mult >= pop_mult - 0.11)&\
                               (dmg_data.ns_std <= NS_std + 0.11)&\
                               (dmg_data.ns_std >= NS_std - 0.11)]

    
    
    frequency = base_frequency * (1 + frequency_mod)

    
    upgraded_reach = copy.deepcopy(reach_object)
    #upgraded_reach['height'] = reach_object['reachHeight'] + crest_height_upgrade
    upgraded_reach['reachHeight'] = reach_object['reachHeight'] + crest_height_upgrade   
    #for overtopping and fragility calculations we must treat sections
    #of reaches with flood walls as separate from sections without floodwalls

    
    #####
    #storms
    
    swe_dist_by_storm = pd.DataFrame(columns = ['depths','depth_probs','storm_id']) #initialize empty dataframe
    storm_ids = storm_params.index
    
    for storm_id in storm_ids:
        this_storm = storm_params.loc[storm_id,:]
        surgeObject = stm.construct_surge_objects(this_storm,\
                                                   locParams.loc[storm_id,:],\
                                                   surgeSurfaceParams, waveSurfaceParams,\
                                                   wavePeriodSurfaceParams,sigmaSurfaceParams.loc[storm_id,:],\
                                                   sea_lvl) #make sure it references id and not row
        #construct_surge_objects is partially written but needs some functions that others have been working on, 
        #please see the storm file - NG
        #also I'm making some assumptions about column names in storm_params, loc_params, need to adjust in the storm file
        #when those are settled - NG
        rainfall = rain.predict_rainfall_generic(rain_params,this_storm['c_p'],this_storm['delta_p'],\
                                                 this_storm['radius'],this_storm['lon'],this_storm['angle'])*(1+rainfall_modifier)
        
        flood_elevs = stm.calcFloodElevs(upgraded_reach,surgeObject,rainfall,polderObject,storm_id,MCIterates,oldFragFlag = oldFragFlag) #add column with storm id [wait, no, we have storm_id already from the loop - NGs]
        
        swe_dist_by_storm = swe_dist_by_storm.append(flood_elevs) 
    
    prob_by_storm = jpm.get_storm_probs(intensity, radii, historic_theta_bar, historic_theta_var,\
                                         historic_x_freq, storm_params, lon, historic_c_p_a0, historic_c_p_a1)

    swe_cdf = jpm.get_swe_cdf(swe_dist_by_storm, prob_by_storm, frequency, min_swe= -13.0)
    
    if calc_cost:
        COST = mtc.get_cost(nsc_data, NS_std, pop_mult, partic_rate, ns_cost_mult, acq_cost_mult, crest_height_upgrade, unit_price, reach_df, str_cost_mult, planning_horizon, discount_rate)[0]
    else:
        COST = 0
        
    dmg_cdf = mtc.dmg_calc_cdf(active_dmg_data, partic_rate, pop_mult, swe_cdf, NS_std)
    EAD = (mtc.ead_calc(dmg_cdf, frequency))[0]
        
    return (EAD,COST)


# Medium Scenario [

#sea_lvl_scenarios = [0.46, 0.63, 0.83]
#intensity_scenarios = [0.1, 0.125, 0.15]
#frequency_scenarios = [0, -0.14, -0.28]


def larose_over_time(crest_height_upgrade, needed_arguments, time_step = 10, 
                     total_rise = 0.63, intensity_change = 0.125, frequency_change = -0.14, 
                     slr_rate_0 = 0.005, slv_0 = 1.03, planning_horizon = 50, 
                     discount_rate = 0.03):
    
    slr_a = slr_rate_0
    slr_b = (total_rise - (slr_rate_0 * planning_horizon)) / (planning_horizon ** 2)
    
    
    # This is to adjust for the integer values: 0-50 being fed in by rhodium
    #crest_height_upgrade[:] = [x / 10 for x in crest_height_upgrade]
    
    
    # num_steps must be equal to an integer
    num_steps = int(planning_horizon / time_step)
    ead_list = []
    years_from_start = [time_step * x for x in list(range(0,num_steps+1))]
    
    # Calculate the ead for the first time period as well as the cost for the initial upgrades, maintanence costs to be propogated
    
    #crest_height_upgrade, needed_arguments, planning_horizon, discount_rate
    for year in years_from_start:
        
        sea_lvl = (slr_b * (year ** 2) + (slr_a * year)) * (3.28084) + slv_0
        intensity = (intensity_change / planning_horizon) * year
        frequency_mod = (frequency_change / planning_horizon) * year
        
        if year == 0:
            initial_ead, total_costs = larose_future_updated(sea_lvl, intensity, frequency_mod, crest_height_upgrade, needed_arguments, 1, planning_horizon, discount_rate)
            ead_list.append(initial_ead)
        else:
            ead, cost = larose_future_updated(sea_lvl, intensity, frequency_mod, crest_height_upgrade, needed_arguments, 0, planning_horizon, discount_rate)
       #     present_value = ead / ((1 + discount_rate) ** (time))
            ead_list.append(ead)
        
        
    num_endpoints = len(ead_list)
    for x in range(0, num_endpoints - 1):
        index = x + ((time_step - 1) * x) 
        start = ead_list[index]
        end = ead_list[index+1]
        slope = end - start
        for y in range(index + 1, index + time_step):
            ead_list.insert(y, ((slope / time_step) * (y - index) + start))
                      
    present_value_ead = []
    for year in range(0, len(ead_list)):
        present_value_ead.append(ead_list[year] / ((1 + discount_rate) ** (year)))
        
    economic_damage = sum(present_value_ead)
    COST = total_costs
    
    return (economic_damage, COST)
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


def min_cost_ead_updated(crest_height_upgrade,needed_arguments, oldFragFlag = False,\
                   planning_horizon = 50, discount_rate=0.03): 
   
    
    reach_object, dmg_data, nms_data, bfe_data, nsc_data, storm_params, \
    locParams, surgeSurfaceParams, waveSurfaceParams, wavePeriodSurfaceParams, \
    sigmaSurfaceParams, radii, historic_theta_bar, historic_theta_var, \
    historic_x_freq, lon, polderObject, MCIterates, base_frequency, \
    unit_price, rainfall_modifier, base_crest_heights, partic_rate, pop_mult, \
    sea_lvl, intensity, frequency_mod, ns_cost_mult, acq_cost_mult, \
    str_cost_mult, NS_std, acq_threshold, res_fp, rain_params, historic_c_p_a0, historic_c_p_a1, reach_df = needed_arguments
    
    
    
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

    
    upgraded_reach = reach_object
    upgraded_reach['height'] = reach_object['reachHeight'] + crest_height_upgrade
    
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

def min_cost_ead(crest_height_upgrade, needed_arguments, planning_horizon, discount_rate): 
    
    reach_objects, dmg_data, nms_data, bfe_data, nsc_data, storm_params, \
    locParams, surgeSurfaceParams, waveSurfaceParams, wavePeriodSurfaceParams, \
    sigmaSurfaceParams, radii, historic_theta_bar, historic_theta_var, \
    historic_x_freq, lon, polderObject, MCIterates, base_frequency, \
    unit_price, rainfall, base_crest_heights, partic_rate, pop_mult, \
    sea_lvl, intensity, frequency_mod, ns_cost_mult, acq_cost_mult, \
    str_cost_mult, NS_std, acq_threshold, res_fp, rain_params, \
    historic_c_p_a0, historic_c_p_a1, reach_df = needed_arguments
    
        
    
    
    active_dmg_data = dmg_data[(dmg_data.partic <= partic_rate + 0.11) &\
                               (dmg_data.partic >= partic_rate - 0.11) &\
                               (dmg_data.pop_mult <= pop_mult + 0.11)&\
                               (dmg_data.pop_mult >= pop_mult - 0.11)&\
                               (dmg_data.ns_std <= NS_std + 0.11)&\
                               (dmg_data.ns_std >= NS_std - 0.11)]

    
    crest_heights = [sum(x) for x in zip(base_crest_heights, crest_height_upgrade)]
    
    frequency = base_frequency * (1 + frequency_mod)
    
    reach_objects = stm.setReachHeights(reach_objects,crest_heights)
    
    #####
    #storms
    
    swe_dist_by_storm = pd.DataFrame(columns = ['depths','depth_probs','storm_id']) #initialize empty dataframe
    storm_ids = storm_params['storm_id'].values
    print("starting storm loop")
    #initialize df for freq/elev/storm/stormprob
    
    for storm_id in storm_ids:
        this_storm = storm_params.loc[storm_params['storm_id']==storm_id]
        surgeObjects = stm.construct_surge_objects(this_storm,\
                                                   locParams.loc[locParams['storm_id']==storm_id],\
                                                   surgeSurfaceParams, waveSurfaceParams,\
                                                   wavePeriodSurfaceParams,sigmaSurfaceParams,\
                                                   sea_lvl) #make sure it references id and not row
        #construct_surge_objects is partially written but needs some functions that others have been working on, 
        #please see the storm file - NG
        #also I'm making some assumptions about column names in storm_params, loc_params, need to adjust in the storm file
        #when those are settled - NG
        rainfall = rain.predict_rainfall_generic(rain_params,this_storm['c_p'].iloc[0],this_storm['delta_p'].iloc[0],\
                                                 this_storm['radius'].iloc[0],this_storm['lon'].iloc[0],this_storm['angle'].iloc[0] * (1 + rainfall))
        
        flood_elevs = stm.calcFloodElevs(reach_objects,surgeObjects,rainfall,polderObject,storm_id,MCIterates,oldFragFlag = False) #add column with storm id [wait, no, we have storm_id already from the loop - NGs]
        
        swe_dist_by_storm = swe_dist_by_storm.append(flood_elevs) 
        
    
    
    #######
    #JPMOS#
    #######
    #print(swe_dist_by_storm)
    
    prob_by_storm = jpm.get_storm_probs(intensity, radii, historic_theta_bar, historic_theta_var,\
                                         historic_x_freq, storm_params, lon, historic_c_p_a0, historic_c_p_a1)

    swe_cdf = jpm.get_swe_cdf(swe_dist_by_storm, prob_by_storm, frequency, min_swe= -13.0)
    
    #####
    #gathering metrics
    
    COST = mtc.get_cost(nsc_data, NS_std, pop_mult, partic_rate, ns_cost_mult, acq_cost_mult, crest_height_upgrade, unit_price, reach_df, str_cost_mult, planning_horizon, discount_rate)[0]

    EAD = (mtc.get_ead(active_dmg_data, partic_rate, pop_mult, swe_cdf, NS_std, frequency))[0]
    
    
    return (EAD,COST)


def larose_future_updated(sea_lvl, intensity, frequency_mod, crest_height_upgrade,needed_arguments, calc_cost, oldFragFlag = False,\
                   planning_horizon = 50, discount_rate=0.03): 
   
    
    reach_object, dmg_data, nms_data, bfe_data, nsc_data, storm_params, \
    locParams, surgeSurfaceParams, waveSurfaceParams, wavePeriodSurfaceParams, \
    sigmaSurfaceParams, radii, historic_theta_bar, historic_theta_var, \
    historic_x_freq, lon, polderObject, MCIterates, base_frequency, \
    unit_price, rainfall_modifier, base_crest_heights, partic_rate, pop_mult, \
    ns_cost_mult, acq_cost_mult, \
    str_cost_mult, NS_std, acq_threshold, res_fp, rain_params, historic_c_p_a0, \
    historic_c_p_a1, reach_df  = needed_arguments
    
    
    
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

    
    upgraded_reach = reach_object
    upgraded_reach['height'] = reach_object['reachHeight'] + crest_height_upgrade
    
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


def larose_future(sea_lvl, intensity, frequency_mod, crest_height_upgrade, needed_arguments, planning_horizon, discount_rate, calc_cost): 
    
    reach_objects, dmg_data, nms_data, bfe_data, nsc_data, storm_params, \
    locParams, surgeSurfaceParams, waveSurfaceParams, wavePeriodSurfaceParams, \
    sigmaSurfaceParams, radii, historic_theta_bar, historic_theta_var, \
    historic_x_freq, lon, polderObject, MCIterates, base_frequency, \
    unit_price, rainfall, base_crest_heights, partic_rate, pop_mult, \
    ns_cost_mult, acq_cost_mult, \
    str_cost_mult, NS_std, acq_threshold, res_fp, rain_params, \
    historic_c_p_a0, historic_c_p_a1, reach_df = needed_arguments
    
        
    
    
    active_dmg_data = dmg_data[(dmg_data.partic <= partic_rate + 0.11) &\
                               (dmg_data.partic >= partic_rate - 0.11) &\
                               (dmg_data.pop_mult <= pop_mult + 0.11)&\
                               (dmg_data.pop_mult >= pop_mult - 0.11)&\
                               (dmg_data.ns_std <= NS_std + 0.11)&\
                               (dmg_data.ns_std >= NS_std - 0.11)]

    
    crest_heights = [sum(x) for x in zip(base_crest_heights, crest_height_upgrade)]
    
    frequency = base_frequency * (1 + frequency_mod)
    
    reach_objects = stm.setReachHeights(reach_objects,crest_heights)
    
    #####
    #storms
    
    swe_dist_by_storm = pd.DataFrame(columns = ['depths','depth_probs','storm_id']) #initialize empty dataframe
    storm_ids = storm_params['storm_id'].values
    print("starting storm loop")
    #initialize df for freq/elev/storm/stormprob
    
    for storm_id in storm_ids:
        this_storm = storm_params.loc[storm_params['storm_id']==storm_id]
        surgeObjects = stm.construct_surge_objects(this_storm,\
                                                   locParams.loc[locParams['storm_id']==storm_id],\
                                                   surgeSurfaceParams, waveSurfaceParams,\
                                                   wavePeriodSurfaceParams,sigmaSurfaceParams,\
                                                   sea_lvl) #make sure it references id and not row
        #construct_surge_objects is partially written but needs some functions that others have been working on, 
        #please see the storm file - NG
        #also I'm making some assumptions about column names in storm_params, loc_params, need to adjust in the storm file
        #when those are settled - NG
        rainfall = rain.predict_rainfall_generic(rain_params,this_storm['c_p'].iloc[0],this_storm['delta_p'].iloc[0],\
                                                 this_storm['radius'].iloc[0],this_storm['lon'].iloc[0],this_storm['angle'].iloc[0] * (1 + rainfall))
        
        flood_elevs = stm.calcFloodElevs(reach_objects,surgeObjects,rainfall,polderObject,storm_id,MCIterates,oldFragFlag = False) #add column with storm id [wait, no, we have storm_id already from the loop - NGs]
        
        swe_dist_by_storm = swe_dist_by_storm.append(flood_elevs) 
        
    
    
    #######
    #JPMOS#
    #######
    #print(swe_dist_by_storm)
    
    prob_by_storm = jpm.get_storm_probs(intensity, radii, historic_theta_bar, historic_theta_var,\
                                         historic_x_freq, storm_params, lon, historic_c_p_a0, historic_c_p_a1)

    swe_cdf = jpm.get_swe_cdf(swe_dist_by_storm, prob_by_storm, frequency, min_swe= -13.0)
    
    #####
    #gathering metrics
    if  calc_cost:
        COST = mtc.get_cost(nsc_data, NS_std, pop_mult, partic_rate, ns_cost_mult, acq_cost_mult, crest_height_upgrade, unit_price, reach_df, str_cost_mult, planning_horizon, discount_rate)[0]
    else: 
        COST = 0

    EAD = (mtc.get_ead(active_dmg_data, partic_rate, pop_mult, swe_cdf, NS_std, frequency))[0]
    
    
    return (EAD,COST)
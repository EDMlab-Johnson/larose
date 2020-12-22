# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:16:15 2019

@author: richardz
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:16:15 2019

@author: richardz
"""


import pandas as pd
from stormObject import surge
import numpy as np
from otopParent import calcOtopFragTotal







def sigma_coefs_config(dd, file="/sigma_reach_track_by_track.csv"):
    
    fits=pd.read_csv(dd+file)
	
    
    return fits

def calc_sigma(surge,surface):
    """Assumes surge is a float, loc_params is a dataframe with a single row
    corresponding to a given reach and storm, fits is a dataframe with 
    a single row containing the response surface parameters for the corresponding reach"""
    
    pred_l = surface['intercept.l'].to_numpy() + surface['coef.peak.surge.l'].to_numpy()*surge
    
    pred_r = surface['intercept.r'].to_numpy() + surface['coef.peak.surge.r'].to_numpy()*surge
    
    return np.maximum(pred_l,0.00000000000001), np.maximum(pred_r,0.00000000000001)
    


def get_storm_params(directory, file="/larose_storm_parameters.csv"):
    """draws storm parameters from a csv file."""
    params = pd.read_csv(directory+file)
    return params
    
def get_loc_params(directory,file="/loc_data.csv"):
    "draws location information on a per-reach per-storm basis from a csv file"""
    params = pd.read_csv(directory+file)
    return params

def get_surge_surface_params(directory,file="/surge_coefficients.csv"):
    """draws the coefficients for the surge height response surface from config file"""
    params = pd.read_csv(directory+file)
    return(params)
    
    
def get_wave_surface_params(directory, file="/wave_coefficients.csv"):
    """draws the coefficients for the wave height response surface from config file"""
    params = pd.read_csv(directory+file)
    return(params)
    
def get_wave_period_surface_params(directory, file="/wp_coefficients.csv"):
    """draws the coefficients for the wave height response surface from config file"""
    params = pd.read_csv(directory+file)
    return(params)

def construct_reach_objects(directory,file = "/reach_config_file.csv"):
    """reads in config file to construct reach objects. requires config file to 
    have column names: reachID, reachName, totalReachLengthInMeters,crownWidth,
    existingTopOfLevee,existingGrade,rearSlope,forwardSlope,tWallLength, charLength,
    pMax,k, and x_c, where charLength, pMax, k, and x_c are parameters of the 
    reach's fragility curve"""
    
    myFrame = pd.read_csv(directory + file)
    myFrame = myFrame.rename(columns = {'totalReachLengthInMeters': 'reachLength',\
                                        'tWallLength': 'reachLengthWall',\
                                        'existingTopOfLevee': 'reachHeight',\
                                        'existingGrade': 'groundHeight'})
    
    myFrame['reachLengthWall'] = myFrame['reachLengthWall']*0.3048
    myFrame['crownWidth'] = myFrame['crownWidth']*0.3048
    myFrame['reachHeight'] = myFrame['reachHeight']*0.3048
    myFrame['groundHeight'] = myFrame['groundHeight']*0.3048
    myFrame['reachLengthNoWall'] = myFrame['reachLength'] - myFrame['reachLengthWall']
    
    myFrame.sort_values('reachID')
    return myFrame

    
    

def get_reach_df(directory, file = "/reach_config_file.csv"):
    return pd.read_csv(directory + file)


def construct_surge_objects(storm_params,locParams,surgeSurface,\
    waveSurface,wavePeriodSurface,sigmaSurface,\
    sea_lvl, time_res = 0.25):
    """Loops through reach_ids in the surgeSurfaceParams dataframe, calculates surge
    heights, wave heights, wave periods, sigmas, and uses these to produce storm/surge objects
    for each reach"""
    
    
    surge_height = pred_surge_wave_height(storm_params,locParams,\
                                              surgeSurface,\
                                              sea_lvl)
    
    wave = pred_surge_wave_height(storm_params,locParams,\
                                      waveSurface,\
                                      sea_lvl)
    
    wavePeriod = pred_wave_period(surge_height, storm_params,locParams,\
                                      wavePeriodSurface)
    
    sigmaL, sigmaR = calc_sigma(surge_height, sigmaSurface)
    
    surge_object = surge(surge_height*0.3048,sigmaL,sigmaR,wave*0.3048,wavePeriod, deltaT = time_res)
    

    return surge_object


def pred_surge_wave_height(storm_params,loc_params,surface,sea_lvl):
    """uses the storm_params (pandas dataframe), loc_params (pandas dataframe),
    and sea level with the regression parameters in surface(pandas dataframe)
    to predict surge or wave height"""
    
    

    pred = surface['intercept'].to_numpy() + surface['c_p_coef'].to_numpy()*storm_params['c_p'] +\
    surface['radius_coef'].to_numpy()*storm_params['radius']  + surface['track_angle_coef'].to_numpy()*\
    loc_params.track_angle.to_numpy() + surface['dist_coef'].to_numpy()*loc_params.dist_from_lf.to_numpy() +\
    surface['dist2_coef'].to_numpy()*loc_params.dist2.to_numpy() + surface['dist3_coef'].to_numpy()*loc_params.dist3.to_numpy() +\
    surface['sin_deg_cw_coef'].to_numpy()*loc_params.sin_deg_cw.to_numpy() + surface['lon_coef'].to_numpy()*loc_params.lon.to_numpy() +\
    surface['sea_level_coef'].to_numpy()*sea_lvl 
    

    
    
    return np.maximum(pred,0.0)

def pred_wave_period(surge, storm_params, loc_params, surface):
    
    #this function performes a numpy vectorized version of the following, commented out set
    #of conditionals
    """
    if surge <= surface['bound_l']:
        period = surface.left_lin_intercept + surface.left_lin_surge1 * surge\
        + surface.dist_from_lf_coef* loc_params.dist_from_lf

    elif surface.bound_l < surge <= surface.knot_l.to_numpy():
        period = surface.spline_1_intercept.to_numpy() + surface.spline_1_surge1.to_numpy()*surge\
        + surface.spline_1_surge2.to_numpy()*(surge**2) + surface.spline_1_surge3.to_numpy()*(surge**3)\
        + surface.dist_from_lf_coef.to_numpy()* loc_params.dist_from_lf.to_numpy()

        
    elif surface.knot_l< surge <= surface['knot_r'].to_numpy():
        period = surface.spline_2_intercept.to_numpy() + surface.spline_2_surge1.to_numpy()*surge\
        + surface.spline_2_surge2.to_numpy()*(surge**2) + surface.spline_2_surge3.to_numpy()*(surge**3)\
        + surface.dist_from_lf_coef.to_numpy()* loc_params.dist_from_lf.to_numpy()
        
    elif surface.knot_r < surge <= surface['bound_r'].to_numpy():
        period = surface.spline_3_intercept.to_numpy() + surface.spline_3_surge1.to_numpy()*surge \
        + surface.spline_3_surge2.to_numpy()*(surge**2) + surface.spline_3_surge3.to_numpy()*(surge**3)\
        + surface.dist_from_lf_coef.to_numpy()* loc_params.dist_from_lf.to_numpy()
        
    elif surface.bound_r < surge:
        period = surface.right_lin_intercept.to_numpy() + surface.right_lin_surge1.to_numpy() * surge\
        + surface.dist_from_lf_coef.to_numpy()* loc_params.dist_from_lf.to_numpy()"""
        
    period = np.where((surge < surface['bound_l'].to_numpy()),\
                      surface.left_lin_intercept.to_numpy() + surface.left_lin_surge1.to_numpy() * surge\
                      + surface.dist_from_lf_coef.to_numpy()* loc_params.dist_from_lf.to_numpy(),\
                      np.where((surge < surface['knot_l'].to_numpy()), \
                               surface.spline_1_intercept.to_numpy() + surface.spline_1_surge1.to_numpy()*surge\
                               + surface.spline_1_surge2.to_numpy()*(surge**2) + surface.spline_1_surge3.to_numpy()*(surge**3)\
                               + surface.dist_from_lf_coef.to_numpy()* loc_params.dist_from_lf.to_numpy(),\
                               np.where((surge < surface['knot_r'].to_numpy()),\
                                        surface.spline_2_intercept.to_numpy() + surface.spline_2_surge1.to_numpy()*surge\
                                        + surface.spline_2_surge2.to_numpy()*(surge**2) + surface.spline_2_surge3.to_numpy()*(surge**3)\
                                        + surface.dist_from_lf_coef.to_numpy()* loc_params.dist_from_lf.to_numpy(),\
                                        np.where((surge < surface['bound_r'].to_numpy()),\
                                                  surface.spline_3_intercept.to_numpy() + surface.spline_3_surge1.to_numpy()*surge \
                                                  + surface.spline_3_surge2.to_numpy()*(surge**2) + surface.spline_3_surge3.to_numpy()*(surge**3)\
                                                  + surface.dist_from_lf_coef.to_numpy()* loc_params.dist_from_lf.to_numpy(),\
                                                  surface.right_lin_intercept.to_numpy() + surface.right_lin_surge1.to_numpy() * surge\
                                                  + surface.dist_from_lf_coef.to_numpy()* loc_params.dist_from_lf.to_numpy()))))
    return np.where(period > 0, period, 1)


    

def calcFloodElevs(reaches,storm_object,rain,myPolder,storm_id,MCIterates,oldFragFlag = False):
    """takes in a list of reach objects (defined in reach.py), a list of 
    surge objects (defined in stormObject.py), rainfall volume in cubic meters,
    a polder object (defined in config.py), a storm_id corresponding to the storm
    in question, and a number of monte carlo iterates to average over. 
    returns a oandas data frame containing the unique depths which result from overtopping
    for the storm in question (depthsUnique), associated frequencies (depth_probs),
    and the storm id (storm_id), the latter of which is not calculated anew but is
    included for combatibility other functions"""
    
    
    otopVols = calcOtopFragTotal(storm_object, reaches, MCIterates, oldFragFlag = oldFragFlag)

    floodVolsMeters = np.maximum(otopVols - myPolder.getPumpRate()*myPolder.getPumpTime()*60/2 + rain, 0)
    

    
    floodVols = floodVolsMeters*0.000810713194
    #converting cubic meters to acre feet
    

    
    #print(floodVols)
    
    storageVolumes = myPolder.getVolumes()
    
    storageDepths = myPolder.getDepths()
    
    #max_depth = min([max(reaches[i]['reachHeight'], storm_object[i].peakSurge) for i in range(len(reaches))])*3.28084
    
    max_depth = np.min(np.maximum(reaches['reachHeight'].to_numpy(), storm_object.peakSurge))*3.28084
    
    
    #this is an odd artifact of the damage data

   
    
    depths = [np.minimum(interpolate(vol,storageVolumes,storageDepths),max_depth) for vol in floodVols]


    
    depths.sort()
    depthsUnique = list(set(depths))
    depthsUnique.sort()
    
    
    
    depthFrequencies = [float(depths.count(x)) for x in depthsUnique]
    
    sumFreq = sum(depthFrequencies)
    
    depth_probs = [x/sumFreq for x in depthFrequencies]
    

    return pd.DataFrame(np.column_stack([depthsUnique,depth_probs,[storm_id]*len(depthsUnique)]),columns=["depths","depth_probs","storm_id"])

    
    
def interpolate(value,valueList,outputList):
    """takes in a float (value), and two lists of equal length (valueList, outputList)
    the elements of output list are known function outputs of an arbitrary monotonic function
    corresponding to the valueList elements of the same index. This function
    linearly interpolates the corresponding output of the input value. Used to 
    calculate stillwater elevations corresponding to overtopping volumes"""
    
    
    if value >= max(valueList):
        # print('overtopping volume exceeds maximum overtopping in stage-storage curve' )

        return max(outputList)
    elif value < min(valueList):
        return(min(outputList))
    else:
        lowVal = max(x for x in valueList if x <= value)
        highVal = min(x for x in valueList if x > value)
        lowInd = valueList.index(lowVal)
        highInd = valueList.index(highVal)
        
        output = outputList[lowInd] + ((value-lowVal)/(highVal-lowVal))*\
        (outputList[highInd] - outputList[lowInd])
    
    return round(output,2)
    
    
class polder:
    def __init__(self,dd,configFile="/polderConfig.csv",stageStorageFile="/lgm_stage_storage_curve.csv"):
        """area should be in square meters, volumes in stage-storage are in acre-feet, 
        and depths in stage-storage are in feet"""
        myFrame = pd.read_csv(dd+configFile)
        row = myFrame.iloc[0,:]
        self.pumpRate = row.pumpRate
        self.pumpTime = row.pumpTime
        
        myFrame = pd.read_csv(dd+stageStorageFile)
        
        self.volumes = list(myFrame.volume_AF)
        
        self.depths = list(myFrame.elevation_navd88_ft)
        
    def getPumpRate(self):
        return self.pumpRate
    def getPumpTime(self):
        return self.pumpTime

    def getVolumes(self):
        return self.volumes
    def getDepths(self):
        return self.depths



        
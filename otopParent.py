# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 23:46:05 2019

@author: ngeldner
"""

import numpy as np

from otopFunctions import calc_otop_all_reaches, otopFrag2_loop, otopFrag1_loop






def calcOtopFragTotal(stormPerReach, reaches, MCIterates, oldFragFlag = False):
    """assumes reach lengths are in meters, outputs cubic meters"""

        #g is the gravitational constant
    g = 9.80665
    #wave breaking scalar factor, taken from CLARA
    breakParam = 0.4
    #influence of berm, taken from CLARA
    bermParam = 0.7
    #influence of armoring, taken from CLARA
    frictionParam = 1.0
    #influence of wave angle of attack, we assume head-on as in CLARA
    waveAngleParam = 1.0
    #influence of vertical flood wall is 0.65 if there's a wall, 1 otherwise
    wallParam = 0.65
    noWallParam = 1.0
    #Weir coefficient for levee is 1.45
    weirCoef = 1.45
    #coefficient for floodwall geometry influence
    geometryParam = 1
    

    
    
    coef1 = np.random.normal(4.75,0.50, size=[1,MCIterates])
    coef2 = np.random.normal(2.60, 0.35, size = [1,MCIterates])
    coef3 = np.random.normal(-0.92,0.24, size= [1,MCIterates])
    
    #axis 0 is reach, axis 1 is time, axis 2 is MCIterates
    #but we'll be doing one reach at a time
    
    otop = calcOtopFragReach(stormPerReach,reaches,g,\
                                         breakParam, bermParam, frictionParam,\
                                         waveAngleParam, wallParam, noWallParam,\
                                         geometryParam,weirCoef,coef1,coef2,coef3,\
                                         oldFragFlag = oldFragFlag)
    
    #otop_sum = np.sum(otop_by_reach,axis=0) we're gonna sum earlier
    
    
    return otop
    
    
    


def calcOtopFragReach(myStorm, reaches,g,breakParam,bermParam,frictionParam,\
                      waveAngleParam, wallParam, noWallParam,geometryParam,\
                      weirCoef, coef1, coef2, coef3, oldFragFlag = False):
    """calculates total overtopping given a storm object with features surgeTS,
    waveHeight, wavePeriod, and deltaT, and a reach object as well as several
    relevant parameters"""


    slope = np.arctan(1/reaches['forwardSlope'].to_numpy())
    surgeTS = myStorm.hydrograph
    crestHeight = reaches['reachHeight'].to_numpy()
    toeHeight = reaches['groundHeight'].to_numpy()
    waveHeight = myStorm.waveHeight
    wavePeriod = myStorm.wavePeriod
    
    wall_lengths = reaches['reachLengthWall'].to_numpy()
    
    wall_indices = wall_lengths > 0
    
    nowall_lengths = reaches['reachLengthNoWall'].to_numpy()
    
    pmax_nowall = reaches['pMaxNoWall'].to_numpy()
    k_nowall = reaches['kNoWall'].to_numpy()
    x_c_nowall = reaches['x_cNoWall'].to_numpy()
    char_length_nowall = reaches['charLengthNoWall'].to_numpy()
    
    pmax_wall = reaches['pMaxWall'].to_numpy()[wall_indices]
    k_wall = reaches['kWall'].to_numpy()[wall_indices]
    x_c_wall = reaches['x_cWall'].to_numpy()[wall_indices]
    char_length_wall = reaches['charLengthWall'].to_numpy()[wall_indices]
    

    #Evaluate the overtopping without breach where there's no wall

    
    noBreachNoWall = calc_otop_all_reaches(g,slope,surgeTS,crestHeight,toeHeight,waveHeight,wavePeriod,\
                                            breakParam,frictionParam,waveAngleParam, noWallParam,\
                                            bermParam,weirCoef,coef1,coef2,coef3)
    

    #evaluate the overtopping with breach (crest height reduced to toe height
    #where there's no wall

    
    breachNoWall = calc_otop_all_reaches(g,slope,surgeTS,toeHeight,toeHeight,waveHeight,wavePeriod,\
                                          breakParam,frictionParam,waveAngleParam, noWallParam,\
                                          bermParam,weirCoef,coef1,coef2,coef3)
    
    #note that the overtoppings in case of wall will have fewer rows, since it will only cover
    #reaches with a wall
    noBreachWall = calc_otop_all_reaches(g,slope[wall_indices],surgeTS[wall_indices],crestHeight[wall_indices],\
                                    toeHeight[wall_indices],waveHeight[wall_indices],wavePeriod[wall_indices],\
                                            breakParam,frictionParam,waveAngleParam, wallParam,\
                                            bermParam,weirCoef,coef1,coef2,coef3)
    
    breachWall = calc_otop_all_reaches(g,slope[wall_indices],surgeTS[wall_indices],toeHeight[wall_indices],\
                                    toeHeight[wall_indices],waveHeight[wall_indices],wavePeriod[wall_indices],\
                                            breakParam,frictionParam,waveAngleParam, wallParam,\
                                            bermParam,weirCoef,coef1,coef2,coef3)
    
    if oldFragFlag == False:
        otopRateWall = otopFrag2_loop(noBreachWall, breachWall,pmax_wall, k_wall, x_c_wall ,\
                                 char_length_wall,wall_lengths[wall_indices])
        otopRateNoWall = otopFrag2_loop(noBreachNoWall, breachNoWall, pmax_nowall, k_nowall, x_c_nowall,\
                                   char_length_nowall,nowall_lengths)
        
        return (np.sum(otopRateWall,axis = 0) + np.sum(otopRateNoWall,axis = 0))*myStorm.deltaT*3600
    
    else:
        otopRateWall = otopFrag1_loop(noBreachWall, breachWall,pmax_wall, k_wall, x_c_wall ,\
                                 char_length_wall,wall_lengths[wall_indices])
        otopRateNoWall = otopFrag1_loop(noBreachNoWall, breachNoWall, pmax_nowall, k_nowall, x_c_nowall,\
                                   char_length_nowall,nowall_lengths)
        return (np.sum(otopRateWall,axis = 0) + np.sum(otopRateNoWall,axis = 0))*myStorm.deltaT*3600
    

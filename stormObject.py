# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 00:19:07 2019

@author: ngeldner
"""
import numpy as np



class surge:
    def __init__(self,peakSurge,sigmaL,sigmaR, waveHeight, wavePeriod, deltaT = 0.25):
        """constructor takes in peak surge height, surge width parameter for forward
        and backward side, wave height, wave period, rain in m^3 and deltaT in hours, 
        the time resolution for the surge hydrograph. The surge is modeled
        as following a gaussian function with different sigma parameters before and
        after peak surge. Note that deltaT must evenly divide 1 day"""
        self.waveHeight = waveHeight
        self.wavePeriod = wavePeriod
        self.deltaT = float(deltaT)
        self.peakSurge = peakSurge
        #hydrograph is 3 days long, 24 hours per day
        hydrographTime = 3*24
        #hydrograph goes from 0 to 3 days inclusive
        hydrographLength = int(hydrographTime/deltaT)
        #peak is at the end of the first day
        hydrographPeakInd = (hydrographLength)*2/3
        
        #want peak_surge and the sigmas as column vectors
        peak_surge = peakSurge.reshape([-1,1])
        sigmaL = sigmaL.reshape([-1,1])
        sigmaR = sigmaR.reshape([-1,1])
        
        i_front = np.arange(hydrographPeakInd).reshape([1,-1])
        
        i_back = np.arange(hydrographPeakInd, hydrographLength).reshape([1,-1])
        
        hydrographFront = peak_surge*np.exp(-(((i_front-hydrographPeakInd)*deltaT)**2)/(2*(sigmaL**2)))
        
        hydrographBack = peak_surge*np.exp(-(((i_back-hydrographPeakInd)*deltaT)**2)/(2*(sigmaR**2)))
        
        
        
        self.hydrograph = np.append(hydrographFront, hydrographBack,axis = 1)
        
        
   

        
        
        
        
        
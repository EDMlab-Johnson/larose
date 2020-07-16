# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:54:52 2019

@author: ngeldner

to feed into an overtopping function, I'll need:
    have - g, slope, crest height, wave height, wave period, wallParam, coef1-3
    don't have - frictionParam, bermParam, wier coef?
"""

import numpy as np
from numba import jit



def vanDerMeer(g, slope, freeBoard, waveHeight, surfSim, frictionParam, waveAngleParam, wallParam, bermParam, coef1, coef2, coef3 ):
    """Assumes that waves come in head on (gamma_beta=1). g is gravitational acceleration
    slope is slope of levy, wave height is the height of waves breaking at levy
    foot, sufSim is calculated from g, slope, waveHeight, and wave period
    not sure where frictionParam comes from, wallParam is 0.65 for floodwall, 1
    otherwise. coef1-coef3 are coefficients for the equation which are empirically
    determined with some uncertainty"""
    
    eq1_ind = surfSim<5.0
    eq2_ind = surfSim>7.0
    
    interpolate_ind = (surfSim>=5.0) & (surfSim<=7.0)

    #we want to return a matrix with as many columns as there are elements of freeBoard
    #and as many rows as there are elements of coef1-coef3
    vdm_out = np.zeros(shape=[len(freeBoard),coef1.size])
    
    vdm_out[eq1_ind,:] = vanDerMeerEq1(g, slope, freeBoard[eq1_ind].reshape((-1,1)), waveHeight[eq1_ind].reshape((-1,1)),\
           surfSim[eq1_ind].reshape((-1,1)),frictionParam, waveAngleParam, wallParam, bermParam, coef1, coef2)
    
    vdm_out[eq2_ind,:] = vanDerMeerEq2(g, slope, freeBoard[eq2_ind].reshape((-1,1)), waveHeight[eq2_ind].reshape((-1,1)),\
           surfSim[eq2_ind].reshape((-1,1)), frictionParam, waveAngleParam,wallParam, coef3)
    
    vdm_out[interpolate_ind,:] = vanDerMeerInterpolate(g, slope, freeBoard[interpolate_ind].reshape((-1,1)), \
           waveHeight[interpolate_ind].reshape((-1,1)), surfSim[interpolate_ind].reshape((-1,1)), frictionParam, waveAngleParam,\
           wallParam, bermParam, coef1, coef2, coef3 )
    
    return vdm_out

def vanDerMeerEq1(g, slope, freeBoard, waveHeight, surfSim, frictionParam, waveAngleParam, wallParam,bermParam, coef1, coef2):
    
    exponent = -coef1 * freeBoard/waveHeight *1.0/(surfSim *bermParam *frictionParam*waveAngleParam*wallParam)
    part1 = ((g * waveHeight**3.0)**0.5) * 0.067/((np.tan(slope))**0.5)*bermParam*surfSim*np.exp(exponent)
    
    exponent = -coef2 * freeBoard/waveHeight*1.0/(frictionParam*waveAngleParam)
    part2 = ((g * waveHeight**3.0)**0.5) * 0.2 * np.exp(exponent)
    return np.minimum(part1,part2)


    
def vanDerMeerEq2(g, slope, freeBoard, waveHeight, surfSim, frictionParam, waveAngleParam, wallParam, coef3):
    exponent = (-1*freeBoard)/(frictionParam*waveAngleParam*waveHeight*(0.33 + 0.022*surfSim))
    return ((g*waveHeight**3.0)**0.5)*10.0**coef3 * np.exp(exponent)
    
def vanDerMeerInterpolate(g, slope, freeBoard, waveHeight, surfSim, frictionParam, waveAngleParam, wallParam,bermParam, coef1, coef2, coef3):
    
    interpolate_out = np.zeros(shape = (freeBoard.size,coef1.size))
    waveLow = vanDerMeerEq1(g, slope, freeBoard, waveHeight, 4.999, frictionParam, waveAngleParam, wallParam,bermParam, coef1, coef2)
    waveHigh = vanDerMeerEq2(g, slope, freeBoard, waveHeight, 7.001, frictionParam, waveAngleParam, wallParam, coef3)
    

    
    interpolate_out[(waveLow == 0.0) | (waveHigh == 0.0)] = 0.0
    
    inds = (waveLow!= 0.0) & (waveHigh!= 0.0)
    
    #surfSim is currently a 1d array. need it to be 2d with 1 row
    #surfSim_row = np.reshape(surfSim,[-1,1])
    
    #now we make an array with the same dimension of interpolate_out
    surfSim_array = np.repeat(surfSim, coef1.size, axis=1)
    
    #every row of inds should be the same, so to access surfSim, we'll just use the first row
    interpolate_out[inds] = np.exp(np.log(waveLow[inds]) + (surfSim_array[inds] - 4.999)/(7.001-4.999) *\
                   (np.log(waveHigh[inds]) - np.log(waveLow[inds])))
    
    return interpolate_out


def otopSurgeWave(g,weirCoef,surgeHeight, crestHeight,waveHeight):
    otop = weirCoef*(surgeHeight - crestHeight)**(3.0/2.0) + 0.13*((g*(waveHeight**3.0))**0.5)
    return otop


def franco(g,freeBoard,waveHeight,waveAngleParam,geometryParam):
    otop = ((g*(waveHeight)**3.0)**0.5) * 0.082 * np.exp(-3.0*(freeBoard/waveHeight)*(1.0/(waveAngleParam*geometryParam)))
    return otop


def calcOtopRateOnce(g, slope, surgeTS, crestHeight, toeHeight, waveHeightRaw, wavePeriod, breakParam, frictionParam, waveAngleParam, wallParam, bermParam, weirCoef, coef1, coef2, coef3):
    #The only calculation that is different between MC Iterates is the van der meer equation
    #so we compute singly until the final step, then convert what we have into a 2d array
    otop_ts = np.zeros(surgeTS.shape)
    
    otop_ts[surgeTS<=toeHeight] = 0.0
    
    surge_indices = surgeTS>0.0
    
    wave_height_ts = waveBreak(breakParam,surgeTS,waveHeightRaw,toeHeight)
    
    freeBoard = crestHeight - surgeTS
    
    #use surge wave equation where surge is positive  and freeboard is non-positive
    sw_indices = (freeBoard<= 0.0) & (surge_indices)
    
    otop_ts[sw_indices] = otopSurgeWave(g,weirCoef,surgeTS[sw_indices],crestHeight,wave_height_ts[sw_indices])
    
    #interested in positive freeboard only where surge is positive
    positive_freeBoard_indices = (freeBoard>0.0) & surge_indices
    
    
    otop_ts[(positive_freeBoard_indices) & ((wave_height_ts<=0.00000000000001) )] = 0.0
    
    #only run van der meer where waveHeight > 0, wavePeriod >0, and freeboard>0
    vdm_indices = ((wave_height_ts >0.00000000000001)) & positive_freeBoard_indices
    
    waveSteep = waveSteepness(wave_height_ts[vdm_indices],wavePeriod,g)
    
    surfSim = calcSurfSim(slope,waveSteep)
    
    otop_vec = np.reshape(otop_ts, [-1,1])
    otop_array = np.repeat(otop_vec, coef1.size, axis=1)
    
    otop_array[vdm_indices,:] = vanDerMeer(g,slope,freeBoard[vdm_indices], wave_height_ts[vdm_indices],\
           surfSim,frictionParam,waveAngleParam, wallParam, bermParam, coef1, coef2, coef3)
    
    return otop_array

#@jit(fastmath = True)
def calc_otop_all_reaches(g, slope, surgeTS, crestHeight, toeHeight, waveHeightRaw, wavePeriod, breakParam, frictionParam, waveAngleParam, wallParam, bermParam, weirCoef, coef1, coef2, coef3):
    #axis 0 is reach, axis 1 is time, axis 2 is MCIterates
    otops = np.zeros(shape=(crestHeight.size, surgeTS.shape[1], coef1.size))
    
    for i in range(otops.shape[0]):
        otops[i,:,:] = calcOtopRateOnce(g, slope[i], surgeTS[i,:], \
             crestHeight[i], toeHeight[i], waveHeightRaw[i], wavePeriod[i], \
             breakParam, frictionParam, waveAngleParam, wallParam, bermParam,\
             weirCoef, coef1, coef2, coef3)
        
    return otops
    
#@jit(fastmath = True)
def otopFrag2_loop(noBreachTS, breachTS, pMax ,k ,x_c ,char_length, length):
    
    otop_totes = np.zeros(shape = (noBreachTS.shape[0], noBreachTS.shape[2]))
    
    for i in range(otop_totes.shape[0]):
        otop_totes[i,:] = otopFrag2(noBreachTS[i,:,:], breachTS[i,:,:], pMax[i],\
                  k[i], x_c[i], char_length[i], length[i])
    return otop_totes
    
def otopFrag1_loop(noBreachTS, breachTS, pMax ,k ,x_c ,char_length, length):
    
    otop_totes = np.zeros(shape = (noBreachTS.shape[0], noBreachTS.shape[2]))
    
    for i in range(otop_totes.shape[0]):
        otop_totes[i,:] = otopFrag1(noBreachTS[i,:,:], breachTS[i,:,:], pMax[i],\
                  k[i], x_c[i], char_length[i], length[i])
    return otop_totes
    
#@jit(fastmath = True)
def otopFrag2(noBreachTS, breachTS, pMax ,k ,x_c ,char_length, length):

    char_lengths = int(np.floor(length/char_length))
    
    remainder = np.mod(length, char_length)
    
    timePoints = breachTS.shape[0]
    peakInd = int((timePoints)*2/3)
    runupTS = noBreachTS[0:peakInd+1,:]

    #draw a uniform variate for each section for each MCIterate row of our otop TS's
    unifs = np.random.random_sample((char_lengths+1,breachTS.shape[1]))
    

    #calculate 2d breach probabilities

    breachProbs = fragCalc2d(runupTS, pMax,k,x_c)
    
    #get vector of the number of characteristic lengths where the failure prob
    #exceeds or is equal to its uniform variate
    #print(max(breachProbs))
    if char_lengths > 0:
        
        breachesRunup = calcBreaches_try(unifs[0:-1,:],breachProbs)
        

        breachesTotal = np.append(breachesRunup, np.ones((int(peakInd/2-1),breachTS.shape[1],))*\
                                  np.reshape(breachesRunup[-1,:],(1,-1)), axis=0)
    else:
        breachesTotal=np.zeros((timePoints,breachTS.shape[1]))
        
    breachProbsRemainder = 1-(1-breachProbs)**(remainder/char_length)
    
   
    
    breachesRunupRemainder = (breachProbsRemainder >= np.reshape(unifs[-1,:],(1,-1)))
    
    ######
    
    
    
    breachesRemainder = np.append(breachesRunupRemainder, np.ones((int(peakInd/2-1),breachTS.shape[1]))*\
                                  np.reshape(breachesRunupRemainder[-1,:],(1,-1)), axis=0)
    

    
    totalRate = otopAggregate(breachesTotal, breachesRemainder, breachTS, noBreachTS,\
                             length, char_length, remainder)

    return np.sum(totalRate,axis=0)

def otopFrag1(noBreachTS, breachTS, pmax,k,x_c , char_length, length):

    charLengths = int(np.floor(length/char_length))
    remainder = length % char_length
    #extract number of collumns from our overtopping info
    timePoints = breachTS.shape[0]
    #timePoints = len(breachTS)
    peakInd = int((timePoints)*2/3)
    #we want the maximum otop for each row, and we want it as a column vector
    peak_otop = np.reshape(np.max(noBreachTS, axis=0),[1,-1])
    
    
    #draw a uniform variate for each section (# cols) and for each MC iterate (# rows)
    
    unifs = np.random.uniform(0,1, size=(charLengths+1, peak_otop.size))

    #calculate 2d breach probabilities


    breachProb = fragCalc2d_singleton(peak_otop, pmax,k,x_c)
    
    #print(breachProb)
    #get vector of the number of characteristic lengths where the failure prob
    #exceeds or is equal to its uniform variate
    
    #n_char_breaches = np.reshape(np.sum(unifs[:,0:-1]<= breachProb, axis = 1),[-1,1])
    
    n_char_breaches = np.reshape(np.sum(unifs[0:-1,:]<= breachProb, axis = 0),(1,-1))

    #n_char_breaches = np.sum(unifs[0:-1] <= breachProb)
    
    
    breachProbRemainder = 1-(1-breachProb)**(remainder/char_length)
    
   
    
    breachRemainder = (breachProbRemainder >= np.reshape(unifs[-1,:], (1, -1)))


    
    failLength = char_length*n_char_breaches + breachRemainder*remainder
    fineLength = length - failLength
    

    final_ts = np.append(noBreachTS[0:peakInd,:]*length, \
                         noBreachTS[peakInd:,:]*fineLength + breachTS[peakInd:,:]*failLength, axis=0)
    
    total_rate = np.sum(final_ts, axis = 0)
    
    

    
    return total_rate
    
    
#@jit(nopython = True, fastmath = True)    
def otopAggregate(failNum,remainderFail,breach, noBreach,length, charLength, remainder):
    failLength = failNum*charLength + remainderFail*remainder
    fineLength = length - failLength

    return failLength*breach + fineLength*noBreach



#@jit(nopython = True, fastmath = True)
def calcBreaches_try(unifs, probTS):
    adj_unifs = np.transpose(unifs)[np.newaxis,:,:]
    return np.sum(adj_unifs <= probTS[:,:,np.newaxis],axis=2)

    
        
        
def calcBreaches(unifs, probTS):
    
    
    
    breachNums = np.zeros(probTS.shape)
    #j is the column index for the uniforms
    for j in range(unifs.shape[0]):
        #for each iteration in this loop, we get an array
        #of True (equal to 1 when used in addition) for a breach)
        breachNums = breachNums + (np.reshape(unifs[j,:],(1,-1)) <= probTS)
    return breachNums
        
        
        





#@jit(nopython = True, fastmath = True)
def fragCalc2d_singleton(overtopping, pMax, k, x_c):
    #simplest way to get an array of zeros with dims of overtopping
    prob = np.zeros(shape=overtopping.shape)
    
    prob[overtopping<=0.0001] = 0
    
    prob[overtopping > 0.0001] = pMax/(1.0+np.exp(- k*(overtopping[overtopping > 0.0001] - x_c)))
        
    return prob

#@jit(fastmath = True)
def fragCalc2d(overtopping, pMax,k,x_c):
    breachProb2d = np.zeros(overtopping.shape)
    breachProb2d[overtopping<=0.0001] = 0.0
    breachProb2d[overtopping>0.0001] = pMax/(1.0+np.exp(- k*(overtopping[overtopping>0.0001] - x_c)))
    #print(max(breachProb2d))
    return breachProb2d
        
    
#@jit(fastmath = True)    
def waveBreak(breakParam, surgeHeight, waveHeightRaw, toeHeight):
    """ wave height is bounded above by (surgeHeight - toeHeight)*breakParam"""
    return np.minimum(waveHeightRaw, breakParam*np.maximum(surgeHeight-toeHeight,0.0))

#@jit(fastmath = True)    
def waveSteepness(waveHeight,wavePeriod,g):
    """used for surf similarity parameter"""
    return (2*np.pi*waveHeight)/(g*(wavePeriod**2))

#@jit(fastmath = True)    
def calcSurfSim(slope,waveSteep):
    """surfSim used for van der meer"""
    return np.tan(slope)/(waveSteep**0.5)
    


"""

 """
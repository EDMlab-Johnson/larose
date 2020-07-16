# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:16:15 2019

@author: Jingya Wang
"""
import pandas as pd
import storms as stm
import math
import scipy.stats as st
import numpy as np

#configuration functions

def get_jpmos_coefs(dd):
  
  historic_c_p_a0 = pd.read_csv(dd + '/JPM-OS/historic_c_p_a0.csv')
  historic_c_p_a1= pd.read_csv(dd + '/JPM-OS/historic_c_p_a1.csv')
  
  return historic_c_p_a0, historic_c_p_a1

  
def get_storm_stats(dd):
  
  radii=pd.read_csv(dd + '/JPM-OS/radius.csv')
  historic_theta_bar=pd.read_csv(dd + '/JPM-OS/historic_theta_bar.csv')
  historic_theta_var=pd.read_csv(dd + '/JPM-OS/historic_theta_var.csv')
  historic_x_freq=pd.read_csv(dd + '/JPM-OS/historic_x_freq.csv')
  #storm_params=pd.read_csv(dd + '/JPM-OS/JMP-OS_storm_parameters.csv')
  #storm_params=stm.get_storm_params(dd)
  lon=pd.read_csv(dd + '/JPM-OS/lon.csv')
  
  return radii, historic_theta_bar, historic_theta_var, historic_x_freq, lon

# calculation functions
## Make bins, to find bounds of intervals of each parameter
# input: vector, global bounds
# output: 2 vectors
def make_bins(para_vec, para_min, para_max):
  para_vec=list(para_vec)
  n=len(para_vec)
  para_vec.sort()
  quantiles=[0]*(n+1)
  for i in range(1,n):
    quantiles[i]=(para_vec[i-1]+para_vec[i])/2
  quantiles[0]=para_min
  quantiles[n]=para_max 
  mins=quantiles[0:n]
  maxs=quantiles[1:n+1]
  return (mins, maxs)

# track to x
def track_to_x(lon, zero_lon):
  return(zero_lon-math.floor(lon))

## Lambda 1
def lambda1(intensity, p_min, p_max, p_global_min, p_global_max, location, historic_c_p_a0, historic_c_p_a1):
    # p_min and p_ max are the bounds of intervals
    # p_global_min and p_global_max are the global minimum and maximum values
    # x is the distance in degrees of longtitude (location)
    # a0 and a1 are the coefficient of Gumbel Distribution 
    a0_mid=historic_c_p_a0.loc[historic_c_p_a0['x'] == location]
    a0=a0_mid.iloc[0,1]
    a1_mid=historic_c_p_a1.loc[historic_c_p_a1['x'] == location]
    a1=a1_mid.iloc[0,1]
    lambda1_cum_1=st.gumbel_r.cdf(x=1013.25-p_max,loc=a0*(1+intensity) + intensity*a1*0.577216, scale=a1)
    lambda1_cum_2=st.gumbel_r.cdf(x=1013.25-p_min,loc=a0*(1+intensity) + intensity*a1*0.577216, scale=a1)
    lambda1_cum=lambda1_cum_1-lambda1_cum_2
    scaling_factor_1=st.gumbel_r.cdf(x=1013.25-p_global_max,loc=a0*(1+intensity) + intensity*a1*0.577216, scale=a1)
    scaling_factor_2=st.gumbel_r.cdf(x=1013.25-p_global_min,loc=a0*(1+intensity) + intensity*a1*0.577216, scale=a1)
    scaling_factor=scaling_factor_1-scaling_factor_2
    return (lambda1_cum/scaling_factor)
    
## Lambda2
def lambda2(r_min, r_max, r_bar, r_sd, r_global_min, r_global_max):
    r_lower=(r_global_min-r_bar)/r_sd
    r_upper=(r_global_max-r_bar)/r_sd
    lambda2_cum_low=st.truncnorm.cdf(x=r_min,a=r_lower,b=r_upper,loc=r_bar,scale=r_sd)
    lambda2_cum_high=st.truncnorm.cdf(x=r_max,a=r_lower,b=r_upper,loc=r_bar,scale=r_sd)
    return (lambda2_cum_high-lambda2_cum_low)

## Lambda3
def lambda3(v_min, v_max, v_bar, v_sd, v_global_min, v_global_max):    
#    v_lower=(v_global_min-v_bar)/v_sd
#    v_upper=(v_global_max-v_bar)/v_sd
#    lambda3_cum_1=st.truncnorm.cdf(x=v_min,a=v_lower,b=v_upper,loc=v_bar,scale=v_sd)
#    lambda3_cum_2=st.truncnorm.cdf(x=v_max,a=v_lower,b=v_upper,loc=v_bar,scale=v_sd)
#    return (lambda3_cum_2-lambda3_cum_1)
    return (1)
     
## Lambda4
def lambda4(theta_min, theta_max, theta_global_min, theta_global_max, location):   
#    theta_bar_mid=historic_theta_bar.loc[historic_theta_bar['x'] == location]
#    theta_bar=theta_bar_mid.iloc[0,1]
#    theta_var_mid=historic_theta_var.loc[historic_theta_var['x'] == location]
#    theta_var=theta_var_mid.iloc[0,1]
#    theta_sd=math.sqrt(theta_var)
#    theta_lower=(theta_global_min-theta_bar)/theta_sd
#    theta_upper=(theta_global_max-theta_bar)/theta_sd
#    lambda4_cum_1=st.truncnorm.cdf(x=theta_min,a=theta_lower,b=theta_upper,loc=theta_bar,scale=theta_sd)
#    lambda4_cum_2=st.truncnorm.cdf(x=theta_max,a=theta_lower,b=theta_upper,loc=theta_bar,scale=theta_sd)
#    return (lambda4_cum_2-lambda4_cum_1)
    return (1)
    
## Lambda 5
def lambda5(lon_min, lon_max, zero_lon, historic_x_freq, x_global_min, x_global_max):
    def track_to_x(lon, zero_lon):
      return(zero_lon-math.floor(lon))

    region_freq = historic_x_freq.loc[(historic_x_freq['x'] >= x_global_min) & \
                                      (historic_x_freq['x'] <= x_global_max), 'freq'].sum()
    
    x1 = track_to_x(lon_max, zero_lon)
    x2 = track_to_x(lon_min, zero_lon)
    
    prop1 = lon_max - math.floor(lon_max)
    prop2 = math.ceil(lon_min) - lon_min
    
    x_freq_x1 = historic_x_freq.loc[historic_x_freq['x'] == x1, 'freq'].sum()
    x_freq_x2 = historic_x_freq.loc[historic_x_freq['x'] == x2, 'freq'].sum()
    
    if x1 == x2:
      return (x_freq_x1*(lon_max-lon_min)/region_freq)
    elif x2-x1 == 1:
      return (x_freq_x1*prop1+x_freq_x2*prop2)/region_freq   
    else:
      x_freq_between = historic_x_freq.loc[(historic_x_freq['x'] >= (x1+1)) & \
                                           (historic_x_freq['x'] <= (x2-1)), 'freq'].sum()
      return((x_freq_x1*prop1 + x_freq_x2*prop2 + x_freq_between)/region_freq)


def get_storm_probs(intensity, radii, historic_theta_bar, historic_theta_var,\
                     historic_x_freq, all_storm_params, lon, historic_c_p_a0, historic_c_p_a1):
    
    ## Assign values to each parameter
    # Central Pressure
    c_p=[900,930,960,975]
    p_global_max=985 # theoretical max value
    p_global_min=882 # theoretical min value

    # R_max
    radius=radii.iloc[:,1]
    r_global_max=40
    r_global_min=5

    # velocity
    v=11
    v_global_min=0
    v_global_max=math.inf


    # theta
    theta=0
    theta_global_min=-90
    theta_global_max=90

    # location
    #lon_mid=parameters.loc[parameters['set2'] == 1]
    #lon=lon_mid.iloc[:,8]
    zero_lon=-83
    x_freq=historic_x_freq.iloc[:,1]
    x=historic_x_freq.iloc[:,0]
    w_bound=-94.42
    e_bound=-88.49
    lon=lon.iloc[:,0]

    p_min,p_max=make_bins(c_p, p_global_min, p_global_max)

    l_x=len(historic_c_p_a0)
    l_p=len(p_max)
    l_lon=len(lon)
    lambda1_matrix=np.zeros((l_lon,l_p))
    for i in range(l_p):
     for j in range(l_lon):
         location=track_to_x(lon[j],zero_lon)
    #     location=historic_c_p_a0.iloc[j,0];
         lambda1_matrix[j,i]=lambda1(intensity, p_min[i], p_max[i], p_global_min, p_global_max, location, historic_c_p_a0, historic_c_p_a1)
    #  np.sum(lambda1_matrix, axis=1)

    l_cp=len(c_p) # number of c_p
    l_Rmax_total=len(radius) # total number of rmax
    l_rmax=int(l_Rmax_total/l_cp) # number of rmax per c_p
    rmax_matrix=np.zeros((l_cp,l_rmax))
    for i in range(l_cp):
        for j in range(l_rmax):
            rmax_matrix[i,j]=radius[l_rmax*i+j]

    # r_min, r_max        
    r_min=np.zeros((l_cp,l_rmax))
    r_max=np.zeros((l_cp,l_rmax))
    for i in range(l_cp):
        r=make_bins(rmax_matrix[i], r_global_min, r_global_max)
        r_min[i]=r[0]
        r_max[i]=r[1]

    # r_bar, r_sd   
    r_bar=np.zeros(l_cp)
    r_sd=np.zeros(l_cp)
    for i in range(l_cp):
       r_bar[i]=14+0.3*(110-(1013.25-c_p[i]))
       r_sd[i]=0.44+r_bar[i]*c_p[i]
        
    lambda2_matrix=np.zeros((l_cp,l_rmax)) 
    for i in range(l_cp):
        for j in range(l_rmax):
         lambda2_matrix[i,j]=lambda2(r_min[i,j],r_max[i,j],r_bar[i],r_sd[i],r_global_min,r_global_max)  


   
    lon_min,lon_max=make_bins(lon, w_bound, e_bound)

    # x_global_min, x_global_max
    x_global_min=track_to_x(e_bound, zero_lon)
    x_global_max=track_to_x(w_bound, zero_lon)


    #region_freq=0
    #x_freq_global_min=x_global_min
    #n_global=x_global_max-x_global_min+1
    #for i in range(n_global):
    #    x_freq_global_min_mid=historic_x_freq.loc[historic_x_freq['x'] == x_freq_global_min]
    #    region_freq += x_freq_global_min_mid.iloc[0,1]
    #    x_freq_global_min=x_freq_global_min+1 

    # lambda5
    lambda5_matrix=np.zeros(l_lon)
    for i in range(l_lon):
     lambda5_matrix[i]=lambda5(lon_min[i], lon_max[i], zero_lon, historic_x_freq, x_global_min, x_global_max)

    ## assign probabilities to each synthetic storm
    n_storms = len(all_storm_params.index)
    prob_by_storm=np.zeros(n_storms)
    for i in range(n_storms):
        storm_params=all_storm_params.iloc[i,:]
        for j in range(l_lon):
          if lon_min[j] <= storm_params['lon'] <= lon_max[j]:
             lambda1_updated=lambda1_matrix[j,:]
             for k in range(l_cp):
               if p_min[k]<= storm_params['c_p']<= p_max[k]:
                  P_1=lambda1_updated[k]
             
        for l in range(l_cp):
          if p_min[l] <= storm_params['c_p'] <= p_max[l]:
                lambda2_updated=lambda2_matrix[l,:]
                for m in range(l_rmax):
                 if r_min[l,m] <= storm_params['radius'] <= r_max[l,m]:
                     P_2=lambda2_updated[m]
        for n in range(l_lon):
            if lon_min[n] <= storm_params['lon'] <= lon_max[n]:
                P_5=lambda5_matrix[n]
        prob_by_storm[i]=P_1*P_2*P_5/lambda5_matrix.sum()
    
    storm_id=all_storm_params['storm_id']
    storm_id=np.array(storm_id)
    storm_id=storm_id.tolist()
    prob_by_storm=np.array(prob_by_storm)
    prob_by_storm=prob_by_storm.tolist()
    prob_by_storm=pd.DataFrame({'storm_id':storm_id, 'prob_by_storm':prob_by_storm})
    return prob_by_storm

#def calc_flood_elevs(storm, reaches, polder, MC_iterates, crest_heights):
#    return storm_SWE
    
def get_swe_cdf(storm_data, prob_by_storm, overall_frequency, min_swe=-99):
    # The probability of observing i storms of interest in a given year
    #overall_frequency=0.253731 
    def P_i(overall_frequency, i):
        P_i = (math.exp(-overall_frequency))*(overall_frequency**i)/math.factorial(i)
        return P_i

    # Fannual
    def sub_F_annual(cum_prob):
        mysum = 0
        for i in range(10): # i is from 0 to 10 since the probability that there 
            #are more than 10 storms in a year is very small so we can ignore it.
            mysum += ((cum_prob*overall_frequency)**i)/math.factorial(i)
        sub_F_annual = math.exp(-overall_frequency)*mysum
        return sub_F_annual
    
    # join storm relative likelihoods to storm frequency distributions
    storm_data = pd.merge(storm_data, prob_by_storm, how='inner', on='storm_id', sort=False)

    # sort elevation
    storm_data=storm_data.sort_values(by='depths',ascending=True)   
    storm_data=storm_data.reset_index(drop=True)

    # PMF of elevation
    storm_data['SWE_prob']=storm_data['depth_probs']*storm_data['prob_by_storm']

    # CDF of elevation (F(dn))
    storm_data['cum_prob']=np.cumsum(storm_data['SWE_prob'])
    #storm_data


    F_annual=[0]*len(storm_data)
    for i in range(len(storm_data)):
        F_annual[i] =  sub_F_annual(storm_data['cum_prob'][i])  
    storm_data['F_annual'] = F_annual

    #F_vals
    years=[5, 8, 10, 13, 15, 20, 25, 33, 42, 50, 75, 100, 125, 150, 200, 250, 300, 350, 400, 500, 1000, 2000]
    F_vals=[0]*len(years)
    for i in range(len(years)):
     F_vals[i]=1-1/years[i]
    F_mid=[0]*len(years)
    
    # initialize with minimum SWE in the polder (needed if overall frequency is 
    # low enough such that the probability of observing a storm is less than  
    # one of the desired return periods)
    SWE=[min_swe]*len(years)
    for i in range(len(years)):
        for j in range(len(storm_data)-1):
            if F_vals[i]<F_annual[j+1] and F_vals[i]>=F_annual[j]:
                F_mid[i]=F_annual[j]
                SWE_mid=storm_data.loc[storm_data['F_annual']==F_mid[i]]
                SWE[i]=max(SWE[i],SWE_mid.iloc[0].loc['depths'])
                
    swe_cdf= pd.DataFrame({'return period (year)': years, 'SWE': SWE})
    #print(swe_cdf)
    return (swe_cdf)



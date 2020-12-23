
directory = 'D:/Projects/Larose' #find actual file path later
dd = directory + '/Data'


import storms as stm
import jpmos as jpm
import cost_model as c_mdl
import metrics_full_interpolation as mtc
import rain as rain
import copy
import numpy as np
import larose_function_variations as lfv
import csv


np.random.seed(42)




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
base_frequency = 0.22

unit_price = c_mdl.get_unit_price(dd)
#reach_df = stm.get_reach_df(dd,file = "/reach_config_file_ipet.csv").sort_values('reachID')
reach_df = stm.get_reach_df(dd,file = "/reach_config_file_new_system_ipet.csv").sort_values('reachID')

#rainfall = rain.rainfall_prediction(storm_params, rain_params)
#rainfall = 0
base_crest_heights = reach_objects['reachHeight'].to_numpy() #something in the reach objects
print("parameters set")


partic_rate = 0
sea_lvl = 1.03
rainfall = 0
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
MCIterates = 25

arguments = [reach_objects, dmg_data, nms_data, bfe_data, nsc_data, storm_params,
             locParams, surgeSurfaceParams, waveSurfaceParams, wavePeriodSurfaceParams,
             sigmaSurfaceParams, radii, historic_theta_bar, historic_theta_var,
             historic_x_freq, lon, polderObject, MCIterates, base_frequency,
             unit_price, rainfall, base_crest_heights, partic_rate, pop_mult,
             ns_cost_mult, acq_cost_mult, str_cost_mult, NS_std, acq_threshold,
             res_fp, rain_params, historic_c_p_a0, historic_c_p_a1, reach_df]

sea_lvl_scenarios = [0.46, 0.63, 0.83]
intensity_scenarios = [0.1, 0.125, 0.15]
frequency_scenarios = [0, -0.14, -0.28]

initial_reach_height = [2]*12
current_upgrades = copy.deepcopy(initial_reach_height)
upgrade_order = []
cost_progression = []
ead_progression = []
increment_height = 0.1

# index for the scenarios defined above; 1 - Low, 2 - Medium, 3 - High
slr_to_run = 3
int_to_run = 3
freq_to_run = 3

random_seed = 42

initial_ead, initial_cost = lfv.larose_over_time(crest_height_upgrade = initial_reach_height, 
                                                   total_rise = sea_lvl_scenarios[slr_to_run-1],
                                                   intensity_change = intensity_scenarios[int_to_run-1],
                                                   frequency_change = frequency_scenarios[freq_to_run-1],
                                                   needed_arguments = copy.deepcopy(arguments))
cost = initial_cost

strat_count = 0
ead = 100000 # initializing arbitrary value greater than while loop control

while (cost < 1000000000 and strat_count <= 500 and ead > 10000):
    
    best_ratio = 0
    best_reach = -1
    
    for reach_num in range(0,12):
        np.random.seed(random_seed)
        temp_upgrade = copy.deepcopy(current_upgrades)
        temp_upgrade[reach_num] = temp_upgrade[reach_num] + increment_height
        
        temp_ead, temp_cost = lfv.larose_over_time(crest_height_upgrade = temp_upgrade, 
                                                   total_rise = sea_lvl_scenarios[slr_to_run-1],
                                                   intensity_change = intensity_scenarios[int_to_run-1],
                                                   frequency_change = frequency_scenarios[freq_to_run-1],
                                                   needed_arguments = copy.deepcopy(arguments))
        
        marginal_ead = initial_ead - temp_ead
        marginal_cost = temp_cost - initial_cost
        if (marginal_cost > 0):
            ratio = marginal_ead / marginal_cost
            if (ratio > best_ratio):
                best_ratio = ratio
                best_reach = reach_num
                cost = temp_cost
                ead = temp_ead
            
    if (best_reach > -1):
        upgrade_order.append(best_reach)
        cost_progression.append(cost)
        ead_progression.append(ead)
        current_upgrades[best_reach] = current_upgrades[best_reach] + increment_height
        initial_ead = ead
        initial_cost = cost   
        
    strat_count = strat_count + 1
    
    if (strat_count % 10 == 0):
        print("Attempt " + str(strat_count))
    
outputs = [upgrade_order, ead_progression, cost_progression]

file = open('D:/Projects/Larose/Analysis_9_10/incremental_upgrades_S' + str(slr_to_run) + '_I' + str(int_to_run) + '_F' + str(freq_to_run) + '_RS' + str(random_seed) + '_2020-11-20.csv',
            'w+', newline='')
#file = open('D:/Projects/Larose/Analysis_9_10/incremental_upgrades_scen' + str(scen_to_run+1) + '_' + str(random_seed) + '_2020-11-03.csv',
#            'w+', newline='')
with file:
    write = csv.writer(file)
    write.writerows(outputs)
    
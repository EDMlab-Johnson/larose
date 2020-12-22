# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 21:20:53 2019

@author: richardz
"""

import pandas as pd
import numpy as np
import math
  
#def get_reach_objects(dd):
#  
#   reach_objects = pd.read_csv(dd + '/reach_config_file_updated.csv')
#  
#   return reach_objects
  
# should this be from: get_storm_stats()  ?
  
def get_unit_price(dd):

 unit_price= pd.read_csv(dd + '/cost_config_file.csv')
 
 return unit_price
  
 
def calc_str_cost(crest_height_upgrade, reach_objects, unit_price, str_cost_mult, planning_horizon=50, discount_rate=0.03):
    reach=reach_objects
    cost=np.mat(unit_price)
    
      # Levee function

    def Levee_Reach_Quantity(crest_height_lift, totalReachLengthInMeters, tWallLength,crownWidth, rearSlope, forwardSlope, existingTopOfLevee, existingGrade, overbuild, clearGrubDepth, existingCrownWidth, existingRearSlope, existingForwardSlope, rearInspectionCorridor, forwardInspectionCorridor):
        
        crest_height_lift_in_feet = crest_height_lift/0.3048 # from meters to feet
        # Levee Reach Length
        # totalReachLengthInMeters=input('Total Reach Length in meters=');
        totalReachLength=totalReachLengthInMeters/0.3048; # from meters to feet
        # tWallLength=input('T-wall Length (Reach)=');
        L = totalReachLength - tWallLength; # Length of Levee Reach
    
        # 4.1 Clear & Grub
        # crownWidth=input('crownWidth in feet=')
        # rearSlope=input('PS Slope=');
        # forwardSlope=input('FS Slope=');
        # exitingTopOfLevee=input('Existing Top of Levee=');
        # existingGrade=input('Existing Grade=');
        ELH = existingTopOfLevee - existingGrade;  # Existing Levee Height
        Y1 = ELH;
        if crest_height_lift < 0:
            print("warning: the crest height upgrade is negative")
            H = 0
        else:
            H = crest_height_lift_in_feet # Levee Lift Height
        # overbuild=input('% Overbuild=');
        #H = leveeDesignElevation - existingTopOfLevee; 
        Ho = H * overbuild; # Overbuild height
        Hc = H + Ho; # Construction height
        Y2 = ELH + Hc; # total height
        X5 = Y2 * rearSlope;
        X6 = Y2 * forwardSlope;
        X =crownWidth + X5 + X6; # Cross Section Length
        Fp_SF = L * X; # Footprint of Levee in SF
        Fp_AC = Fp_SF / 43560; # transfer footprint of levee from SF to AC
    
        # 4.2 Embankment and Compaction
        # clearGrubDepth=input('Clear and Grub Depth in feet='); 
        CGV = clearGrubDepth * X * totalReachLength;  # Clear and Grub Vol
        # existingCrownWidth=input('Existing CrowncrowdWidthidth in feet=');
        X1 = existingCrownWidth;
        A1 = Y1 * X1; 
        # existingRearSlope=input('Existing PS Slope=');
        X3 = Y1 * existingRearSlope;
        A2 = 0.5 * Y1 * X3;
        # existingForwardSlope=input('Existing FS Slope=');
        X2 = Y1 * existingForwardSlope;
        A3 = 0.5 * Y1 * X2;
        X4 =crownWidth;
        A4 = Y2 * X4;
        A5 = 0.5 * Y2 * X5;
        A6 = 0.5 * Y2 * X6;
        Anew = A4 + A5 + A6;
        Aexisting = A1 + A2 + A3;
        Adelta = Anew - Aexisting;
        A = Adelta;
        V_CF = L * A + CGV; #Volume of Levee in CF
        V_CY = V_CF / 27; # transfer Volume of Levee from CF to CY
    
        # 4.3 Turff
        import math
        L1 = math.sqrt(pow(X5,2) + pow(Y2,2));
        L2 = math.sqrt(pow(X6,2) + pow(Y2,2));
        # rearInspectionCorridor=input('PS inspection Corridor=');
        # forwardInspectionCorridor=input('FS inspection Corridor=');
        As_sf = (L1 + L2 + X4 + rearInspectionCorridor + forwardInspectionCorridor) * L; # Surface area of levee in sf
        As_ac = As_sf / 43560; # transfer Surface area of levee from sf to ac
    
        # 4.4 SWPPP
        SWPPP = 2 * L;
    
        return L, Fp_AC, V_CY, As_ac, SWPPP


    # T-wall Function
    def Twall_Reach_Quantity(baseSlabThickness, structuralSuperiority, leveeDesignElevation, tWallLength, wallThickness, existingGrade, baseSlabWidth, foundationPiles, eachPileLength, pileSpacing, pileRows, sheetPileLength):
        
        # T-wall Reach Height
         if baseSlabThickness==0:
             Tow=structuralSuperiority # Tow: Wall Design Elevation (Top of wall)
         else:
             Tow=leveeDesignElevation+structuralSuperiority
        
        # T-wall Reach Length (assumed to be in feet)
         tWallLength=tWallLength;
        
        # 5.1 Wall Concrete
         if baseSlabThickness==0:
             existingGrade=0
         else:
             existingGrade=existingGrade 
         Tos=existingGrade # Top of slab elevation
         Hw=Tow-Tos # Wall height
         Wv=(Hw*wallThickness*tWallLength)/27 #Wall Volume converted to CY

        
        # 5.2 Base Concrete
         Bv=(tWallLength*baseSlabThickness*baseSlabWidth)/27 # Base Volume converted to CY
        
        # 5.3 Foundation Piles(PPC)
         foundationPiles=0 
        
        # 5.4 Foundation Piles(HP)
         Lw=tWallLength # Length of wall
         Np=(Lw/pileSpacing)*pileRows # Number of piles req'd
         Lpt=eachPileLength*Np #Pile length (total)
         
        # 5.5 Sheetpile Cutoff Wall
         Asp=sheetPileLength*tWallLength # Area of sheetpile
        
        # 5.6 3-bulb Waterstop
         Ws=tWallLength 
        
        # 5.7 Excavation/Fill
         EF=Bv
        
         return Wv, Bv, foundationPiles, Lpt, Asp, Ws, EF

    # Reach parameters
    #Name=reach['reachName']
    totalReachLengthInMeters=reach['totalReachLengthInMeters']
    tWallLength=reach['tWallLength']
    crownWidth=reach['crownWidth']
    rearSlope=reach['rearSlope']
    forwardSlope=reach['forwardSlope']
    existingTopOfLevee=reach['existingTopOfLevee']
    existingGrade=reach['existingGrade']
    overbuild=reach['overbuild(%)']
    clearGrubDepth=reach['clearGrubDepth']
    existingCrownWidth=reach['existingCrownWidth']
    existingRearSlope=reach['existingRearSlope']
    existingForwardSlope=reach['existingForwardSlope']
    rearInspectionCorridor=reach['rearInspectionCorridor']
    forwardInspectionCorridor=reach['forwardInspectionCorridor']


    #T-wall Parameters
    tWallValue=reach['tWallValue'] # whether t-wall exists or not
    baseSlabThickness=reach['baseSlabThickness']
    structuralSuperiority=reach['structuralSuperiority']
    tWallLength=reach['tWallLength']
    wallThickness=reach['wallThickness']
    existingGrade=reach['existingGrade']
    baseSlabWidth=reach['baseSlabWidth']
    foundationPiles=reach['foundationPiles']
    eachPileLength=reach['eachPileLength']
    pileSpacing=reach['pileSpacing']    
    pileRows=reach['pileRows']
    sheetPileLength=reach['sheetPileLength']

    # total quantity for construction part
    length=len(reach);
    totalLeveeQuantity=[0]*5;
    totalTwallQuantity=[0]*7; 
    for i in range(length):
      #print("Levee Design Elevation of ",Name[i])
      crest_height_lift = float(crest_height_upgrade[i]) # in meters
      crest_height_lift_in_feet_2 = crest_height_lift/0.3048 # in feet
      leveeDesignElevation = crest_height_lift_in_feet_2 + existingTopOfLevee[i] # in feet
      if crest_height_lift == 0:
         reachResult = [0]*5 # if  there is no upgrade for the reach, then the no construction cost needs to be calculated
      else: 
         reachResult = Levee_Reach_Quantity(crest_height_lift, totalReachLengthInMeters[i], 
            tWallLength[i], crownWidth[i], rearSlope[i], forwardSlope[i],existingTopOfLevee[i], 
            existingGrade[i], overbuild[i], clearGrubDepth[i], existingCrownWidth[i], existingRearSlope[i], 
            existingForwardSlope[i],rearInspectionCorridor[i], forwardInspectionCorridor[i])
      for j in range(5):
         totalLeveeQuantity[j] += reachResult[j]; # total reach quantity, need times unit cost then
      
      if  tWallValue[i]==0:
          twallResult=[0]*7;
      else:
          twallResult = Twall_Reach_Quantity(baseSlabThickness[i], structuralSuperiority[i], leveeDesignElevation, tWallLength[i], wallThickness[i], existingGrade[i], baseSlabWidth[i], foundationPiles[i], eachPileLength[i], pileSpacing[i], pileRows[i], sheetPileLength[i]);
      for j in range(7):
         totalTwallQuantity[j] += twallResult[j]; # total T-wall quantity, need times unit cost then
    
    
    # total quantity for O&M part
    totalLeveeQuantityForOM=[0]*5
    #totalTwallQuantityForOM=[0]*7
    for i in range(length):
      #print("Levee Design Elevation of ",Name[i])
      crest_height_lift = float(crest_height_upgrade[i])
      # if there is no upgrade, OM cost still needs to be considered
      reachResultForOM = Levee_Reach_Quantity(crest_height_lift, totalReachLengthInMeters[i], 
            tWallLength[i], crownWidth[i], rearSlope[i], forwardSlope[i],existingTopOfLevee[i], 
            existingGrade[i], overbuild[i], clearGrubDepth[i], existingCrownWidth[i], existingRearSlope[i], 
            existingForwardSlope[i],rearInspectionCorridor[i], forwardInspectionCorridor[i])
      for j in range(5):
         totalLeveeQuantityForOM[j] += reachResultForOM[j]; # total reach quantity, need times unit cost then
       
    
    #Construction cost
    leveeUnitCost=cost[0:4,1]
    twallUnitCost=cost[4:11,1]
    totalLeveeQuantity=np.asarray(totalLeveeQuantity)
    leveeTotalCost=np.dot(np.transpose(leveeUnitCost),totalLeveeQuantity[1:5])
    twallTotalCost=np.dot(np.transpose(twallUnitCost),totalTwallQuantity)
    subtotal=leveeTotalCost+twallTotalCost
    mobilizationRate=cost[11,1]
    mobilization=subtotal*mobilizationRate
    subtotalWithoutContingency=subtotal+mobilization
    contingencyRate=cost[13,1]
    contingency=subtotalWithoutContingency*contingencyRate
    # total estimated construction cost
    constructionCost=float(subtotalWithoutContingency+contingency)
    
    
    # Construction Management Cost
    constructionMgtCost=subtotal*cost[12,1]
    # Planning, Engineering and Design cost
    PEDRate=cost[14,1]
    PED=float(constructionCost*PEDRate)
    
    
    # Operations and Maintenance cost(per 50 yr)
    QTY=[0]*11
    QTY[0]=float(totalLeveeQuantityForOM[3])
    QTY[1]=float(totalLeveeQuantityForOM[0])/5280
    QTY[2]=1
    QTY[3]=1
    QTY[4]=float(totalLeveeQuantityForOM[0])/5280
    QTY[5]=float(totalLeveeQuantityForOM[0])
    QTY[6]=float(sum(tWallLength))/5280
    frequency=cost[0:11,4]
    unitprice=cost[0:11,5]
    subtotal1=np.multiply(QTY,np.transpose(frequency))
    subtotal2=np.dot(subtotal1,unitprice)
    subtotal=float(subtotal2) # Subtotal O and M cost Without Contingency
    #years=cost[12,3]
    contingencyRate_OandM=cost[11,3]
    contingency_OandM=subtotal*contingencyRate_OandM
    OandMCostPerYear=subtotal+contingency_OandM
    OandMCostPerYearWithoutAction = cost[13,3]
    AnnualOandMCost = OandMCostPerYear - OandMCostPerYearWithoutAction
    if discount_rate == 0:
        conversionFactor = planning_horizon
    else:
        conversionFactor = (math.pow((1+discount_rate), planning_horizon)-1)/(discount_rate*math.pow(discount_rate+1,planning_horizon))
    OandMCost=conversionFactor * AnnualOandMCost

    #Total cost
    # print("constructionCost")
    # print(constructionCost)
    # print("constructionMgtCost")
    # print(constructionMgtCost)
    # print("PED")
    # print(PED)
    # print("OandMCost")
    # print(OandMCost)
    totalcost=float(constructionCost+constructionMgtCost+PED+OandMCost)
    totalcostwithUncertaintyFactor=totalcost*str_cost_mult
    #print('Estimated Construction Cost=',constructionCost)
    #print('Planning, Engineering and Design=', PED)
    #print('Operations and Maintenance=',OandMCost)
    #print('Total Cost=',totalcost)
    #print('Cost Including Uncertainty Factor',totalcostwithUncertaintyFactor)
    return totalcostwithUncertaintyFactor


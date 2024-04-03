# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 18:17:49 2020

@author: Liisa
"""
#Options
#Development of human actvity scenarios
#Scenario Most likely, Scenario BAU, Scenario Low chnage, Scenario high change
Scenario="Scenario BAU"
#Application extent for new measures
#For new measures: LOWEX,MEDEX,MAXEX
Extent="MEDEX"
#Include New measures Yes/No
IncludeNew="Yes"

#Following quantities N_ are defined to make sure that data is read correctly
#Number of activities, pressures, state variables, basins,countries,measure types   
N_act=42
N_pres=84
N_state=29
N_basins=18
N_count=10 
N_mtypes=161
#Number of measure type effectiveness grid questions,maximum NO of experts assessing measure type effectiveness,
N_MTEQ=126
N_expM=29
#Number of actual measures, maximum number of experts assessing activity pressure contributions
N_actMeas=264 
N_expAP=6
#Number of Human development activity changes and links between pressures
N_DEV=39 
#Number of equivalence links for pressures
N_SubPres=65
#No of experts and state components by state topic
N_expPS=22
N_state_com=73
N_expPS_Ben=19
N_state_comBen=20
N_expPS_Bird=16
N_state_comBird=6
N_expPS_FishM=16
N_state_com_FishM=11
N_expPS_FishCoast=14
N_state_com_FishCoast=16
N_expPS_FishC=14
N_state_com_FishC=8
N_expPS_HZ=22
N_state_comHZ=4
N_expPS_Mam=7
N_state_comMam=8
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:45:36 2020

@author: Liisa Saikkonen
"""
#Ran and edited with anaconda spyder and python 3.7.7 

#Import libraries and initialize correct input files  in "initialize.py"
#The libraries need to be installed before they can be imported in initialize.py. Use anaconda navigator.
#ownFunctions file also depend on pandas library
exec(open("initialize.py").read())

#If the numbers of items (activities, pressures, states...) have been changed these need to be updated in item_numbers.py.
#This is done so that the data is read correctly
#Also the options for the development sceanrio,inclusion of new measures and their application extent are defined here
exec(open("item_numbers.py").read())

#The following 3 files calculate the actual results, and they should be run in this order
#These files call functions in file ownFuncltions which has to be in the same folder

#defines distributions and draws values for measure effectivenessa and activity pressure distributions
exec(open('measure_effectivenessCEA.py').read())
#Calculates reductions in prohjected pressures based on the distributions and drawn values, human activity scenarios actual measures etc.
exec(open('pressure_reductionCEA.py').read())
#Estmates the imapcts of pressure reductions on state, upcoming
#exec(open('pressure_stateCEA.py').read())

#Following line saves the workspace variables of current session
#To run following line you need to install dill library
dill.dump_session("session_name.pkl")
#dill.load can be used to load the workspace variables

#Following 2 files export main results to excel .
#First one can be run after measure_pressure.py
exec(open('export_measure_pressure_CEA.py').read())
#Second one can be run after pressure_state.py
#exec(open('export_pressure_state.py').read())


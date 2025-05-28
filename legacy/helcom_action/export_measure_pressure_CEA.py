# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:41:28 2020

@author: E1007607
"""
#Probablity distribution
#Next row for litter
#probsREffPress=probsTREffPress
#probsREffPress.to_excel('Total pressure reduction with Overlap and chain effect.xlsx')
#probsEffPress.to_excel('Total pressure reduction without Overlap and chain effect.xlsx')
#probsEffPress2.to_excel('Total pressure reduction of direct to pressure.xlsx')

#pressure reductions from existing measures
#For litter
#Reff_on_press_sum=TReff_on_press
#28_7:changed this to AAR2
#RReff_on_press_sum=Reff_on_press_sum.groupby(['Basins','Pressure']).mean().reset_index()  # can also not group then use #part for orgnizes data fromate
RReff_on_press_sum=AAR4
#28_7: basins to GA
RReff_on_press_sum.columns=RReff_on_press_sum.columns.map(str)
#ReportPresRed=pd.DataFrame({"Pressure":RReff_on_press_sum['Pressure'],"GA":RReff_on_press_sum['GA'],\
#                            "Mean":RReff_on_press_sum.loc[:,"0":"999"].mean(axis=1),"Stdev":RReff_on_press_sum.loc[:,"0":"999"].std(axis=1)})

ReportPresRed=pd.DataFrame({"Pressure":RReff_on_press_sum['presName'],"GA":RReff_on_press_sum['basName'],\
                            "Mean":RReff_on_press_sum.loc[:,"0":"999"].mean(axis=1),"Stdev":RReff_on_press_sum['1001'],\
                            "Relstdev":RReff_on_press_sum['1001'].div(RReff_on_press_sum.loc[:,"0":"999"].mean(axis=1))})
#Organize data to the formate match the template
PresRedmean=ReportPresRed.pivot(index='GA', columns='Pressure', values='Mean')
PresRedSD=ReportPresRed.pivot(index='GA', columns='Pressure', values='Stdev')
#PresRedmean = pd.pivot_table(ReportPresRed, values='Mean', index=['Basins'], columns=['Pressure'], aggfunc=np.mean)
#PresRedSD = pd.pivot_table(ReportPresRed, values='Stdev', index=['Basins'], columns=['Pressure'], aggfunc=np.mean)

#Exprt results
ReportPresRed.to_excel('pupdated\pressure reductions from existing measures (list form).xlsx')
PresRedmean.to_excel('pupdated\pressure reductions from existing measures (Expected value).xlsx')
PresRedSD.to_excel('pupdated\pressure reductions from existing measures (SD).xlsx')
#writer = pd.ExcelWriter('pressure reductions from existing measures.xlsx', engine='xlsxwriter')
#ReportPresRed.to_excel(writer, sheet_name='list form')
#PresRedmean.to_excel(writer, sheet_name='Expected value')
#PresRedSD.to_excel(writer, sheet_name='SD')

#replace ID to name?

#Export measure effectivness
RGMT_sims=GMT_sims.copy()
RGMT_sims.columns=GMT_sims.columns.map(str)
#Organize data to another the formate match the template

RGMT_sims['presName']="None"
RGMT_sims['actName']="None"
RGMT_sims['MTName']="None"
for index, row in RGMT_sims.iterrows():
   if row["Pressure"]>0:  
       RGMT_sims.at[index,'presName']=pressures[pressures.index==row["Pressure"]]["Pressures"].iloc[0]
       RGMT_sims.at[index,'actName']=activities[activities.index==row["Activity"]]["Activity"].iloc[0]
   else:
       RGMT_sims.at[index,'presName']="Direct to state"
   RGMT_sims.at[index,'MTName']=mtypes[mtypes.index==row["Measure type"]]["Measure type"].iloc[0]
#Exprt results
ReportMeaEff=pd.DataFrame({"Pressure":RGMT_sims['presName'],"Activity":RGMT_sims['actName'],\
                           "Measure type":RGMT_sims['MTName'],"Mean":RGMT_sims.loc[:,"0":"999"].mean(axis=1),\
                           "Stdev":RGMT_sims.loc[:,"0":"999"].std(axis=1),"Relstdev":RGMT_sims.loc[:,"0":"999"].std(axis=1).div(RGMT_sims.loc[:,"0":"999"].mean(axis=1))})   
ReportMeaEff.to_excel('pupdated\Effectiveness of measure types (list form).xlsx')  
ReportMeaEff=pd.DataFrame({"Pressure":RGMT_sims['Pressure'],"Activity":RGMT_sims['Activity'],\
                           "Measure type":RGMT_sims['Measure type'],"Mean":RGMT_sims.loc[:,"0":"999"].mean(axis=1),\
                           "Stdev":RGMT_sims.loc[:,"0":"999"].std(axis=1),"Relstdev":RGMT_sims.loc[:,"0":"999"].std(axis=1).div(RGMT_sims.loc[:,"0":"999"].mean(axis=1))})
MeaEffmean = pd.pivot_table(ReportMeaEff, values='Mean', index=['Pressure', 'Measure type'], columns=['Activity'], aggfunc=np.mean)
MeaEffSD = pd.pivot_table(ReportMeaEff, values='Stdev', index=['Pressure', 'Measure type'], columns=['Activity'], aggfunc=np.mean)
MeaEffmean.to_excel('pupdated\Effectiveness of measure types (Expected value).xlsx')
MeaEffSD.to_excel('pupdated\Effectiveness of measure types (SD).xlsx')

#Export Act-Pressure contibutions
RDAP_sims=DAP_sims.copy()
RDAP_sims.columns=RDAP_sims.columns.map(str)

RDAP_sims1= RDAP_sims.merge(AA,on="Basins")
RDAP_sims3=RDAP_sims.groupby(["Activity","Pressure","GA"])["Basins"].apply(list).reset_index()
RDAP_sims1.drop("Basins",axis=1,inplace=True)

RDAP_sims2=RDAP_sims1.groupby(["Pressure","Activity","GA"]).apply(lambda x:pd.Series(np.average(x.loc[:,"0":"999"],weights=x["Weight"],axis=0))).reset_index()
RDAP_sims4=pd.merge(RDAP_sims3,RDAP_sims2,on=['Pressure','GA','Activity'])
RDAP_sims4['presName']="None"
RDAP_sims4['basName']="None"
RDAP_sims4['actName']="None"
RDAP_sims4.columns=RDAP_sims4.columns.map(str)
for index, row in RDAP_sims4.iterrows():
   RDAP_sims4.at[index,'presName']=pressures[pressures.index==row["Pressure"]]["Pressures"].iloc[0]
   RDAP_sims4.at[index,'actName']=activities[activities.index==row["Activity"]]["Activity"].iloc[0]
   RDAP_sims4.at[index,'basName']=",".join(basin_names.iloc[row["Basins"]]['Basin'].to_list())

ReportMActPresC=pd.DataFrame({"Pressure":RDAP_sims4['presName'],"Activity":RDAP_sims4['actName'],"Basins":RDAP_sims4['basName'],"GA":RDAP_sims4['GA'],\
                              "Mean":RDAP_sims4.loc[:,"0":"999"].mean(axis=1),"Stdev":RDAP_sims4.loc[:,"0":"999"].std(axis=1),\
                              "relStdev":RDAP_sims4.loc[:,"0":"999"].std(axis=1).div(RDAP_sims4.loc[:,"0":"999"].mean(axis=1))})
ReportMActPresC.to_excel('pupdated\Act-Pressure contibution (list form).xlsx')
    
#ActPresCmean = pd.pivot_table(ReportMActPresC, values='Mean', index=['GA', 'Basins'], columns=['Pressure','Activity'], aggfunc=np.mean)
ReportMActPresC=pd.DataFrame({"Pressure":RDAP_sims4['Pressure'],"Activity":RDAP_sims4['Activity'],"GA":RDAP_sims4['GA'],\
                              "Mean":RDAP_sims4.loc[:,"0":"999"].mean(axis=1),"Stdev":RDAP_sims4.loc[:,"0":"999"].std(axis=1),\
                              "relStdev":RDAP_sims4.loc[:,"0":"999"].std(axis=1).div(RDAP_sims4.loc[:,"0":"999"].mean(axis=1))})
ActPresCmean = pd.pivot_table(ReportMActPresC, values='Mean', index=['GA'], columns=['Pressure','Activity'], aggfunc=np.mean)
ActPresCSD = pd.pivot_table(ReportMActPresC, values='Stdev', index=['GA'], columns=['Pressure','Activity'], aggfunc=np.mean)
ActPresCmean.to_excel('pupdated\Act-Pressure contibution (Expected value).xlsx')
ActPresCSD.to_excel('pupdated\Act-Pressure contibution (SD).xlsx')

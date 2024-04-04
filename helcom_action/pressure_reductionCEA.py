# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 12:36:11 2020
@author: E1005457
"""

#Use only literature estimates: MT_probs_L , expert: MT_probs, both: MT_probs_L
#MTL1 only literature
#L combined keeping literarture value if doubles
#probs are used for calculating measure effectiveness
MT_probsP=MT_probs.copy()

#Add AP contributions for measures that affect pressures directly 
#after measures affecting the same pressures through activities
Direct["1"]=1
Drest=np.tile(Direct[["1"]],pop)
Direct.drop("1",axis=1,inplace=True)
Direct=Direct.join(pd.DataFrame(Drest))
Direct.columns=Direct.columns.map(str)
#concat with other AP distributions
DAP_sims.columns=DAP_sims.columns.map(str)
DAP_simsD=pd.concat([DAP_sims,Direct],axis=0)

#Merge measure type effects to actual measures->actual measure effects (AMEff)
AMEff=pd.merge(actMeas,MT_probsP,left_on="MT_ID",right_on="Measure type")
AMEff=AMEff.explode("actBas")
AMEff = AMEff[AMEff["actBas"]!='']
AMEff.reset_index(inplace=True)
AMEff=AMEff.drop("index",axis=1)
AMEff.rename(columns={"level_0":"ID"},inplace=True)

#Same for new measures (AMEff5)
AMEff5=probsNE1.rename(columns={"Measure ID":"ID"})
AMEff5=AMEff5.explode("actBas")
AMEff5 = AMEff5[AMEff5["actBas"]!='']
AMEff5.reset_index(inplace=True)
AMEff5=AMEff5.drop("index",axis=1)
AMEff5=AMEff5.groupby(["ID","Activity","Pressure","State","C_ID","actBas","Description","Name M1","Group"]).mean().reset_index()
AMEff5["MT_ID"]=AMEff5["Group"]+1000 #round(AMEff5["MT_ID"])
AMEff5.drop("Group",inplace=True,axis=1)
AMEff5["In_Activities"]=0;
AMEff5["In_Pressure"]=0;
AMEff5["In_State_components"]=0;
AMEff5["multiplier_OL"]=AMEff5[Extent];
AMEff5["ID"]=AMEff.iloc[len(AMEff)-1]["ID"]+AMEff5["ID"]

#Column names as strings for concat
AMEff5.columns=AMEff5.columns.map(str)
AMEff.columns=AMEff.columns.map(str)

#Are new measures included
if (IncludeNew=="Yes"):
    AMEff=pd.concat([AMEff,AMEff5],sort=False)

#Aactual measure effect multipliers based on basin shares
amef=np.zeros((len(AMEff)))
for i in range(0,len(AMEff)):
    amef[i]=basin_effects.iloc[int(AMEff.iloc[i]["C_ID"])][int(AMEff.iloc[i]["actBas"])]
AMEff['bef']=amef
del amef

#process actual measure effects
AMEff=AMEff.rename(columns={"actBas": "Basins"})
AMEff['Basins'] = AMEff['Basins'].astype(int)
AMEff=AMEff[(AMEff["In_Activities"].astype(int)==0)|(AMEff["In_Activities"].astype(int)==AMEff["Activity"].astype(int))]
AMEff.loc[pd.isna(AMEff["Pressure"]),"Pressure"]=0
AMEff.loc[pd.isna(AMEff["State"]),"State"]=0
AMEff=AMEff[(AMEff["In_Pressure"].astype(int)==0)|((AMEff["In_Pressure"].astype(int))==(AMEff["Pressure"].astype(int)))]
AMEff.reset_index(drop=True,inplace=True)
#delete effects that have no effect or affetc marine litter
AMEff=AMEff[AMEff["0"]<1]
AMEff=AMEff[AMEff["Pressure"]<4500]

#Measures affecting state directly
AMEff3=AMEff[(AMEff["State"]!=0)]
AMEff3.reset_index(inplace=True,drop=True)
AMEff3.columns=AMEff3.columns.map(str)

#Expected values and "confidence intervals" for state improvements (increase in the probability to...) EffSimSDG
EffSimS=np.zeros((len(AMEff3),3))
for i in range(0,len(AMEff3)):
    EffSimS[i,0]=np.mean(np.random.choice(values,pop,p=AMEff3.loc[i,"0":"99"])*0.01*AMEff3.iloc[i]["bef"]*AMEff3.iloc[i]["multiplier_OL"])
    EffSimS[i,1]=np.percentile(np.random.choice(values,pop,p=AMEff3.loc[i,"0":"99"])*0.01*AMEff3.iloc[i]["bef"]*AMEff3.iloc[i]["multiplier_OL"],10)
    EffSimS[i,2]=np.percentile(np.random.choice(values,pop,p=AMEff3.loc[i,"0":"99"])*0.01*AMEff3.iloc[i]["bef"]*AMEff3.iloc[i]["multiplier_OL"],90)
EffSimsSD=pd.DataFrame({"State":AMEff3["State"], "Basin":AMEff3["Basins"],"Expected":EffSimS[:,0],"10percentile":EffSimS[:,1], "90percentile":EffSimS[:,2]})
abu=pd.DataFrame({"State":AMEff3["State"], "Basin":AMEff3["Basins"],"MT_ID":AMEff3["MT_ID"]}).groupby(["Basin","State"]).mean().reset_index()
EffSimSD=np.zeros((len(abu),3))
for index, row in abu.iterrows(): 
    rivit=EffSimsSD[((EffSimsSD["State"]==row["State"])&(EffSimsSD["Basin"]==row["Basin"]))]
    EffSimSD[index,0]=caus_impact(int(len(rivit)-1),rivit.iloc[:,2].to_numpy())
    EffSimSD[index,1]=caus_impact(int(len(rivit)-1),rivit.iloc[:,3].to_numpy())
    EffSimSD[index,2]=caus_impact(int(len(rivit)-1),rivit.iloc[:,4].to_numpy())
EffSimSD=pd.DataFrame({"State":abu["State"], "Basin":abu["Basin"],"Expected":EffSimSD[:,0],"10percentile":EffSimSD[:,1], "90percentile":EffSimSD[:,2]})
EffSimSD["State"]=EffSimSD["State"].astype(int)
EffSimSDG=EffSimSD.groupby(["State","Basin"]).mean().reset_index()

#Merge activity pressures on actual measure effects 
#until bef, distribution of measure effects
#after bef simulated activity pressure contributions
DAP_simsD.columns=DAP_simsD.columns.map(str)
AMEffP=pd.merge(AMEff,DAP_simsD,how="left",on=['Activity','Pressure','Basins']) 
AMEffP.drop_duplicates(inplace=True)

#Simulate measure effects taking into account country share of the basin
EffSim=np.zeros((len(AMEffP),pop))
EffSimWO=np.zeros((len(AMEffP),pop))
EffPress=np.zeros((len(AMEffP),pop))
probsEffPress=np.zeros((int(len(AMEffP)),100))
Aprobs=AMEffP.loc[:,"0_x":"99_x"]
Abef=AMEffP["bef"]
Amul=AMEffP["multiplier_OL"]
for i in range(0,len(AMEffP)):
    EffSim[i,0:pop]=np.random.choice(values,pop,p=Aprobs.iloc[i])*0.01*Abef.iloc[i]*Amul.iloc[i] 
EffSimD=pd.DataFrame(EffSim)

#Recursive effects and overlaps start here
#Makes sure that pressure is not reduced more than 100% from one activity per basin
#average impact in Baltic Sea and share of total area where has effect
recEff=AMEffP.loc[:,["Activity","Pressure","Basins","GA"]].drop_duplicates()
allEff=AMEffP.loc[:,["Activity","Pressure","Basins","GA","MT_ID","C_ID"]].join(EffSimD)

#Go through actual measures by basin, activity and pressure
rece=np.zeros((len(recEff),pop))
j=0;
for index, row in recEff.iterrows():
    rivit=allEff[(allEff['Activity']==row['Activity'])&(allEff['Pressure']==row['Pressure'])&(allEff['Basins']==row['Basins'])]
    #Remove same measure types in the same countries and basins, set to maximum
    rivit=rivit.groupby(['Activity','Pressure','Basins','MT_ID','C_ID']).max().reset_index()
    #Which overlaps effect given presure and activity
    overlapit=overlaps[(overlaps['Activity']==row['Activity'])&(overlaps['Pressure']==row['Pressure'])]
    ovf5 = pd.DataFrame(columns = ['Activity','Pressure','Basins','MT_ID','C_ID',"kerroin"])
    #thematic overlaps start here, now code is pretty messy and unnecessarily ccomplex 
    for x in overlapit.iloc[:]["Overlap"].drop_duplicates():
        ovf4 = pd.DataFrame(columns = ['Activity','Pressure','Basins','MT_ID','C_ID',"kerroin"])
        #define the rows for given overlap
        overlapit2=overlapit[overlapit['Overlap']==x]
        #overlapped measure type and multiplier
        kerroin=overlapit2.iloc[0]["Multiplier"]
        ov1=overlapit2.iloc[0]["Overlapped"] 
        #Are there measures of overlapped type?
        if(rivit[rivit["MT_ID"]==ov1].empty==False):
            for y in overlapit2.iloc[:]["Overlapping"]:
                #Are there measures of the overlapping type?
                if(rivit[rivit["MT_ID"]==y].empty==False):
                    #Are there basins and countries where these both are implemented
                    ovf1=rivit[rivit["MT_ID"]==ov1][['Activity','Pressure','Basins','MT_ID','C_ID']]
                    ovf2=rivit[rivit["MT_ID"]==y][["Basins","C_ID"]]                      
                    #merge to find the areas where these overlap
                    ovf3=pd.merge(ovf1, ovf2, on=["Basins","C_ID"], how='inner')
                    ovf3["kerroin"]=1
                    #overlaps for different overlapping measures
                    ovf4=ovf4.append(ovf3)
        #Drop identific rows= two opverlapping multipliers
        ovf4=ovf4.T.drop_duplicates().T
        #set multiplier
        ovf4["kerroin"]=kerroin
        #add to the set of all overlap effects
        ovf5=ovf5.append(ovf4)
        
    if(ovf5.empty==False):
        ovf5=ovf5.groupby(['Activity','Pressure','Basins','MT_ID','C_ID']).prod().reset_index()
    rivit=pd.merge(rivit, ovf5, on=['Activity','Pressure','Basins','MT_ID','C_ID'], how='left')
    rivit["kerroin"]=rivit["kerroin"].fillna(1)
    rivit.iloc[:,6:6+pop]=rivit.iloc[:,6:6+pop].multiply(rivit.loc[:]["kerroin"], axis="index")
    #Recursive effects, could work faster if the recursive function was defined in the main file
    for i in range(6,6+pop):
        rece[j,i-6]=caus_impact(int(len(rivit)-1),rivit.iloc[:,i].to_numpy())
    j=j+1;

#Effects of development in human activties
#Merge development scenarios to basin,pressure and activity  
DEV_scen=DEV_scen[["Activity",Scenario]]
DEV_scen=DEV_scen.fillna(0)
DEV_ScenF=np.tile(DEV_scen[[Scenario]],pop)
DEV_ScenF1=np.add(DEV_ScenF, 1)
DEV_scenB=pd.DataFrame(DEV_scen["Activity"])
DEV_scenA=DEV_scenB.reset_index(drop=True).join(pd.DataFrame(DEV_ScenF))
DEV_scenE=DEV_scenB.reset_index(drop=True).join(pd.DataFrame(DEV_ScenF1))
DEV_scenW=pd.merge(recEff,DEV_scenE,how='left', on=['Activity'])
# Join activities, pressures and basins used for APs to development scanario 
devEff=pd.merge(DAP_simsD[["Activity","Pressure","Basins","GA"]],DEV_scenA,how='left', on=['Activity'])
DEV_scenW.columns=DEV_scenW.columns.map(str)
rece1=np.multiply(rece, DEV_scenW.loc[:,"0":"999"]) 
#Join pressure reduction multiplied by scenario+1 to activity, pressure and basins used for pressure reductions    
receEff= recEff[["Activity","Pressure","Basins","GA"]].reset_index(drop=True).join(pd.DataFrame(rece1))
#Merge AP contributions with measure effects multiplied by scenario plus 1  
AreceEff=pd.merge(DAP_simsD,receEff,how="left",on=['Activity','Pressure','Basins','GA'])
#Merge prevoius to development sceanrio, now includes AP contributions, pressure reduction multiplied by scenario +1, and scenario
AreceEff=pd.merge(AreceEff,devEff,how="left",on=['Activity','Pressure','Basins','GA'])

#Convert integer column names to string
AreceEff.columns=AreceEff.columns.map(str)
AreceEff=AreceEff.fillna(0)
#x= AP, -=scenario, y=E(1+scenario)
#Calculate impacts taking into account changes in human activities
REffPress=np.multiply(np.subtract(AreceEff.loc[:,"0_y":"999_y"],AreceEff.loc[:,"0":"999"]),AreceEff.loc[:,"0_x":"999_x"])
REffPress=AreceEff[["Activity","Pressure","Basins","GA"]].join(pd.DataFrame(REffPress)) 
#Measures affecting and not effecting pressures directly
REffPress1=REffPress[REffPress["Activity"]!=100]
REffPress2=REffPress[REffPress["Activity"]==100]
REffPress1=REffPress1.groupby(['Basins','Pressure','GA']).sum().reset_index()

#concat diretc impacts and impacts of measures on pressures through activities
for index, row in REffPress2.iterrows():
    rivi=REffPress1[(REffPress1["Basins"]==row["Basins"])&(REffPress1["Pressure"]==row["Pressure"])]
    if(rivi.empty==False):
        REffPress2.loc[(REffPress2["Basins"]==row["Basins"])&(REffPress2["Pressure"]==row["Pressure"]),"0_y":"999_y"]=\
        np.multiply(np.subtract(1,rivi.loc[:,"0_y":"999_y"]),row.loc["0_y":"999_y"])
        #REffPress2.loc[(REffPress2["Basins"]==row["Basins"])&(REffPress2["Pressure"]==row["Pressure"]),"0_x":"999_x"]=row.loc["0_x":"999_x"]
REffPress2=REffPress2.groupby(['Basins','Pressure','GA']).sum().reset_index()
REffPress3=pd.concat([REffPress1,REffPress2])
Reff_on_press_sum=REffPress3.groupby(['Basins','Pressure','GA']).sum().reset_index()

#ReportPresRed=pd.DataFrame({"Pressure":Reff_on_press_sum['Pressure'],"Basins":Reff_on_press_sum['Basins'],\
#                            "Mean":Reff_on_press_sum.loc[:,"0_y":"999_y"].mean(axis=1),"Stdev":Reff_on_press_sum.loc[:,"0_y":"999_y"].std(axis=1)})

#Histogram probability distributions for projected pressure reductions   
probsREffPress=pd.DataFrame(np.zeros((len(Reff_on_press_sum),100)))
for i in range(0,len(Reff_on_press_sum)):    
    probsREffPress.iloc[i,0:100]=plt.hist(Reff_on_press_sum.loc[i,"0_y":"999_y"]*100, bins, density=1)[0]
data={'Pressure':Reff_on_press_sum['Pressure'],'Basins':Reff_on_press_sum['Basins'],'GA':Reff_on_press_sum['GA']}
data=pd.DataFrame(data)
probsREffPress=data.join(probsREffPress)

#pressure reductions passed on to pressure-state, add litter and nutrients
#nutrientResults.columns = nutrientResults.columns.str.replace("_y", "_x")
#litterResults.columns = litterResults.columns.str.replace("_y", "_x")
presRedForState=pd.concat([Reff_on_press_sum,litterResults,nutrientResults],sort=False)
presRedForState.drop(["mean","stdev","Activity"],inplace=True,axis=1)
presRedForState.reset_index(inplace=True,drop=True)

#Add the effects of new measures for litter and nutrients
rivit=presRedForState[presRedForState.Pressure.isin([49,51])]
for j in range(1,18):
    rivit2=rivit[rivit["Basins"]==j]
    presRedForState["0_y"][(presRedForState["Pressure"]==49)&(presRedForState["Basins"]==j)]=caus_impact(1,rivit2.loc[:,"0_y":"999_y"].mean(axis=1,skipna=True).to_numpy())
    presRedForState["1_y"][(presRedForState["Pressure"]==49)&(presRedForState["Basins"]==j)]=caus_impact(1,rivit2.loc[:,"0_y":"999_y"].min(axis=1,skipna=True).to_numpy())
    presRedForState["2_y"][(presRedForState["Pressure"]==49)&(presRedForState["Basins"]==j)]=caus_impact(1,rivit2.loc[:,"0_y":"999_y"].max(axis=1,skipna=True).to_numpy())
rivit=presRedForState[presRedForState.Pressure.isin([50,52])]
for j in range(1,18):
    rivit2=rivit[rivit["Basins"]==j]
    presRedForState["0_y"][(presRedForState["Pressure"]==50)&(presRedForState["Basins"]==j)]=caus_impact(1,rivit2.loc[:,"0_y":"999_y"].mean(axis=1,skipna=True).to_numpy())
    presRedForState["1_y"][(presRedForState["Pressure"]==50)&(presRedForState["Basins"]==j)]=caus_impact(1,rivit2.loc[:,"0_y":"999_y"].min(axis=1,skipna=True).to_numpy())
    presRedForState["2_y"][(presRedForState["Pressure"]==50)&(presRedForState["Basins"]==j)]=caus_impact(1,rivit2.loc[:,"0_y":"999_y"].max(axis=1,skipna=True).to_numpy())

#Basin weights
AA={'Basins':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],"Weight":[23973,10662,921,3485,4620,17600,42014,5876,75133,34528,18787,32831,29877,16829,59349,8159,32225]}    
AA = pd.DataFrame(AA)
AAR= presRedForState.merge(AA,on="Basins")
AAR3=AAR.groupby(["Pressure","GA"])["Basins"].apply(list).reset_index()
AAR.drop("Basins",axis=1,inplace=True)
#Aggregate results from sub basin to assessment unit (GA) for exporting results 
AAR["WSTDEV"]= AAR.loc[:,"0_y":"999_y"].std(axis=1)
AAR2=AAR.groupby(["Pressure","GA"]).apply(lambda x:pd.Series(np.average(x.loc[:,"0_y":"WSTDEV"],weights=x["Weight"],axis=0))).reset_index()
AAR4=pd.merge(AAR3,AAR2,on=['Pressure','GA'])
AAR4['presName']="None"
AAR4['basName']="None"
for index, row in AAR4.iterrows():
   AAR4.at[index,'presName']=pressures[pressures.index==row["Pressure"]]["Pressures"].iloc[0]
   AAR4.at[index,'basName']=",".join(basin_names.iloc[row["Basins"]]['Basin'].to_list())
#AAR4 contains projected pressure reductions for assessment areas weighted by basin area

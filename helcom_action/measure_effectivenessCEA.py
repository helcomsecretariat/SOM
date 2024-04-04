# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:44:05 2019
@author: Liisa Saikkonen
"""

#Get input excel sheets
Background = wb.get_sheet_by_name('Background')
SubPres = wb.get_sheet_by_name('SubPres')
MTEQ = wb1.get_sheet_by_name('MTEQ')
MT_surv_Benthic = wb1.get_sheet_by_name('MT_surv_Benthic')
MT_surv_Birds = wb1.get_sheet_by_name('MT_surv_Birds')
MT_surv_Fish = wb1.get_sheet_by_name('MT_surv_Fish')
MT_surv_HZ = wb1.get_sheet_by_name('MT_surv_HZ')
MT_surv_NIS = wb1.get_sheet_by_name('MT_surv_NIS')
MT_surv_Noise = wb1.get_sheet_by_name('MT_surv_Noise')
MT_surv_Mammals = wb1.get_sheet_by_name('MT_surv_Mammals')
ActMeas = wb.get_sheet_by_name('ActMeas')
ActPres = wb3.get_sheet_by_name('ActPres')
Direct = wb3.get_sheet_by_name('Direct')
CountBas = wb.get_sheet_by_name('CountBas')
LitterResults = wb4.get_sheet_by_name('Sheet1') 
NutrientResults=wb5.get_sheet_by_name('Taul1') 
Literature=wb7.get_sheet_by_name('Taul1') 
Overlaps = wb.get_sheet_by_name('Overlaps')
DEV_scenarios= wb.get_sheet_by_name('DEV_scenarios') 

#For new measures
NewEffectiveness=wb8.get_sheet_by_name('Effectiveness') 
Names=wb8.get_sheet_by_name('Names') 
Typesgroups=wb8.get_sheet_by_name('Typesgroups') 

#READING DATA FROM EXCEL SHEETS FOR OTHER THAN PRESSURE-STATE STARTS HERE
#Read background info
#Read Activities
Ncur=1
activities=readData(Background,Ncur,N_act,2)
#Read pressure names and IDs
Ncur=Ncur+2+N_act
pressures=readData(Background,Ncur,N_pres,2)
pressures.rename(columns={"Pressure":"Pressures"},inplace=True)
#Read state names and IDs
Ncur=Ncur+2+N_pres  
states=readData(Background,Ncur,N_state,2)
#Read basin names and IDs
Ncur=Ncur+2+N_state
basin_names=readData(Background,Ncur,N_basins,2)
#Read country names and IDs
Ncur=Ncur+2+N_basins
countries=readData(Background,Ncur,N_count,2)
#Read measure names and IDs
Ncur=Ncur+3+N_count
mtypes=readData(Background,Ncur,N_mtypes,2)

#Read data on measure type links to activity-pressures
Ncur=1
mlinks=readData(sheet=MTEQ,minRow=Ncur,items=N_MTEQ)

#Read pressure reduction results for nutrient and litter reductions and add them later to results 
Ncur=1
litterResults=readData(sheet=LitterResults)
litterResults=litterResults[litterResults.columns.dropna()]
nutrientResults=readData(sheet=NutrientResults)
literature=readData(sheet=Literature)

#extract measure effectiveness expert weights
#The indexing should be changed to non-numeric
#This should also be added to pressure state
expWeightsM=mlinks.iloc[:,7:7+N_expM]
expWeightsM.fillna(value=pd.np.nan, inplace=True)
expWeightsM.fillna(1,inplace=True)

#Read survey data on measure type effectiveness by topic
msurvBenthic=readData(sheet=MT_surv_Benthic,minRow=Ncur,items=N_expM,ex_col=1)
msurvBirds=readData(sheet=MT_surv_Birds,minRow=Ncur,items=N_expM,ex_col=1)
msurvFish=readData(sheet=MT_surv_Fish,minRow=Ncur,items=N_expM,ex_col=1)
msurvHz=readData(sheet=MT_surv_HZ,minRow=Ncur,items=N_expM,ex_col=1)
msurvNIS=readData(sheet=MT_surv_NIS,minRow=Ncur,items=N_expM,ex_col=1)
msurvNoise=readData(sheet=MT_surv_Noise,minRow=Ncur,items=N_expM,ex_col=1)
msurvMammals=readData(sheet=MT_surv_Mammals,minRow=Ncur,items=N_expM,ex_col=1)
msurv=pd.concat([msurvBenthic, msurvBirds,msurvFish,msurvHz,msurvNIS,msurvNoise,msurvMammals],axis=1, sort=False)

#read activity pressure data
Ncur=1;
aPres=readData(sheet=ActPres)  
Direct=readData(sheet=Direct)
#Read overlap data
Ncur=1;
overlaps=readData(sheet=Overlaps)
#read effectiveness data for new measures
Ncur=1;
NewEffectiveness=readData(sheet=NewEffectiveness)
NewNames=readData(sheet=Names)
Typesgroups=readData(sheet=Typesgroups)

#Read data on measure type links to activity-pressures, Liisa added 2_6
Ncur=1
subPres=readData(SubPres,Ncur,N_SubPres,5)
subPres['State'][subPres['State']=='0;']="1;2;3;4;5;6;7;8;9;21;22;23;24;25;26;27;28;29;30;35;40;41;46;52;53;62;66;68;69;" 
subPres['State']=subPres.State.str.split(';')
subPres=subPres.explode('State')
subPres=subPres[subPres["State"] != ""]
subPres['State'] = subPres['State'].astype(int)
subPres=subPres.reset_index()
subPres=subPres.drop(['ID'], axis=1)

#read data on actual measures, explode based on country
actMeas=readData(ActMeas,Ncur,N_actMeas)

#explode based on included activties
actMeas['In_Activities']=actMeas.In_Activities.str.split(';')
actMeas=actMeas.explode('In_Activities')
actMeas = actMeas[actMeas["In_Activities"] != ""]
actMeas.reset_index(drop=True,inplace=True)
#explode based on included state componenet
actMeas['In_Pressure']=actMeas.In_Pressure.str.split(';')
actMeas=actMeas.explode('In_Pressure')
actMeas = actMeas[actMeas["In_Pressure"] != ""]
actMeas.reset_index(drop=True,inplace=True)

#Read basins data
Ncur=1;
basins=readData(sheet=CountBas,minRow=Ncur,items=N_count,columns=19)
basin_effects=readData(sheet=CountBas,minRow=Ncur+N_count+3,items=N_count)
actMeas=areas(actMeas,basins).reset_index(drop=True)

#New measures
NewEffectiveness.loc[(pd.isna(NewEffectiveness.MT_ID)),"MT_ID"]=NewEffectiveness.loc[(pd.isna(NewEffectiveness.MT_ID)),"Measure ID"]+1000
NewEffectiveness["MT_ID"]=NewEffectiveness["MT_ID"].astype(str)
NewEffectiveness["MT_ID"]=NewEffectiveness["MT_ID"]+str(";")
NewEffectiveness['MT_ID']=NewEffectiveness.MT_ID.str.split(';')
NewEffectiveness=NewEffectiveness.explode('MT_ID')
NewEffectiveness = NewEffectiveness[NewEffectiveness["MT_ID"] != ""]
NewEffectiveness.reset_index(drop=True,inplace=True)

#Read DEV scenario data
Ncur=2;
DEV_scen=readData(sheet=DEV_scenarios,minRow=Ncur,items=N_DEV)

#Drop empty rows where there are no values for activity pressure contributions
aPres=aPres.dropna(subset=aPres.columns[4:22], how='all')
aPres.reset_index(drop=True, inplace=True)
#READING DATA FROM EXCEL SHEETS FOR OTHER THAN PRESSURE-STATE STARTS HERE

#DISTRIBUTIONS AND SIMULATIONS (=drawing values from distributions)  START HERE
#Number of randomly drawn values per expert (sims) for pooled distributions
#And the number of randomly drawn values to simulate the results (pop)
sims=1000
pop=1000
#Parameter of PERT beta distribution
lam=4
#Discrete distribution bins and values
#These could be changed for different data formats and "more" discrete distributions
#100 histogram bins with intervals of 1
bins=np.arange(0,101,1)
#bin values: 0.5%....99.5%
values=np.arange(1,101,1)-0.5

#DISTRIBUTIONS AND SIMULATIONS FOR ACTIVITY PRESSURE CONTRIBUTIONS START HERE
#Create distributions for activity pressure contributions and draw values

#Empty array for the 100 probabilities of activity pressure contributions: 0-1%, 1-2%...99-100%
probsAP=np.zeros((int(len(aPres)),100))
#Expert specific distributions
AP_dists=np.zeros((int(len(aPres)),N_expAP,sims))
#Array for pooled values from experts
AP_pooled=np.zeros((int(len(aPres)),sims*N_expAP))
#size of pooled values 
poolLen=len(AP_pooled[0])
#Array for simulated values from pooled distribution
AP_sims=np.zeros((len(aPres),pop))

#Extract maximum, minimum and most likely for activity pressure contributions
AP_ml=aPres[[col for col in aPres if col.startswith('Ml')]]
AP_min=aPres[[col for col in aPres if col.startswith('Min')]]
AP_max=aPres[[col for col in aPres if col.startswith('Max')]]

for i in range(0,len(aPres)):
    for j in range(0,N_expAP):
        #Define minimum, maximum and most likely activity pressure contributions to define expert specific contributions
        mini=np.nanmin([float(AP_min.iloc[i][j]),float(AP_ml.iloc[i][j]),float(AP_max.iloc[i][j])])
        maxi=np.nanmax([float(AP_min.iloc[i][j]),float(AP_ml.iloc[i][j]),float(AP_max.iloc[i][j])])
        ml=np.nanmedian([float(AP_min.iloc[i][j]),float(AP_ml.iloc[i][j]),float(AP_max.iloc[i][j])])
        AP_min.iloc[i][j]=mini;AP_max.iloc[i][j]=maxi;AP_ml.iloc[i][j]=ml;
        #Create PERT distributions based on ml, min and max values
        if(mt.isnan(AP_min.iloc[i][j]) & mt.isnan(AP_ml.iloc[i][j]) & mt.isnan(AP_max.iloc[i][j])):
            AP_dists[i,j,:]=np.nan
        if((AP_min.iloc[i][j]<AP_ml.iloc[i][j])|(AP_max.iloc[i][j]>AP_ml.iloc[i][j])):
            AP_dists[i,j,:]=(np.random.beta(\
                    1+lam*(AP_ml.iloc[i][j]-AP_min.iloc[i][j])/(AP_max.iloc[i][j]-AP_min.iloc[i][j]),\
                    1+lam*(AP_max.iloc[i][j]-AP_ml.iloc[i][j])/(AP_max.iloc[i][j]-AP_min.iloc[i][j]),sims)\
                    *(AP_max.iloc[i][j]*0.01-AP_min.iloc[i][j]*0.01)+AP_min.iloc[i][j]*0.01)*100
            """
            Any other distribution type (uniform, triangular...) could be used
            AP_dists[i,j,:]=np.random.triangular(AP_min.iloc[i][j],AP_ml.iloc[i][j],AP_max.iloc[i][j],sims)  
            """
        else:
            AP_dists[i,j,:]=AP_ml.iloc[i][j] 
    #Concatenate simulated values from expert base distributions to the pooled distributions
    AP_pooled[i,0:poolLen]=np.concatenate(AP_dists[i,0:N_expAP,:])
    #Create probability distribution of pooled distributions using histograms
    #AP includes the probability for each interval 0-1%,...99-100%
    probsAP[i,:]=plt.hist(AP_pooled[i,0:poolLen], bins, density=1)[0]
    #Draw pop values from the distributions
    AP_sims[i,0:pop]=np.random.choice(values,pop,p=probsAP[i,:])*0.01
    
#Add IDs of pressures, activities, basins and geographic assessment areas (GAs) to the simulated activity pressure contributions     
DAP_sims=pd.DataFrame(AP_sims)
DAP_sims["Pressure"]=aPres['Pressure']
DAP_sims["Activity"]=aPres['Activity']
DAP_sims["Basins"]=aPres['Basins']
DAP_sims["GA"]=aPres['GA']

#The same for activity pressure probabilities
DAP_probs=pd.DataFrame(probsAP)
data={"Pressure":aPres['Pressure'],"Activity":aPres['Activity'],"Basins":aPres['Basins'], "GA":aPres['GA']}
DAP_probs=pd.DataFrame(data).join(DAP_probs)

#It is not guaranteed that simulated activity pressure contributions (1 to pop) sum to 1 by pressure and assessment area GA
#Set the sum of contributions to 1 in DAP_sims by dividing the values by their sum
Nsums=np.zeros((len(DAP_sims),pop))
for i in range(0,len(DAP_sims)):
    DFsumma=DAP_sims.iloc[:,0:pop][(DAP_sims["Pressure"]==DAP_sims.iloc[i]["Pressure"]) &\
                         (DAP_sims["Basins"]==DAP_sims.iloc[i]["Basins"])]
    Nsums[i,:]=DFsumma.sum()
DAP_sims.iloc[:,0:pop]=DAP_sims.iloc[:,0:pop].div(Nsums)

#GAP_Sims are simulated activity pressure contributions by GA
#These were used for data validation, they are ont used for anything else
GAP_sims=DAP_sims.copy()
GAP_sims.drop("Basins",axis=1,inplace=True)

#Explode DAP_sims by basins to create activity pressure contribution for each basin
#The drawn values (1 to pop) are the same for each basin within the same assessment unit GA
DAP_sims['Basins'][DAP_sims['Basins']=='0;']="1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;" 
DAP_sims['Basins']=DAP_sims.Basins.str.split(';')
DAP_sims=DAP_sims.explode('Basins')
DAP_sims=DAP_sims[DAP_sims["Basins"] != ""]
DAP_sims['Basins'] = DAP_sims['Basins'].astype(int)
DAP_sims=DAP_sims.reset_index()
DAP_sims=DAP_sims.drop(['index'], axis=1)

#Direct includes activity pressure contributions that affect pressures directly, for pressures that also have measures affecting them through activities
#Applies for new measures
#Explode Direct by basins
Direct['Basins'][Direct['Basins']=='0;']="1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;" 
Direct['Basins']=Direct.Basins.str.split(';')
Direct=Direct.explode('Basins')
Direct=Direct[Direct["Basins"] != ""]
Direct['Basins'] = Direct['Basins'].astype(int)
Direct=Direct.reset_index()
Direct=Direct.drop(['index'], axis=1)

#DISTRIBUTIONS AND SIMULATIONS FOR ACTIVITY PRESSURE CONTRIBUTIONS END HERE
#DAP_sims contains the pop drawn random values for each activity pressure contribution for each basin of all assessment areas (GA)
#DAP_probs contains the probabilities for each value interval 0-1%,1-2%,....99-100% for each activity pressure contribution for all assessment areas (GA)


#DISTRIBUTIONS AND SIMULATIONS FOR MEASURE TYPE EFFECTIVENESS START HERE
#Ordered list of measure types used in measure type effectiveness surve                     
mtlinksT=list(msurv.columns);
mtlinksT[:] = (value for value in mtlinksT if value != 'ME')
mtlinksT=mtlinksT[::2]

#set column names to measure type effectiveness data
nimet=[];
mtlinksA=[];
mtlinksP=[];
mtlinksMTEQ=[];
mtlinksDP=[]; #direct to pressures
mtlinksDS=[]; #direct to state
for i in range(0,N_MTEQ):
    for j in range(0,mlinks['AMT'][i]):
        nimet.extend(["MTE"+str(mlinks['Activity'][i])+"_"+str(mlinks['Pressure'][i])+"_"+str(i)+"_"+str(j),
                      "MTU"+str(mlinks['Activity'][i])+"_"+str(mlinks['Pressure'][i])+"_"+str(i)+"_"+str(j)])
    nimet.extend(["ME"+str(mlinks['Activity'][i])+"_"+str(mlinks['Pressure'][i])+"_"+str(i)])
    #append values
    mtlinksA += mlinks['AMT'][i] * [mlinks['Activity'][i]]
    mtlinksP += mlinks['AMT'][i] * [mlinks['Pressure'][i]]
    mtlinksMTEQ += mlinks['AMT'][i] * [i]
    mtlinksDP+= mlinks['AMT'][i] * [mlinks['Direct_to_pressure'][i]] 
    mtlinksDS+= mlinks['AMT'][i] * [mlinks['Direct_to_state'][i]]  
#nimet=pd.Series(nimet).str.cat(types)
msurv.columns=nimet
del nimet


#Extract measure effects, uncertainty and maximum effects to different data frames
measureT_effects=msurv[[col for col in msurv if col.startswith('MTE')]]
measureT_uncer=msurv[[col for col in msurv if col.startswith('MTU')]]
measureT_uncer=100-measureT_uncer
measureTMax=msurv[[col for col in msurv if col.startswith('ME')]]
#replace nones by nans
measureTMax.fillna(value=pd.np.nan, inplace=True)

#Define the maximum surveyed effectiveness for each measure effectiveness grid by expert
# the index to maximum value is also extracted,it is not used for anything
j=0
#Data frame for relative effectiveness (from grid-question) of the most effective measure
MEFFT=[]
#Data frame for relative uncertainty (from grid-question) of the most effective measure
MEFUT=[]
for i in range(0,N_MTEQ):
    MEFF=[]
    MEFU=[]
    for k in range (0,N_expM):
           measureT_effects.iloc[k,j:j+mlinks.iloc[i]["AMT"]] 
           a=np.nanmax(measureT_effects.iloc[k,j:j+mlinks.iloc[i]["AMT"]])
           MEFF.append(a) 
           if(mt.isnan(a)==False):        
               b=np.nanargmax(measureT_effects.iloc[k,j:j+mlinks.iloc[i]["AMT"]])
               MEFU.append(measureT_uncer.iloc[k,j+b])
           else:
               MEFU.append(float('NaN'))
    j=j+mlinks.iloc[i]["AMT"]
    MEFFT.append(MEFF)
    MEFUT.append(MEFU)  
#All-NaN axis encountered                 
pMEFFT=pd.DataFrame(list(map(list, zip(*MEFFT))))
pMEFFT.columns=[f'MM{i}' for i in range(1,N_MTEQ+1)]
pMEFUT=pd.DataFrame(list(map(list, zip(*MEFUT))))
pMEFUT.columns=[f'MM{i}' for i in range(1,N_MTEQ+1)]

#Define maximum, minimum and most likely values to create expert distributions
measureT_min=np.zeros((N_expM,sum(mlinks['AMT'])))
measureT_max=np.zeros((N_expM,sum(mlinks['AMT'])))
measureT_ml=np.zeros((N_expM,sum(mlinks['AMT'])))
measureT_weight=np.zeros((N_expM,sum(mlinks['AMT'])))
probsMT=np.zeros((sum(mlinks['AMT']),100))
#Create distributions for measure type effects
MT_dists=np.zeros((sum(mlinks['AMT']),N_expM,sims))
GMT_sims=np.zeros((sum(mlinks['AMT']),pop))
del j

for i in range(0,sum(mlinks['AMT'])):    
    for j in range(0,N_expM):
        if (mt.isnan(measureT_effects.iloc[j,i])|mt.isnan(measureT_uncer.iloc[j,i])):
            measureT_effects.iloc[j,i]=np.nan
            measureT_uncer.iloc[j,i]=np.nan
        #multiplier to scale measure effectiveness based on the effectiveness of the most effective measure in each grid question 
        kerroin=measureTMax.iloc[j,mtlinksMTEQ[i]]/pMEFFT.iloc[j,mtlinksMTEQ[i]]        
        measureT_ml[j,i]=measureT_effects.iloc[j,i]*kerroin
        measureT_weight[j,i]=expWeightsM.iloc[mtlinksMTEQ[i],j]
        if (measureT_effects.iloc[j,i]< measureT_uncer.iloc[j,i]/2):
            measureT_min[j,i]=0
            measureT_max[j,i]=measureT_uncer.iloc[j,i]*kerroin
        elif ((measureT_effects.iloc[j,i]+measureT_uncer.iloc[j,i]/2) > 100):
            measureT_max[j,i]=100*kerroin
            measureT_min[j,i]=(100-measureT_uncer.iloc[j,i])*kerroin
        else:
            measureT_max[j,i]=(measureT_effects.iloc[j,i]+measureT_uncer.iloc[j,i]/2)*kerroin
            measureT_min[j,i]=(measureT_effects.iloc[j,i]-measureT_uncer.iloc[j,i]/2)*kerroin
        if (measureT_max[j,i]>100):
            measureT_max[j,i]=100

        if(mt.isnan(measureT_ml[j,i])==False):      
            #Calibrate lambda according to what was discussed in SOM meeting
            lamp=deflambda(measureT_min[j,i],measureT_max[j,i],measureT_ml[j,i],5)
            if((measureT_min[j,i]< measureT_ml[j,i])|(measureT_max[j,i]>measureT_ml[j,i])):                
                MT_dists[i,j,:]=(np.random.beta(\
                    1+lamp*(measureT_ml[j][i]-measureT_min[j][i])/(measureT_max[j][i]-measureT_min[j][i]),\
                    1+lamp*(measureT_max[j][i]-measureT_ml[j][i])/(measureT_max[j][i]-measureT_min[j][i]),sims)\
                    *(measureT_max[j][i]*0.01-measureT_min[j][i]*0.01)+measureT_min[j][i]*0.01)*100
                """
                MT_dists[i,j,:]=np.random.triangular(measureT_min[j,i],measureT_ml[j,i],measureT_max[j,i],sims) 
                """
            else:
                MT_dists[i,j,:]=measureT_ml[j,i] 
        else:
            MT_dists[i,j,:]=[np.nan]*sims
            measureT_weight[j,i]=np.nan
    #Pool distributions using expert weights
    MT_pooled=np.concatenate(np.repeat(MT_dists[i,0:N_expM,:],expWeightsM.iloc[mtlinksMTEQ[i],:],axis=0))
    #Probability distributions for pooled values
    probsMT[i,:]=plt.hist(MT_pooled, bins, density=1)[0]
    #Simulated values from pooled probability distributions
    GMT_sims[i,0:pop]=np.random.choice(values,pop,p=probsMT[i,:])*0.01

#Include literature estimates
literature2=literature[["Measure type","Activity","Pressure","Model"]]
literature2=literature2.groupby(['Measure type','Activity','Pressure']).count().reset_index()
pooledMTL=np.empty((len(literature2),max(literature2["Model"])*sims))
pooledMTL[:]=np.nan
probsMTL=np.zeros((len(literature2),100))
simsMTL=np.zeros((len(literature2),pop))
for index1, row in literature2.iterrows():
    literature3=literature[(literature["Measure type"]==row["Measure type"])&(literature["Activity"]==row["Activity"])&(literature["Pressure"]==row["Pressure"])]
    index3=0
    for index2, rivi in literature3.iterrows():
        mini=np.nanmin([float(rivi["Minimum "]),float(rivi["Most Likely"]),float(rivi["Maximum"])])
        maxi=np.nanmax([float(rivi["Minimum "]),float(rivi["Most Likely"]),float(rivi["Maximum"])])
        ml=np.nanmedian([float(rivi["Minimum "]),float(rivi["Most Likely"]),float(rivi["Maximum"])])
        if ((ml>mini)|(ml<maxi)):
            pooledMTL[index1,index3*sims:(index3+1)*sims]=(np.random.beta(1+lam*(ml-mini)/(maxi-mini),\
                     1+lam*(maxi-ml)/(maxi-mini),sims)*(maxi*0.01-mini*0.01)+mini*0.01)*100
        else:
            pooledMTL[index1,index3*sims:(index3+1)*sims]=ml
        index3=index3+1
    probsMTL[index1,:]=plt.hist(pooledMTL[index1,0:], bins, density=1)[0]
    simsMTL[index1,0:pop]=np.random.choice(values,pop,p=probsMTL[index1,:])*0.01
simsMTL1=literature2.join(pd.DataFrame(simsMTL)) 
literature2.drop(["Model"],inplace=True,axis=1) 
probsMTL1=literature2.join(pd.DataFrame(probsMTL))

#merge GMT_sims to DAP_sims to simulate effects of measures on total pressure reductions by GA
data={'Activity':mtlinksA,'Pressure':mtlinksP,'Measure type':mtlinksT, "State":mtlinksDS}
#Create dataframes 
MTlinks=pd.DataFrame(data)
#Probability distributions for measure types
MT_probs=pd.DataFrame(probsMT)
#simulated values for measure types
GMT_sims=pd.DataFrame(GMT_sims)
#This is used later for calculating pressure reductions
MT_probs=MTlinks.join(MT_probs).reset_index(drop=True)  

#This is used later for creating measure effectiveness results
GMT_sims=MTlinks.join(GMT_sims).reset_index(drop=True)

#New measures have measures affecting state directly, therefore include the column also for existing measures 
GMT_sims["State"]=GMT_sims.State.str.split(';')
GMT_sims=GMT_sims.explode("State")
GMT_sims=GMT_sims[GMT_sims["State"]!='']
GMT_sims.reset_index(inplace=True,drop=True)

MT_probs["State"]=MT_probs.State.str.split(';')
MT_probs=MT_probs.explode("State")
MT_probs=MT_probs[MT_probs["State"]!='']
MT_probs.reset_index(inplace=True,drop=True)
MT_probs.columns=MT_probs.columns.map(str)
GMT_sims.columns=GMT_sims.columns.map(str)
MT_probs.loc[pd.isna(MT_probs["State"]),"State"]=0
GMT_sims.loc[pd.isna(GMT_sims["State"]),"State"]=0

#DISTRIBUTIONS AND SIMULATIONS FOR MEASURE EFFECTIVENESS FOR Existing measures END HERE
#MT_probs contains probabilities of measure effectiveness intervals 0-1%, 1-2%...99-100% (to reduce pressures from activities) 
#GMT_sims contains pop randomly drawn measure effectiveness values from measure effectiveness distributions 
#probsMTL1 and simsMTL1 contain the same for literature estimates

#DISTRIBUTIONS AND SIMULATIONS FOR NEW MEASURES START HERE
#Set State to 0 if does not affect state or Pressure to 0 if does not affect Pressure
NewEffectiveness.loc[pd.isna(NewEffectiveness["Pressure"]),"Pressure"]=0
NewEffectiveness.loc[pd.isna(NewEffectiveness["State"]),"State"]=0
NewEffectiveness["MT_ID"]=NewEffectiveness["MT_ID"].astype(int)
NewEffectiveness.loc[pd.isna(NewEffectiveness["LOWEX"]),"LOWEX"]=1
NewEffectiveness.loc[pd.isna(NewEffectiveness["MAXEX"]),"MAXEX"]=1
NewEffectiveness2=NewEffectiveness[["Measure ID","MT_ID","Activity","Pressure","State","B_ID","C_ID","LOWEX","MEDEX","MAXEX","Group"]]
pooledNE=np.empty((len(NewEffectiveness2),sims))
pooledNE[:]=np.nan
probsNE=np.zeros((len(NewEffectiveness2),100))
simsNE=np.zeros((len(NewEffectiveness2),pop))

for index1, rivi in NewEffectiveness.iterrows():
    mini=np.nanmin([float(rivi["MINEF"]),float(rivi["MLEF"]),float(rivi["MAXEF"])])
    maxi=np.nanmax([float(rivi["MINEF"]),float(rivi["MLEF"]),float(rivi["MAXEF"])])
    ml=np.nanmedian([float(rivi["MINEF"]),float(rivi["MLEF"]),float(rivi["MAXEF"])])
    if ((ml>mini)|(ml<maxi)):
        pooledNE[index1,0:sims]=(np.random.beta(1+lam*(ml-mini)/(maxi-mini),\
                 1+lam*(maxi-ml)/(maxi-mini),sims)*(maxi*0.01-mini*0.01)+mini*0.01)*100
        probsNE[index1,:]=plt.hist(pooledNE[index1,:], bins, density=1)[0]
        simsNE[index1,0:pop]=np.random.choice(values,pop,p=probsNE[index1,:])*0.01
    elif(np.isnan(ml)==False):
        #pooledNE[index1,0:sims]=ml
        if(ml!=0):
            maxi=ml*1.6
            if(maxi>100):
                maxi=100
            mini=ml*0.4
            pooledNE[index1,0:sims]=(np.random.beta(1+lam*(ml-mini)/(maxi-mini),\
                    1+lam*(maxi-ml)/(maxi-mini),sims)*(maxi*0.01-mini*0.01)+mini*0.01)*100
            probsNE[index1,:]=plt.hist(pooledNE[index1,:], bins, density=1)[0]
            simsNE[index1,0:pop]=np.random.choice(values,pop,p=probsNE[index1,:])*0.01
        else:
            pooledNE[index1,0:sims]=0
            probsNE[index1,:]=plt.hist(pooledNE[index1,:], bins, density=1)[0]
            simsNE[index1,0:pop]=np.random.choice(values,pop,p=probsNE[index1,:])*0.01            
    else:
        if(int(rivi["MT_ID"])>1000):
            pooledNE[index1,0:sims]=0
            probsNE[index1,:]=plt.hist(pooledNE[index1,:], bins, density=1)[0]
            simsNE[index1,0:pop]=np.random.choice(values,pop,p=probsNE[index1,:])*0.01
        else:
            if(MT_probs.loc[((MT_probs["Measure type"]==rivi["MT_ID"])\
            &(MT_probs["Activity"]==rivi["Activity"])&(MT_probs["Pressure"]==rivi["Pressure"])&(MT_probs["State"]==rivi["State"])),"0":"99"].empty!=True):
                probsNE[index1,:]=MT_probs.loc[((MT_probs["Measure type"]==rivi["MT_ID"])\
                       &(MT_probs["Activity"]==rivi["Activity"])&(MT_probs["Pressure"]==rivi["Pressure"])&(MT_probs["State"]==rivi["State"])),"0":"99"]
                simsNE[index1,:]=GMT_sims.loc[((GMT_sims["Measure type"]==rivi["MT_ID"])\
                      &(GMT_sims["Activity"]==rivi["Activity"])&(GMT_sims["Pressure"]==rivi["Pressure"])&(GMT_sims["State"]==rivi["State"])),"0":"999"]       
            else:
                pooledNE[index1,0:sims]=0
                probsNE[index1,:]=plt.hist(pooledNE[index1,0:], bins, density=1)[0]
                simsNE[index1,0:pop]=np.random.choice(values,pop,p=probsNE[index1,:])*0.01

NewEffectiveness3=pd.merge(NewEffectiveness2,NewNames,on=['Measure ID'])
simsNE2=NewEffectiveness3.join(pd.DataFrame(simsNE))
probsNE2=NewEffectiveness3.join(pd.DataFrame(probsNE))

#Define actual basins for actual measures taking into account 
#that measures can only affet certain basins and countries border different basins
simsNE2.columns=simsNE2.columns.map(str)
#Explode by country and basin for new measures
probsNE1=areas(probsNE2,basins).reset_index(drop=True)
simsNE1=areas(simsNE2,basins).reset_index(drop=True)
#probsNE1=simsNE1.groupby([]).mean().reset_index()

GMT_sims2=simsNE2[["Activity","Pressure","Measure ID","State"]]
GMT_sims2=pd.concat([GMT_sims2,simsNE2.loc[:,"0":"999"]],axis=1)
GMT_sims2.drop_duplicates(inplace=True)
GMT_sims2.reset_index(drop=True,inplace=True)
GMT_sims2=GMT_sims2[GMT_sims2["Pressure"]<4500]


#DISTRIBUTIONS AND SIMULATIONS FOR NEW MEASURES END HERE
#For new measures probsNE2 are probabilities for measure effectiveness to reduce pressures (0-100%), at 1% intervals 
#probsNE1 is the same data for all country shares of basins that they are applied in
#GMT_sims2 contain pop randomly drawn measure effectiveness values from measure effectiveness distributions 
#and simsNE1 measure randomly drawn effectiveness values exploded by country shares of basins 

#Merge literature estimates with expert data starts here, not needed for anything else
GMT_sims1 = GMT_sims.sort_values(by=['Measure type','Activity','Pressure'])
simsMTL1 = simsMTL1.sort_values(by=['Measure type','Activity','Pressure'])
GMT_sims1["Model"]=0
GMT_sims_L=pd.concat([GMT_sims1,simsMTL1]).reset_index(drop=True)
GMT_sims_L.drop_duplicates(subset=['Measure type','Activity','Pressure'],keep='last',inplace=True)
GMT_sims_LB=pd.concat([GMT_sims1,simsMTL1]).reset_index(drop=True)
MT_probs1 = MT_probs.sort_values(by=['Measure type','Activity','Pressure'])
probsMTL1 = probsMTL1.sort_values(by=['Measure type','Activity','Pressure'])
MT_probs_L=pd.concat([MT_probs1,probsMTL1]).reset_index(drop=True)
MT_probs_L.drop_duplicates(subset=['Measure type','Activity','Pressure'],keep='last',inplace=True)

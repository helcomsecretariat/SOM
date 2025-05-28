# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 20:18:43 2021

@author: E1005457
"""
import pandas as pd

#Functions called by other files

#Reads data from input data
def readData(sheet,minRow=1,items=0,columns=0,ex_col=0):
    if(columns==0):
        columns=sheet.max_column
    if(items==0):
        items=sheet.max_row-1    
    tieto=[]
    for value in sheet.iter_rows(minRow,minRow+items,ex_col+1,columns,True):
        tieto.append(value)  
    tieto=pd.DataFrame(tieto[1:minRow+items],columns=tieto[0])
    if 'ID' in tieto:
        tieto=tieto.set_index(["ID"])
    return tieto

#Chain/recursive impact for measures that target pressure from same activity in same area
def caus_impact(n,impacts):
    if (n==0):
        return impacts[0]
    else:
        imp=caus_impact(n-1,impacts)
        return (1-imp)*impacts[n]+imp

#split (measure) rows by country and basin
def areas(measures,basins):
    #actMeas=actMeas.drop(['In_Pressure', 'Name M1', 'Note'], axis=1)
    measures.loc[measures["C_ID"]=="0;","C_ID"]="1;2;3;4;5;6;7;8;9"
    measures['C_ID']=measures.C_ID.str.split(';')
    measures=measures.explode('C_ID')
    measures = measures[measures["C_ID"] != ""]
    measures.reset_index(inplace=True)
    
    act_bas=[]
    for i in range(0,len(measures)):
        if (int(measures.iloc[i]["C_ID"])==0):
            bl={'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17'}
        else: 
            bl=set(basins.iloc[int(measures.iloc[i]["C_ID"])-1]["B_ID"].split(';')) 
        if (int(measures.iloc[i]["B_ID"].split(';')[0]) != 0):
            al=set(measures.iloc[i]["B_ID"].split(';'))           
            if(len(al&bl)>0):
                act_bas.append(list(al&bl))
            else:
                act_bas.append(666) 
        else:
            act_bas.append(list(bl)) 
    #Create all measure basin combinations from actual meaasure data
    actMeas2=pd.DataFrame({"actBas":act_bas})   
    measures["actBas"]=actMeas2["actBas"].str.join(';')   
    measures=measures.drop("B_ID",axis=1)
    measures = measures[measures["actBas"].notnull()]
    measures['actBas']=measures.actBas.str.split(';')
    return measures

#Redefine lambda for measure effectiveness distributions
# Max lambda 10.
def deflambda(minim,maxim,mostl,diff):
    if(abs(((minim+maxim+6*mostl)/8)-mostl)<diff):
        #print((minim+maxim+6*mostl)/8)
        lambd=4
    elif(minim+maxim-2*(mostl+diff)>0):
        lambd=(minim+maxim-2*(mostl+diff))/(3/2*diff)
    else: 
        lambd=(minim+maxim-2*(mostl-diff))/(-diff*3/2)
    if(lambd>=10):
        lambd=10;
    return lambd;

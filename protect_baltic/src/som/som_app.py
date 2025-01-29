"""
Copyright (c) 2024 Baltic Marine Environment Protection Commission
Copyright (c) 2022 Antti-Jussi Kieloaho (Natural Resources Institute Finland)

LICENSE available under 
local: 'SOM/protect_baltic/LICENSE'
url: 'https://github.com/helcomsecretariat/SOM/blob/main/protect_baltic/LICENCE'
"""

from copy import deepcopy

import numpy as np
import pandas as pd

from som.som_tools import *
from utilities import Timer, exception_traceback

def process_input_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads in data and processes to usable form.

    Arguments:
        config (dict): dictionary loaded from configuration file

    Returns:
        measure_survey_df (DataFrame): contains the measure survey data of expert panels
        pressure_survey_df (DataFrame): contains the pressure survey data of expert panels
        data (dict): 
            'measure' (DataFrame):
                'ID': unique measure identifier
                'measure': name / description column
            'activity' (DataFrame):
                'ID': unique activity identifier
                'activity': name / description column
            'pressure' (DataFrame):
                'ID': unique pressure identifier
                'pressure': name / description column
            'state' (DataFrame):
                'ID': unique state identifier
                'state': name / description column
            'measure_effects' (DataFrame): measure effects on activities / pressures / states
            'pressure_contributions' (DataFrame): pressure contributions to states
            'thresholds' (DataFrame): changes in states required to meet specific target thresholds
            'domain'
            'cases'
            'activity_contributions'
            'overlaps'
            'development_scenarios'
    """
    #
    # measure survey data
    #

    file_name = config['input_data']['measure_effect_input']
    measure_effects = process_measure_survey_data(file_name)

    #
    # pressure survey data (combined pressure contributions and GES threshold)
    #

    file_name = config['input_data']['pressure_state_input']
    pressure_contributions, thresholds = process_pressure_survey_data(file_name)

    #
    # measure / pressure / activity / state links
    #

    # read core object descriptions
    # i.e. ids for measures, activities, pressures and states
    file_name = config['input_data']['general_input']
    id_sheets = config['input_data']['general_input_sheets']['ID']
    data = read_ids(file_name=file_name, id_sheets=id_sheets)

    #
    # read case input
    #

    file_name = config['input_data']['general_input']
    sheet_name = config['input_data']['general_input_sheets']['case']
    cases = read_cases(file_name=file_name, sheet_name=sheet_name)

    #
    # read activity contribution data
    #

    file_name = config['input_data']['general_input']
    sheet_name = config['input_data']['general_input_sheets']['postprocess']
    activity_contributions = read_activity_contributions(file_name=file_name, sheet_name=sheet_name)

    #
    # read overlap data
    #

    file_name = config['input_data']['general_input']
    sheet_name = config['input_data']['general_input_sheets']['overlaps']
    overlaps = read_overlaps(file_name=file_name, sheet_name=sheet_name)

    #
    # read activity development scenario data
    #

    file_name = config['input_data']['general_input']
    sheet_name = config['input_data']['general_input_sheets']['development_scenarios']
    development_scenarios = read_development_scenarios(file_name=file_name, sheet_name=sheet_name)

    data.update({
        'measure_effects': measure_effects, 
        'pressure_contributions': pressure_contributions, 
        'thresholds': thresholds, 
        'cases': cases, 
        'activity_contributions': activity_contributions, 
        'overlaps': overlaps, 
        'development_scenarios': development_scenarios
    })

    data = link_area_ids(data)

    return data


def build_links(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Builds links.

    Arguments:
        data (dict):

    Returns:
        links (DataFrame) = Measure-Activity-Pressure-State reduction table
    """
    msdf = data['measure_effects']

    # verify that there are no duplicate links
    assert len(msdf[msdf.duplicated(['measure', 'activity', 'pressure', 'state'])]) == 0

    # create a new dataframe for links
    links = pd.DataFrame(msdf)

    # get picks from cumulative distribution
    links['reduction'] = links['probability'].apply(get_pick)
    links = links.drop(columns=['probability'])

    # initialize multiplier column
    links['multiplier'] = np.ones(len(msdf))

    #
    # Overlaps (measure-measure interaction)
    #

    measure_ids = links['measure'].unique()
    overlaps = data['overlaps']
    for id in measure_ids:
        rows = overlaps.loc[overlaps['Overlapping'] == id, :]
        for i, row in rows.iterrows():
            overlapping_id = row['Overlapping']
            overlapped_id = row['Overlapped']
            pressure_id = row['Pressure']
            activity_id = row['Activity']
            multiplier = row['Multiplier']
            query = (links['measure'] == overlapped_id) & (links['pressure'] == pressure_id)
            if activity_id != 0:
                query = query & (links['activity'] == activity_id)
            links.loc[query, 'multiplier'] = links.loc[query, 'multiplier'] * multiplier

    return links


def build_scenario(data: dict[str, pd.DataFrame], scenario: str) -> pd.DataFrame:
    """
    Build scenario
    """
    act_to_press = data['activity_contributions']
    dev_scen = data['development_scenarios']

    # for each pressure, save the total contribution of activities for later normalization
    actual_sum = {}
    for pressure_id in act_to_press['Pressure'].unique():
        actual_sum[pressure_id] = {}
        activities = act_to_press.loc[act_to_press['Pressure'] == pressure_id, :]
        for area in activities['area_id'].unique():
            actual_sum[pressure_id][area] = activities.loc[activities['area_id'] == area, 'value'].sum()
    
    # multiply activities by scenario multiplier
    def get_scenario(activity_id):
        multiplier = dev_scen.loc[dev_scen['Activity'] == activity_id, scenario]
        if len(multiplier) == 0:
            return 1
        multiplier = multiplier.values[0]
        return multiplier
    act_to_press['value'] = act_to_press['value'] * act_to_press['Activity'].apply(get_scenario)

    # normalize
    normalize_factor = {}
    for pressure_id in act_to_press['Pressure'].unique():
        normalize_factor[pressure_id] = {}
        activities = act_to_press.loc[act_to_press['Pressure'] == pressure_id, :]
        for area in activities['area_id'].unique():
            scenario_sum = activities.loc[activities['area_id'] == area, 'value'].sum()
            normalize_factor[pressure_id][area] = 1 + scenario_sum - actual_sum[pressure_id][area]

    def normalize(value, pressure_id, area_id):
        return value * normalize_factor[pressure_id][area_id]

    act_to_press['value'] = act_to_press.apply(lambda x: normalize(x['value'], x['Pressure'], x['area_id']), axis=1)
    
    return act_to_press


def build_cases(cases: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """
    Builds cases.
    """
    # replace all zeros (0) in activity / pressure / state columns with full list of values
    # filter those lists to only include relevant IDs (from links)
    # finally explode to only have single IDs per row
    cols = ['activity', 'pressure', 'state']
    for col in cols:
        cases[col] = cases[col].astype(object)
    for i, row in cases.iterrows():
        maps_links = links.loc[links['measure'] == row['measure'], cols]    # select relevant measure/activity/pressure/state links
        if len(maps_links) == 0:
            cases.drop(i, inplace=True) # drop rows where measure has no effect
            continue
        for col in cols:
            cases.at[i, col] = maps_links[col].unique().tolist() if row[col] == 0 else row[col]
    for col in cols:
        cases = cases.explode(col)
    
    cases = cases.reset_index(drop=True)

    # filter out links that don't have associated reduction
    m = cases['measure'].isin(links['measure'])
    a = cases['activity'].isin(links['activity'])
    p = cases['pressure'].isin(links['pressure'])
    s = cases['state'].isin(links['state'])
    existing_links = (m & a & p & s)
    cases = cases.loc[existing_links, :]

    cases = cases.reset_index(drop=True)

    # remove duplicate measures in areas, measure with highest coverage and implementation is chosen
    cases = cases.sort_values(by=['coverage', 'implementation'], ascending=[False, False])
    cases = cases.drop_duplicates(subset=['measure', 'activity', 'pressure', 'state', 'area_id'], keep='first')
    cases = cases.reset_index(drop=True)

    return cases


def build_changes(data: dict[str, pd.DataFrame], links: pd.DataFrame, time_steps: int = 1) -> pd.DataFrame:
    """
    Simulate the reduction in activities and pressures caused by measures and 
    return the change observed in state. 
    """
    cases = data['cases']
    areas = cases['area_id'].unique()

    # create dataframes to store changes in pressure and state, one column per area_id
    # NOTE: the DataFrames are created on one line to avoid PerformanceWarning

    # represents the amount of the pressure ('ID' column) that is left
    # 1 = unchanged pressure, 0 = no pressure left
    pressure_levels = pd.DataFrame(data['pressure']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(1.0)
    # represents the amount of the total pressure load that is left affecting the given state ('ID' column)
    # 1 = unchanged pressure load, 0 = no pressure load left affecting the state
    total_pressure_load_levels = pd.DataFrame(data['state']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(1.0)
    # stores the change in individual pressures from activity reductions between simulation rounds
    pressure_changes = pd.DataFrame(data['pressure']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(0.0)
    # stores the change in total pressure loads from pressure reductions between simulation rounds
    total_pressure_load_changes = pd.DataFrame(data['pressure']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(0.0)

    # make sure activity contributions don't exceed 100 % 
    for area in areas:
        for p_i, p in pressure_levels.iterrows():
            mask = (data['activity_contributions']['area_id'] == area) & (data['activity_contributions']['Pressure'] == p['ID'])
            relevant_contributions = data['activity_contributions'].loc[mask, :]
            if len(relevant_contributions) > 0:
                contribution_sum = relevant_contributions['value'].sum()
                if contribution_sum > 1:
                    data['activity_contributions'].loc[mask, 'value'] = relevant_contributions['value'] / contribution_sum

    #
    # simulation loop
    #

    for time_step in range(time_steps):

        # save previous time step levels before new reductions
        previous_pressure_levels = pressure_levels
        previous_total_pressure_load_levels = total_pressure_load_levels

        #
        # pressure reductions
        #

        # TODO: check if overlaps should be done on a per area basis
        # TODO: make sure pressure reductions don't exceed 100 %

        # activity contributions
        for area in areas: # for each area
            c = cases.loc[cases['area_id'] == area, :]  # select cases for current area
            for p_i, p in pressure_levels.iterrows():
                relevant_measures = c.loc[c['pressure'] == p['ID'], :]
                for m_i, m in relevant_measures.iterrows(): # for each measure implementation affecting the current pressure in the current area
                    mask = (links['measure'] == m['measure']) & (links['activity'] == m['activity']) & (links['pressure'] == m['pressure']) & (links['state'] == m['state'])
                    row = links.loc[mask, :]    # find the reduction of the current measure implementation
                    if len(row) == 0:
                        continue    # skip measure if data on the effect is not known
                    assert len(row) == 1
                    red = row['reduction'].values[0]
                    multiplier = row['multiplier'].values[0]
                    for mod in ['coverage', 'implementation']:
                        multiplier = multiplier * m[mod]
                    reduction = red * multiplier
                    # if activity is 0 (= straight to pressure), contribution will be 1
                    if m['activity'] == 0:
                        contribution = 1
                    # if activity is not in contribution list, contribution will be 0
                    mask = (data['activity_contributions']['Activity'] == m['activity']) & (data['activity_contributions']['Pressure'] == m['pressure']) & (data['activity_contributions']['area_id'] == area)
                    contribution = data['activity_contributions'].loc[mask, 'value']
                    if len(contribution) == 0:
                        contribution = 0
                    else:
                        contribution = contribution.values[0]
                    # reduce pressure
                    pressure_levels.at[p_i, area] = pressure_levels.at[p_i, area] * (1 - reduction * contribution)
                    # normalize activity contributions to reflect pressure reduction
                    norm_mask = (data['activity_contributions']['area_id'] == area) & (data['activity_contributions']['Pressure'] == p['ID'])
                    relevant_contributions = data['activity_contributions'].loc[norm_mask, 'value']
                    data['activity_contributions'].loc[norm_mask, 'value'] = relevant_contributions / (1 - reduction * contribution)

        #
        # state reductions
        #

        # straight to state measures
        for area in areas:
            c = cases.loc[cases['area_id'] == area, :]
            for s_i, s in total_pressure_load_levels.iterrows():
                relevant_measures = c.loc[c['state'] == s['ID'], :]
                for m_i, m in relevant_measures.iterrows():
                    mask = (links['measure'] == m['measure']) & (links['activity'] == m['activity']) & (links['pressure'] == m['pressure']) & (links['state'] == m['state'])
                    row = links.loc[mask, :]
                    if len(row) == 0:
                        continue
                    else:
                        red = row['reduction'].values[0]
                        multiplier = row['multiplier'].values[0]
                    for mod in ['coverage', 'implementation']:
                        multiplier = multiplier * m[mod]
                    reduction = red * multiplier
                    total_pressure_load_levels.at[s_i, area] = total_pressure_load_levels.at[s_i, area] * (1 - reduction)
        
        # pressure contributions
        pressure_contributions = data['pressure_contributions']
        for area in areas:
            a_i = pressure_levels.columns.get_loc(area)
            for s_i, s in total_pressure_load_levels.iterrows():
                relevant_pressures = pressure_contributions.loc[pressure_contributions['State'] == s['ID'], :]
                for p_i, p in relevant_pressures.iterrows():
                    row_i = pressure_levels.loc[pressure_levels['ID'] == p['pressure']].index[0]
                    reduction = 1 - pressure_levels.iloc[row_i, a_i]    # reduction = 100 % - the part that is left of the pressure
                    contribution = p['average']
                    total_pressure_load_levels.at[s_i, area] = total_pressure_load_levels.at[s_i, area] * (1 - reduction * contribution)
        
        # compare state reduction to GES threshold
        thresholds = data['thresholds']
        cols = ['PR', '10', '25', '50']
        state_ges = {}
        for col in cols:
            state_ges[col] = pd.DataFrame(data['state']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(1.0)
        for area in areas:
            a_i = total_pressure_load_levels.columns.get_loc(area)
            for s_i, s in total_pressure_load_levels.iterrows():
                row = thresholds.loc[(thresholds['State'] == s['ID']) & (thresholds['area_id'] == area), cols]
                if len(row) == 0:
                    continue
                for col in cols:
                    state_ges[col].iloc[s_i, a_i] = row.loc[:, col].values[0]
        
        #
        # Next time step updates
        #

        # TODO: update activity contributions, according to which contributions that got reduced

    data.update({
        'pressure_levels': pressure_levels, 
        'total_pressure_load_levels': total_pressure_load_levels, 
        'state_ges': state_ges
    })

    return data


#EOF
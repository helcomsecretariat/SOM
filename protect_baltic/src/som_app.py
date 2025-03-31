"""
Copyright (c) 2024 Baltic Marine Environment Protection Commission

LICENSE available under 
local: 'SOM/protect_baltic/LICENSE'
url: 'https://github.com/helcomsecretariat/SOM/blob/main/protect_baltic/LICENCE'
"""

import numpy as np
import pandas as pd
import os
from som_tools import *
from utilities import *
import matplotlib.pyplot as plt

def build_input(config: dict) -> dict[str, pd.DataFrame]:
    """
    Loads input data. If loading already processed data, probability distributions need to be converted back to arrays. 
    """
    path = os.path.realpath(config['input_data']['path'])
    if not os.path.isfile(path): path = os.path.join(os.path.dirname(os.path.realpath(__file__)), config['input_data']['path'])
    if config['use_legacy_input_data']:
        input_data = process_input_data(config)
        with pd.ExcelWriter(path) as writer:
            for key in input_data:
                input_data[key].to_excel(writer, sheet_name=key, index=False)
    else:
        input_data = pd.read_excel(io=path, sheet_name=None)
        conversion_sheet = [
            ('measure_effects', 'probability'), 
            ('activity_contributions', 'contribution'), 
            ('pressure_contributions', 'contribution'), 
            ('thresholds', 'PR'), 
            ('thresholds', '10'), 
            ('thresholds', '25'), 
            ('thresholds', '50')
        ]
        def str_to_arr(s):
            if type(s) is float: return s
            arr = []
            for a in [x for x in s.replace('[', '').replace(']', '').split(' ')]:
                if a != '':
                    arr.append(a)
            arr = np.array(arr)
            arr = arr.astype(float)
            arr = arr / np.sum(arr)
            return arr
        for sheet in conversion_sheet:
            input_data[sheet[0]][sheet[1]] = input_data[sheet[0]][sheet[1]].apply(str_to_arr)
    return input_data


def build_links(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Build links by picking random samples using probability distributions.
    """
    #
    # measure effects
    #

    # verify that there are no duplicate links
    try: assert len(data['measure_effects'][data['measure_effects'].duplicated(['measure', 'activity', 'pressure', 'state'])]) == 0
    except Exception as e: fail_with_message(f'Duplicate measure effects in input data!', e)

    # create a new dataframe for links
    links = pd.DataFrame(data['measure_effects'])

    # get picks from cumulative distribution
    links['reduction'] = links['probability'].apply(get_pick)
    links = links.drop(columns=['probability'])

    data['measure_effects'] = links
    
    #
    # activity contributions
    #

    data['activity_contributions']['contribution'] = data['activity_contributions']['contribution'].apply(get_pick)

    #
    # pressure contributions
    #

    # get picks from cumulative distribution
    data['pressure_contributions']['contribution'] = data['pressure_contributions']['contribution'].apply(lambda x: get_pick(x) if not np.any(np.isnan(x)) else np.nan)
    
    data['pressure_contributions'] = data['pressure_contributions'].drop_duplicates(subset=['State', 'pressure', 'area_id'], keep='first').reset_index(drop=True)

    # verify that there are no duplicate links
    try: assert len(data['pressure_contributions'][data['pressure_contributions'].duplicated(['State', 'pressure', 'area_id'])]) == 0
    except Exception as e: fail_with_message(f'Duplicate pressure contributions in input data!', e)
    
    # make sure pressure contributions for each state / area are 100 %
    for area in data['area']['ID']:
        for state in data['state']['ID']:
            mask = (data['pressure_contributions']['area_id'] == area) & (data['pressure_contributions']['State'] == state)
            relevant_contributions = data['pressure_contributions'].loc[mask, :]
            if len(relevant_contributions) > 0:
                data['pressure_contributions'].loc[mask, 'contribution'] = relevant_contributions['contribution'] / relevant_contributions['contribution'].sum()

    #
    # thresholds
    #

    threshold_cols = ['PR', '10', '25', '50']   # target thresholds (PR=GES)

    # get picks from cumulative distribution
    for col in threshold_cols:
        data['thresholds'][col] = data['thresholds'][col].apply(lambda x: get_pick(x) if not np.any(np.isnan(x)) else np.nan)

    data['thresholds'] = data['thresholds'].drop_duplicates(subset=['State', 'area_id'], keep='first').reset_index(drop=True)

    # verify that there are no duplicate links
    try: assert len(data['thresholds'][data['thresholds'].duplicated(['State', 'area_id'])]) == 0
    except Exception as e: fail_with_message(f'Duplicate GES targets in input data!', e)

    return data


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
            actual_sum[pressure_id][area] = activities.loc[activities['area_id'] == area, 'contribution'].sum()
    
    # multiply activities by scenario multiplier
    def get_scenario(activity_id):
        multiplier = dev_scen.loc[dev_scen['Activity'] == activity_id, scenario]
        if len(multiplier) == 0:
            return 1
        multiplier = multiplier.values[0]
        return multiplier
    act_to_press['contribution'] = act_to_press['contribution'] * act_to_press['Activity'].apply(get_scenario)

    # normalize
    normalize_factor = {}
    for pressure_id in act_to_press['Pressure'].unique():
        normalize_factor[pressure_id] = {}
        activities = act_to_press.loc[act_to_press['Pressure'] == pressure_id, :]
        for area in activities['area_id'].unique():
            scenario_sum = activities.loc[activities['area_id'] == area, 'contribution'].sum()
            normalize_factor[pressure_id][area] = 1 + scenario_sum - actual_sum[pressure_id][area]

    def normalize(value, pressure_id, area_id):
        return value * normalize_factor[pressure_id][area_id]

    act_to_press['contribution'] = act_to_press.apply(lambda x: normalize(x['contribution'], x['Pressure'], x['area_id']), axis=1)
    
    return act_to_press


def build_cases(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Builds cases.
    """
    cases = data['cases']
    links = data['measure_effects']
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

    data['cases'] = cases

    return data


def build_changes(data: dict[str, pd.DataFrame], time_steps: int = 1, warnings = False) -> dict[str, pd.DataFrame]:
    """
    Simulate the reduction in activities and pressures caused by measures and 
    return the change observed in state. 
    """
    # this variable is used in assertions where float number error might affect comparisons
    allowed_error = 0.00001     

    cases = data['cases']
    links = data['measure_effects']
    areas = data['area']['ID']

    # create dataframes to store changes in pressure and state, one column per area_id
    # NOTE: the DataFrames are created on one line to avoid PerformanceWarning

    # represents the amount of the pressure ('ID' column) that is left
    # 1 = unchanged pressure, 0 = no pressure left
    pressure_levels = pd.DataFrame(data['pressure']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(1.0)
    # represents the amount of the total pressure load that is left affecting the given state ('ID' column)
    # 1 = unchanged pressure load, 0 = no pressure load left affecting the state
    total_pressure_load_levels = pd.DataFrame(data['state']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(1.0)

    # represents the reduction observed in the pressure ('ID' column)
    pressure_reductions = pd.DataFrame(data['pressure']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(0.0)
    # represents the reduction observed in the total pressure load ('ID' column)
    total_pressure_load_reductions = pd.DataFrame(data['state']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(0.0)

    # make sure activity contributions don't exceed 100 %
    for area in areas:
        for p_i, p in pressure_levels.iterrows():
            mask = (data['activity_contributions']['area_id'] == area) & (data['activity_contributions']['Pressure'] == p['ID'])
            relevant_contributions = data['activity_contributions'].loc[mask, :]
            if len(relevant_contributions) > 0:
                contribution_sum = relevant_contributions['contribution'].sum()
                if contribution_sum > 1:
                    data['activity_contributions'].loc[mask, 'contribution'] = relevant_contributions['contribution'] / contribution_sum
            try: assert data['activity_contributions'].loc[mask, 'contribution'].sum() <= 1 + allowed_error
            except Exception as e: fail_with_message(f'Failed to verify that activity contributions do not exceed 100 % for area {area}, pressure {p["ID"]} with contribution sum {data['activity_contributions'].loc[mask, 'contribution'].sum()}', e)

    # make sure pressure contributions don't exceed 100 %
    for area in areas:
        for s_i, s in total_pressure_load_levels.iterrows():
            mask = (data['pressure_contributions']['area_id'] == area) & (data['pressure_contributions']['State'] == s['ID'])
            relevant_contributions = data['pressure_contributions'].loc[mask, :]
            if len(relevant_contributions) > 0:
                contribution_sum = relevant_contributions['contribution'].sum()
                if contribution_sum > 1:
                    data['pressure_contributions'].loc[mask, 'contribution'] = relevant_contributions['contribution'] / contribution_sum
            try: assert data['pressure_contributions'].loc[mask, 'contribution'].sum() <= 1 + allowed_error
            except Exception as e: fail_with_message(f'Failed to verify that pressure contributions do not exceed 100 % for area {area}, state {s["ID"]} with contribution sum {data['pressure_contributions'].loc[mask, 'contribution'].sum()}', e)

    #
    # simulation loop
    #

    for time_step in range(time_steps):

        #
        # pressure reductions
        #

        # activity contributions
        for area in areas:
            c = cases.loc[cases['area_id'] == area, :]  # select cases for current area
            for p_i, p in pressure_levels.iterrows():
                relevant_measures = c.loc[c['pressure'] == p['ID'], :]  # select all measures affecting the current pressure in the current area
                relevant_overlaps = data['overlaps'].loc[data['overlaps']['Pressure'] == p['ID'], :]    # select all overlaps affecting current pressure
                for m_i, m in relevant_measures.iterrows():
                    #
                    # get measure effect (= reduction), and apply modifiers
                    #
                    mask = (links['measure'] == m['measure']) & (links['activity'] == m['activity']) & (links['pressure'] == m['pressure']) & (links['state'] == m['state'])
                    row = links.loc[mask, :]    # find the reduction of the current measure implementation
                    if len(row) == 0:
                        if warnings: print(f'WARNING! Effect of measure {m["measure"]} on activity {m["activity"]} and pressure {m["pressure"]} not known! Measure {m["measure"]} will be skipped in area {area}.')
                        continue    # skip measure if data on the effect is not known
                    try: assert len(row) == 1
                    except Exception as e: fail_with_message(f'ERROR! Multiple instances of measure {m["measure"]} effect on activity {m["activity"]} and pressure {m["pressure"]} given in input data!', e)
                    reduction = row['reduction'].values[0]
                    for mod in ['coverage', 'implementation']:
                        reduction = reduction * m[mod]
                    #
                    # overlaps (measure-measure interaction)
                    #
                    for o_i, o in relevant_overlaps.loc[(relevant_overlaps['Overlapped'] == m['measure']) & (relevant_overlaps['Activity'] == m['activity']), :].iterrows():
                        if o['Overlapping'] in relevant_measures.loc[relevant_measures['activity'] == m['activity'], 'measure'].values: # ensure the overlapping measure is also for the current activity
                            reduction = reduction * o['Multiplier']
                    #
                    # contribution
                    #
                    if m['activity'] == 0:
                        contribution = 1    # if activity is 0 (= straight to pressure), contribution will be 1
                    else:
                        cont_mask = (data['activity_contributions']['Activity'] == m['activity']) & (data['activity_contributions']['Pressure'] == m['pressure']) & (data['activity_contributions']['area_id'] == area)
                        contribution = data['activity_contributions'].loc[cont_mask, 'contribution']
                        if len(contribution) == 0:
                            if warnings: print(f'WARNING! Contribution of activity {m["activity"]} to pressure {m["pressure"]} not known! Measure {m["measure"]} will be skipped in area {area}.')
                            continue    # skip measure if activity is not in contribution list
                        else:
                            try: assert len(contribution) == 1
                            except Exception as e: fail_with_message(f'ERROR! Multiple instances of activity {m["activity"]} contribution on pressure {m["pressure"]} given in input data!', e)
                            contribution = contribution.values[0]
                    #
                    # reduce pressure
                    #
                    pressure_levels.at[p_i, area] = pressure_levels.at[p_i, area] * (1 - reduction * contribution)
                    if pressure_levels.at[p_i, area] < 0:
                        print(f'area {area}, pressure {p["ID"]} => level = {pressure_levels.at[p_i, area]}, red = {reduction}, cont = {contribution}')
                    #
                    # normalize activity contributions to reflect pressure reduction
                    #
                    if abs(1 - contribution) > allowed_error and contribution != 0:     # only normalize if there is change in contributions
                        data['activity_contributions'].loc[cont_mask, 'contribution'] = contribution * (1 - reduction)   # reduce the current contribution before normalizing
                        norm_mask = (data['activity_contributions']['area_id'] == area) & (data['activity_contributions']['Pressure'] == p['ID'])
                        relevant_contributions = data['activity_contributions'].loc[norm_mask, 'contribution']
                        data['activity_contributions'].loc[norm_mask, 'contribution'] = relevant_contributions / (1 - reduction * contribution)

        #
        # total pressure load reductions
        #

        # straight to state measures
        for area in areas:
            c = cases.loc[cases['area_id'] == area, :]  # select cases for current area
            for s_i, s in total_pressure_load_levels.iterrows():
                relevant_measures = c.loc[c['state'] == s['ID'], :] # select all measures affecting current state in current the area
                for m_i, m in relevant_measures.iterrows():
                    #
                    # get measure effect (= reduction), and apply modifiers
                    #
                    mask = (links['measure'] == m['measure']) & (links['activity'] == m['activity']) & (links['pressure'] == m['pressure']) & (links['state'] == m['state'])
                    row = links.loc[mask, :]
                    if len(row) == 0:
                        continue
                    reduction = row['reduction'].values[0]
                    for mod in ['coverage', 'implementation']:
                        reduction = reduction * m[mod]
                    #
                    # overlaps (measure-measure interaction)
                    #
                    for o_i, o in data['overlaps'].loc[(data['overlaps']['Overlapped'] == m['measure']) & (data['overlaps']['Activity'] == m['activity']) & (data['overlaps']['Pressure'] == m['pressure']), :].iterrows():
                        if o['Overlapping'] in relevant_measures['measure'].values:
                            reduction = reduction * o['Multiplier']
                    #
                    # reduce pressure
                    #
                    total_pressure_load_levels.at[s_i, area] = total_pressure_load_levels.at[s_i, area] * (1 - reduction)

        # pressure contributions
        for area in areas:
            a_i = pressure_levels.columns.get_loc(area)
            for s_i, s in total_pressure_load_levels.iterrows():
                relevant_pressures = data['pressure_contributions'].loc[(data['pressure_contributions']['area_id'] == area) & (data['pressure_contributions']['State'] == s['ID']), :]  # select contributions of pressures affecting current state in current area
                for p_i, p in relevant_pressures.iterrows():
                    #
                    # main pressure reduction
                    #
                    row_i = pressure_levels.loc[pressure_levels['ID'] == p['pressure']].index[0]
                    reduction = 1 - pressure_levels.iloc[row_i, a_i]    # reduction = 100 % - the part that is left of the pressure
                    contribution = data['pressure_contributions'].loc[(data['pressure_contributions']['area_id'] == area) & (data['pressure_contributions']['State'] == s['ID']) & (data['pressure_contributions']['pressure'] == p['pressure']), 'contribution'].values[0]
                    #
                    # subpressures
                    #
                    relevant_subpressures = data['subpressures'].loc[(data['subpressures']['State'] == s['ID']) & (data['subpressures']['State pressure'] == p['pressure']), :]
                    for sp_i, sp in relevant_subpressures.iterrows():
                        sp_row_i = pressure_levels.loc[pressure_levels['ID'] == sp['Reduced pressure']].index[0]
                        multiplier = sp['Multiplier']
                        red = 1 - pressure_levels.iloc[sp_row_i, a_i]    # reduction = 100 % - the part that is left of the pressure
                        reduction = reduction + multiplier * red
                    try: assert reduction <= 1 + allowed_error
                    except Exception as e: fail_with_message(f'Failed on area {area}, state {s["ID"]}, pressure {p["pressure"]} with reduction {reduction}', e)
                    #
                    # reduce total pressure load
                    #
                    total_pressure_load_levels.at[s_i, area] = total_pressure_load_levels.at[s_i, area] * (1 - reduction * contribution)
                    #
                    # normalize pressure contributions to reflect pressure reduction
                    #
                    if abs(1 - contribution) > allowed_error and contribution != 0:     # only normalize if there is change in contributions
                        data['pressure_contributions'].loc[(data['pressure_contributions']['area_id'] == area) & (data['pressure_contributions']['State'] == s['ID']) & (data['pressure_contributions']['pressure'] == p['pressure']), 'contribution'] = contribution * (1 - reduction)   # reduce the current contribution before normalizing
                        norm_mask = (data['pressure_contributions']['area_id'] == area) & (data['pressure_contributions']['State'] == s['ID'])
                        relevant_contributions = data['pressure_contributions'].loc[norm_mask, 'contribution']
                        data['pressure_contributions'].loc[norm_mask, 'contribution'] = relevant_contributions / (1 - reduction * contribution)
                        try: assert abs(1 - data['pressure_contributions'].loc[norm_mask, 'contribution'].sum()) <= allowed_error
                        except Exception as e: fail_with_message(f'Failed on area {area}, state {s["ID"]}, pressure {p["pressure"]} with pressure contribution sum not equal to 1', e)
    
    # total reduction observed in total pressure loads
    for area in areas:
        for s_i, s in total_pressure_load_levels.iterrows():
            total_pressure_load_reductions.at[s_i, area] = 1 - total_pressure_load_levels.at[s_i, area]

    # GES thresholds
    cols = ['PR', '10', '25', '50']
    thresholds = {}
    for col in cols:
        thresholds[col] = pd.DataFrame(data['state']['ID']).reindex(columns=['ID']+areas.tolist())
    for area in areas:
        a_i = total_pressure_load_levels.columns.get_loc(area)
        for s_i, s in total_pressure_load_levels.iterrows():
            row = data['thresholds'].loc[(data['thresholds']['State'] == s['ID']) & (data['thresholds']['area_id'] == area), cols]
            if len(row) == 0:
                continue
            for col in cols:
                thresholds[col].iloc[s_i, a_i] = row.loc[:, col].values[0]
    
    data.update({
        'pressure_levels': pressure_levels, 
        'total_pressure_load_levels': total_pressure_load_levels, 
        'total_pressure_load_reductions': total_pressure_load_reductions, 
        'thresholds': thresholds
    })

    return data


def build_results(sim_res: str, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Process the simulated results to calculate uncertainties.

    Uncertainty is determined as standard error of the mean.
    """
    files = [os.path.join(sim_res, x) for x in os.listdir(sim_res) if x.endswith('.xlsx') and 'sim_res' in x]

    areas = data['area']['ID']
    pressures = data['pressure']['ID']
    states = data['state']['ID']

    #
    # pressure levels
    #
    pressure_levels_average = pd.DataFrame(data['pressure']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(1.0)
    pressure_levels_error = pd.DataFrame(data['pressure']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(1.0)
    arr = np.empty(shape=(len(pressures.tolist()), len(areas.tolist()), len(files)))
    for i in range(len(files)):
        df = pd.read_excel(io=files[i], sheet_name='PressureLevels')
        arr[:, :, i] = df.values[:, 1:]
    pressure_levels_average.iloc[:, 1:] = np.mean(arr, axis=2)
    pressure_levels_error.iloc[:, 1:] = np.std(arr, axis=2, ddof=1) / np.sqrt(arr.shape[2])    # calculate standard error
    #
    # total pressure load levels
    #
    total_pressure_load_levels_average = pd.DataFrame(data['state']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(1.0)
    total_pressure_load_levels_error = pd.DataFrame(data['state']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(1.0)
    arr = np.empty(shape=(len(states.tolist()), len(areas.tolist()), len(files)))
    for i in range(len(files)):
        df = pd.read_excel(io=files[i], sheet_name='TPLLevels')
        arr[:, :, i] = df.values[:, 1:]
    total_pressure_load_levels_average.iloc[:, 1:] = np.mean(arr, axis=2)
    total_pressure_load_levels_error.iloc[:, 1:] = np.std(arr, axis=2, ddof=1) / np.sqrt(arr.shape[2])    # calculate standard error
    #
    # total pressure load reductions
    #
    total_pressure_load_reductions_average = pd.DataFrame(data['state']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(1.0)
    total_pressure_load_reductions_error = pd.DataFrame(data['state']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(1.0)
    arr = np.empty(shape=(len(states.tolist()), len(areas.tolist()), len(files)))
    for i in range(len(files)):
        df = pd.read_excel(io=files[i], sheet_name='TPLReductions')
        arr[:, :, i] = df.values[:, 1:]
    total_pressure_load_reductions_average.iloc[:, 1:] = np.mean(arr, axis=2)
    total_pressure_load_reductions_error.iloc[:, 1:] = np.std(arr, axis=2, ddof=1) / np.sqrt(arr.shape[2])    # calculate standard error
    #
    # thresholds
    #
    thresholds_average = pd.DataFrame(data['state']['ID']).reindex(columns=['ID']+areas.tolist())
    thresholds_error = pd.DataFrame(data['state']['ID']).reindex(columns=['ID']+areas.tolist())
    arr = np.empty(shape=(len(states.tolist()), len(areas.tolist()), len(files)))
    for i in range(len(files)):
        df = pd.read_excel(io=files[i], sheet_name='RequiredReductionsForGES')
        arr[:, :, i] = df.values[:, 1:]
    thresholds_average.iloc[:, 1:] = np.mean(arr, axis=2)
    thresholds_error.iloc[:, 1:] = np.std(arr, axis=2, ddof=1) / np.sqrt(arr.shape[2])    # calculate standard error
    #
    # measure effects
    #
    measure_effects_mean = pd.DataFrame(data['measure_effects']).drop(columns=['probability'])
    measure_effects_error = pd.DataFrame(data['measure_effects']).drop(columns=['probability'])
    arr = np.empty(shape=([x for x in data['measure_effects'].values.shape]+[len(files)]))
    for i in range(len(files)):
        df = pd.read_excel(io=files[i], sheet_name='MeasureEffects')
        arr[:, :, i] = df.values
    measure_effects_mean['reduction'] = np.mean(arr[:, -1, :], axis=1)
    measure_effects_error['reduction'] = np.std(arr[:, -1, :], axis=1, ddof=1) / np.sqrt(arr.shape[2])
    suffixes = ('_mean', '_error')
    measure_effects = pd.merge(measure_effects_mean, measure_effects_error, on=['measure', 'activity', 'pressure', 'state'], suffixes=suffixes)
    #
    # activity contributions
    #
    activity_contributions_mean = pd.DataFrame(data['activity_contributions'])
    activity_contributions_error = pd.DataFrame(data['activity_contributions'])
    arr = np.empty(shape=([x for x in data['activity_contributions'].values.shape]+[len(files)]))
    for i in range(len(files)):
        df = pd.read_excel(io=files[i], sheet_name='ActivityContributions')
        arr[:, :, i] = df.values
    activity_contributions_mean['contribution'] = np.mean(arr[:, -1, :], axis=1)
    activity_contributions_error['contribution'] = np.std(arr[:, -1, :], axis=1, ddof=1) / np.sqrt(arr.shape[2])
    suffixes = ('_mean', '_error')
    activity_contributions = pd.merge(activity_contributions_mean, activity_contributions_error, on=['Activity', 'Pressure', 'area_id'], suffixes=suffixes)
    #
    # pressure contributions
    #
    pressure_contributions_mean = pd.DataFrame(data['pressure_contributions'])
    pressure_contributions_error = pd.DataFrame(data['pressure_contributions'])
    arr = np.empty(shape=([x for x in data['pressure_contributions'].values.shape]+[len(files)]))
    for i in range(len(files)):
        df = pd.read_excel(io=files[i], sheet_name='PressureContributions')
        arr[:, :, i] = df.values
    pressure_contributions_mean['contribution'] = np.mean(arr[:, -1, :], axis=1)
    pressure_contributions_error['contribution'] = np.std(arr[:, -1, :], axis=1, ddof=1) / np.sqrt(arr.shape[2])
    suffixes = ('_mean', '_error')
    pressure_contributions = pd.merge(pressure_contributions_mean, pressure_contributions_error, on=['State', 'pressure', 'area_id'], suffixes=suffixes)

    # create new dict of dataframes
    res = {
        'PressureMean': pressure_levels_average, 
        'PressureError': pressure_levels_error, 
        'TPLMean': total_pressure_load_levels_average, 
        'TPLError': total_pressure_load_levels_error, 
        'TPLRedMean': total_pressure_load_reductions_average, 
        'TPLRedError': total_pressure_load_reductions_error, 
        'ThresholdsMean': thresholds_average, 
        'ThresholdsError': thresholds_error, 
        'MeasureEffects': measure_effects, 
        'ActivityContributions': activity_contributions, 
        'PressureContributions': pressure_contributions
    }

    return res


def build_display(res: dict[str, pd.DataFrame], data: dict[str, pd.DataFrame], out_dir: str):
    """
    Constructs plots to visualize results.
    """
    areas = data['area']['ID']

    # area dependent plots
    for area in areas:

        # create new directory for the plots
        area_name = data['area'].loc[areas == area, 'area'].values[0]
        temp_dir = os.path.join(out_dir, f'{area}_{area_name}')
        os.makedirs(temp_dir, exist_ok=True)

        #
        # General plot settings
        #

        marker = 's'
        markersize = 5
        markercolor = 'black'
        capsize = 3
        capthick = 1
        elinewidth = 1
        ecolor = 'salmon'
        label_angle = 60
        char_limit = 25
        bar_width = 0.4
        bar_color_1 = 'turquoise'
        bar_color_2 = 'seagreen'
        edge_color = 'black'

        #
        # TPL
        #

        fig, ax = plt.subplots(figsize=(16, 12), constrained_layout=True)

        # adjust data
        suffixes = ('_mean', '_error')
        df = pd.merge(res['TPLMean'].loc[:, ['ID', area]], res['TPLError'].loc[:, ['ID', area]], on='ID', suffixes=suffixes)
        x_vals = data['state'].loc[:, 'state'].values
        x_vals = np.array([x[:char_limit]+'...' if len(x) > char_limit else x for x in x_vals])     # limit characters to char_limit
        y_vals = df[str(area)+'_mean'] * 100    # convert to %
        y_err = df[str(area)+'_error'] * 100    # conver to %

        # create plot
        ax.errorbar(np.arange(len(x_vals)), y_vals, yerr=y_err, linestyle='None', marker=marker, capsize=capsize, capthick=capthick, elinewidth=elinewidth, markersize=markersize, color=markercolor, ecolor=ecolor)
        ax.set_xlabel('Environmental State')
        ax.set_ylabel('Level (%)')
        ax.set_title(f'Total Pressure Load on Environmental States\n({area_name})')
        ax.set_xticks(np.arange(len(x_vals)), x_vals, rotation=label_angle, ha='right')
        ax.yaxis.grid(True, linestyle='--', color='lavender')

        # adjust axis limits
        x_lim = [- 0.5, len(x_vals) - 0.5]
        ax.set_xlim(x_lim)
        y_lim = [-5, 105]
        ax.set_ylim(y_lim)

        # export
        plt.savefig(os.path.join(temp_dir, f'{area}_{area_name}_TotalPressureLoadLevels.png'), dpi=200)

        plt.close(fig)

        #
        # Pressures
        #

        fig, ax = plt.subplots(figsize=(25, 12), constrained_layout=True)

        # adjust data
        suffixes = ('_mean', '_error')
        df = pd.merge(res['PressureMean'].loc[:, ['ID', area]], res['PressureError'].loc[:, ['ID', area]], on='ID', suffixes=suffixes)
        x_vals = data['pressure'].loc[:, 'pressure'].values
        x_vals = np.array([x[:char_limit]+'...' if len(x) > char_limit else x for x in x_vals])     # limit characters to char_limit
        y_vals = df[str(area)+'_mean'] * 100    # convert to %
        y_err = df[str(area)+'_error'] * 100    # conver to %

        # create plot
        ax.errorbar(np.arange(len(x_vals)), y_vals, yerr=y_err, linestyle='None', marker=marker, capsize=capsize, capthick=capthick, elinewidth=elinewidth, markersize=markersize, color=markercolor, ecolor=ecolor)
        ax.set_xlabel('Pressure')
        ax.set_ylabel('Level (%)')
        ax.set_title(f'Pressure Levels\n({area_name})')
        ax.set_xticks(np.arange(len(x_vals)), x_vals, rotation=label_angle, ha='right')
        ax.yaxis.grid(True, linestyle='--', color='lavender')

        # adjust axis limits
        x_lim = [- 0.5, len(x_vals) - 0.5]
        ax.set_xlim(x_lim)
        y_lim = [-5, 105]
        ax.set_ylim(y_lim)

        # export
        plt.savefig(os.path.join(temp_dir, f'{area}_{area_name}_PressureLevels.png'), dpi=200)

        plt.close(fig)

        #
        # GES thresholds
        #

        fig, ax = plt.subplots(figsize=(16, 12), constrained_layout=True)

        # adjust data
        x_labels = np.array([x[:char_limit]+'...' if len(x) > char_limit else x for x in data['state'].loc[:, 'state'].values])     # limit characters to char_limit
        x_vals = np.arange(len(x_labels))
        suffixes = ('_mean', '_error')
        df = pd.merge(res['TPLRedMean'].loc[:, ['ID', area]], res['TPLRedError'].loc[:, ['ID', area]], on='ID', suffixes=suffixes)
        y_vals_tpl = df[str(area)+'_mean'] * 100    # convert to %
        y_err_tpl = df[str(area)+'_error'] * 100    # conver to %
        df = pd.merge(res['ThresholdsMean'].loc[:, ['ID', area]], res['ThresholdsError'].loc[:, ['ID', area]], on='ID', suffixes=suffixes)
        y_vals_ges = df[str(area)+'_mean'] * 100    # convert to %
        y_err_ges = df[str(area)+'_error'] * 100    # convert to %

        # create plot
        label_tpl = 'Reduction with measures'
        ax.bar(x_vals-bar_width/2, y_vals_tpl, width=bar_width, align='center', color=bar_color_1, label=label_tpl, edgecolor=edge_color)
        ax.errorbar(x_vals-bar_width/2, y_vals_tpl, yerr=y_err_tpl, linestyle='None', marker='None', capsize=capsize, capthick=capthick, elinewidth=elinewidth, ecolor=ecolor)
        label_ges = 'GES'
        ax.bar(x_vals+bar_width/2, y_vals_ges, width=bar_width, align='center', color=bar_color_2, label=label_ges, edgecolor=edge_color)
        ax.errorbar(x_vals+bar_width/2, y_vals_ges, yerr=y_err_ges, linestyle='None', marker='None', capsize=capsize, capthick=capthick, elinewidth=elinewidth, ecolor=ecolor)
        ax.set_xlabel('Environmental State')
        ax.set_ylabel('Reduction (%)')
        ax.set_title(f'Total Pressure Load Reduction vs. GES Reduction Thresholds\n({area_name})')
        ax.set_xticks(x_vals, x_labels, rotation=label_angle, ha='right')
        ax.yaxis.grid(True, linestyle='--', color='lavender')
        ax.legend()

        # export
        plt.savefig(os.path.join(temp_dir, f'{area}_{area_name}_Thresholds.png'), dpi=200)

        # adjust axis limits
        x_lim = [- 0.5, len(x_vals) - 0.5]
        ax.set_xlim(x_lim)
        y_lim = [0, 100]
        ax.set_ylim(y_lim)

        plt.close(fig)

    #
    # Measure effects
    #

    fig, ax = plt.subplots(figsize=(100, 14), constrained_layout=True)

    bar_width = 0.8
    edge_color = 'black'
    activity_font_size = 8

    # adjust data
    df = res['MeasureEffects'].sort_values(by=['measure', 'pressure', 'state', 'activity'])
    suffixes = ('', '_name')
    for col in ['measure', 'activity', 'pressure', 'state']:
        df = df.merge(data[col].loc[:, [col, 'ID']], left_on=col, right_on='ID', how='left', suffixes=suffixes)
        df = df.drop(columns=[col, 'ID'])
        df = df.rename(columns={col+'_name': col})
        df.loc[:, col] = np.array([(x[:char_limit]+'...' if len(x) > char_limit else x) if type(x) == str else 'All' for x in df.loc[:, col].values])
    df['index'] = np.arange(len(df))
    x_ticks = {x: df[df['measure'] == x]['index'].mean() for x in df['measure'].unique()}

    # set colors
    df['color_key'] = df['pressure'].astype(str) + '_' + df['state'].astype(str)
    unique_keys = df['color_key'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_keys)))
    color_map = {key: colors[i] for i, key in enumerate(unique_keys)}

    # create plot
    for key in unique_keys:
        subset = df[df['color_key'] == key]
        bars = ax.bar(subset['index'], subset['reduction_mean'] * 100, width=bar_width, color=color_map[key], label=key if key not in ax.get_legend_handles_labels()[1] else '', edgecolor=edge_color)
        ax.errorbar(subset['index'], subset['reduction_mean'] * 100, yerr=subset['reduction_error'] * 100, linestyle='None', marker='None', capsize=capsize, capthick=capthick, elinewidth=elinewidth, ecolor=ecolor)
        for bar, (_, row) in zip(bars, subset.iterrows()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, str(row['activity']), 
                    ha='center', va='center', rotation=90, fontsize=activity_font_size, color='white')

    ax.set_xlabel('Measure')
    ax.set_ylabel('Reduction effect (%)')
    ax.set_title(f'Measure Reduction Effects')
    ax.set_xticks(list(x_ticks.values()), list(x_ticks.keys()), rotation=label_angle, ha='right')
    ax.yaxis.grid(True, linestyle='--', color='lavender')
    ax.legend(title='Pressure/State', bbox_to_anchor=(1.05, 1), loc='upper left')

    # adjust axis limits
    x_lim = [- 0.5, len(df) - 0.5]
    ax.set_xlim(x_lim)
    y_lim = [0, 100]
    ax.set_ylim(y_lim)

    # export
    for area in areas:
        area_name = data['area'].loc[areas == area, 'area'].values[0]
        temp_dir = os.path.join(out_dir, f'{area}_{area_name}')
        plt.savefig(os.path.join(temp_dir, f'{area}_{area_name}_MeasureEffects.png'), dpi=200)

    plt.close(fig)


def set_id_columns(res: dict[str, pd.DataFrame], data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Replaces id column values with the name of the corresponding measure/activity/pressure/state in the result dataframes
    """
    relations = {
        'PressureMean': 'pressure', 
        'PressureError': 'pressure', 
        'TPLMean': 'state', 
        'TPLError': 'state', 
        'TPLRedMean': 'state', 
        'TPLRedError': 'state', 
        'ThresholdsMean': 'state', 
        'ThresholdsError': 'state'
    }
    def replace_ids(id, k):
        return data[k].loc[data[k]['ID'] == id, k].values[0]
    for key in relations:
        res[key]['ID'] = res[key]['ID'].apply(lambda x: replace_ids(x, relations[key]))
        res[key] = res[key].rename(columns={col: data['area'].loc[data['area']['ID'] == col, 'area'].values[0] for col in [c for c in res[key].columns if c != 'ID']})
    relations = {
        'MeasureEffects': ['measure', 'activity', 'pressure', 'state'], 
        'ActivityContributions': ['Activity', 'Pressure', 'area_id'], 
        'PressureContributions': ['State', 'pressure', 'area_id']
    }
    conversions = {
        'Activity': 'activity', 
        'Pressure': 'pressure', 
        'State': 'state', 
        'area_id': 'area'
    }
    for key in relations:
        for col in relations[key]:
            k = conversions[col] if col in conversions else col
            res[key][col] = res[key][col].apply(lambda id: data[k].loc[data[k]['ID'] == id, k].values[0] if id != 0 else '-')

    return res

#EOF
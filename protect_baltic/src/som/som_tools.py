"""
Copyright (c) 2024 Baltic Marine Environment Protection Commission
Copyright (c) 2022 Antti-Jussi Kieloaho (Natural Resources Institute Finland)

LICENSE available under 
local: 'SOM/protect_baltic/LICENSE'
url: 'https://github.com/helcomsecretariat/SOM/blob/main/protect_baltic/LICENCE'
"""

import numpy as np
import pandas as pd

def read_survey_data(file_name: str, sheet_names: dict[int, str]) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """
    Measure survey data: Part 1

    Arguments:
        file_name (str): path of survey excel file
        sheet_names (dict): dict of survey sheet ids in file_name
    
    Return:
        mteq (DataFrame): survey data information
        measure_survey_data (dict): survey data
    """
    # measure effect input Excel file
    mteq = pd.read_excel(io=file_name, sheet_name=sheet_names[0])

    # preprocess values
    mteq['Direct_to_state'] = [x.split(';') if type(x) == str else x for x in mteq['Direct_to_state']]

    measure_survey_data = {}

    for id, sheet in enumerate(sheet_names.values()):

        if id == 0:
            continue

        measure_survey_data[id] = pd.read_excel(io=file_name, sheet_name=sheet, header=None)

    return mteq, measure_survey_data


def preprocess_survey_data(mteq: pd.DataFrame, measure_survey_data: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Measure survey data: Part 2

    This method parses the survey sheets from webropol

    Arguments:
        mteq (DataFrame): survey data information
        measure_survey_data (dict): survey data
    
    Return:
        survey_df (DataFrame): survey data information
            survey_id: unique id for each questionnaire / survey
            title: type of value ('expected value' / 'variance' / 'max effectivness' / 'expert weights')
            block: id for each set of questions within each survey, unique across all surveys
            measure: measure id of the question
            activity: activity id of the question
            pressure: pressure id of the question
            state: state id (if defined) of the question ([nan] if no state)
            1...n (=expert ids): answer value for each expert (NaN if no answer)
    """

    cols = ['survey_id', 'title', 'block', 'measure', 'activity', 'pressure', 'state']
    survey_df = pd.DataFrame(columns=cols)  # create new DataFrame

    block_number = 0    # represents the survey block

    # for every survey sheet
    for survey_id in measure_survey_data:
        
        survey_info = mteq[mteq['Survey ID'] == survey_id]  # select the rows linked to current survey

        end = 0     # represents last column index of the question set
        for row, amt in enumerate(survey_info['AMT']):  # for each set of questions (row in MTEQ)
            
            end = end + (2 * amt + 1)     # end column for data
            start = end - (2 * amt)     # start column for data
            
            titles = ['expected value', 'variance'] * amt
            titles.append('max effectivness')
            titles.append('expert weights')

            measures = measure_survey_data[survey_id].iloc[0, start:end].tolist() # select current question column names as measure ids
            measures.append(np.nan) # append NaN for max effectivness (ME)
            measures.append(np.nan )   # append NaN for expert weights

            activity_id = survey_info['Activity'].iloc[row] # select current row Activity
            activities = [activity_id] * amt * 2
            activities.append(np.nan)
            activities.append(np.nan)

            pressure_id = survey_info['Pressure'].iloc[row] # select current row Pressure
            pressures = [pressure_id] * amt * 2
            pressures.append(np.nan)
            pressures.append(np.nan)

            direct_ids = survey_info['Direct_to_state'].iloc[row]   # select current row state
            if isinstance(direct_ids, str):
                directs = [[int(x) for x in direct_ids.split(';') if x != '']] * amt * 2
            elif isinstance(direct_ids, list):
                directs = [[int(x) for x in direct_ids if x != '']] * amt * 2
            elif isinstance(direct_ids, float) or isinstance(direct_ids, int):
                directs = [[direct_ids if not np.isnan(direct_ids) else np.nan]] * amt * 2
            else:
                directs = [direct_ids] * amt * 2
            directs.append(np.nan)
            directs.append(np.nan)

            expert_cols = [True if 'exp' in col.lower() else False for col in survey_info.columns]  # find all expert columns
            expert_weights = survey_info.loc[:, expert_cols].iloc[row]  # select expert weight values
            expert_weights.fillna(1, inplace=True)  # replace NaN values in weights with ones
            
            data = measure_survey_data[survey_id].loc[1:, start:end]    # select current question answers
            data[end+1] = expert_weights  # create column for expert weights
            for expert, weight in enumerate(expert_weights, 1): # for each row (expert) in weights
                data.loc[expert, end+1] = weight    # set the weight as the value
            data = data.transpose() # transpose so that experts are columns and measures are rows

            # add survey info to each entry in the data 
            data['survey_id'] = [survey_id] * len(data) # new column with survey_id for every row
            data['title'] = titles
            data['block'] = [block_number] * len(data)  # new column with block_number for every row
            data['measure'] = measures
            data['activity'] = activities
            data['pressure'] = pressures
            data['state'] = directs

            survey_df = pd.concat([survey_df, data], ignore_index=True, sort=False)
            block_number = block_number + 1

    return survey_df


def get_expert_ids(df: pd.DataFrame) -> list:
    '''
    Returns list of expert id column names from dataframe using regex
    '''
    return df.filter(regex='^(100|[1-9]?[0-9])$').columns


def process_survey_data(survey_df: pd.DataFrame) -> pd.DataFrame:
    r'''
    Measure survey data: part 3

    1. Adjust expert answers by scaling factor

    2. Calculate effectivness range boundaries

    3. Calculate mean for expected values and variance

    4. New id for 'measure' and 'activity' by multiplying id by 10000

        This is done as we need to track specific measure-activity-pressure and measure-state combinations
        'pressure' and 'state' id:s are not multiplied!

    5. Scaling factor ('title' with value 'max effectivness') is removed

    Arguments:
        survey_df (DataFrame): survey data information
            survey_id: unique id for each questionnaire / survey
            title: type of value ('expected value' / 'variance' / 'max effectivness' / 'expert weights')
            block: id for each set of questions within each survey, unique across all surveys
            measure: measure id of the question
            activity: activity id of the question
            pressure: pressure id of the question
            state: state id (if defined) of the question ([nan] if no state)
            1...n (=expert ids): answer value for each expert (NaN if no answer)
    Returns:
        survey_df (DataFrame): processed survey data information
    '''
    id_multiplier = 10000

    # select column names corresponding to expert ids (any number between 1 and 100)
    expert_ids = get_expert_ids(survey_df)

    # Step 1: adjust answers by scaling factor
    
    block_ids = survey_df.loc[:,'block'].unique()   # find unique block ids
    for b_id in block_ids:  # for each block
        block = survey_df.loc[survey_df['block'] == b_id, :]    # select all rows with current block id
        for col in block:   # for each column
            if isinstance(col, int):    # if it is an expert answer
                # from the column, select the expected values and variances
                expected_value = block.loc[block['title']=='expected value', col]
                variance = block.loc[block['title']=='variance', col]
                # skip if no questions were answered
                if expected_value.isnull().all():
                    block.loc[block['title']=='variance', col] = np.nan     # also set all variances to null
                    continue
                if variance.isnull().all():
                    block.loc[block['title']=='expected value', col] = np.nan     # also set all expected values to null
                    continue
                # find the highest value of the answers
                max_expected_value = expected_value.max()
                # find the max effectivness estimated by the expert
                max_effectivness = block.loc[block['title']=='max effectivness', col].values[0]
                # calculate scaling factor
                if np.isnan(max_effectivness):
                    # set all values to null if no max effectivness (in column, for current block)
                    survey_df.loc[survey_df['block'] == b_id, col] = np.nan
                elif max_effectivness == 0 or max_expected_value == 0:
                    # scale all expected values to 0 if max effectivness is zero or all expected values are zero
                    survey_df.loc[(survey_df['block'] == b_id) & (survey_df['title'] == 'expected value'), col] = 0
                else:
                    # get the scaling factor
                    scaling_factor = np.divide(max_expected_value, max_effectivness)
                    # divide the expected values by the new scaling factor
                    survey_df.loc[(survey_df['block'] == b_id) & (survey_df['title'] == 'expected value'), col] = np.divide(expected_value, scaling_factor)

    # Step 2: calculate effectivness range boundaries

    # create new rows for 'effectivness lower' and 'effectivness upper' bounds after variance rows
    new_rows = []
    for i, row in survey_df.iterrows():
        new_rows.append(row)
        if row['title'] == 'variance':
            # create lower bound
            min_row = survey_df.loc[i].copy()
            min_row['title'] = 'effectivness lower'
            min_row[expert_ids] = np.nan
            new_rows.append(min_row)
            # create upper bound
            max_row = survey_df.loc[i].copy()
            max_row['title'] = 'effectivness upper'
            max_row[expert_ids] = np.nan
            new_rows.append(max_row)
    survey_df = pd.DataFrame(new_rows, columns=survey_df.columns)
    survey_df.reset_index(drop=True, inplace=True)
    # set values for 'effectivness lower' and 'effectivness upper' bounds rows
    # calculated as follows:
    #   lower boundary:
    #       if expected_value + variance / 2 > 100:
    #           boundary = 100 - variance
    #       else:
    #           if expected_value - variance / 2 < 0:
    #               boundary = 0
    #           else:
    #               boundary = expected_value - variance / 2
    #   upper boundary:
    #       if expected_value - variance / 2 < 0:
    #           boundary = variance
    #       else:
    #           if expected_value + variance / 2 > 100:
    #               boundary = 100
    #           else:
    #               boundary = expected_value + variance / 2
    for i, row in survey_df.iterrows():
        if row['title'] == 'effectivness lower':
            expected_value = survey_df.iloc[i-2][expert_ids]
            variance = survey_df.iloc[i-1][expert_ids]
            reach_upper_limit = expected_value + variance / 2 > 100 # boolean array
            row_values = survey_df.loc[i, expert_ids]
            row_values[reach_upper_limit] = 100 - variance
            row_values[~reach_upper_limit] = expected_value - variance / 2
            row_values[row_values < 0] = 0
            survey_df.loc[i, expert_ids] = row_values
        if row['title'] == 'effectivness upper':
            expected_value = survey_df.iloc[i-3][expert_ids]
            variance = survey_df.iloc[i-2][expert_ids]
            reach_lower_limit = expected_value - variance / 2 < 0   # boolean array
            row_values = survey_df.loc[i, expert_ids]
            row_values[reach_lower_limit] = variance
            row_values[~reach_lower_limit] = expected_value + variance / 2
            row_values[row_values > 100] = 100
            survey_df.loc[i, expert_ids] = row_values

    # Step 3: calculate aggregated mean and variance

    # create individual PERT distributions for each expert
    # draw equal and large number of values from each expert distribution (=picks)
    # amount of picks should be scaled by participating expert weights
    # pool picks together
    # apply a discrete probability distribution to picks
    # from these distributions

    # create a new 'aggregated' column for expected value rows and set their value as the mean of the expert answers per question
    # expert answers are weighted by amount of participating experts per answer
    expected_values = survey_df.loc[survey_df['title'] == 'expected value', np.insert(expert_ids, 0, 'block')]  # select expected value rows
    variances = survey_df.loc[survey_df['title'] == 'variance', np.insert(expert_ids, 0, 'block')]  # select variance rows
    expert_weights = survey_df.loc[survey_df['title'] == 'expert weights', np.insert(expert_ids, 0, 'block')]   # select expert weight rows
    for b_id in block_ids:  # for each block

        expert_weights_block = expert_weights.loc[expert_weights['block'] == b_id, expert_ids]  # weights, should only be one row for each block
        
        # expected values

        # select expected values of block
        expected_values_block = expected_values.loc[expected_values['block'] == b_id, expert_ids]
        # select values that are not nan, bool matrix
        expected_values_non_nan = ~np.isnan(expected_values_block)
        # multiply those values with weights, True = 1 and False = 0
        expected_values_non_nan_weights = expected_values_non_nan * expert_weights_block.reset_index(drop=True).values
        # sum weights to get number of participating experts
        expected_values_non_nan_weights_sum = expected_values_non_nan_weights.sum(axis=1)
        # multiply values by weights and sum
        expected_values_sum = (expected_values_block * expert_weights_block.values).sum(axis=1)
        # divide sum by amount of experts to get mean
        expected_values_aggregated = expected_values_sum / expected_values_non_nan_weights_sum
        # add aggregated values to dataframe
        survey_df.loc[(survey_df['block'] == b_id) & (survey_df['title'] == 'expected value'), 'aggregated'] = expected_values_aggregated

        # variance, same way as above
        
        variances_block = variances.loc[variances['block'] == b_id, expert_ids]
        variances_non_nan = ~np.isnan(variances_block)
        variances_non_nan_weights = variances_non_nan * expert_weights_block.reset_index(drop=True).values
        variances_non_nan_weights_sum = variances_non_nan_weights.sum(axis=1)
        variances_sum = (variances_block * expert_weights_block.values).sum(axis=1)
        variances_aggregated = variances_sum / variances_non_nan_weights_sum
        survey_df.loc[(survey_df['block'] == b_id) & (survey_df['title'] == 'variance'), 'aggregated'] = variances_aggregated

    # Step 4: Update measure and activity ids

    # multiply every measure and activity id with the multiplier
    survey_df['measure'] = survey_df['measure'] * id_multiplier
    survey_df['activity'] = survey_df['activity'] * id_multiplier

    measure_ids = survey_df['measure'].unique() # identify unique measure ids
    measure_ids = measure_ids[~pd.isnull(measure_ids)]  # remove null value ids

    for m_id in measure_ids:    # for every measure
        measures = survey_df.loc[(survey_df['measure'] == m_id) & (survey_df['title'] == 'expected value')]   # select expected value rows for current measure
        indices = measures.index.values # select indices
        
        if len(measures) > 1:   # if there are more than one row for each measure
            for num, id in enumerate(measures['measure']):  # for each measure
                new_id = id + (num + 1)  # the new id is the old one + the current counter + 1
                index = indices[num]    # select the index of the current measure row
                # set new measure id for both expected value and variance rows
                survey_df.loc[index, 'measure'] = new_id
                survey_df.loc[index+1, 'measure'] = new_id

    # Step 5: Scaling factor removed

    survey_df = survey_df.loc[survey_df['title'] != 'max effectivness']     # remove scaling factor from survey data
    # survey_df = survey_df.loc[survey_df['title'] != 'effectivness lower']     # remove lower effectivness boundary for now
    # survey_df = survey_df.loc[survey_df['title'] != 'effectivness upper']     # remove upper effectivness boundary for now

    return survey_df


def read_core_object_descriptions(file_name: str) -> dict[str, dict]:
    """
    Reads in model object descriptions from general input files

    - Core object descriptions
    - Model domain descriptions
    - Case descriptions
    - Linkage descriptions

    Arguments:
        file_name (str): source excel file name containing 
            'Measure type list', 'Activity list', 'Pressure list', 'State list' in sheets

    Returns:
        object_data (dict): dictionary containing measure, activity, pressure and state ids and descriptions in separate sub-dictionaries
    """
    general_input = pd.read_excel(io=file_name, sheet_name=None)    # read excel file into DataFrame

    def create_dict(sheet_name, id_col_name, obj_col_name):
        df = general_input[sheet_name]
        obj_dict = {}
        [obj_dict.update({id: name}) if isinstance(name, str) else obj_dict.update({id: None}) for id, name in zip(df[id_col_name], df[obj_col_name])]
        return obj_dict

    # Create dict for measures
    # convert id number muptiplying it by 1000
    measures_dict = create_dict('Measure type list', 'ID', 'Measure type')

    # Create dict for activities
    # convert id number muptiplying it by 1000
    activities_dict = create_dict('Activity list', 'ID', 'Activity')

    # Create dict for pressures
    # id number does not need converting
    pressure_dict = create_dict('Pressure list', 'ID', 'Pressure')

    # Create dict for states
    # id number does not need converting
    state_dict = create_dict('State list', 'ID', 'States')

    object_data = {
        'measure': measures_dict,
        'activity': activities_dict,
        'pressure': pressure_dict,
        'state': state_dict,
    }

    return object_data


def read_domain_input(file_name: str, countries_exclude: list[str], basins_exclude: list[str]) -> dict[str, pd.DataFrame]:
    """
    Reads in calculation domain descriptions

    Arguments:
        file_name (str): source excel file name containing 
            'CountBas', 'Country list' and 'Basin list' sheets
        countries_exclude (list): list of countries to exclude
        basins_exclude (list): list of basins to exclude
    
    Returns:
        domain (dict): {
            countries_by_basins (DataFrame): area fractions of basins (column) per country (row)
            countries (DataFrame): country ids
            basins (DataFrame): basin ids
        }
    """
    # countries
    sheet_name = 'Country list'
    countries = pd.read_excel(io=file_name, sheet_name=sheet_name, index_col='ID')  # note that column 'ID' is changed to dataframe index
    countries = countries[~np.isin(countries.values, countries_exclude)]    # remove rows to be excluded

    # basins
    sheet_name = 'Basin list'
    basins = pd.read_excel(io=file_name, sheet_name=sheet_name, index_col='ID') # note that column 'ID' is changed to dataframe index
    basins = basins[~np.isin(basins.values, basins_exclude)]    # remove rows to be excluded

    # country-basin links
    sheet_name = 'CountBas'
    countries_by_basins = pd.read_excel(io=file_name, sheet_name=sheet_name)
    countries_by_basins = countries_by_basins[np.isin(countries_by_basins['ID'], countries.index)]  # remove excluded countries
    countries_by_basins = countries_by_basins.drop(columns=[x for x in countries_by_basins.columns if x not in basins.index])   # remove excluded basins (+ ID column)

    domain = {
        'countries_by_basins': countries_by_basins,
        'countries': countries,
        'basins': basins
    }

    return domain


def read_case_input(file_name: str) -> pd.DataFrame:
    """
    Reading in and processing data for cases. Each row represents one case. 
    
    In columns of 'ActMeas' sheet ('in_Activities', 'in_Pressure' and 'In_State_components') the value 0 == 'all relevant'.
    Relevant activities, pressures and state can be found in measure-wise from 'MT_to_A_to_S' sheets (linkages)
    
    - multiply MT_ID id by 10000 to get right measure_id
    - multiply In_Activities ids by 10000 to get right activity_id
    - multiply B_ID by 1000 to get right basin_id

    Arguments:
        file_name (str): name of source excel file name containing 'ActMeas' sheet

    Returns:
        cases (DataFrame): case data
    """
    sheet_name = 'ActMeas'
    cases = pd.read_excel(io=file_name, sheet_name=sheet_name)
    
    # separate activities grouped together in sheet on the same row with ';' into separate rows
    cases['In_Activities'] = [list(filter(None, x.split(';'))) if type(x) == str else x for x in cases['In_Activities']]
    cases = cases.explode('In_Activities')

    # separate pressures grouped together in sheet on the same row with ';' into separate rows
    cases['In_Pressure'] = [list(filter(None, x.split(';'))) if type(x) == str else x for x in cases['In_Pressure']]
    cases = cases.explode('In_Pressure')

    # separate basins grouped together in sheet on the same row with ';' into separate rows
    cases['B_ID'] = [list(filter(None, x.split(';'))) if type(x) == str else x for x in cases['B_ID']]
    cases = cases.explode('B_ID')

    # separate countries grouped together in sheet on the same row with ';' into separate rows
    cases['C_ID'] = [list(filter(None, x.split(';'))) if type(x) == str else x for x in cases['C_ID']]
    cases = cases.explode('C_ID')

    # In_State_components is in input data just for book keeping
    cases['In_State_components'] = [list(filter(None, x.split(';'))) if type(x) == str else x for x in cases['In_State_components']]
    # cases = cases.explode('In_State_components')
    # cases['In_State_components'].astype('int')

    # change types of split values from str to int
    cases = cases.astype({
        'In_Activities': 'int',
        'In_Pressure': 'int',
        'B_ID': 'int',
        'C_ID': 'int'
    })

    # create new column 'countrybasin_id' to link basins and countries, and create the unique ids
    cases['countrybasin_id'] = cases['B_ID'] * 1000 + cases['C_ID']

    return cases


def read_linkage_descriptions(file_name: str):
    """
    Reads description of links between Measures, Activities, Pressures, and States.

     Arguments:
        file_name (str): name of source excel file name containing 'ActMeas' sheet
        sheet_name (str): name of sheet in source excel ('ActMeas')

    Returns:
        linkages (DataFrame): dataframe containing mappings between measures to actitivities to pressures to states
    """
    sheet_name = 'MT_to_A_to_S'
    linkages = pd.read_excel(io=file_name, sheet_name=sheet_name)
    
    # separate measures grouped together in sheet on the same row with ';' into separate rows
    linkages['MT'] = [list(filter(None, x.split(';'))) if type(x) == str else x for x in linkages['MT']]
    linkages = linkages.explode('MT')
    linkages['MT'].notna().astype('int')    # convert non-nan values to int

    # separate activities grouped together in sheet on the same row with ';' into separate rows
    linkages['Activities'] = [list(filter(None, x.split(';'))) if type(x) == str else x for x in linkages['Activities']]
    linkages = linkages.explode('Activities')
    linkages['Activities'].notna().astype('int')    # convert non-nan values to int

    # separate pressures grouped together in sheet on the same row with ';' into separate rows
    linkages['Pressure'] = [list(filter(None, x.split(';'))) if type(x) == str else x for x in linkages['Pressure']]
    linkages = linkages.explode('Pressure')
    linkages['Pressure'].notna().astype('int')  # convert non-nan values to int

    # separate states grouped together in sheet on the same row with ';' into separate rows
    linkages['State (if needed)'] = [list(filter(None, x.split(';'))) if type(x) == str else x for x in linkages['State (if needed)']]
    linkages = linkages.explode('State (if needed)')
    linkages['State (if needed)'].notna().astype('int') # convert non-nan values to int

    return linkages


def read_postprocess_data(file_name: str) -> pd.DataFrame:
    """
    Reads input data of activities to pressures in Baltic Sea basins. 

    Arguments:
        file_name (str): name of source excel file name containing 'ActPres' sheet

    Returns:
        act_to_press (DataFrame): dataframe containing mappings between activities and pressures
    """
    sheet_name = 'ActPres'
    act_to_press = pd.read_excel(file_name, sheet_name=sheet_name)

    # read all most likely, min and max column values into lists in new columns
    act_to_press['expected'] = act_to_press.filter(regex='Ml[1-6]').values.tolist()
    act_to_press['minimun'] = act_to_press.filter(regex='Min[1-6]').values.tolist()
    act_to_press['maximum'] = act_to_press.filter(regex='Max[1-6]').values.tolist()

    # remove all most likely, min and max columns
    act_to_press.drop(act_to_press.filter(regex='Ml[1-6]').columns, axis=1, inplace=True)
    act_to_press.drop(act_to_press.filter(regex='Min[1-6]').columns, axis=1, inplace=True)
    act_to_press.drop(act_to_press.filter(regex='Max[1-6]').columns, axis=1, inplace=True)

    # separate basins grouped together in sheet on the same row with ';' into separate rows
    act_to_press['Basins'] = [list(filter(None, x.split(';'))) if type(x) == str else x for x in act_to_press['Basins']]
    act_to_press = act_to_press.explode('Basins')

    return act_to_press


def pert_dist(peak, low, high, size) -> np.ndarray:
    '''
    Returns a set of random picks from a PERT distribution.
    '''
    # weight, controls probability of edge values (higher -> more emphasis on most likely, lower -> extreme values more probable)
    # 4 is standard used in unmodified PERT distributions
    gamma = 4
    # calculate expected value
    # mu = ((low + gamma) * (peak + high)) / (gamma + 2)
    r = high - low
    alpha = 1 + gamma * (peak - low) / r
    beta = 1 + gamma * (high - peak) / r
    return low + np.random.default_rng().beta(alpha, beta, size=int(size)) * r


def get_prob_dist(expecteds: pd.DataFrame, 
                  lower_boundaries: pd.DataFrame, 
                  upper_boundaries: pd.DataFrame, 
                  weights: pd.DataFrame) -> np.ndarray:
    '''
    Returns a cumulative probability distribution.
    '''
    # select values that are not nan, bool matrix
    non_nan = ~np.isnan(expecteds) & ~np.isnan(lower_boundaries) & ~np.isnan(upper_boundaries)
    # multiply those values with weights, True = 1 and False = 0
    weights_non_nan = (non_nan.values * weights.values).flatten()

    # create a PERT distribution for each expert
    # from each distribution, draw a large number of picks
    # pool the picks together
    number_of_picks = 200
    picks = []
    for i in range(len(expecteds)):
        peak = expecteds.values[i]
        low = lower_boundaries.values[i]
        high = upper_boundaries.values[i]
        w = weights_non_nan[i]
        if None in [peak, low, high, w]:
            continue    # skip if any value is None
        dist = pert_dist(peak, low, high, w * number_of_picks)
        picks += dist.tolist()
    
    # fit picks to discrete distribution
    # the distribution has 100 elements
    # every element at index i represents the probability of a value below i percent
    picks = np.array(picks)
    disc_dist = np.zeros(shape=100)
    for i in range(disc_dist.size):
        disc_dist[i] = np.sum(picks < i) / picks.size
    
    print(disc_dist)
    exit()

    return disc_dist


#EOF
"""
Copyright (c) 2024 Baltic Marine Environment Protection Commission
Copyright (c) 2022 Antti-Jussi Kieloaho (Natural Resources Institute Finland)

LICENSE available under 
local: 'SOM/protect_baltic/LICENSE'
url: 'https://github.com/helcomsecretariat/SOM/blob/main/protect_baltic/LICENCE'
"""

import numpy as np
import pandas as pd
import warnings     # for suppressing deprecated warnings
import traceback

def read_survey_data(file_name: str, sheet_names: dict[int, str]) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """
    Measure survey data: Part 1

    Arguments:
        file_name (str): path of survey excel file
        sheet_names (dict): dict of survey sheet ids in file_name
    
    Return:
        info (DataFrame): survey data information
        survey_data (dict): survey data
    """
    # read information sheet from input Excel file
    info = pd.read_excel(io=file_name, sheet_name=sheet_names[0])

    measure_survey_data = {}
    for id, sheet in enumerate(sheet_names.values()):
        # skip if information sheet
        if id == 0:
            continue
        # read data sheet from input Excel file, set header to None to include top row in data
        measure_survey_data[id] = pd.read_excel(io=file_name, sheet_name=sheet, header=None)

    return info, measure_survey_data


def preprocess_measure_survey_data(mteq: pd.DataFrame, measure_survey_data: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Measure survey data: Part 2

    This method parses the survey sheets from webropol

    Arguments:
        mteq (DataFrame): survey data information
        measure_survey_data (dict): survey data
    
    Return:
        survey_df (DataFrame): survey data information
            survey_id: unique id for each questionnaire / survey
            title: type of value ('expected value' / 'variance' / 'max effectiveness' / 'expert weights')
            block: id for each set of questions within each survey, unique across all surveys
            measure: measure id of the question
            activity: activity id of the question
            pressure: pressure id of the question
            state: state id (if defined) of the question ([nan] if no state)
            1...n (=expert ids): answer value for each expert (NaN if no answer)
    """
    # preprocess values
    mteq['Direct_to_state'] = [x.split(';') if type(x) == str else x for x in mteq['Direct_to_state']]

    # create new dataframe
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
            titles.append('max effectiveness')
            titles.append('expert weights')

            measures = measure_survey_data[survey_id].iloc[0, start:end].tolist() # select current question column names as measure ids
            measures.append(np.nan) # append NaN for max effectiveness (ME)
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

            with warnings.catch_warnings(action='ignore'):
                survey_df = pd.concat([survey_df, data], ignore_index=True, sort=False)
            block_number = block_number + 1

    return survey_df


def get_expert_ids(df: pd.DataFrame) -> list:
    '''
    Returns list of expert id column names from dataframe using regex
    '''
    return df.filter(regex='^(100|[1-9]?[0-9])$').columns


def process_measure_survey_data(survey_df: pd.DataFrame) -> pd.DataFrame:
    r'''
    Measure survey data: part 3

    - Adjust expert answers by scaling factor
    - Calculate effectiveness range boundaries
    - New id for 'measure' and 'activity' by multiplying id by 10000
        This is done as we need to track specific measure-activity-pressure and measure-state combinations
        'pressure' and 'state' id:s are not multiplied!
    - Calculate probability distributions
    - Remove rows and columns that are not needed anymore

    Arguments:
        survey_df (DataFrame): survey data information
            survey_id: unique id for each questionnaire / survey
            title: type of value ('expected value' / 'variance' / 'max effectiveness' / 'expert weights')
            block: id for each set of questions within each survey, unique across all surveys
            measure: measure id of the question
            activity: activity id of the question
            pressure: pressure id of the question
            state: state id (if defined) of the question ([nan] if no state)
            1...n (=expert ids): answer value for each expert (NaN if no answer)
    Returns:
        survey_df (DataFrame): processed survey data information
            measure: measure id
            activity: activity id
            pressure: pressure id
            state: state id (if defined, [nan] if no state)
            cumulative probability: cum. prob. distribution represented as list
    '''
    # select column names corresponding to expert ids (any number between 1 and 100)
    expert_ids = get_expert_ids(survey_df)

    #
    # Adjust answers by scaling factor
    #

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
                # find the max effectiveness estimated by the expert
                max_effectiveness = block.loc[block['title']=='max effectiveness', col].values[0]
                # calculate scaling factor
                if np.isnan(max_effectiveness):
                    # set all values to null if no max effectiveness (in column, for current block)
                    survey_df.loc[survey_df['block'] == b_id, col] = np.nan
                elif max_effectiveness == 0 or max_expected_value == 0:
                    # scale all expected values to 0 if max effectiveness is zero or all expected values are zero
                    survey_df.loc[(survey_df['block'] == b_id) & (survey_df['title'] == 'expected value'), col] = 0
                else:
                    # get the scaling factor
                    scaling_factor = np.divide(max_expected_value, max_effectiveness)
                    # divide the expected values by the new scaling factor
                    survey_df.loc[(survey_df['block'] == b_id) & (survey_df['title'] == 'expected value'), col] = np.divide(expected_value, scaling_factor)

    #
    # Calculate effectiveness range boundaries
    #

    # create new rows for 'effectiveness lower' and 'effectiveness upper' bounds after variance rows
    new_rows = []
    for i, row in survey_df.iterrows():
        new_rows.append(row)
        if row['title'] == 'variance':
            # create lower bound
            min_row = survey_df.loc[i].copy()
            min_row['title'] = 'effectiveness lower'
            min_row[expert_ids] = np.nan
            new_rows.append(min_row)
            # create upper bound
            max_row = survey_df.loc[i].copy()
            max_row['title'] = 'effectiveness upper'
            max_row[expert_ids] = np.nan
            new_rows.append(max_row)
    survey_df = pd.DataFrame(new_rows, columns=survey_df.columns)
    survey_df.reset_index(drop=True, inplace=True)
    # set values for 'effectiveness lower' and 'effectiveness upper' bounds rows
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
        if row['title'] == 'effectiveness lower':
            expected_value = survey_df.iloc[i-2][expert_ids]
            variance = survey_df.iloc[i-1][expert_ids]
            reach_upper_limit = expected_value + variance / 2 > 100 # boolean array
            row_values = survey_df.loc[i, expert_ids]
            row_values[reach_upper_limit] = 100 - variance
            row_values[~reach_upper_limit] = expected_value - variance / 2
            row_values[row_values < 0] = 0
            survey_df.loc[i, expert_ids] = row_values
        if row['title'] == 'effectiveness upper':
            expected_value = survey_df.iloc[i-3][expert_ids]
            variance = survey_df.iloc[i-2][expert_ids]
            reach_lower_limit = expected_value - variance / 2 < 0   # boolean array
            row_values = survey_df.loc[i, expert_ids]
            row_values[reach_lower_limit] = variance
            row_values[~reach_lower_limit] = expected_value + variance / 2
            row_values[row_values > 100] = 100
            survey_df.loc[i, expert_ids] = row_values

    #
    # Update measure and activity ids
    #

    # id_multiplier = 10000

    # # multiply every measure and activity id with the multiplier
    # survey_df['measure'] = survey_df['measure'] * id_multiplier
    # survey_df['activity'] = survey_df['activity'] * id_multiplier

    # measure_ids = survey_df['measure'].unique() # identify unique measure ids
    # measure_ids = measure_ids[~pd.isnull(measure_ids)]  # remove null value ids

    # for m_id in measure_ids:    # for every measure
    #     measures = survey_df.loc[(survey_df['measure'] == m_id) & (survey_df['title'] == 'expected value')]   # select expected value rows for current measure
    #     indices = measures.index.values # select indices
        
    #     if len(measures) > 1:   # if there are more than one row for each measure
    #         for num, id in enumerate(measures['measure']):  # for each measure
    #             new_id = id + (num + 1)  # the new id is the old one + the current counter + 1
    #             index = indices[num]    # select the index of the current measure row
    #             # set new measure id for both expected value and variance rows
    #             survey_df.loc[index, 'measure'] = new_id
    #             survey_df.loc[index+1, 'measure'] = new_id
    #             survey_df.loc[index+2, 'measure'] = new_id
    #             survey_df.loc[index+3, 'measure'] = new_id

    #
    # Calculate probability distributions
    #

    # add a new column for the probability
    survey_df['cumulative probability'] = pd.Series([np.nan] * len(survey_df), dtype='object')

    # access expert answer columns, separate rows by type of answer
    expecteds = survey_df[expert_ids].loc[survey_df['title'] == 'expected value']
    lower_boundaries = survey_df[expert_ids].loc[survey_df['title'] == 'effectiveness lower']
    upper_boundaries = survey_df[expert_ids].loc[survey_df['title'] == 'effectiveness upper']
    weights = survey_df.loc[survey_df['title'] == 'expert weights', np.insert(expert_ids, 0, 'block')]
    blocks = survey_df['block'].loc[(survey_df['title'] == 'expected value')]
    # go through each measure-activity-pressure link
    for num in expecteds.index:
        # access current row data and convert to 1-D arrays
        b_id = blocks.loc[num]
        e = expecteds.loc[num].to_numpy().astype(float)
        l = lower_boundaries.loc[num+2].to_numpy().astype(float)
        u = upper_boundaries.loc[num+3].to_numpy().astype(float)
        w = weights.loc[weights['block'] == b_id, expert_ids].to_numpy().astype(float).flatten()
        # get expert probability distribution
        prob_dist = get_prob_dist(expecteds=e, 
                                  lower_boundaries=l, 
                                  upper_boundaries=u, 
                                  weights=w)
        
        survey_df.at[num, 'cumulative probability'] = prob_dist

    #
    # Remove rows and columns that are not needed anymore
    #

    for title in ['max effectiveness', 'variance', 'effectiveness lower', 'effectiveness upper', 'expert weights']:
        survey_df = survey_df.loc[survey_df['title'] != title]
    survey_df = survey_df.drop(columns=expert_ids)
    survey_df = survey_df.drop(columns=['survey_id', 'title', 'block'])

    #
    # Split states into separate rows, and finally reset index
    #

    survey_df = survey_df.explode(column='state')
    survey_df = survey_df.reset_index(drop=True)

    #
    # Replace nan values with zeros and convert columns to integers
    #

    for column in ['measure', 'activity', 'pressure', 'state']:
        with warnings.catch_warnings(action='ignore'):
            survey_df[column] = survey_df[column].fillna(0)
        survey_df[column] = survey_df[column].astype(int)

    return survey_df


def preprocess_pressure_survey_data(psq: pd.DataFrame, pressure_survey_data: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Pressure survey data: Part 2

    This method parses the survey sheets

    Arguments:
        psq (DataFrame): survey data information
        pressure_survey_data (dict): survey data
    
    Return:
        survey_df (DataFrame): survey data information
            survey_id: unique id for each questionnaire / survey
            question_id: unique id for each question across questionnaires
            State: state id
            Basins: basin id
            GES known: 0 or 1, is the GES threshold known
            Weight: amount of experts participating in answer
            Expert: expert id
            PX: pressure id of pressure X
            SX: significance of pressure X
            MIN, MAX, ML: minimum, maximum, most likely pressure reduction required to reach GES, 10, 25 and 50 % improvement
    """
    # preprocess values
    psq['Basins'] = [x.split(';') if type(x) == str else x for x in psq['Basins']]
    psq['Countries'] = [x.split(';') if type(x) == str else x for x in psq['Countries']]

    # add question id column
    psq['question_id'] = list(range(len(psq)))

    # create new dataframe
    survey_df = pd.DataFrame(columns=['survey_id', 'question_id', 'State', 'Basins', 'Countries', 'GES known', 'Weight'])
    
    # survey columns from which to take data
    cols = ['Expert']
    cols += [x + str(i+1) for x in ['P', 'S'] for i in range(6)]    # up to 6 different pressures related to state, and their significance
    cols += [x + y for x in ['MIN', 'MAX', 'ML'] for y in ['PR', '10', '25', '50']]     # required pressure reduction to reach GES (if known) or X % improvement in state

    start = 0    # keep track of where to access data in psq

    # for every survey sheet
    for survey_id in pressure_survey_data:

        # set first row as header
        pressure_survey_data[survey_id].columns = pressure_survey_data[survey_id].iloc[0].values
        pressure_survey_data[survey_id].drop(index=0, axis=0, inplace=True)

        # identify amount of experts in survey
        expert_ids = pressure_survey_data[survey_id]['Expert'].unique()
        # identify amount of questions in survey
        questions = np.sum(pressure_survey_data[survey_id]['Expert'] == expert_ids[0])

        # use number of questions to get state, basins and GES known
        question_id = psq['question_id'].iloc[start:start+questions].reset_index(drop=True)
        state = psq['State'].iloc[start:start+questions].reset_index(drop=True)
        basins = psq['Basins'].iloc[start:start+questions].reset_index(drop=True)
        countries = psq['Countries'].iloc[start:start+questions].reset_index(drop=True)
        ges_known = psq['GES known'].iloc[start:start+questions].reset_index(drop=True)

        # find all expert weight columns and values
        expert_cols = [True if 'exp' in col.lower() else False for col in psq.columns]
        expert_weights = psq.loc[start:start+questions, expert_cols].reset_index(drop=True)
        expert_weights = expert_weights.fillna(1)

        survey_answers = 0
        for expert in expert_ids:

            # select expert answers
            data = pressure_survey_data[survey_id][cols].loc[pressure_survey_data[survey_id][cols]['Expert'] == expert].reset_index(drop=True)

            # verify that the amount of answers is correct
            if len(data) != questions: raise Exception('Not same amount of answers for each expert in survey sheet!')

            survey_answers += len(data)
            
            # set survey id, state, basins and GES known for data
            data['survey_id'] = survey_id
            data['question_id'] = question_id
            data['State'] = state
            data['Basins'] = basins
            data['Countries'] = countries
            data['GES known'] = ges_known

            # set expert weights
            data['Weight'] = expert_weights['Exp' + str(int(expert))]

            # add data to final dataframe
            with warnings.catch_warnings(action='ignore'):
                survey_df = pd.concat([survey_df, data], ignore_index=True, sort=False)
        
        # verify that the correct number of answers was saved
        if survey_answers != len(expert_ids) * questions: raise Exception('Incorrect amount of answers found for survey!')

        # increase counter
        start += questions

    return survey_df


def process_pressure_survey_data(survey_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pressure survey data: part 3
    """
    # create new dataframe for merged rows
    cols = ['survey_id', 'question_id', 'State', 'Basins', 'Countries', 'GES known']
    new_df = pd.DataFrame(columns=cols+['Pressures', 'Averages', 'Uncertainties']+['PR', '10', '25', '50'])
    # remove empty elements from Basins and Countries, and convert ids to integers
    survey_df['Basins'] = survey_df['Basins'].apply(lambda x: [int(basin) for basin in x if basin != ''])
    survey_df['Countries'] = survey_df['Countries'].apply(lambda x: [int(country) for country in x if country != ''])
    # identify all unique questions
    questions = survey_df['question_id'].unique()
    # process each state
    for question in questions:
        # select current question rows
        data = survey_df.loc[survey_df['question_id'] == question].reset_index(drop=True)
        #
        # pressure contributions and uncertainties
        #
        # select pressures and significances, and find non nan values
        pressures = data[['P'+str(x+1) for x in range(6)]].to_numpy().astype(float)
        significances = data[['S'+str(x+1) for x in range(6)]].to_numpy().astype(float)
        mask = ~np.isnan(pressures)
        # weigh significancs by amount of participating experts
        w = data[['Weight']].to_numpy().astype(float)
        significances = significances * w
        # go through each expert answer and calculate weights
        weights = {}
        for i, e in enumerate(pressures):
            s_tot = np.sum(significances[i][mask[i]])
            for p, s in zip(pressures[i][mask[i]], significances[i][mask[i]]):
                if int(p) not in weights:
                    weights[int(p)] = []
                weights[int(p)].append(s / s_tot)
        # using weights, calculate contributions and uncertainties
        average = {p: np.mean(weights[p]) for p in weights}
        stddev = {p: np.std(weights[p]) for p in weights}
        # scale contributions and uncertainties so contributions sum up to 100 %
        factor = np.sum([average[p] for p in average])
        average = {p: average[p] / factor for p in average}
        stddev = {p: stddev[p] / factor for p in stddev}
        # round values
        decimals = 4
        average = {p: np.round(average[p], decimals) for p in average}
        stddev = {p: np.round(stddev[p], decimals) for p in stddev}
        # convert to lists
        pressures = list(average.keys())
        average = [average[p] for p in pressures]
        stddev = [stddev[p] for p in pressures]
        #
        # probability distributions for pressure reductions
        #
        reductions = {}
        for r in ['PR', '10', '25', '50']:
            # get min, max and ml data
            r_min = data['MIN'+r].to_numpy().astype(float)
            r_max = data['MAX'+r].to_numpy().astype(float)
            r_ml = data['ML'+r].to_numpy().astype(float)
            # get weighted cumulative probability distribution
            dist = get_prob_dist(r_ml, r_min, r_max, w.flatten())
            reductions[r] = dist
        #
        # merge processed data with dataframe
        #
        # create a new dataframe row and merge with new dataframe
        data = survey_df[cols].loc[survey_df['question_id'] == question].reset_index(drop=True).iloc[0]
        data = pd.DataFrame([data])
        # initialize new columns
        for c in ['Pressures', 'Averages', 'Uncertainties']+['PR', '10', '25', '50']:
            data[c] = np.nan
            data[c] = data[c].astype(object)
        # change data type to allow for lists
        data.at[0, 'Pressures'] = pressures
        data.at[0, 'Averages'] = average
        data.at[0, 'Uncertainties'] = stddev
        for r in ['PR', '10', '25', '50']:
            data.at[0, r] = reductions[r]
        with warnings.catch_warnings(action='ignore'):
            new_df = pd.concat([new_df, data], ignore_index=True, sort=False)
    #
    # get probability from distribution for each of the thresholds
    #
    for col in ['PR', '10', '25', '50']:
        new_df[col] = new_df[col].apply(lambda x: get_pick(x) if not np.any(np.isnan(x)) else np.nan)
    #
    # explode basin and country columns and create area_id
    #
    for col in ['Basins', 'Countries']:
        new_df = new_df.explode(col)
    new_df = new_df.reset_index(drop=True)
    new_df['area_id'] = None
    for i, row in new_df.iterrows():
        new_df.at[i, 'area_id'] = (row['Basins'], row['Countries'])
    new_df = new_df.drop(columns=['Basins', 'Countries'])
    #
    # split pressures into separate rows
    #
    new_df = new_df.assign(pressure=[list(zip(*row)) for row in zip(new_df['Pressures'], new_df['Averages'], new_df['Uncertainties'])])
    new_df = new_df.explode('pressure')
    new_df = new_df.reset_index(drop=True)
    new_df = new_df.drop(columns=['Pressures', 'Averages', 'Uncertainties'])
    new_df[['pressure', 'average', 'uncertainty']] = pd.DataFrame(new_df['pressure'].tolist())
    #
    # remove rows with missing data (no pressure or no thresholds)
    #
    new_df = new_df.loc[~new_df['pressure'].isna(), :]
    mask = (new_df['PR'].isna()) & (new_df['10'].isna()) & (new_df['25'].isna()) & (new_df['50'].isna())
    new_df = new_df.loc[~mask, :]
    #
    # final steps
    #
    new_df['pressure'] = new_df['pressure'].astype(int)
    new_df = new_df.drop(columns=['survey_id', 'question_id', 'GES known'])

    return new_df


def read_core_object_descriptions(file_name: str, id_sheets: dict) -> dict[str, dict]:
    """
    Reads in model object descriptions from general input files

    Arguments:
        file_name (str): source excel file name containing measure, activity, pressure and state id sheets
        id_sheets (dict): should have structure {'measure': sheet_name, 'activity': sheet_name, ...}

    Returns:
        object_data (dict): dictionary containing measure, activity, pressure and state ids and descriptions in separate sub-dictionaries
    """
    # create dicts for each category
    object_data = {}
    for category in id_sheets:
        # read excel sheet into dataframe
        df = pd.read_excel(io=file_name, sheet_name=id_sheets[category])
        # remove non-necessary columns
        df.drop(columns=[col for col in df.columns if col not in ['ID', category]])
        # remove rows where id is nan or empty string
        df = df.dropna(subset=['ID'])
        df = df[df['ID'] != '']
        # convert id column to integer (if not already)
        df['ID'] = df['ID'].astype(int)
        object_data[category] = df
        
        # # convert to dict
        # obj_dict = {}
        # [obj_dict.update({id: name}) if isinstance(name, str) else obj_dict.update({id: None}) for id, name in zip(df['ID'], df[category])]
        # object_data[category] = obj_dict

    return object_data


def read_domain_input(file_name: str, id_sheets: dict, countries_exclude: list[str], basins_exclude: list[str]) -> dict[str, pd.DataFrame]:
    """
    Reads in calculation domain descriptions

    Arguments:
        file_name (str): source excel file name containing country, basin and country-basin sheets
        id_sheets (dict): dict containing sheet names
        countries_exclude (list): list of countries to exclude
        basins_exclude (list): list of basins to exclude
    
    Returns:
        domain (dict): {
            countries_by_basins (DataFrame): area fractions of basins (column) per country (row)
            countries (DataFrame): country ids
            basins (DataFrame): basin ids
        }
    """
    # create dicts for each category
    domain = {}
    for category in id_sheets:
        # read excel sheet into dataframe
        df = pd.read_excel(io=file_name, sheet_name=id_sheets[category])
        # remove rows where id is nan or empty string
        df = df.dropna(subset=['ID'])
        df = df[df['ID'] != '']
        # convert id column to integer (if not already)
        df['ID'] = df['ID'].astype(int)
        domain[category] = df
    
    # process data
    for category in domain:
        if category == 'countries_by_basins':
            # remove excluded countries
            domain[category] = domain[category][np.isin(domain[category]['ID'], domain['country'].index)]
            # remove excluded basins (+ ID column)
            domain[category] = domain[category].drop(columns=[x for x in domain[category].columns if x not in domain['basin'].index])
        if category == 'country':
            # remove rows to be excluded
            domain[category] = domain[category][~np.isin(domain[category][category], countries_exclude)]
            domain[category] = domain[category].set_index('ID')
        if category == 'basin':
            # remove rows to be excluded
            domain[category] = domain[category][~np.isin(domain[category][category], basins_exclude)]
            domain[category] = domain[category].set_index('ID')

    return domain


def read_case_input(file_name: str, sheet_name: str) -> pd.DataFrame:
    """
    Reading in and processing data for cases. Each row represents one case. 
    
    In columns of 'ActMeas' sheet ('activities', 'pressure' and 'state') the value 0 == 'all relevant'.

    Arguments:
        file_name (str): name of source excel file name
        sheet_name (str): name of excel sheet

    Returns:
        cases (DataFrame): case data
    """
    cases = pd.read_excel(io=file_name, sheet_name=sheet_name)

    assert len(cases[cases.duplicated(['ID'])]) == 0

    for col in ['activity', 'pressure', 'state', 'basin', 'country']:
        # separate ids grouped together in sheet on the same row with ';' into separate rows
        cases[col] = [list(filter(None, x.split(';'))) if type(x) == str else x for x in cases[col]]
        cases = cases.explode(col)
        # change types of split values from str to int
        cases[col] = cases[col].astype(int)
    
    for col in ['coverage', 'implementation']:
        cases[col] = cases[col].astype(float)

    cases = cases.reset_index()

    # create new column 'area_id' to link basins and countries, and create the unique ids
    cases['area_id'] = None
    for i, row in cases.iterrows():
        cases.at[i, 'area_id'] = (row['basin'], row['country'])

    return cases


def read_postprocess_data(file_name: str, sheet_name: str) -> pd.DataFrame:
    """
    Reads input data of activities to pressures in Baltic Sea basins. 

    Arguments:
        file_name (str): name of source excel file name containing 'ActPres' sheet

    Returns:
        act_to_press (DataFrame): dataframe containing mappings between activities and pressures
    """
    act_to_press = pd.read_excel(file_name, sheet_name=sheet_name)

    # read all most likely, min and max column values into lists in new columns
    for col, regex_str in zip(['expected', 'minimum', 'maximum'], ['Ml[1-6]', 'Min[1-6]', 'Max[1-6]']):
        act_to_press[col] = act_to_press.filter(regex=regex_str).values.tolist()

    # remove all most likely, min and max columns
    for regex_str in ['Ml[1-6]', 'Min[1-6]', 'Max[1-6]']:
        act_to_press.drop(act_to_press.filter(regex=regex_str).columns, axis=1, inplace=True)

    # separate basins grouped together in sheet on the same row with ';' into separate rows
    act_to_press['Basins'] = [list(filter(None, x.split(';'))) if type(x) == str else x for x in act_to_press['Basins']]
    act_to_press = act_to_press.explode('Basins')

    # process each column
    for category in ['Activity', 'Pressure', 'Basins']:
        # split each id merged with ';', the column value becomes a list (unless it is already integer)
        act_to_press[category] = [[y for y in x.split(';') if y != ''] if type(x) == str else x for x in act_to_press[category]]
        # find empty column value lists, replace with nan
        f = lambda x: np.nan if type(x) == list and len(x) == 0 else x
        act_to_press[category] = [f(x) for x in act_to_press[category]]
        # explode column lists into separate rows
        act_to_press = act_to_press.explode(category)
        # convert non-nan values to int
        act_to_press.loc[act_to_press[category].notna(), category] = act_to_press.loc[act_to_press[category].notna(), category].astype(int)

    act_to_press = act_to_press.reset_index(drop=True)

    # calculate probability distributions
    act_to_press['cumulative probability'] = pd.Series([np.nan] * len(act_to_press), dtype='object')
    for num in act_to_press.index:
        # convert expert answers to array
        expected = np.array(list(act_to_press.loc[num, ['expected']])).flatten()
        lower = np.array(list(act_to_press.loc[num, ['minimum']])).flatten()
        upper = np.array(list(act_to_press.loc[num, ['maximum']])).flatten()
        weights = np.full(len(expected), 1)
        # if boundaries are unknown, set to same as expected
        lower[np.isnan(lower)] = expected[np.isnan(lower)]
        upper[np.isnan(upper)] = expected[np.isnan(upper)]
        # get probability distribution
        prob_dist = get_prob_dist(expected, lower, upper, weights)
        act_to_press.at[num, 'cumulative probability'] = prob_dist

    act_to_press = act_to_press.drop(columns=['expected', 'minimum', 'maximum'])

    act_to_press['value'] = act_to_press['cumulative probability'].apply(get_pick)

    return act_to_press


def read_development_scenarios(file_name: str, sheet_name: str) -> pd.DataFrame:
    """
    Reads input data of activity development scnearios. 

    Arguments:
        file_name (str): name of source excel file name
        sheet_name (str): name of sheet in excel file

    Returns:
        development_scenarios (DataFrame): dataframe containing activity development scenarios
    """
    development_scenarios = pd.read_excel(file_name, sheet_name=sheet_name)

    # replace nan values with 0, assuming that no value means no change
    for category in ['Scenario BAU', 'Scenario low change', 'Scenario most likely change', 'Scenario high change']:
        development_scenarios.loc[np.isnan(development_scenarios[category]), category] = 0
    
    development_scenarios['Activity'] = development_scenarios['Activity'].astype(int)

    return development_scenarios


def read_overlaps(file_name: str, sheet_name: str) -> pd.DataFrame:
    """
    Reads input data of measure-measure interactions. 

    Arguments:
        file_name (str): name of source excel file name
        sheet_name (str): name of sheet in excel file

    Returns:
        overlaps (DataFrame): dataframe containing overlaps between individual measures
    """
    overlaps = pd.read_excel(file_name, sheet_name=sheet_name)

    # replace nan values in ID columns with 0 and make sure they are integers
    for category in ['Overlap', 'Pressure', 'Activity', 'Overlapping', 'Overlapped']:
        overlaps.loc[np.isnan(overlaps[category]), category] = 0
        overlaps[category] = overlaps[category].astype(int)

    return overlaps


def pert_dist(peak, low, high, size) -> np.ndarray:
    '''
    Returns a set of random picks from a PERT distribution.
    '''
    # weight, controls probability of edge values (higher -> more emphasis on most likely, lower -> extreme values more probable)
    # 4 is standard used in unmodified PERT distributions
    gamma = 4
    # calculate expected value
    # mu = ((low + gamma) * (peak + high)) / (gamma + 2)
    if low == high and low == peak:
        return np.full(int(size), peak)
    r = high - low
    alpha = 1 + gamma * (peak - low) / r
    beta = 1 + gamma * (high - peak) / r
    return low + np.random.default_rng().beta(alpha, beta, size=int(size)) * r


def get_prob_dist(expecteds: np.ndarray, 
                  lower_boundaries: np.ndarray, 
                  upper_boundaries: np.ndarray, 
                  weights: np.ndarray) -> np.ndarray:
    '''
    Returns a cumulative probability distribution. All arguments should be 1D arrays.
    '''
    # verify that all arrays have the same size
    assert expecteds.size == lower_boundaries.size == upper_boundaries.size == weights.size

    #
    # TODO: remove uncomment in future to not accept faulty data
    # for now, sort arrays to have values in correct order
    #
    # # verify that all lower boundaries are lower than the upper boundaries
    # assert np.sum(lower_boundaries > upper_boundaries) == 0
    # # verify that most likely values are between lower and upper boundaries
    # assert np.sum((expecteds < lower_boundaries) & (expecteds > upper_boundaries)) == 0
    arr = np.full((len(expecteds), 3), np.nan)
    arr[:, 0] = lower_boundaries
    arr[:, 1] = expecteds
    arr[:, 2] = upper_boundaries
    arr = np.array([np.sort(row) for row in arr])
    lower_boundaries = arr[:, 0]
    expecteds = arr[:, 1]
    upper_boundaries = arr[:, 2]
    
    # select values that are not nan, bool matrix
    non_nan = ~np.isnan(expecteds) & ~np.isnan(lower_boundaries) & ~np.isnan(upper_boundaries)
    # multiply those values with weights, True = 1 and False = 0
    weights_non_nan = (non_nan * weights)

    # create a PERT distribution for each expert
    # from each distribution, draw a large number of picks
    # pool the picks together
    number_of_picks = 200
    picks = []
    for i in range(len(expecteds)):
        peak = expecteds[i]
        low = lower_boundaries[i]
        high = upper_boundaries[i]
        w = weights_non_nan[i]
        if ~non_nan[i]: # note the tilde ~ to check for nan value
            continue    # skip if any value is nan
        dist = pert_dist(peak, low, high, w * number_of_picks)
        picks += dist.tolist()
    
    # return nan if no distributions (= no expert answers)
    if len(picks) == 0:
        return np.nan
    
    # fit picks to discrete distribution
    # the distribution has 100 elements
    # every element at index i represents the probability of a value below i percent
    picks = np.array(picks)
    disc_dist = np.zeros(shape=100)
    for i in range(disc_dist.size):
        disc_dist[i] = np.sum(picks < i) / picks.size

    return disc_dist


def get_pick(dist) -> float:
    if dist is not None:
        weights = np.zeros(dist.shape)
        for i in range(1, weights.size):
            weights[i] = dist[i] - dist[i-1]
        pick = np.random.random() * np.sum(weights)
        for k, val in enumerate(weights):
            if pick < val:
                break
            pick -= val
        return pick
    else:
        return np.nan


#EOF
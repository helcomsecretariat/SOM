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

def get_expert_ids(df: pd.DataFrame) -> list:
    '''
    Returns list of expert id column names from dataframe using regex
    '''
    return df.filter(regex='^(100|[1-9]?[0-9])$').columns


def process_measure_survey_data(file_name: str, sheet_names: dict[int, str]) -> pd.DataFrame:
    """
    This method reads input from the excel file containing data about measure reduction efficiencies 
    on activities, pressures and states.

    Arguments:
        file_name (str): path of survey excel file
        sheet_names (dict): dict of survey sheet ids in file_name

    Returns:
        survey_df (DataFrame): processed survey data information
            measure (int): measure id
            activity (int): activity id
            pressure (int): pressure id
            state (int): state id (if defined, [nan] if no state)
            cumulative probability (list[float]): cum. prob. distribution represented as list
    """
    #
    # read information sheet from input Excel file
    #

    mteq = pd.read_excel(io=file_name, sheet_name=sheet_names[0])

    measure_survey_data = {}
    for id, sheet in enumerate(sheet_names.values()):
        # skip if information sheet
        if id == 0:
            continue
        # read data sheet from input Excel file, set header to None to include top row in data
        measure_survey_data[id] = pd.read_excel(io=file_name, sheet_name=sheet, header=None)
    
    #
    # preprocess values
    #

    mteq['State'] = [x.split(';') if type(x) == str else x for x in mteq['State']]

    #
    # create new dataframe
    #

    cols = ['survey_id', 'title', 'block', 'measure', 'activity', 'pressure', 'state']
    survey_df = pd.DataFrame(columns=cols)

    block_number = 0    # represents the survey block

    # for every survey sheet
    for survey_id in measure_survey_data:
        
        survey_info = mteq[mteq['Survey ID'] == survey_id]  # select the rows linked to current survey

        end = 0     # represents last column index of the question set
        for row, amt in enumerate(survey_info['AMT']):  # for each set of questions (row in MTEQ)
            
            end = end + (2 * amt + 1)     # end column index for data
            start = end - (2 * amt)     # start column index for data
            
            # create list to describe the data on each row
            titles = ['expected value', 'variance'] * amt
            titles.append('max effectiveness')
            titles.append('expert weights')

            # select current question column names as measure ids
            measures = measure_survey_data[survey_id].iloc[0, start:end].tolist()
            measures.append(np.nan)
            measures.append(np.nan)

            # create lists to hold ids and format each row as a list
            ids = {}
            for category in ['Activity', 'Pressure', 'State']:
                category_ids = survey_info[category].iloc[row]
                if isinstance(category_ids, str):
                    ids[category] = [[int(x) for x in category_ids.split(';') if x != '']] * amt * 2
                elif isinstance(category_ids, list):
                    ids[category] = [[int(x) for x in category_ids if x != '']] * amt * 2
                elif isinstance(category_ids, float) or isinstance(category_ids, int):
                    ids[category] = [[category_ids if not np.isnan(category_ids) else np.nan]] * amt * 2
                else:
                    ids[category] = [category_ids] * amt * 2
                ids[category].append(np.nan)
                ids[category].append(np.nan)

            # in MTEQ sheet, find all expert weight columns, get the values for the current row, set empty cells to 1
            expert_cols = [True if 'exp' in col.lower() else False for col in survey_info.columns]
            expert_weights = survey_info.loc[:, expert_cols].iloc[row]
            expert_weights.fillna(1, inplace=True)
            
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
            data['activity'] = ids['Activity']
            data['pressure'] = ids['Pressure']
            data['state'] = ids['State']

            with warnings.catch_warnings(action='ignore'):
                survey_df = pd.concat([survey_df, data], ignore_index=True, sort=False)
            block_number = block_number + 1

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
    # Split activity / pressure / state lists into separate rows, and reset index
    #

    for col in ['activity', 'pressure', 'state']:
        survey_df = survey_df.explode(column=col)
        survey_df = survey_df.reset_index(drop=True)

    #
    # Replace nan values with zeros and convert columns to integers
    #

    for column in ['measure', 'activity', 'pressure', 'state']:
        with warnings.catch_warnings(action='ignore'):
            survey_df[column] = survey_df[column].fillna(0)
        survey_df[column] = survey_df[column].astype(int)

    return survey_df


def process_pressure_survey_data(file_name: str, sheet_names: dict[int, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This method reads input from the excel file containing data about pressure contributions to states 
    and the changes in state required to reach required thresholds of improvement.

    Arguments:
        file_name (str): path of survey excel file
        sheet_names (dict): dict of survey sheet ids in file_name

    Returns:
        pressure_contributions (DataFrame):
            State: state id
            pressure: pressure id
            area_id: area id
            average: average contribution of pressure
            uncertainty: standard deveiation of pressure contribution
        thresholds (DataFrame):
            State: state id
            area_id: area id
            PR: reduction in state required to reach GES target
            10: reduction in state required to reach 10 % improvement
            25: reduction in state required to reach 25 % improvement
            50: reduction in state required to reach 50 % improvement
    """
    #
    # set parameter values
    #

    expert_number = 6   # max number of experts per question
    threshold_cols = ['PR', '10', '25', '50']   # target thresholds (PR=GES)

    #
    # read information sheet from input Excel file
    #

    psq = pd.read_excel(io=file_name, sheet_name=sheet_names[0])

    pressure_survey_data = {}
    for id, sheet in enumerate(sheet_names.values()):
        # skip if information sheet
        if id == 0:
            continue
        # read data sheet from input Excel file, set header to None to include top row in data
        pressure_survey_data[id] = pd.read_excel(io=file_name, sheet_name=sheet, header=None)
    
    #
    # preprocess values
    #

    psq['Basins'] = [x.split(';') if type(x) == str else x for x in psq['Basins']]
    psq['Countries'] = [x.split(';') if type(x) == str else x for x in psq['Countries']]

    # add question id column
    psq['question_id'] = list(range(len(psq)))

    #
    # create new dataframe
    #

    survey_df = pd.DataFrame(columns=['survey_id', 'question_id', 'State', 'Basins', 'Countries', 'GES known', 'Weight'])
    
    # survey columns from which to take data
    cols = ['Expert']
    cols += [x + str(i+1) for x in ['P', 'S'] for i in range(expert_number)]    # up to expert_number different pressures related to state, and their significance
    cols += [x + y for x in ['MIN', 'MAX', 'ML'] for y in threshold_cols]     # required pressure reduction to reach GES (if known) or X % improvement in state

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

    # create new dataframe for merged rows
    cols = ['survey_id', 'question_id', 'State', 'Basins', 'Countries', 'GES known']
    new_df = pd.DataFrame(columns=cols+['Pressures', 'Averages', 'Uncertainties']+threshold_cols)
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
        pressures = data[['P'+str(x+1) for x in range(expert_number)]].to_numpy().astype(float)
        significances = data[['S'+str(x+1) for x in range(expert_number)]].to_numpy().astype(float)
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
        for r in threshold_cols:
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
        for c in ['Pressures', 'Averages', 'Uncertainties']+threshold_cols:
            data[c] = np.nan
            data[c] = data[c].astype(object)
        # change data type to allow for lists
        data.at[0, 'Pressures'] = pressures
        data.at[0, 'Averages'] = average
        data.at[0, 'Uncertainties'] = stddev
        for r in threshold_cols:
            data.at[0, r] = reductions[r]
        with warnings.catch_warnings(action='ignore'):
            new_df = pd.concat([new_df, data], ignore_index=True, sort=False)
    #
    # get probability from distribution for each of the thresholds
    #
    for col in threshold_cols:
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
    new_df = new_df.loc[new_df['pressure'].notna(), :]
    new_df = new_df[new_df[threshold_cols].notna().any(axis=1)]
    new_df = new_df.reset_index(drop=True)
    #
    # make sure pressure ids are integers, remove unnecessary columns
    #
    new_df['pressure'] = new_df['pressure'].astype(int)
    new_df = new_df.drop(columns=['survey_id', 'question_id', 'GES known'])
    #
    # split new_df into two dataframes, one for pressure contributions and one for thresholds
    #
    pressure_contributions = pd.DataFrame(new_df.loc[:, ['State', 'pressure', 'area_id', 'average', 'uncertainty']])
    thresholds = pd.DataFrame(new_df.loc[:, ['State', 'area_id'] + threshold_cols]).drop_duplicates(subset=['State', 'area_id'], keep='first').reset_index(drop=True)

    return pressure_contributions, thresholds


def read_ids(file_name: str, id_sheets: dict) -> dict[str, dict]:
    """
    Reads in model object descriptions from general input files

    Arguments:
        file_name (str): source excel file name containing measure, activity, pressure and state id sheets
        id_sheets (dict): should have structure {'measure': sheet_name, 'activity': sheet_name, ...}

    Returns:
        object_data (dict): dictionary containing measure, activity, pressure and state ids and descriptions in separate dataframes
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
        
    return object_data


def read_cases(file_name: str, sheet_name: str) -> pd.DataFrame:
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

    cases = cases.reset_index(drop=True)

    # create new column 'area_id' to link basins and countries, and create the unique ids
    cases['area_id'] = None
    for i, row in cases.iterrows():
        cases.at[i, 'area_id'] = (row['basin'], row['country'])
    
    cases = cases.drop(columns=['basin', 'country'])

    return cases


def link_area_ids(data: dict[str, pd.DataFrame]):
    """
    Links area ids in the area dataframe to the (basin, country) tuples of the data and updates 
    area_id fields in the data to use actual area ids from the area dataframe.
    """
    # create area_id column
    data['area']['area_id'] = None
    for i, row in data['area'].iterrows():
        data['area'].at[i, 'area_id'] = (row['basin'], row['country'])
    def find_area_id(area_id):
        new_id = data['area'].loc[data['area']['area_id'] == area_id, 'ID']
        assert len(new_id.values) == 1
        new_id = new_id.values[0]
        return new_id
    for category in data.keys():
        if type(data[category]) == pd.DataFrame and category != 'area':
            if 'area_id' in data[category].columns:
                # filter out non-existent areas
                mask = data[category]['area_id'].isin(data['area']['area_id'])
                data[category] = data[category].loc[mask, :]
                data[category] = data[category].reset_index(drop=True)
                # change all other columns using area_id to use actual id
                data[category]['area_id'] = data[category]['area_id'].apply(find_area_id)

    return data


def read_activity_contributions(file_name: str, sheet_name: str) -> pd.DataFrame:
    """
    Reads input data of activities to pressures in Baltic Sea basins. 

    Arguments:
        file_name (str): name of source excel file name
        sheet_name (str): name of excel sheet
        areas (DataFrame): table to link countries/basins with area ids

    Returns:
        act_to_press (DataFrame): dataframe containing mappings between activities and pressures
    """
    act_to_press = pd.read_excel(file_name, sheet_name=sheet_name)

    # read all most likely, min and max column values into lists in new columns
    for col, regex_str in zip(['expected', 'minimum', 'maximum'], ['ML[1-6]', 'Min[1-6]', 'Max[1-6]']):
        act_to_press[col] = act_to_press.filter(regex=regex_str).values.tolist()

    # remove all most likely, min and max columns
    for regex_str in ['ML[1-6]', 'Min[1-6]', 'Max[1-6]']:
        act_to_press.drop(act_to_press.filter(regex=regex_str).columns, axis=1, inplace=True)

    # separate valuesgrouped together in sheet on the same row with ';' into separate rows
    for category in ['Activity', 'Pressure', 'Basin', 'Country']:
        act_to_press[category] = [list(filter(None, x.split(';'))) if type(x) == str else x for x in act_to_press[category]]
        act_to_press = act_to_press.explode(category)
        act_to_press[category] = act_to_press[category].astype(int)
    act_to_press = act_to_press.reset_index(drop=True)

    # create area_id column
    act_to_press['area_id'] = None
    for i, row in act_to_press.iterrows():
        act_to_press.at[i, 'area_id'] = (row['Basin'], row['Country'])
    act_to_press = act_to_press.drop(columns=['Basin', 'Country'])

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

    act_to_press['value'] = act_to_press['cumulative probability'].apply(get_pick)

    act_to_press = act_to_press.drop(columns=['expected', 'minimum', 'maximum', 'cumulative probability'])

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
    for category in ['BAU', 'ChangeMin', 'ChangeML', 'ChangeMax']:
        development_scenarios.loc[np.isnan(development_scenarios[category]), category] = 0
        development_scenarios[category] = development_scenarios[category].astype(float)
    
    development_scenarios['Activity'] = development_scenarios['Activity'].astype(int)

    development_scenarios = development_scenarios.drop(columns=['Activity Description'])

    # change values from percentual change to multiplier type by adding 1
    for category in ['BAU', 'ChangeMin', 'ChangeML', 'ChangeMax']:
        development_scenarios[category] = development_scenarios[category] + 1

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
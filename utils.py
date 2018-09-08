from __future__ import division
import numpy as np
import pandas as pd
from preprocess.preprocessing import relevant_data
import os


def load_complete_data():
    """
    Loads the complete loan application dataset
    :return: pandas dataframe
    """

    base_dir = '/home/sagar/Desktop/DecisionTree'
    file = os.path.join(base_dir, 'data/loan_application.csv')

    app_df = pd.read_csv(file)

    print 'Your dataset has {} rows and {} columns'.format(app_df.shape[0],
                                                           app_df.shape[1])

    return app_df


def load_relevant_data():
    """
    Loads the complete loan application dataset without Application ID
    :return: pandas dataframe
    """

    base_dir = '/home/sagar/Desktop/DecisionTree'
    file = os.path.join(base_dir, 'data/loan_application.csv')

    app_df = pd.read_csv(file)

    app_df = relevant_data(app_df)

    print 'Your dataset has {} rows and {} columns'.format(app_df.shape[0],
                                                           app_df.shape[1])

    return app_df


def load_manual_calculations():
    """
    Loads only the first 10 rows of applicaiton dataset
    :return: pandas dataframe
    """

    base_dir = '/home/sagar/Desktop/DecisionTree'
    file = os.path.join(base_dir, 'data/loan_application.csv')

    manual_df = pd.read_csv(file, nrows=10)

    print 'Your dataset has {} rows and {} columns'.format(manual_df.shape[0],
                                                           manual_df.shape[1])

    return manual_df


def get_probablities(data_df, variable):
    """
    Provides probablities for Y and N outcomes for a given variable
    :param
        variable: Column name / Predictors
        data_df: raw pandas dataframe
    :return: pandas dataframe with count, probabilities, squared probabilities
             and probabilities multiplied by its log
    """

    # Count of every Y and N for variable's different values
    count = pd.DataFrame(data_df.groupby([variable, 'Application_Status'])[
                             'Application_Status'].count())
    count.columns = ['count']

    # Count of every Y and N for the whole subset
    target_count = pd.DataFrame(data_df.groupby(variable)[
                                    'Application_Status'].count())
    target_count.columns = ['target_count']
    target_count['target_weight'] = target_count['target_count'].map(
        lambda x: x / target_count['target_count'].sum())

    count = count.merge(target_count, left_index=True, right_index=True,
                        how='left')

    # Probability of every Y and N for variable's different values
    prob = pd.DataFrame(data_df.groupby([variable, 'Application_Status'])[
                            'Application_Status'].count()).groupby(level=0).\
        apply(lambda x: x / float(x.sum())).round(3)
    prob.columns = ['prob']

    # Merging these 2 dataframes
    result_df = count.merge(prob, left_index=True, right_index=True)

    result_df['sqrd_prob'] = result_df['prob'].map(lambda x: x**2)

    result_df['log_prob'] = result_df['prob'].map(lambda x: x*np.log2(x))

    # Calculate Gini Index for individual variable's outcomes
    gini_resp = pd.DataFrame(result_df.groupby(level=0).
                             apply(lambda x: 1 - float(x.sqrd_prob.sum())))
    gini_resp.columns = ['gini_respective']

    result_df = result_df.merge(gini_resp, left_index=True, right_index=True,
                                how='left')

    # Calculate Entropy for individual variable's outcomes
    entropy_resp = pd.DataFrame(result_df.groupby(level=0).
                                apply(lambda x: -1*float(x.log_prob.sum())))
    entropy_resp.columns = ['entropy_resp']

    result_df = result_df.merge(entropy_resp, left_index=True,
                                right_index=True, how='left')

    return result_df.round(3)


def get_gini_index(data_df):
    """
    Provides Gini Index for every variable except for unique App ID and target
    :param data_df: your test / train dataset
    :return: pandas dataframe with gini score for each variable
    """

    # Initiate a list to save the results
    gini_ls = []

    # Iterate over every columns and get its probabilities
    for col in ['Gender', 'Married', 'Dependents', 'Education',
                'Self_Employed', 'Credit_History', 'Property_Area', 'Income']:
        data = get_probablities(data_df, col)

        # Retaining only required columns
        data = data[['target_weight', 'gini_respective']].drop_duplicates()

        # Calculate Gini for the variable and append results to list
        gini = pd.DataFrame(data['target_weight']*data['gini_respective']). \
            sum()[0]
        gini_ls.append((col, gini))

    res_df = pd.DataFrame(gini_ls, columns=['Variable', 'Gini']).round(2)

    return res_df.sort_values('Gini')


def get_target_entropy(data_df):
    """
    Calculates target entropy for a subset
    :param data_df: subset pandas df
    :return: float - entropy
    """

    # Count of every Y and N for the whole subset
    target_count = pd.DataFrame(data_df.groupby('Application_Status')[
                                    'Application_Status'].count())
    target_count.columns = ['target_count']
    target_count['target_weight'] = target_count['target_count'].map(
        lambda x: x / target_count['target_count'].sum())
    target_count['log_weight'] = target_count['target_weight'].map(
        lambda x: x*np.log2(x))

    return round(-1 * (target_count['log_weight'].sum()), 2)


def get_info_gain(data_df):
    """
    Provides Information Gain for every variable except for unique App ID and
    target
    :param data_df: subset of data
    :return: pandas dataframe with information gain on each variables
    """

    target_entropy = get_target_entropy(data_df)

    entropy_ls = []

    for col in ['Gender', 'Married', 'Dependents', 'Education',
                'Self_Employed', 'Credit_History', 'Property_Area', 'Income']:
        data = get_probablities(data_df, col)

        # Retaining only required columns
        data = data[['target_weight', 'entropy_resp']].drop_duplicates()

        # Calculate Gini for the variable and append results to list
        entropy = pd.DataFrame(data['target_weight'] * data['entropy_resp']). \
            sum()[0]
        ig = target_entropy - entropy
        entropy_ls.append((col, ig))

    res_df = pd.DataFrame(entropy_ls,
                          columns=['Variable', 'InformationGain']).round(2)

    return res_df.sort_values('InformationGain', ascending=False)

if __name__ == '__main__':
    data = load_relevant_data()
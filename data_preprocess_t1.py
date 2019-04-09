#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Pre-process the BRFSS data.

Created on Wed Aug 22 21:47:38 2018
"""
import logging
import pandas as pd
import numpy as np
from bisect import bisect_left

logging.basicConfig(filename="data-preprocess.log", level=logging.DEBUG)

"""
========================================================
Manually curated processers for each question
========================================================
"""

"""
How to decide which question is binary question and which is categorical question?

Run the following code:
    [(type1_df.columns[i], len(np.unique(type1_df.iloc[:, i].values)),
     np.unique(type1_df.iloc[:, i].values)) for i in range(52)]
then we get a list of <question name, unique answers to this question>.
By referring to the BRFSS codebook, we can decide

1. Question category
2. Nan_filler
3. No_response_droper
4. Special_processor


Question that is removed:
1. 'DIABAGE2' - follow-up question
"""
QUESTIONS = {
    'binary': ['EXERANY2', 'ADDEPEV2', 'ASTHMA3', 'ASTHNOW', 'CHCCOPD1',
                   'CHCKIDNY', 'CHCOCNCR', 'CHCSCNCR', 'CVDCRHD4', 'CVDINFR4',
                   'CVDSTRK3', 'HAVARTH3', 'CPDEMO1', 'DECIDE', 'DIFFALON',
                   'DIFFDRES', 'DIFFWALK', 'INTERNET', 'SEX', 'ECIGARET',
                   'SMOKE100', 'BLIND', 'DEAF', 'VETERAN3'],
    'continuous': ['MENTHLTH', 'PHYSHLTH', 'POORHLTH', 'SLEPTIM1',
                       'ALCDAY5', 'AVEDRNK2', 'DRNK3GE5', 'MAXDRNKS', 'CHILDREN',
                       'FALL12MN', 'FALLINJ2'],
    'categorical': ['_AGEG5YR', '_BMI5CAT', '_EDUCAG', '_INCOMG', 'GENHLTH',
                        'DIABETE3', 'EMPLOY1', 'MARITAL', 'RENTHOM1', 'ECIGNOW',
                        'LASTSMK2', 'SMOKDAY2', 'USENOW3'],
    'blacklist': ['STOPSMK2', 'NUMHHOL2', 'NUMPHON2', 'DIABAGE2']
}

# Value to replace NaN (missing, not asked). Also note how many Nan are there for such Q.
Nan_filler = {
    # binary
    'EXERANY2': 2, # 0
    'ADDEPEV2': 2, # 0
    'ASTHMA3': 2, # 0
    'ASTHNOW': 2, # 150,000+
    'CHCCOPD1': 2, # 0
    'CHCKIDNY': 2, # 0
    'CHCOCNCR': 2, # 0
    'CHCSCNCR': 2, # 0
    'CVDCRHD4': 2, #0
    'CVDINFR4': 2, #0
    'CVDSTRK3': 2, #0
    'HAVARTH3': 2, #0
    'CPDEMO1': 1, # 45,000+
    'DECIDE': 2, # 4,100+
    'DIFFALON': 2, # 4,600+
    'DIFFDRES': 2, # 4,400+
    'DIFFWALK': 2, # 4,300+
    'INTERNET': 2, # 1,400+
    'SEX': 9, # 0
    'ECIGARET': 2, # 5,300+
    'SMOKE100': 2, # 4,900+
    'BLIND': 2, # 3,900+
    'DEAF': 2, # 3,600+
    'VETERAN3': 2, # 20+
    # continuous
    'MENTHLTH': 88, # 0
    'PHYSHLTH': 88, # 1+
    'POORHLTH': 88, # 89,000+
    'SLEPTIM1': 77, # 0
    'ALCDAY5': 888, # 5,600+
    'AVEDRNK2': 0, # 100,000+
    'DRNK3GE5': 88, # 100,000+
    'MAXDRNKS': 0, # 100,000+
    'CHILDREN': 88, # 40+
    'FALL12MN': 88, # 7,500+
    'FALLINJ2': 88, # 120,000+
    # categorical
    '_BMI5CAT': 9, # 10,900+
    'GENHLTH': 7, # 1+
    'DIABETE3': 7, # 0
    'EMPLOY1': 9, # 30+
    'MARITAL': 9, # 1+
    'RENTHOM1': 7, # 1+
    'ECIGNOW': 3, # 161,000+
    'LASTSMK2': 8, # 104,400+
    'SMOKDAY2': 3, # 90,300+
    'USENOW3': 3, # 5,100+
}

# Value to drop (Don't know - 7, Refused - 9)
No_Response_droper = {
    # binary
    'EXERANY2': [7, 9],
    'ADDEPEV2': [7, 9],
    'ASTHMA3': [7, 9],
    'ASTHNOW': [7, 9],
    'CHCCOPD1': [7, 9],
    'CHCKIDNY': [7, 9],
    'CHCOCNCR': [7, 9],
    'CHCSCNCR': [7, 9],
    'CVDCRHD4': [7, 9],
    'CVDINFR4': [7, 9],
    'CVDSTRK3': [7, 9],
    'HAVARTH3': [7, 9],
    'CPDEMO1': [7, 9],
    'DECIDE': [7, 9],
    'DIFFALON': [7, 9],
    'DIFFDRES': [7, 9],
    'DIFFWALK': [7, 9],
    'INTERNET': [7, 9],
    'SEX': [9],
    'ECIGARET': [7, 9],
    'SMOKE100': [7, 9],
    'BLIND': [7, 9],
    'DEAF': [7, 9],
    'VETERAN3': [7, 9],
    # continuous
    'MENTHLTH': [77, 99],
    'PHYSHLTH': [77, 99],
    'POORHLTH': [77, 99],
    'SLEPTIM1': [77, 99],
    'ALCDAY5': [777, 999],
    'AVEDRNK2': [77, 99],
    'DRNK3GE5': [77, 99],
    'MAXDRNKS': [77, 99],
    'CHILDREN': [99],
    'FALL12MN': [77, 99],
    'FALLINJ2': [77, 99],
    # categorical
    '_BMI5CAT': [9],
    '_EDUCAG': [9],
    '_INCOMG': [9], # 16%
    'GENHLTH': [7, 9],
    'DIABETE3': [7, 9],
    'EMPLOY1': [9],
    'MARITAL': [9],
    'RENTHOM1': [7, 9],
    'ECIGNOW': [7, 9],
    'LASTSMK2': [77, 99],
    'SMOKDAY2': [7, 9],
    'USENOW3': [7, 9]
}

# Special 0 counts
# The following values will be converted to 0.
# e.g. for some questions, 88 means 0 (none)
Zero_converter = {
    # continuous
    'MENTHLTH': 88,
    'PHYSHLTH': 88,
    'POORHLTH': 88,
    'ALCDAY5': 888,
    'DRNK3GE5': 88,
    'CHILDREN': 88,
    'FALL12MN': 88,
    'FALLINJ2': 88,
}

# Special logistics
Special_processor = {
    'ALCDAY5': lambda x: ((x-100)*4.2 if x > 100 else x) if x < 200 else x - 200,
}




"""
========================================================
Data preparation functions
========================================================
"""


def df_chain_processor(df):
    """
    A chain of preprocessor for dataframe.
    The order has to be preserved.
    
    
    1. Drop questions in `blacklist`.
    2. Fill Nan answer with correct values from `fill_map`.
    3. Drop no response according to `No_Response_droper`.
    4. Convert 0 count from 88 and 888.
    5. Apply special processor.
    6. Drop columns with constant features.
    """
    logging.info("1. Remove questions in `blacklist`")
    s1 = df.shape
    res = df.drop(QUESTIONS['blacklist'], axis=1, errors="ignore")
    s2 = res.shape
    assert s1[1] - s2[1] <= len(QUESTIONS['blacklist'])
    
    logging.info("2. Fill in Nan with `Nan_filler` map")
    for col, val in Nan_filler.items():
        res[col].fillna(val, inplace=True)
    assert np.any(np.isnan(res)) == False
    
    logging.info("3. Remove `No Response` rows for each questions - (column, #remove)")
    for col, vals in No_Response_droper.items():
        cnt = 0
        for val in vals:
            idx_to_drop = res[res[col] == val].index
            cnt += len(idx_to_drop)
            res.drop(idx_to_drop, inplace=True)
        logging.info("%s, %s", col, cnt)
    
    logging.info("4. Set proper 0")
    for col, val in Zero_converter.items():
        res.loc[res[col] == val, col] = 0
    assert np.any(res == 88) == False
    assert np.any(res == 888) == False

    for col in QUESTIONS['binary']:
        res.loc[res[col] == 2, col] = 0
    
    logging.info("5. Apply special processor")
    for col, func in Special_processor.items():
        res[col] = res[col].apply(func)

    logging.info("6. Drop column with constant values")
    columns_std_0 = res.loc[:, res.std()==0].columns.tolist()
    logging.info("Dropped columns are: {}".format(columns_std_0))
    res = res.drop(columns_std_0, axis=1)
    
    return res


def construct_features(df):
    """
    Construct feature dataframe for clustering based on following rules:
    
    1. binary question is feature, take value either 0 or 1
    2. categorical question take n feature, where n is number of possible answer
    3. continuous question is discretized into K-bins
    """
    nf = 0
    feature_columns_map = {}
    # add binary
    features = df[QUESTIONS['binary']]
    for question in QUESTIONS['binary']:
        feature_columns_map[question] = [question]
    nf += len(QUESTIONS['binary'])
    
    # add categorical
    for cateQ in QUESTIONS['categorical']:
        org_val = df[cateQ].astype(int).values
        answers = np.unique(org_val)
        nf += len(answers)
        feature_names = ["{}_{}".format(cateQ, a) for a in answers]
        feature_map = dict(zip(answers, range(len(answers))))
        new_vals = np.zeros(shape=(len(org_val), len(answers)))
        for i, v in enumerate(org_val):
            new_vals[i, feature_map[v]] = 1
        new_df = pd.DataFrame(data=new_vals, index=features.index, columns=feature_names)
        lp = features.shape[0]
        features = pd.concat([features, new_df], axis=1)
        lc = features.shape[0]
        assert lp == lc
        feature_columns_map[cateQ] = feature_names
    
    # add continuous
    for cateQ in QUESTIONS['continuous']:
        splits = np.unique(np.percentile(df[cateQ], range(20, 100, 20)))
        if len(splits) <= 1:
            upperHalf = df.loc[df[cateQ] > splits[-1], cateQ]
            uhSplits = np.unique(np.percentile(upperHalf, range(25, 100, 25)))
            splits = np.concatenate((splits, uhSplits))
        assert len(splits) >= 2
        logging.info('{} == split == {}'.format(cateQ, splits))
        
        # convert continuous value to category index
        cont_col = df[cateQ].apply(lambda x: bisect_left(splits, x))
        answers = np.unique(cont_col)
        nf += len(answers)
        feature_names = ["{}_{}".format(cateQ, a) for a in answers]
        feature_map = dict(zip(answers, range(len(answers))))
        new_vals = np.zeros(shape=(cont_col.shape[0], len(answers)))
        for i, v in enumerate(cont_col.astype(int).values):
            new_vals[i, feature_map[v]] = 1
        new_df = pd.DataFrame(data=new_vals, index=features.index, columns=feature_names)
        lp = features.shape[0]
        features = pd.concat([features, new_df], axis=1)
        lc = features.shape[0]
        assert lp == lc
        feature_columns_map[cateQ] = feature_names

    assert features.shape[1] == nf
    assert features.max().max() <= 1
    return features, feature_columns_map


def data_preparation_type_I(remove_selected=False):
    # load input data, set the validation set aside
    df = pd.read_csv("BRFSS_Dataset/BRFSS_2016_AgeOver65.csv", sep=",", header=0, index_col=0)
    validation = df.sample(30, random_state=2018)
    df = df.drop(validation.index)
    
    """Process type I data"""
    type1_generalized = []
    with open('Type1_generalized.names') as fin1:
        for line in fin1:
            type1_generalized.append(line.strip())
            
    if remove_selected:
        with open('Type1_remove.names') as fin1:
            for line in fin1:
                QUESTIONS['blacklist'].append(line.strip())

    t1_df = df[type1_generalized]
    t1_df = df_chain_processor(t1_df)
    t1_features, feature_columns_map = construct_features(t1_df)

    # sample validation
    validation_feat = t1_features.sample(30, random_state=2018)
    validation_feat.to_csv("validation_features.csv")

    t1_df = t1_df.drop(validation_feat.index)
    t1_features = t1_features.drop(validation_feat.index)
    df = df.loc[t1_df.index]
    print t1_df.shape, validation.shape
    return t1_df, t1_features, feature_columns_map, validation, df


def get_validation_set_features(columns):
    validation_features = pd.read_csv("validation_features.csv", index_col=0)
    validation_features = validation_features[columns]
    print validation_features.shape
    return validation_features


if __name__ == '__main__':
    t1_df, t1_features, _, _, _ = data_preparation_type_I()

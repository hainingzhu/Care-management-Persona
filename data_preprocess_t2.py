#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Pre-process the BRFSS data.

Created on Wed Aug 22 21:47:38 2018
"""
import logging
import numpy as np
import pandas as pd

logging.basicConfig(filename="data-preprocess-t2.log", level=logging.DEBUG)

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
"""
QUESTIONS = {
    'binary': ['CDDISCUS', 'CIMEMLOS'],
    'continuous': [],
    'categorical': ['EMTSUPRT', 'LSATISFY', 'MEDADVIC', 'UNDRSTND', 'WRITTEN', 'CDASSIST', 'CDHOUSE',
                    'CDSOCIAL']
}

# Value to replace NaN (missing, not asked). Also note how many Nan are there for such Q.
Nan_filler = {
    # binary
    'CDDISCUS': 7,
    'CIMEMLOS': 7,
    # categorical
    'EMTSUPRT': 7,
    'LSATISFY': 7,
    'MEDADVIC': 7,
    'UNDRSTND': 7,
    'WRITTEN': 7,
    'CDASSIST': 7,
    'CDHOUSE': 7,
    'CDSOCIAL': 7
}

# Value to drop (Don't know - 7, Refused - 9)
No_Response_droper = {
    # binary
    'CDDISCUS': [7, 9],
    'CIMEMLOS': [7, 9],
    # categorical
    'EMTSUPRT': [7, 9],
    'LSATISFY': [7, 9],
    'MEDADVIC': [7, 9],
    'UNDRSTND': [7, 9],
    'WRITTEN': [7, 9],
    'CDASSIST': [7, 9],
    'CDHOUSE': [7, 9],
    'CDSOCIAL': [7, 9]
}

# Special 0 counts
# The following values will be converted to 0.
# e.g. for some questions, 88 means 0 (none)
Zero_converter = {
    # continuous

}

# Special logistics
Special_processor = {

}

NUM_CLUSTER = 4



"""
========================================================
Data preparation functions
========================================================
"""


def df_chain_processor(df):
    """
    A chain of preprocessor for dataframe.
    The order has to be preserved.

    2. Fill Nan answer with correct values from `fill_map`.
    3. Drop no response according to `No_Response_droper`.
    4. Convert 0 count from 88 and 888.
    5. Apply special processor.
    6. Drop columns with constant features.
    """
    for q in df.columns:
        # Fill in Nan with `Nan_filler` map
        if q in Nan_filler:
            df[q].fillna(Nan_filler[q], inplace=True)
        # Remove `No Response` rows for each questions - (column, #remove)
        if q in No_Response_droper:
            for val in No_Response_droper[q]:
                idx_to_drop = df[df[q] == val].index
                df.drop(idx_to_drop, inplace=True)
        # Set proper 0 for binary feature
        if q in QUESTIONS['binary']:
            df.loc[df[q] == 2, q] = 0

    assert np.any(np.isnan(df)) == False
    return df


def construct_features(df):
    """
    Construct feature dataframe for clustering based on following rules:

    1. binary question is feature, take value either 0 or 1
    2. categorical question take n feature, where n is number of possible answer
    3. continuous question is discretized into K-bins
    """
    nf = 0
    feature_columns_map = {}
    features = None

    for q in df.columns:
        if q in QUESTIONS['binary']:
            features = pd.concat((features, df[q]), axis=1)
            feature_columns_map[q] = [q]
            nf += 1
        if q in QUESTIONS['categorical']:
            org_val = df[q].astype(int).values
            answers = np.unique(org_val)
            nf += len(answers)
            feature_names = ["{}_{}".format(q, a) for a in answers]
            feature_map = dict(zip(answers, range(len(answers))))
            new_vals = np.zeros(shape=(len(org_val), len(answers)))
            for i, v in enumerate(org_val):
                new_vals[i, feature_map[v]] = 1
            new_df = pd.DataFrame(data=new_vals, index=df.index, columns=feature_names)
            lp = df.shape[0]
            features = pd.concat([features, new_df], axis=1)
            lc = features.shape[0]
            assert lp == lc
            feature_columns_map[q] = feature_names

    assert features.shape[1] == nf
    assert features.max().max() <= 1
    assert features.shape[0] == df.shape[0]
    return features, feature_columns_map


def data_process_type_II(df, cls1):
    df["labels"] = cls1.labels_

    """Process type II data"""
    type2_conditional = []
    with open('Type2_conditional.names') as fin2:
        _group = []
        for line in fin2:
            f = line.strip()
            if f != "":
                _group.append(line.strip())
            else:
                type2_conditional.append(_group)
                _group = []
    logging.info("{0} groups of Type II data found".format(len(type2_conditional)))

    for idx, _group in enumerate(type2_conditional):
        grp_df = df[_group]
        grp_df = df_chain_processor(grp_df)
        grp_features, fcm = construct_features(grp_df)
        grp_df["labels"] = df.loc[grp_df.index, "labels"]

        logging.info(">" * 20)
        logging.info("Conditional group {0} has {1} questions, aka {2} features"
                     .format(idx, len(_group), grp_features.shape[1]))
        res = []
        for j in range(NUM_CLUSTER):
            type2_df = grp_features.loc[grp_df["labels"] == j]
            logging.info("{0} rows available within cluster {1}".format(type2_df.shape[0], j))
            res_df = type2_df.mean()
            res_df["count_{}".format(idx)] = type2_df.shape[0]
            res.append(res_df)
        grp_res = pd.concat(res, axis=1).transpose()
        logging.info(grp_res)
        grp_res.to_csv("conditional_question_centers_grp_{}.csv".format(idx), sep=",")


if __name__ == '__main__':
    pass

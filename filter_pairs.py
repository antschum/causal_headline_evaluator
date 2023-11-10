import pandas as pd
from itertools import combinations
import pickle as pkl
import os
# This is preparing the Upworthy Dataset

# Load data
df = pd.read_csv("C:/Users/mldem/Downloads/upworthy-archive-datasets/upworthy-archive-confirmatory-packages-03.12.2020.csv",low_memory=False)
# add index as column value
df.reset_index(inplace=True)

# 1. Filter out all datapoints that dont contain more than 2 Headlines when filtering for experimental ID and image
# create mask. 
df['headline_count'] = df.groupby(['clickability_test_id', 'eyecatcher_id']).headline.transform('count')

df_pairs = df.loc[df['headline_count']>=2, ['clickability_test_id', 'excerpt', 'headline', 'lede', 'eyecatcher_id', 'clicks', 'headline_count']]
df_pairs.columns
# probs not necessary - still have the original indices. 
# df_pairs.reset_index(inplace=True)

examples = df_pairs.loc[df_pairs['clickability_test_id']=='546dd17e26714c82cc00001c', ['headline', 'eyecatcher_id']]
print(examples)

# goal at the end:
# df with columns [sentence1, sentence2, click_difference, headline1_more_clicks] containing the indices at that location which can then be matched again to the embeddings. 
combo = df.groupby(['clickability_test_id', 'eyecatcher_id'])['index'].apply(combinations,2)\
                     .apply(list).apply(pd.Series)\
                     .stack().apply(pd.Series)\
                     .set_axis(['Idx_Headline1','Idx_Headline2'],axis = 1)\
                     .reset_index(level=0).reset_index()
combo["click_difference"][0]
combo['click_difference'] = df.loc[combo['Idx_Headline1'], ['clicks']].to_numpy()-df.loc[combo['Idx_Headline2'], ['clicks']].to_numpy()
combo['headline1_more_clicks'] = combo['click_difference']>=0

condition = combo["click_difference"] < 0
combo.loc[condition, ['Idx_Headline1', 'Idx_Headline2']] = combo.loc[condition, ['Idx_Headline2', 'Idx_Headline1']].values   

combo.drop('level_1',axis=1)


combo.to_csv("C:/Projects/CausalClicker/causal_headline_evaluator/headline_pair_indices.csv")

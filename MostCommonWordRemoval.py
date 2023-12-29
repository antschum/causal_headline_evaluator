from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import pandas as pd
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.linear_model import RidgeCV, LogisticRegression, LinearRegression
from torch.utils.data import DataLoader, TensorDataset
import csv
import torch 
import pickle
import random
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
import math
import shap
#import langid

# Set random seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

#Load data (like in Jupyter notebook)
df = pd.read_csv("upworthy-archive-confirmatory-packages-03.12.2020.csv", low_memory=False)
#adding index
df.reset_index(inplace=True,names=["embedding_id"])

#remove rows without eyecatcher_id (about 100)
has_eyecatcher_id = df['eyecatcher_id'].notna()
df = df.loc[has_eyecatcher_id]
#Create a new column for clickrate
df["clickrate"] = round((df["clicks"]/ df["impressions"]), ndigits=10)

#filter data based on same clickability_id and eyecatcher_id
df['headline_count'] = df.groupby(['clickability_test_id', 'eyecatcher_id']).headline.transform('count')
df.columns
# filter for all headlines with at least 2 pairs. 
df = df.loc[df['headline_count']>=2, ['clickability_test_id', 'excerpt', 'headline', 'lede', 'eyecatcher_id', 'clicks', 'headline_count',"embedding_id","clickrate","impressions"]]

# drop all rows with same headline, clickability_test_id and eyecatcher_id

#df = df.drop_duplicates(subset=["headline","clickability_test_id","eyecatcher_id"],keep=False)

cti = df[df.duplicated(subset=["headline","clickability_test_id","eyecatcher_id"],keep=False)].clickability_test_id
eti = df[df.duplicated(subset=["headline","clickability_test_id","eyecatcher_id"],keep=False)].eyecatcher_id
print(df.shape)
print("we removed: ",((df['clickability_test_id'].isin(cti) & df['eyecatcher_id'].isin(eti))).sum())
print(df[((df['clickability_test_id'].isin(cti) & df['eyecatcher_id'].isin(eti)))][['clickability_test_id', 'eyecatcher_id', 'headline']][:20])
df = df[~(df['clickability_test_id'].isin(cti) & df['eyecatcher_id'].isin(eti))]
print(df.shape)

#checking if it was successful
# -> could there be cases where there are duplicates and we only delete the headlines and not the whole experiment?
print(df[df["clickability_test_id"] == "546de9399ad54eca4800003c"]) #this is an example of matching headline, clickability test id and eyecatcher id
df = df.sort_values(by='headline_count', ascending=False)
#print(df.head())
#checking if there are any duplicates
#df = df.drop_duplicates(subset=["headline"]) ##this was before
##this is a new version
dupl_headline = df[df.duplicated(subset=["headline"])] 
## I believe we are removing too many values here - one of the duplicates we want to keep right?

#duplicated = df[dupl_headline]
ids = dupl_headline["clickability_test_id"]
eid = dupl_headline["eyecatcher_id"]


mask = df['clickability_test_id'].isin(ids)&df['eyecatcher_id'].isin(eid)
df = df[~mask]


## Idea: 
#1. Get 500 Most common words. 
#2. Pull all headlines with eg. first word. 
#3. Run new embedding space for all combinations - with word and without word. -> create strings without words, couple it to original somehow so that we can get difference later. 
#4. Predict number of clicks with these new embeddigns (without that word.)


#1. Get 500 Most common words. 
# Python program to find the k most frequent words (Geek Website)
# from data set 
from collections import Counter 
  
# should we capitalize the words? Do upper and lower case letters make a difference in the embedding space? 
# Are the headlines in all capitalizations and all lower case close to each other?
# split() returns list of all the words in the string 
ignore = {'the', "The", "A",'a','if','in','it','of','or', 'To', 'And', 'Of', 'In', 'Is', 'This', 'It', 'That', 'On'}
split_it = [word for headline in df.headline for word in headline.split() if word not in ignore]
  
# Pass the split_it list to instance of Counter class. 
counter = Counter(split_it) 

# Remove filler words like: 
# a, the, 
  
# most_common() produces k frequently encountered 
# input values and their respective counts. 
most_occur = counter.most_common(10) 
most_occur


#2. Pull all headlines with eg. first word. 
# take for example 
word = 'What'

# Its very wierd, there are more headlines in the word_headlines dataset than 
# the counter counts.. 
word_headlines = df[df.headline.str.contains(word)]

# removes word from headline, adds new headline to df as removed column
word_headlines['removed'] = word_headlines.headline.str.replace(word, '')

#3. Run new embedding space for the headlines with words removed. 
#4. Predict number of clicks with these new embeddigns and add as column to df. 
#5. Visualize somehow and run for all words. 
## How can we summarize that data? 


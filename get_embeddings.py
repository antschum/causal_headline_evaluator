import torch 


import pickle
from sentence_transformers import SentenceTransformer
import pandas as pd

#cpu/gpu
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#Define model - pretrained
model = SentenceTransformer('all-mpnet-base-v2')

#Load data
df = pd.read_csv("C:/Projects/CausalClicker/causal_headline_evaluator/upworthy-archive-confirmatory-packages-03.12.2020.csv", low_memory=False)
#Delete some unnecessary columns
df.columns
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
print(df.shape) #too few observations :(
#print(df[df["clickability_test_id"] == "545181f8763e26efef000001"]) #those are staying because there are a lot of observations
#print(df[df["clickability_test_id"] == "54518b6da54be28ef000000b"]) #those are not staying because there a less packages within the test than then in the one above

df = df[df["clickability_test_id"] != "51436075220cb800020007b3"]
#checking if i dropped exactly 4
print(df.shape)


#Extract only headlines
headlines =df.headline.values

#Embeddings
embeddings = model.encode(headlines,convert_to_tensor=True,batch_size=32,show_progress_bar=True)
print(embeddings.shape)
with open('duplicates_removed_embeddings.pkl', "wb") as fOut:
     pickle.dump({'headlines': headlines, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
print(embeddings.shape)



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
#adding index
df.reset_index(inplace=True,names=["embedding_id"])

#remove rows without eyecatcher_id (about 100)
has_eyecatcher_id = df['eyecatcher_id'].notna()
df = df.loc[has_eyecatcher_id]
#Create a new column for clickrate
df["clickrate"] = round((df["clicks"]/ df["impressions"]), ndigits=10)
print(df.groupby(["clickability_test_id"]).count().mean()) #average of 4.64 packages within one test
print(df.groupby(["clickability_test_id","eyecatcher_id"]).count().mean()) #average of 2.11 packages with the same eyecatcher id and same clickability_test_id
#filter data based on same clickability_id and eyecatcher_id
df['headline_count'] = df.groupby(['clickability_test_id', 'eyecatcher_id']).headline.transform('count')
df.columns
# filter for all headlines with at least 2 pairs. 
df = df.loc[df['headline_count']>=2, ['clickability_test_id', 'excerpt', 'headline', 'lede', 'eyecatcher_id', 'clicks', 'headline_count',"embedding_id","clickrate"]]

# drop all rows with same headline,clickability_test_id and eyecatcher_id
df = df.drop_duplicates(subset=["headline","clickability_test_id","eyecatcher_id"])
df = df.sort_values(by='headline_count', ascending=False)
df = df.drop_duplicates(subset=["headline"])
df = df[df["clickability_test_id"] != "51436075220cb800020007b3"]
# Make tensor
print(df.shape)

#Extract only headlines
#headlines =df.headline.values

#Embeddings
#embeddings = model.encode(headlines,convert_to_tensor=True,batch_size=32,show_progress_bar=True)

#with open('all-mpnet-base-v2_embeddings.pkl', "wb") as fOut:
   #  pickle.dump({'headlines': headlines, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)


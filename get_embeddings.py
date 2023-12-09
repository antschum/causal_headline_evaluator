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
df = pd.read_csv("upworthy-archive-confirmatory-packages-03.12.2020.csv", low_memory=False)
#Delete some unnecessary columns
df.columns
# add index as column value
df.reset_index(inplace=True, names=['embedding_id'])
## maybe we just order them from the beginning and then check if there are any negative differences? That should almost be enough. 

# 1. Filter out all datapoints that dont contain more than 2 Headlines when filtering for experimental ID and image
# create mask. 
df['headline_count'] = df.groupby(['clickability_test_id', 'eyecatcher_id']).headline.transform('count')

# filter for all headlines with at least 2 pairs. 
df = df.loc[df['headline_count']>=2, ['clickability_test_id', 'excerpt', 'headline', 'lede', 'eyecatcher_id', 'clicks', 'headline_count', 'embedding_id']]

#Extract only headlines
headlines =df.headline.values

#Embeddings
embeddings = model.encode(headlines,convert_to_tensor=True,batch_size=32,show_progress_bar=True,
                          )

with open('all-mpnet-base-v2_embeddings.pkl', "wb") as fOut:
     pickle.dump({'headlines': headlines, 'embeddings': embeddings, 'embedding_id':df.embedding_id}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

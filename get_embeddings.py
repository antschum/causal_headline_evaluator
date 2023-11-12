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
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

#Load data
df = pd.read_csv("/Users/tonia/Dropbox/2023WS_Ash_Research_Causal_Predictor/osfstorage-archive/upworthy-archive-datasets/upworthy-archive-confirmatory-packages-03.12.2020.csv", low_memory=False)
#Delete some unnecessary columns
df.columns
delete_cols = ["created_at","updated_at","share_text","square"]
df = df.drop(columns=delete_cols)

#Extract only headlines
headlines =df.headline.values

#Embeddings
embeddings = model.encode(headlines,convert_to_tensor=True,batch_size=32,show_progress_bar=True,
                          )

with open('full_embeddings.pkl', "wb") as fOut:
     pickle.dump({'headlines': headlines, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

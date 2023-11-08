import torch
from transformers import AutoTokenizer,AutoModel
from sentence_transformers import SentenceTransformer,SentencesDataset, InputExample, losses,models
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.linear_model import RidgeCV


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

#Define model - not pretrained
# word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
# pooling_model = models.Pooling(pooling_mode_cls_token=True) #pooling method, ins sbert they use this instead of cls?
# model = SentenceTransformer(modules=[word_embedding_model, pooling_model]) #making the sentence transformer

#Define model - pretrained
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')




#Load data
df = pd.read_csv("C:/Users/mldem/Downloads/upworthy-archive-datasets/upworthy-archive-confirmatory-packages-03.12.2020.csv")
#Delete some unnecessary columns
df.columns
delete_cols = ["created_at","updated_at","share_text","square"]
df = df.drop(columns=delete_cols)
df.sample(10)
#Extract only headlines
headlines =df.headline.values
headlines = headlines[:100]
clicks = df.clicks.values
clicks=clicks[:100]

#Embeddings
embeddings = model.encode(headlines)
embeddings.shape
#To Save embeddings
# with open('embeddings.pkl', "wb") as fOut:
#     pickle.dump({'sentences': sentences, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

# #Load sentences & embeddings from disc
# with open('embeddings.pkl', "rb") as fIn:
#     stored_data = pickle.load(fIn)
#     stored_sentences = stored_data['sentences']
#     stored_embeddings = stored_data['embeddings']

#Define test/train data
X_train, X_test, y_train, y_test = train_test_split(embeddings,clicks, test_size=0.2)
regression_model =RidgeCV(alphas=[0.001,0.002,0.005,0.01,0.05,0.07,0.2,0.4,0.6, 1],store_cv_values=True)
clf = regression_model.fit(X_train, y_train)
clf.score(X_train,y_train)

predictions = regression_model.predict(X_test)
rmse = mean_squared_error(y_test, predictions, squared=False)
print(predictions)
print(y_test)
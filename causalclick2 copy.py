import torch 
import pickle
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
from torch import nn
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset



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
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension()) #pooling method, ins sbert they use this instead of cls?
# dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Sigmoid())
# model = SentenceTransformer(modules=[word_embedding_model, pooling_model,dense_model]) #making the sentence transformer

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
headlines.shape
headlines = headlines[:1000]
clicks = df.clicks.values
clicks=clicks[:1000]
clicks = torch.tensor(clicks)
clicks.shape

#Embeddings
embeddings = model.encode(headlines,convert_to_tensor=True,batch_size=32,show_progress_bar=True,
                          )
embeddings.shape
#To Save embeddings
with open('embeddings.pkl', "wb") as fOut:
     pickle.dump({'headlines': headlines, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

#Load sentences & embeddings from disc
with open('embeddings.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    stored_sentences = stored_data['headlines']
    stored_embeddings = stored_data['embeddings']
   

#Define test/train data
dataset = TensorDataset(embeddings,clicks)
#train_set, test_set = random_split(dataset, [0.8, 0.2])
reader = pd.read_csv("C:\Projects\CausalClicker\headline_pair_indices.csv",header=True)
reader
input_examples = []
for row in reader:
        print(row)
        break
        score = float(row['click_difference'])  
        inp_example = InputExample(texts=[df.loc[row['Idx_Headline1'], "headline"], df.loc[row['Idx_Headline2'], "headline"]], label=score)

        input_examples.append(inp_example)
        # if row['split'] == 'dev':
        #     dev_samples.append(inp_example)
        # elif row['split'] == 'test':
        #     test_samples.append(inp_example)
        # else:
        #     train_samples.append(inp_example)

train_set, test_set = random_split(input_examples, [0.8, 0.2])


train_dataloader_x = DataLoader(
            train_set,  # The training samples.
             # Select batches randomly
            batch_size = 32 # Trains with this batch size.
        )
train_dataloader_y = DataLoader(
            test_set,  # The training samples.
             # Select batches randomly
            batch_size = 32 # Trains with this batch size.
        )

regression_model =RidgeCV(alphas=[0.001,0.002,0.005,0.01,0.05,0.07,0.2,0.4,0.6, 1],store_cv_values=True)
regression_model.cv_values_.mean(axis=0)
clf = regression_model.fit(X_train, y_train)
clf.score(X_train,y_train)

predictions = regression_model.predict(X_test)
rmse = mean_squared_error(y_test, predictions, squared=False)
print(rmse)
print(predictions)
print(y_test)
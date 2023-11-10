
from charset_normalizer import md__mypyc
from sentence_transformers import SentenceTransformer
import pandas as pd
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.linear_model import RidgeCV, LogisticRegression, LinearRegression
from torch import nn

from torch.utils.data import DataLoader, TensorDataset
import csv
import gzip
import torch 
import pickle
from sklearn.metrics import accuracy_score


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
df = pd.read_csv("C:/Users/mldem/Downloads/upworthy-archive-datasets/upworthy-archive-confirmatory-packages-03.12.2020.csv", low_memory=False)
#Delete some unnecessary columns
df.columns
delete_cols = ["created_at","updated_at","share_text","square"]
df = df.drop(columns=delete_cols)
df.sample(10)
#Extract only headlines
headlines =df.headline.values
headlines.shape
clicks = df.clicks.values
clicks = torch.tensor(clicks)
clicks.shape

#Embeddings
# embeddings = model.encode(headlines,convert_to_tensor=True,batch_size=32,show_progress_bar=True)
# embeddings.shape



#To Save embeddings
with open('embeddings.pkl', "wb") as fOut:
     pickle.dump({'headlines': headlines, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

with open('/Users/tonia/Dropbox/2023WS_Ash_Research_Causal_Predictor/causal_headline_evaluator/full_embeddings.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    stored_sentences = stored_data['headlines']
    stored_embeddings = stored_data['embeddings']


#Define test/train data
dataset = TensorDataset(stored_embeddings,clicks)
X_train, X_test, y_train, y_test = train_test_split(stored_embeddings,clicks, test_size=0.2)
clicks.shape
# 1. Initial Training Regression on Embeddings. 

ridge_model =RidgeCV(alphas=[0.001,0.002,0.005,0.01,0.05,0.07,0.2,0.4,0.6, 1],store_cv_values=True)
ridge_fit = ridge_model.fit(X_train, y_train)
ridge_fit.score(X_train,y_train)

predictions = ridge_model.predict(X_test)
rmse = mean_squared_error(y_test, predictions, squared=False)
print("Ridge Regression MSE for clicks difference:", rmse) #result is awful :)

pairs = pd.read_csv("C:/Projects/CausalClicker/causal_headline_evaluator/headline_pair_indices.csv")
# Compute Vector difference
### Need embeddings for all data!
vec1 = pairs.apply(lambda row:(stored_embeddings[row['Idx_Headline1']]), axis=1)
vec1= torch.stack(vec1.tolist())
vec2 = pairs.apply(lambda row:(stored_embeddings[row['Idx_Headline2']]), axis=1)
vec2= torch.stack(vec2.tolist())
# pairs['embedding_diff'] = pairs.apply(lambda row: stored_embeddings[row['Idx_Headline1']] - stored_embeddings[row['Idx_Headline2']], axis=1) previous try with difference
concatenated_vector = torch.cat([vec1, vec2], dim=1)
concatenated_vector.shape #shape is double, so it must be fine :)
#2 Predicting Headline-Winner based on SBert Embeddings with Logistic Regression
headline_more_clicks = torch.tensor(pairs['headline1_more_clicks'])
print(headline_more_clicks.shape)
#embs = torch.stack(pairs['embedding_diff'].tolist()) #because we have a column where each row is a tensor so we kinda unpack them.
#print(embs.shape)

X_train, X_test, y_train, y_test = train_test_split(embs,headline_more_clicks, test_size=0.2)
logistic = LogisticRegression()
logistic.fit(X_train, y_train)
predicted_logistic = logistic.predict(X_test)
accuracy_logistic = accuracy_score(predicted_logistic,y_test)
print("Accuracy for clicks difference:", accuracy_logistic) 


#3 Predicting Click difference based on SBert Embeddings with Ridge Regression
## check shape matching and turning into tensors to work
clicks_diff = torch.tensor(pairs['click_difference'])
print(clicks_diff.shape)

X_train, X_test, y_train, y_test = train_test_split(concatenated_vector,clicks_diff, test_size=0.2)
ridge_model_diff =RidgeCV(alphas=[0.001,0.002,0.005,0.01,0.05,0.07,0.2,0.4,0.6, 1],store_cv_values=True)
ridge_fit_diff = ridge_model_diff.fit(X_train, y_train)
ridge_fit_diff.score(X_train,y_train)
ridge_predictions_diff = ridge_model_diff.predict(X_test)
ridge_rmse_diff = mean_squared_error(y_test, ridge_predictions_diff)
print("Ridge Regression MSE for clicks difference:", ridge_rmse_diff) #result doesnt seem to be that much better so I am not convinced if I did the right thing


#4 Predicting Click difference based on SBert Embeddings with Linear Regression
X_train, X_test, y_train, y_test = train_test_split(concatenated_vector,clicks_diff, test_size=0.2)
lin_model_diff = LinearRegression()
lin_fit_diff = lin_model_diff.fit(X_train, y_train)
lin_fit_diff.score(X_train,y_train)
lin_predictions_diff = lin_model_diff.predict(X_test)
lin_rmse_diff = mean_squared_error(y_test, lin_predictions_diff)
print("Linear Regression MSE for clicks difference:", lin_rmse_diff)




# Fine Tuning Pretrained Model 

with open("/Users/tonia/Dropbox/2023WS_Ash_Research_Causal_Predictor/causal_headline_evaluator/headline_pair_indices.csv", "r") as fIn:
    reader = csv.DictReader(fIn, delimiter=",", quoting=csv.QUOTE_NONE)
    input_examples = []
    for row in reader:
        print(row)
        score = int(row['click_difference'])  
        inp_example = InputExample(texts=[df.loc[int(row['Idx_Headline1']), "headline"], df.loc[int(row['Idx_Headline2']), "headline"]], label=score)

        input_examples.append(inp_example)
        # if row['split'] == 'dev':
        #     dev_samples.append(inp_example)
        # elif row['split'] == 'test':
        #     test_samples.append(inp_example)
        # else:
        #     train_samples.append(inp_example)

# Still not fully working here..
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

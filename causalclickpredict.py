
from sentence_transformers import SentenceTransformer
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
import scipy.stats as stats
path = 'C:/Projects/CausalClicker/causal_headline_evaluator'
dataset = "C:/Users/mldem/Downloads/upworthy-archive-datasets/upworthy-archive-confirmatory-packages-03.12.2020.csv"
# Set random seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

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
df = pd.read_csv(dataset, low_memory=False)
#Delete some unnecessary columns
df.columns
delete_cols = ["created_at","updated_at","share_text","square"]
df = df.drop(columns=delete_cols)
#Create a new column for clickrate
df["clickrate"] = round((df["clicks"]/ df["impressions"]),ndigits=3)
df.columns
clicks =torch.tensor(df.clicks.values)

#Embeddings
#Extract only headlines
#headlines =df.headline.values
#embeddings = model.encode(headlines,convert_to_tensor=True,batch_size=32,show_progress_bar=True)
#embeddings.shape

#To Save embeddings
#with open('full_embeddings.pkl', "wb") as fOut:
     #pickle.dump({'headlines': headlines, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
#To open embeddings
with open(path+'/full_embeddings.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    stored_sentences = stored_data['headlines']
    stored_embeddings = stored_data['embeddings']


# 1. Predicting clicks from headline embeddings with ridge regression 
# Model
X_train, X_test, y_train, y_test = train_test_split(stored_embeddings, clicks, test_size=0.2)
# Ridge Model
ridge_model =RidgeCV(alphas=[0.001,0.002,0.005,0.01,0.05,0.07,0.2,0.4,0.6, 1, 10],store_cv_values=True)
ridge_model.fit(X_train, y_train)
ridge_model.score(X_train,y_train) #0.1629
predictions = ridge_model.predict(X_test) #alpha = 10
rmse = mean_squared_error(y_test, predictions, squared=False)
print("Ridge Regression MSE for click difference:", rmse)
print("Ridge Regression R2 for click difference:", r2_score(y_true=y_test, y_pred=predictions))

df["predictions"] = ridge_model.predict(stored_embeddings)
stats.spearmanr(df.sort_values(["predictions"]).loc[:,"headline"],df.sort_values(["clicks"]).loc[:,"headline"])

#last 20
print("Last 20 predicted:", df.sort_values(["predictions"]).loc[:,['headline',"clicks"]][:20])
print("Last 20 true:",df.sort_values(["clicks"]).loc[:,['headline',"clicks"]][:20])
#first 20
print("Top 20 predicted:",df.sort_values(["predictions"]).loc[:,['headline',"clicks"]][-20:])
print("Top 20 true:",df.sort_values(["clicks"]).loc[:,['headline',"clicks"]][-20:])

# Ridge with clickrate instead of clicks
# Model

clickrate =torch.tensor(df.clickrate.values)

#X_train, X_test, y_train, y_test = train_test_split(stored_embeddings,clickrate, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(stored_embeddings,torch.log(clickrate+1), test_size=0.2)
# Ridge Model

ridge_model_clickrate =RidgeCV(alphas=[0.001,0.002,0.005,0.01,0.05,0.07,0.2,0.4,0.6, 1, 10],store_cv_values=True)
ridge_model_clickrate.fit(X_train, y_train)
ridge_model_clickrate.score(X_train,y_train) #0.1629
predictions_clickrate = ridge_model_clickrate.predict(X_test) #alpha = 10
rmse_clickrate = mean_squared_error(y_test, predictions_clickrate)

predictions_clickrate_all = ridge_model_clickrate.predict(stored_embeddings)

df["predictions_clickrate_all"] = predictions_clickrate_all

df["predictions_clickrate_als"] = np.exp(df["predictions_clickrate_all"])
df["predictions_clickrate_als"].min()
for i in range(predictions_clickrate_all.shape[0]):
    if predictions_clickrate_all[i] < 0:
        predictions_clickrate_all[i] = 0
        
df["predictions_clickrate_all"] = predictions_clickrate_all

print("Ridge Regression MSE for click difference:", rmse_clickrate)
print("Ridge Regression R2 for click difference:", r2_score(y_true=y_test, y_pred=predictions_clickrate))
print(predictions_clickrate_all.min())

#last 20
print("Last 20 predicted:", df.sort_values(["predictions_clickrate"]).loc[:,['headline',"clickrate","predictions_clickrate"]][:20])
print("Last 20 true:",df.sort_values(["clickrate"]).loc[:,['headline',"clickrate","predictions_clickrate"]][:20])
#first 20
print("Top 20 predicted:",df.sort_values(["predictions"]).loc[:,['headline',"clickrate","predictions_clickrate"]][-20:])
print("Top 20 true:",df.sort_values(["clickrate"]).loc[:,['headline',"clickrate"]][-20:])



# Linear Model
linear_model =LinearRegression()
linear_model.fit(X_train, y_train)
linear_model.score(X_train,y_train) #0.1643
predictions = linear_model.predict(X_test)
rmse = mean_squared_error(y_test, predictions, squared=False)
print("Linear Regression MSE for clicks:", rmse)
print("Linear Regression R2 for clicks:", r2_score(y_true=y_test, y_pred=predictions))

#Visualizing predicted clicks vs actual clicks
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
# print stuff. 
predictions = ridge_model.predict(stored_embeddings)

# visualize real and predicted values
fig, ax = plt.subplots()
sns.scatterplot(x = predictions, y = df['clicks'], ax=ax)
ax.set_xlim(0,800)

#Pair headlines based on clickability_test_id and eyecatcher_id
#Import dataset with pairs
df_pairs = pd.read_csv(path+"/headline_pair_indices.csv")
#Compute vector difference
embedding_diff = torch.stack(df_pairs.apply(lambda row: stored_embeddings[row['Idx_Headline1']] - stored_embeddings[row['Idx_Headline2']], axis=1).tolist()) 
#because we have a column where each row is a tensor so we kinda unpack them.
#Sort pairs s.t. headline1 is headline with more clicks. 
df_sorted_pairs = df_pairs.copy()
df_sorted_pairs.loc[~df_sorted_pairs["headline1_more_clicks"], ['Idx_Headline1', 'Idx_Headline2']] = df_sorted_pairs.loc[~df_sorted_pairs["headline1_more_clicks"], ['Idx_Headline2', 'Idx_Headline1']].values   

#2 Predicting Headline-Winner based on SBert Embeddings with Logistic Regression

headline1_more_clicks = torch.tensor(df_pairs['headline1_more_clicks'])
X_train, X_test, y_train, y_test = train_test_split(embedding_diff, headline1_more_clicks, test_size=0.2)
logistic = LogisticRegression(max_iter=200)
logistic.fit(X_train, y_train)
predicted_logistic = logistic.predict(X_test)
accuracy_logistic = accuracy_score(predicted_logistic,y_test)
print("Accuracy predicting winner:", accuracy_logistic)


#3 Predicting Click difference based on SBert embeddings with Ridge Regression
## check shape matching and turning into tensors to work
clicks_diff = torch.tensor(abs(df_pairs['click_difference']))
sorted_embedding_diff = torch.stack(df_sorted_pairs.apply(lambda row: stored_embeddings[row['Idx_Headline1']] - stored_embeddings[row['Idx_Headline2']], axis=1).tolist()) 

#Based on difference vector
X_train, X_test, y_train, y_test = train_test_split(sorted_embedding_diff, clicks_diff, test_size=0.2)
ridge_model_diff =RidgeCV(alphas=[0.001,0.002,0.005,0.01,0.05,0.07,0.2,0.4,0.6, 1, 10],store_cv_values=True)
ridge_fit_diff = ridge_model_diff.fit(X_train, y_train)
ridge_fit_diff.score(X_train,y_train) #0.09528
ridge_predictions_diff = ridge_model_diff.predict(X_test)
ridge_rmse_diff = mean_squared_error(y_test, ridge_predictions_diff)
print("Ridge Regression MSE for clicks difference:", ridge_rmse_diff)
print("Ridge Regression R2 for click difference:", r2_score(y_true=y_test, y_pred=ridge_predictions_diff)) 

#show headlines that scored highly and compare with headlines scored low - using the whole dataset
ridge_predictions_full = ridge_model_diff.predict(sorted_embedding_diff)

df_sorted_pairs["predictions"] = ridge_model_diff.predict(sorted_embedding_diff)
#last 20
df.loc[df_sorted_pairs.sort_values(["predictions"]).loc[:,'Idx_Headline1'][:20], ['headline', 'clicks',"predictions"]]
#first 20
df.loc[df_sorted_pairs.sort_values(["predictions"]).loc[:,'Idx_Headline1'][-20:], ['headline', 'clicks',"predictions"]]


# Compare headline ranking between true click difference and predicted click difference
## there has to be a better way!
predicted_ranking = df_sorted_pairs.sort_values(['predictions'])['Idx_Headline1'].astype(str).values+df_sorted_pairs.sort_values(['predictions'])['Idx_Headline2'].astype(str).values
true_ranking = df_sorted_pairs.sort_values(['click_difference'])['Idx_Headline1'].astype(str).values+df_sorted_pairs.sort_values(['click_difference'])['Idx_Headline2'].astype(str).values

print("Spearman correlation is",spearmanr(predicted_ranking, true_ranking))





# Extra: Prediction based on concatenated full embeddings
## Here we need to make sure the headline ordering is correct! 
#Compute concatenated embeddings of the pairs
vec1 = df_sorted_pairs.apply(lambda row:(stored_embeddings[row['Idx_Headline1']]), axis=1)
vec1= torch.stack(vec1.tolist())
vec2 = df_sorted_pairs.apply(lambda row:(stored_embeddings[row['Idx_Headline2']]), axis=1)
vec2= torch.stack(vec2.tolist())
concatenated_vector = torch.cat([vec1, vec2], dim=1)
print(concatenated_vector.shape) 
#torch.corrcoef(concatenated_vector)


#Based on concatenated full embeddings
X_train, X_test, y_train, y_test = train_test_split(concatenated_vector, clicks_diff, test_size=0.2)
# Ridge Model
ridge_model_diff =RidgeCV(alphas=[0.001,0.002,0.005,0.01,0.05,0.07,0.2,0.4,0.6, 1, 10],store_cv_values=True)
ridge_fit_diff = ridge_model_diff.fit(X_train, y_train)
ridge_fit_diff.score(X_train,y_train) #0.01513
ridge_predictions_diff = ridge_model_diff.predict(X_test)
ridge_rmse_diff = mean_squared_error(y_test, ridge_predictions_diff)
print("Ridge Regression MSE for clicks difference:", ridge_rmse_diff)
print("Ridge Regression R2 for click difference:", r2_score(y_true=y_test, y_pred=ridge_predictions_diff)) 

# Linear Model
lin_model_diff = LinearRegression()
lin_fit_diff = lin_model_diff.fit(X_train, y_train)
lin_fit_diff.score(X_train,y_train)
lin_predictions_diff = lin_model_diff.predict(X_test)
lin_rmse_diff = mean_squared_error(y_test, lin_predictions_diff)
print("Linear Regression MSE for clicks difference:", lin_rmse_diff) #Result is better with linear regression
print("Linear Regression R2 for click difference:", r2_score(y_true=y_test, y_pred=lin_predictions_diff)) 




# Fine Tuning Pretrained Model 

#Define model - not pretrained
# word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension()) #pooling method, ins sbert they use this instead of cls?
# dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Sigmoid())
# model = SentenceTransformer(modules=[word_embedding_model, pooling_model,dense_model]) #making the sentence transformer


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

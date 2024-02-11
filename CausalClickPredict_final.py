from sentence_transformers import SentenceTransformer
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error
import numpy as np
from sklearn.linear_model import RidgeCV, LinearRegression
import torch 
import pickle
import random
import langid
import re
import matplotlib.pyplot as plt
import scipy.stats as stats 

path = "C:/Projects/CausalClicker/causal_headline_evaluator"
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
# Set random seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

#DATA CLEANING & MANIPULATION
#Load data
df = pd.read_csv("upworthy-archive-confirmatory-packages-03.12.2020.csv", low_memory=False)
#adding index
df.reset_index(inplace=True,names=["embedding_id"])
#removing whitespace at the end or at the beginning of the sentence
df.headline = df.headline.apply(lambda h: h.strip())
#remove rows without eyecatcher_id
has_eyecatcher_id = df['eyecatcher_id'].notna()
df = df.loc[has_eyecatcher_id]
#Create a new column for clickrate
df["clickrate"] = round((df["clicks"]/ df["impressions"]), ndigits=10)
#filter data based on same clickability_id and eyecatcher_id
df['headline_count'] = df.groupby(['clickability_test_id', 'eyecatcher_id']).headline.transform('count')
# filter for all headlines with at least 2 pairs. 
df = df.loc[df['headline_count']>=2, ['clickability_test_id', 'excerpt', 'headline', 'lede', 'eyecatcher_id', 'clicks', 'headline_count',"embedding_id","clickrate","impressions"]]
# drop all rows with same headline, clickability_test_id and eyecatcher_id (full duplicates)
cti = df[df.duplicated(subset=["headline","clickability_test_id","eyecatcher_id"],keep=False)].clickability_test_id
eti = df[df.duplicated(subset=["headline","clickability_test_id","eyecatcher_id"],keep=False)].eyecatcher_id
df = df[~(df['clickability_test_id'].isin(cti) & df['eyecatcher_id'].isin(eti))]
#if there are duplicates among different group articles, only keeping that group that has the most headlines within.
df = df.sort_values(by='headline_count', ascending=False)
dupl_headline = df[df.duplicated(subset=["headline"])] 
ids = dupl_headline["clickability_test_id"]
eid = dupl_headline["eyecatcher_id"]
mask = df['clickability_test_id'].isin(ids)&df['eyecatcher_id'].isin(eid)
df = df[~mask]


# check that there are no more duplicate headlines. 
len(df.headline.drop_duplicates()) == len(df)

# Checking for spanish headlines (only works for Windows,comment this part if you are using MAC and use line 81 to remove the headlines)
headlines = df['headline'].astype(str).tolist()
results = [langid.classify(headline) for headline in headlines]
log_file = "output_log.txt"

with open(log_file, "w", encoding="utf-8") as log:
    for headline,result in zip(headlines,results):
        language = result
        log.write(f"Sentence: {headline}, Identified Language: {language}\n")
#These are the Spanish headlines:
#Como Decir Todo … Sin Pronunciar Ninguna Palabra
#Ve La Protesta Que Todos Deben Conocer, Pero Que Nadie Puede Oir
#En Vez De ‘Sí Se Puede,’ Ya Es ‘Sí Se Shhhhhhhhh’?
#¿Cómo Se Dice ‘Nada’ En Español?
#removing the sapnish headlines, they are within the same experiment
df = df[df["clickability_test_id"] != "51436075220cb800020007b3"]
print("Final number of observations:",df.shape[0])
# Make tensor for click rate
clickrate = torch.tensor(df.clickrate.values)


# Calculating mean per clickability_test_id and eyecatcher_id
df["means"] = df.groupby(["clickability_test_id","eyecatcher_id"])["clickrate"].transform("mean")
df["adjusted_clickrate"] = df['clickrate']-df['means']
# Make tensor for adjusted click rate
adjusted_clickrate = torch.tensor(df.adjusted_clickrate.values)


#GENERATING EMBEDDINGS
#Define model - pretrained
model = SentenceTransformer('all-mpnet-base-v2')
#Extract only headlines
""" headlines =df.headline.values
embeddings = model.encode(headlines,convert_to_tensor=True,batch_size=32,show_progress_bar=True)

#To Save embeddings
with open('duplicates_removed_embeddings.pkl', "wb") as fOut:
     pickle.dump({'headlines': headlines, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
 """
#Load prerun embeddings of all-mpnet-base-v2
with open('duplicates_removed_embeddings.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    stored_headlines = stored_data['headlines']
    stored_embeddings = stored_data['embeddings']

# MODELS

# 1. Correlational models
# Splitting data into test and train
X_train, X_test, y_train, y_test = train_test_split(stored_embeddings, clickrate, test_size=0.2, random_state=seed)
# 1.1. Ridge Regression
ridge_model =RidgeCV(alphas=[0.001,0.002,0.005,0.01,0.05,0.07,0.2,0.4,0.6, 1, 10],store_cv_values=True)
ridge_model.fit(X_train, y_train)
ridge_model.score(X_train,y_train)
predictions = ridge_model.predict(X_test)
rmse = mean_squared_error(y_test, predictions)
print("Ridge Regression R2 is:", r2_score(y_true=y_test, y_pred=predictions))
df["predictions_ridge"] = ridge_model.predict(stored_embeddings)

# 1.2. Linear Model
linear_model =LinearRegression()
linear_model.fit(X_train, y_train)
linear_model.score(X_train,y_train)
predictions = linear_model.predict(X_test)
rmse = mean_squared_error(y_test, predictions, squared=False)
print("Linear Regression R2 is:", r2_score(y_true=y_test, y_pred=predictions))
df["predictions_linear"] = linear_model.predict(stored_embeddings)

# 2. Causal models
# Splitting data into test and train
X_train, X_test, y_train, y_test = train_test_split(stored_embeddings, adjusted_clickrate, test_size=0.2, random_state=seed)

# 2.1. Causal ridge model
causal_ridge_model =RidgeCV(alphas=[0.001,0.002,0.005,0.01,0.05,0.07,0.2,0.4,0.6, 1, 10],store_cv_values=True)
causal_ridge_model.fit(X_train, y_train)
causal_ridge_model.score(X_train,y_train)
causal_predictions_rg = causal_ridge_model.predict(X_test)
rmse = mean_squared_error(y_test, causal_predictions_rg)
print("Causal Ridge Regression R2 is:", r2_score(y_true=y_test, y_pred=causal_predictions_rg))
df["causal_predictions_ridge"] = causal_ridge_model.predict(stored_embeddings)

# 2.2. Causal linear model
causal_linear_model =LinearRegression()
causal_linear_model.fit(X_train, y_train)
causal_linear_model.score(X_train,y_train)
causal_predictions_lm = causal_linear_model.predict(X_test)
rmse = mean_squared_error(y_test, causal_predictions_lm, squared=False)
print("Causal Linear Regression R2 is:", r2_score(y_true=y_test, y_pred=causal_predictions_lm))
df["causal_predictions_linear"] = causal_linear_model.predict(stored_embeddings)


# MODELS exploration

# 1.1. Compare bottom 300 causal model with correlational model - Ridge Regression
last300_pred = df.sort_values(["predictions_ridge"],ascending=True).loc[:,['headline']][:300]
last300_pred_causal = df.sort_values(["causal_predictions_ridge"],ascending=True).loc[:,['headline']][:300]
#Checking for intersection
print("The overlap for bottom 300 Ridge is ",round(np.intersect1d(last300_pred.values,last300_pred_causal.values).size/300*100,2),"%")
# 1.2 Compare top 300 causal model with correlational model - Ridge Regression
first300_pred = df.sort_values(["predictions_ridge"],ascending=False).loc[:,['headline']][:300]
first300_pred_causal = df.sort_values(["causal_predictions_ridge"],ascending=False).loc[:,['headline']][:300]
#Checking for intersection
print("The overlap for top 300 Ridge is ",round(np.intersect1d(first300_pred.values,first300_pred_causal.values).size/300*100,2),"%")

# 2. Exploring top 20 most frequent words and perturbing them in order to obtain a new adjusted click rate
import nltk
from nltk.corpus import stopwords
from collections import Counter 
#nltk.download('stopwords')
# Capitalize all words in headline. (we do not have to regenerate embeddings because it does not care) 
df["headline_cap"] = df.headline.str.title()
punctuation = { ".", ",", "?", "!", ";", ":"}
stop_words = stopwords.words('english')
capitalized = [word.capitalize() for word in stop_words]
ignore = stop_words + capitalized + list(punctuation)
split_it = [word for headline_cap in df.headline_cap for word in re.findall(r"[\w']+|[.,!?;]", headline_cap) if word not in ignore]
# Pass the split_it list to instance of Counter class. 
counter = Counter(split_it) 
most_occur = np.array(counter.most_common(50))
# Pull all headlines with the missing word - top 50
words = most_occur[:20,0]
words_removal_matrix_stopwords = pd.DataFrame(columns=["Word","Median","Abs Mean","Max diff","Min diff"])

""" i = 0 
for word in words:
    
    word_headlines = df[[word in re.findall(r"[\w']+|[.,!?;]", headline_cap) for headline_cap in df.headline_cap]].copy()
    word_headlines['removed'] = [re.sub(r'\b{}\b'.format(word), '',headline_cap) for headline_cap in word_headlines.headline_cap]
    model = SentenceTransformer('all-mpnet-base-v2')
# Run new embedding space for the headlines with words removed. 
# Generating embeddings for each headline, where the most frequent word is removed
    removed_word_embeddings = model.encode(word_headlines.removed.values, convert_to_tensor=True,batch_size=32,show_progress_bar=True)
# Predict number of clicks with these new embeddigns and add as column to df. 
    word_headlines['removed_word_clickrate'] = causal_ridge_model.predict(removed_word_embeddings)
    removed_word_diff = word_headlines.removed_word_clickrate - word_headlines.causal_predictions_ridge
# Table: word, mean and max (removing the word increases the clickrate) and min (removing the word decreases the clickrate.)
   
    words_removal_matrix_stopwords.loc[i,["Word","Median","Abs Mean","Max diff","Min diff"]] = word,np.median(removed_word_diff),np.mean(np.abs(removed_word_diff)),np.max(removed_word_diff),np.min(removed_word_diff)
    i += 1
print(words_removal_matrix_stopwords)
 """

# 3. Topic Modelling
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
import spacy
from umap import UMAP
nlp = spacy.load('en_core_web_sm')
#model
model = SentenceTransformer('all-mpnet-base-v2')
#reduce dimensionality
umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine',random_state=42)
#Tokenize topics
vectorizer_model = CountVectorizer(ngram_range=(1,2),
                                   stop_words=list(nlp.Defaults.stop_words))
#cluster reducued embeddings
cluster_model = HDBSCAN(min_cluster_size=150, min_samples=5, metric='euclidean', prediction_data = True)
#representation model
from bertopic.representation import KeyBERTInspired

representation_model = KeyBERTInspired()
topic_model = BERTopic(embedding_model=model, language='English',
                       umap_model = umap_model,
                       representation_model = representation_model,
                       vectorizer_model = vectorizer_model,
                       n_gram_range=(1,2), min_topic_size = 100,
                       hdbscan_model=cluster_model)

topic, probs = topic_model.fit_transform(df.headline, stored_embeddings.numpy())

# Reducing outliers
new_topics = topic_model.reduce_outliers(df.headline.tolist(),topic, strategy="c-tf-idf", threshold = 0.05)
topic_model.update_topics(df.headline.tolist(), topics=new_topics)
# Topic modelling dataframe
topic_info = topic_model.get_document_info(df.headline)
embeddings_topics =topic_model.topic_embeddings_
topic_modeling = pd.merge(df, topic_info, how='left', left_on='headline', right_on='Document')
print("Number of headlines per topic:",topic_model.get_topic_freq().sort_values("Count",ascending=False))


# Extracting top and bottom 300
ridge_topic = topic_modeling.sort_values(["predictions_ridge"],ascending=False)[:300]
causal_ridge_topic = topic_modeling.sort_values(["causal_predictions_ridge"],ascending=False)[:300]
print(ridge_topic["Topic"].value_counts())
print(causal_ridge_topic["Topic"].value_counts())
# Extracting unique topics and removing the outliers group from it - Ridge Regression 
unique_topics_ridge = ridge_topic["Topic"].unique()
unique_topics_ridge = np.delete(unique_topics_ridge,np.where(unique_topics_ridge==-1))
unique_topics_causal_ridge = causal_ridge_topic["Topic"].unique()
unique_topics_causal_ridge = np.delete(unique_topics_causal_ridge,np.where(unique_topics_causal_ridge==-1)) 

# 3.1 Topic Modelling H1: The number of unique topics of the top 300 headlines in the causal model does not differ sufficiently than the number of unique topics from the correlational model.
print("Ratio of topics ridge/causal ridge:",round((unique_topics_ridge.size/unique_topics_causal_ridge.size)*100,2),"%")

# 3.2. Topic Modelling - H2: The histograms of the topics from both models do not differ sufficiently.
plt.figure()
ridge_topic["Topic"].value_counts().plot(kind="bar",title="Histogram of topics from Ridge Model",ylim=(0,90))
plt.savefig('hist_topics_ridge.png')
plt.figure()
causal_ridge_topic["Topic"].value_counts().plot(kind="bar",title="Histogram of topics from Causal Ridge Model",ylim=(0,90))
plt.savefig('hist_topics_causal_ridge.png')


# 4. Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def apply_sentiment_analysis(headlines, model_name = 'Ridge'):
    count_positive = 0
    count_negative = 0
    count_neutral = 0
    for headline in headlines:
        vs = analyzer.polarity_scores(headline)
        sentiment_dict = analyzer.polarity_scores(headline)
        if sentiment_dict['compound'] >= 0.05 :
            count_positive+=1
     
        elif sentiment_dict['compound'] <= - 0.05 :
            count_negative+=1
     
        else:
            count_neutral+=1
    print(f"Within the {len(headlines)} headlines (with the highest clickrate)")
    print(f"for the {model_name} model:")
    print(f" {count_positive} Positive headlines")
    print(f" {count_neutral} Neutral headlines ")
    print(f" {count_negative} Negative headlines")
    print(f"Ratios: {round(count_positive*100/len(headlines))}% positive {round(count_neutral*100/len(headlines))}% neutral {round(count_negative*100/len(headlines))}% negative")

# 4.1. Comparing Sentiment of Headlines with highest clickrates
# Overall distribution
apply_sentiment_analysis(topic_modeling.headline, model_name = "no")
# Top 300 headlines wrt clickrate
apply_sentiment_analysis(topic_modeling.sort_values(["clickrate"], ascending=False).headline[:300], model_name = "no")
# Top 300 headlines
apply_sentiment_analysis(ridge_topic.headline)
# Top 300 headlines
apply_sentiment_analysis(causal_ridge_topic.headline, model_name = "Causal Ridge")



# 4.2. Special: Gender Analysis
word = "Women"
new_word = "Men"
word_headlines = df[[word in re.findall(r"[\w']+|[.,!?;]", headline) for headline in df.headline_cap]].copy()   
word_headlines['new_word'] = [re.sub(r'\b{}\b'.format(word),new_word,headline) for headline in word_headlines.headline]
model = SentenceTransformer('all-mpnet-base-v2')
new_word_embeddings = model.encode(word_headlines.new_word.values, convert_to_tensor=True,batch_size=32,show_progress_bar=True)
#4. Predict number of clicks with these new embeddigns and add as column to df. 
word_headlines['new_word_clickrate'] = causal_ridge_model.predict(new_word_embeddings)
word_headlines["new_word_diff"] = word_headlines.new_word_clickrate - word_headlines.causal_predictions_ridge
#5. Table: word, mean and max (removing the word increases the clickrate) and min (removing the word decreases the clickrate.)
word_headlines["sentiment"] = word_headlines.headline.apply(lambda h: 'positive' if analyzer.polarity_scores(h)['compound'] >= 0.05 else 'negative' if analyzer.polarity_scores(h)['compound'] <= -0.05 else "neutral")

# Difference in clickrate prediction is more or less normally distributed
plt.figure()
plt.hist(word_headlines.new_word_diff, bins=20)
plt.xlabel('Difference in predicted adjusted clickrate')
plt.savefig('new_word_diff_hist.png')
# Paired T-Test
print("P-value of the paired t-test is:",stats.ttest_rel(word_headlines.causal_predictions_ridge, word_headlines.new_word_clickrate))

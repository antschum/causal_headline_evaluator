from sentence_transformers import SentenceTransformer
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import train_test_split
from sklearn import metrics
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
import random

# Set random seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Load Model
transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
ridge = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10],  store_cv_values=True)

# Load Data
df = pd.read_csv("/Users/tonia/Dropbox/2023WS_Ash_Research_Causal_Predictor/osfstorage-archive/upworthy-archive-datasets/upworthy-archive-confirmatory-packages-03.12.2020.csv", delimiter=',', low_memory=False)

# Report the number of sentences.
print('Number of training sentences: {:,}\n'.format(df.shape[0]))

# Extract Sentences and Labels
headlines = df.headline.values[:10000]
clicks = df.clicks.values[:10000]

# Embeddings - training with 10 000
embeddings = transformer_model.encode(headlines)

# train, test split (can also do this with multiple datasets - because we have a lot)
X_train, X_test, y_train, y_test = train_test_split(embeddings, clicks, test_size=0.2, random_state=seed)

ridge.fit(X_train, y_train)
predicted = ridge.predict(X_test)
print("Ridge Regression MSE:",metrics.mean_squared_error(y_test, predicted))
print("Ridge Regression R2:",metrics.r2_score(y_test, predicted))

# Mean Ridge Cross Validation Values
ridge.cv_values_.mean(axis=0)


# Some Ideas: 
# - We are currently not training the model to explicitely learn a transformation that helps predict clicks 
# - We can create our own transformer model with pooling layer and linear layer on top - maybe this gives us better stuff. 
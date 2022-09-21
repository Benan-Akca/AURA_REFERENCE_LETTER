
#%%
import numpy as np
import pandas as pd
import en_core_web_sm
from nltk.tokenize import sent_tokenize
import re
import bokeh
from bokeh.io import output_file, show,save
from bokeh.plotting import figure
from bokeh.transform import factor_cmap, factor_mark

from bokeh.models import HoverTool, ColumnDataSource, FactorRange,ColumnDataSource, Range1d
from bokeh.layouts import row, column, gridplot
from bokeh.models.widgets import Tabs, Panel
#import tensorflow as tf 1

from tqdm import tqdm
import string
import matplotlib.pyplot as plt
from operator import itemgetter
from itertools import islice,product, chain
from termcolor import colored
import spacy
import os
import time

import collections
from collections import OrderedDict

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

#from gensim.models import KeyedVectors
#from gensim import models

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

outdir = 'grace_plots'
import os

model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
sp = spacy.load('en_core_web_sm')



from transformers import AutoModelForSequenceClassification
#from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request



# Tasks:
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# download label mapping
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# PT
model_sentiment = AutoModelForSequenceClassification.from_pretrained(MODEL)

def get_sentiment_score(text):
    #text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    if encoded_input["input_ids"].shape[1]>512:
        encoded_input["input_ids"] = encoded_input["input_ids"][:,0:512]
        encoded_input["attention_mask"] = encoded_input["attention_mask"][:,0:512]
    output = model_sentiment(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return scores

def get_sentiment_df(df,sentence_column,drop_sent_scores = True):
    scores_list = [0]*df[sentence_column].shape[0]
    for inx,sentence in enumerate(df[sentence_column].values):
        score = get_sentiment_score(sentence)
        scores_list[inx] = score
    scores_array = np.array(scores_list)

    df[["negative","neutral","positive"]] = scores_array

    df["sentiment"] = [labels[i] for i in np.argmax(scores_array,axis=1)]

    df = df.loc[df["sentiment"] !="neutral"]

    df["score"] = [-row.negative if row.sentiment == "negative" else row.positive for inx,row in df.iterrows()]
    
    if drop_sent_scores:
        df = df.drop(columns=["negative","neutral","positive"])
        
    return df

def sentiment_analysis(df_new):
    df = df_new.copy()


    p=[]
    negs=[]
    pos=[]

    df_root = []
    
    aspect_neg_counts = collections.defaultdict(int)



    df_senti = get_sentiment_df(df)
    df_senti = df_senti.drop_duplicates(keep="first")





    df_all = df_senti
    df_all.loc[(df_all.sentiment == "negative") & (df_all.subcategory == "brag on a bus driver"),"subcategory"]= "bus services"

    bins = [-1, -0.8, -0.5, 0, 0.5, 0.6, 0.8, 1]
    names = ['poor', 'bad', 'below average',"average", 'somewhat good','good',"great"]

    df_all['sentiment_score'] = pd.cut(df_all['score'], bins, labels=names)
    df_all["abs_score"] = df_all.score.abs()
    df_all= df_all.loc[df_all.groupby(["msg","district"],as_index=False)[["abs_score"]].idxmax().abs_score]
    return df_all
# %%

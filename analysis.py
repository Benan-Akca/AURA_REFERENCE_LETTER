#%%

from nltk.tokenize import sent_tokenize
import pandas as pd
import stanza
import pandas as pd 
pd.options.display.max_rows = 2000
pd.options.display.max_columns = 2000
pd.options.display.max_colwidth = 2000
nlp = stanza.Pipeline('en', processors='tokenize,pos')

df  = pd.read_excel("alperen_100.xlsx")
df = df.dropna(subset = ["lor1"])
df["report_id"] = df.index
# %%
lor = df.iloc[0]["lor1"]
# %%
import re
def clean_text(lor):
    lor = re.sub(r"\n"," ",lor)
    lor = re.sub(r"\"","",lor)
    lor = re.sub(r"\\","",lor)
    lor = re.sub(r"\(","",lor)
    lor = re.sub(r"\)","",lor)
    lor = re.sub(r"\s+"," ",lor)
    return lor
# %%
lor
# %%
def tokenize_sentences(text):
    sentence_list = sent_tokenize(text)
    return sentence_list

def insert_space(text):
    new_text = re.sub(r"([a-zA-Z\)]\.)(\w)", r"\1 \2", text, count=50)
    new_text = re.sub(r"([0-9]\.)([A-Z]+)", r"\1 \2", new_text, count=20)
    new_text = re.sub(
        r"(?:(^[0-9]\.|^\s[0-9]\.))([a-zA-Z]+)", r"\1 \2", new_text, count=20
    )
    new_text = re.sub(r"(?:(^[0-9]\s|^\s[0-9]\s))", r"", new_text, count=20)
    new_text = re.sub(r"(?:\s[0-9]\.\s|^[0-9]\.\s)", r" ", new_text, count=20)
    new_text = new_text.lstrip().rstrip()
    return new_text


def get_df_sentence(df, column_name):
    df[column_name] = df[column_name].astype("string")
    df[column_name] = df[column_name].apply(insert_space)
    sentences = df[column_name].apply(tokenize_sentences)
    df_sentence = pd.DataFrame(sentences.tolist(), index=df.report_id).stack()
    df_sentence = df_sentence.reset_index()[[0, "report_id"]]
    df_sentence.columns = ["sentence_split", "report_id"]
    df_sentence["sentence_split"] = df_sentence["sentence_split"].map(
        lambda x: x.lstrip(".").rstrip(".")
    )
    df_sentence["sentence_split"] = df_sentence["sentence_split"].apply(str.lower)
    return df_sentence

#%%
df["lor1"] = df["lor1"].apply(clean_text)
df_bck_sentence = get_df_sentence(df, "lor1")
#%%
df_bck_sentence.sentence_split = df_bck_sentence.sentence_split.str.replace(r"\s*\/\s*", "/")
# %%

doc = nlp("he has keen eyes on films")
sent_list = [sent.text for sent in doc.sentences]
adjectives = [word.text for sent in doc.sentences for word in sent.words if word.xpos in ["JJ","JJR","JJS"]]
adverbs = [word.text for sent in doc.sentences for word in sent.words if word.xpos in ["RB","RBR","RBS"]] 

#%%
doc = nlp("he has keen eyes looks quickly")
#%%
doc = nlp("he has keen eyes looks quickly")
sent_list = [sent.text for sent in doc.sentences]
print([f'{word.text}_{word.xpos}' for sent in doc.sentences for word in sent.words])
# %%
# 1 lor1_adj lor1_adv lor2_adj,lor2_adv dye yen, sütunlar ekle.
# 2 cümle sayılarını ekle lor1_sentence_count etc.txt
# 3 noun chunkların içinde adject,ve veya adverv olanları çıkar
# Histogramları çıkart anova analiz vs yap

#%%

def get_sentence_adj_pos(sentence):
    doc = nlp(sentence)
    sent_list = [sent.text for sent in doc.sentences]
    adjectives = [word.text for sent in doc.sentences for word in sent.words if word.xpos in ["JJ","JJR","JJS"]]
    return adjectives

def get_sentence_adv_pos(sentence):
    doc = nlp(sentence)
    sent_list = [sent.text for sent in doc.sentences]
    adverbs = [word.text for sent in doc.sentences for word in sent.words if word.xpos in ["RB","RBR","RBS"]] 
    return adverbs


df_bck_sentence["adj"] = df_bck_sentence.sentence_split.apply(get_sentence_adj_pos)
df_bck_sentence["adv"] = df_bck_sentence.sentence_split.apply(get_sentence_adv_pos)
# %%
df_report_adj = df_bck_sentence.groupby('report_id')['adj'].agg(sum).reset_index()
# %%
df_report_adv = df_bck_sentence.groupby('report_id')['adv'].agg(sum).reset_index()
# %%
df_report = df_report_adj.merge(df_report_adv, left_on='report_id', right_on='report_id')
# %%
df_lor_1 = df[['report_id','applicant_id', 'application_year', 'birth_year', 'citizenship',
       'medical_school', 'residency_school', 'md_year', 'residency_year',
       'step1', 'step2', 'step3', 'step_old', 'IMG', 'class_rank', 'honors',
       'received_offer', 'language_skills', 'interests',
       'number_of_publications', 'speciality', 'gender','lor1',
       'lor1_position', 'lor1_university', 'lor1_gender']]
# %%
df_lor_ana = df_lor_1.merge(df_report,left_on='report_id', right_on='report_id')
#%%
df_lor_ana.to_excel("df_lor_analysis.xlsx")
# %%
df_lor1_male = df_lor_ana.loc[df_lor_ana.gender == "male"]
# %%
df_lor1_female  = df_lor_ana.loc[df_lor_ana.gender == "female"]
# %%
df_lor1_male.shape
#%%
df_lor1_female.shape
# %%
df_lor1_female_adj = pd.DataFrame(df_lor1_female.adj.sum(),columns=["adj"])
#%%
df_lor1_female_adv = pd.DataFrame(df_lor1_female.adv.sum(),columns=["adv"])

df_lor1_male_adj = pd.DataFrame(df_lor1_male.adj.sum(),columns=["adj"])

df_lor1_male_adv = pd.DataFrame(df_lor1_male.adv.sum(),columns=["adv"])

# %%
df_lor1_male_adj_freq = df_lor1_male_adj.value_counts()
# %%
df_lor1_female_adj_stats = pd.DataFrame({"frequency":df_lor1_female_adj.value_counts(),
                                        "percentage of statements per letter":(df_lor1_female_adj.value_counts()/df_lor1_female.shape[0])})
# %%
df_lor1_male_adj_stats = pd.DataFrame({"frequency":df_lor1_male_adj.value_counts(),
                                        "percentage of statements per letter":(df_lor1_male_adj.value_counts()/df_lor1_male.shape[0])})

#%%
df_lor1_male_adj_stats.to_excel("df_lor1_male_adj_stats.xlsx")
df_lor1_female_adj_stats.to_excel("df_lor1_female_adj_stats.xlsx")
# %%
import sentiment_score as ss
import importlib
#%%
importlib.reload(ss)
# %%
sentiment_male = ss.get_sentiment_df(df_lor1_male,"lor1",drop_sent_scores = False)
# %%
sentiment_female = ss.get_sentiment_df(df_lor1_female,"lor1",drop_sent_scores = False)
# %%
male_mean_score = sentiment_male.score.mean()

#%%
female_mean_score = sentiment_female.score.mean()
# %%
df_bck_sentence_sent = ss.get_sentiment_df(df_bck_sentence,"sentence_split",drop_sent_scores = False)
# %%
import numpy as np
import spacy
from spacy import displacy

def get_sentiment_score_from_list(word_list):
    score_list = [ss.get_sentiment_score(word) for word in word_list]
    scores_array = np.array(score_list)
    
    labels = ["negative","neutral","positive"]
    print(scores_array)
    
    score_list_new = []
    for i in scores_array:
        sentiment= labels[np.argmax(i)] 
        print(sentiment)
        score = np.max(i)
        
        if sentiment == "negative":
            print("negative")
            score =  -score
        elif sentiment == "neutral":
            print("neutral")
            score =  0
        elif sentiment == "positive":
            print("positive")
            score =  score   
        score_list_new.append(score)
    
    return score_list_new

get_sentiment_score_from_list(["good","paper"])


df_bck_sentence_sent["adj_scores"] = df_bck_sentence_sent.adj.apply(get_sentiment_score_from_list)

df_bck_sentence_sent[["adj","adj_scores"]]

ss.get_sentiment_score("left")



nlp = spacy.load("en_core_web_sm")
doc = nlp("He is an excellent, dedicated, and committed radiologist whose work product and work ethic are superior.")
displacy.serve(doc, style="dep")
# %% gender

#%%

import sentiment_score as ss
import importlib
#%%


import re
def clean_text(lor):
    lor = re.sub(r"\n"," ",lor)
    lor = re.sub(r"\"","",lor)
    lor = re.sub(r"\\","",lor)
    lor = re.sub(r"\(","",lor)
    lor = re.sub(r"\)","",lor)
    lor = re.sub(r"\s+"," ",lor)
    return lor


def tokenize_sentences(text):
    sentence_list = sent_tokenize(text)
    return sentence_list

def insert_space(text):
    new_text = re.sub(r"([a-zA-Z\)]\.)(\w)", r"\1 \2", text, count=50)
    new_text = re.sub(r"([0-9]\.)([A-Z]+)", r"\1 \2", new_text, count=20)
    new_text = re.sub(
        r"(?:(^[0-9]\.|^\s[0-9]\.))([a-zA-Z]+)", r"\1 \2", new_text, count=20
    )
    new_text = re.sub(r"(?:(^[0-9]\s|^\s[0-9]\s))", r"", new_text, count=20)
    new_text = re.sub(r"(?:\s[0-9]\.\s|^[0-9]\.\s)", r" ", new_text, count=20)
    new_text = new_text.lstrip().rstrip()
    return new_text


def get_df_sentence(df, column_name):
    df[column_name] = df[column_name].astype("string")
    df[column_name] = df[column_name].apply(insert_space)
    sentences = df[column_name].apply(tokenize_sentences)
    df_sentence = pd.DataFrame(sentences.tolist(), index=df.lor_id).stack()
    df_sentence = df_sentence.reset_index()[[0, "lor_id"]]
    df_sentence.columns = ["sentence_split", "lor_id"]
    df_sentence["sentence_split"] = df_sentence["sentence_split"].map(
        lambda x: x.lstrip(".").rstrip(".")
    )
    df_sentence["sentence_split"] = df_sentence["sentence_split"].apply(str.lower)
    return df_sentence


#%%


df1  = pd.read_excel(".\\Data\\alperen_100.xlsx")
df2  = pd.read_excel(".\\Data\\Master.xlsx")
df3  = pd.read_excel(".\\Data\\RESEARCH -taranan.xlsx")
#%%
df = pd.concat([df1,df2,df3],axis=0)

#%%
df = df.dropna(subset = ["lor1"])
#%%
applicant_columns = ['applicant_id', 'application_year', 'birth_year', 'citizenship',
       'medical_school', 'residency_school', 'md_year', 'residency_year',
       'step1', 'step2', 'step3', 'step_old', 'IMG', 'class_rank', 'honors',
       'received_offer', 'language_skills', 'interests',
       'number_of_publications', 'speciality', 'gender']

# %%
from collections import defaultdict
df_dic = defaultdict(list)
#%%
for inx,row in df.iterrows():
    for lor in ["lor1","lor2","lor3","lor4"]:
        for column in applicant_columns:
            df_dic[column].append(row[column])
        df_dic["lor"].append(row[lor])
        df_dic["lor_position"].append(row[lor + "_position"])
        df_dic["lor_university"].append(row[lor + "_university"])
        df_dic["lor_gender"].append(row[lor + "_gender"])
        df_dic["lor_number"].append(lor)
  
# %%
df_new = pd.DataFrame(df_dic)

# %%
df_new = df_new.dropna(subset=["lor"])

#%%
df_new.shape
#%%
df_new["lor"] = df_new["lor"].apply(clean_text)
#%%
df_new["lor_id"] = df_new.index
# %%
df_male = df_new.loc[(df_new.gender == "male")]
df_female = df_new.loc[df_new.gender == "female"]
# %%
#8min 47s
df_sentiment_male = ss.get_sentiment_df(df_male,"lor",drop_sent_scores = False)
#2min 50s
df_sentiment_female = ss.get_sentiment_df(df_female,"lor",drop_sent_scores = False)

# %%
df_sentiment_male.to_excel("df_sentiment_male.xlsx")
df_sentiment_female.to_excel("df_sentiment_female.xlsx")
# %%
df_sentiment_male.score.mean()
#0.922
# %%
df_sentiment_female.score.mean()
#0.920
#%%
from scipy.stats import ttest_ind

# %%
ttest_ind(df_sentiment_male['score'], df_sentiment_female['score'])
# Ttest_indResult(statistic=0.4477779636275707, pvalue=0.6544202583903076)

# %%
df_male_sentence = get_df_sentence(df_sentiment_male, "lor")
df_female_sentence = get_df_sentence(df_sentiment_female, "lor")
# %%
male_sentence_ratio = df_male_sentence.shape[0]/df_sentiment_male.shape[0]
# 18.22
#%%
female_sentence_ratio = df_female_sentence.shape[0]/df_sentiment_female.shape[0]
# 18.83
# %%
df_male_lor_male = df_sentiment_male.loc[df_sentiment_male["lor_gender"]=="male"]
df_male_lor_female = df_sentiment_male.loc[df_sentiment_male["lor_gender"]=="female"]
df_female_lor_male = df_sentiment_female.loc[df_sentiment_female["lor_gender"]=="male"]
df_female_lor_female = df_sentiment_female.loc[df_sentiment_female["lor_gender"]=="female"]
# %%
mm_score = df_male_lor_male.score.mean()
print("mm score", mm_score)
mf_score =  df_male_lor_female.score.mean()
print("mf score", mf_score)
fm_score = df_female_lor_male.score.mean()
print("fm score", fm_score)
ff_score = df_female_lor_female.score.mean()
print("ff score", ff_score)

""" 
mm score 0.9233926798046475
mf score 0.9181113643983824
fm score 0.9162155736576427
ff score 0.9328229874372482 """
# %%
ttest_ind(df_female_lor_female['score'], df_female_lor_male['score'])
#Ttest_indResult(statistic=1.6186915448626993, pvalue=0.10697964146838995)
# %%
# %%
df_male_lor_male_sentence = get_df_sentence(df_male_lor_male, "lor")
df_male_lor_female_sentence = get_df_sentence(df_male_lor_female, "lor")
df_female_lor_male_sentence = get_df_sentence(df_female_lor_male, "lor")
df_female_lor_female_sentence = get_df_sentence(df_female_lor_female, "lor")

male_lor_male_sentence_ratio = df_male_lor_male_sentence.shape[0]/df_male_lor_male.shape[0]
male_lor_female_sentence_ratio = df_male_lor_female_sentence.shape[0]/df_male_lor_female.shape[0]
female_lor_male_sentence_ratio = df_female_lor_male_sentence.shape[0]/df_female_lor_male.shape[0]
female_lor_female_sentence_ratio = df_female_lor_female_sentence.shape[0]/df_female_lor_female.shape[0]
print("male_lor_male_sentence_ratio: ",male_lor_male_sentence_ratio)
print("male_lor_female_sentence_ratio: ",male_lor_female_sentence_ratio)
print("female_lor_male_sentence_ratio: ",female_lor_male_sentence_ratio)
print("female_lor_female_sentence_ratio: ",female_lor_female_sentence_ratio)
""" 
male_lor_male_sentence_ratio:  17.883802816901408
male_lor_female_sentence_ratio:  20.17699115044248
female_lor_male_sentence_ratio:  19.01818181818182
female_lor_female_sentence_ratio:  18.442307692307693
"""
#%%
mm_report_size = df_male_lor_male_sentence.groupby("lor_id").size()
mf_report_size = df_male_lor_female_sentence.groupby("lor_id").size()
fm_report_size = df_female_lor_male_sentence.groupby("lor_id").size()
ff_report_size = df_female_lor_female_sentence.groupby("lor_id").size()

#%%
ttest_ind(mf_report_size,mm_report_size)
# Ttest_indResult(statistic=3.506475178061064, pvalue=0.00048397566892492976)
# %%

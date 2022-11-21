#for copying into vim script 
from tqdm.auto import tqdm
from sentence_transformers import evaluation
import torch 
from transformers import BertModel
from transformers import BertTokenizer
from datasets import Dataset
import pandas as pd
from transformers.optimization import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt 
import numpy as np
import sklearn.model_selection
import sklearn
from sentence_transformers import SentenceTransformer, InputExample, losses, util
import csv 

#we can't have any na rows when calculating embeddings on content 
df = pd.read_csv("/shared/3/projects/benlitterer/localNews/mergedNewsData/mergedNER.tsv", sep="\t").dropna(subset=["content"])

#get only institutional national data 
toKeep = ["vox", "cbsnews", "usatoday", "buzzfeednews", "businessinsider", "cbssports", "foxnews", "cnn", "cnbc", "huffingtonpost", "washingtonpost", "msnbc", "yahoonews", "thenewyorktimes", "fivethirtyeight", "forbes", "abcnews", "huffpost", "nytimes", "newyorkpost", "dailymail", "vanityfair"]
institutional = df[(df["national"] == False) | ((df["national"] == True) & (df["source"].isin(toKeep)))]

#get may-september 2021 data 
institutionalSubset = institutional[(institutional["date"] >= "2021-05-01") & (institutional["date"] <= "2021-09-01")]

#sanity check 
print(max(institutionalSubset["date"]))
print(min(institutionalSubset["date"]))
print(institutionalSubset[institutionalSubset["national"] == False].shape)
print(institutionalSubset[institutionalSubset["national"] == True].shape)

modelPath = "/shared/3/projects/benlitterer/localNews/NetworkMVP/SBERTstockEval"
model = SentenceTransformer(modelPath, device="cuda:4")

outPath = "/shared/3/projects/benlitterer/localNews/NetworkMVP/dataWithEmbeddings.tsv"

df = institutionalSubset

#get content and put it in a list for encoding 
textList = df["content"] 

embeddings = []
for article in tqdm(textList): 
    embeddings.append(model.encode(article))
    
df["embedding"] = embeddings 
df.to_csv(outPath, sep="\t", quoting=csv.QUOTE_NONNUMERIC)

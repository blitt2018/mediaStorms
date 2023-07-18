#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm.auto import tqdm
from sentence_transformers import evaluation
import torch 
import torch.nn
from transformers import BertModel
from transformers import BertTokenizer
from datasets import Dataset
import pandas as pd
from transformers.optimization import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt 
import numpy as np
import sklearn.model_selection
import sklearn
from transformers import AutoTokenizer, AutoModel
from torch.nn import CosineEmbeddingLoss
from sentence_transformers import SentenceTransformer, InputExample, losses, util


# In[2]:


#we can't have any na rows when calculating embeddings on content 
df = pd.read_csv("/shared/3/projects/benlitterer/localNews/mergedNewsData/mergedNER.tsv", sep="\t", nrows = 1000).dropna(subset=["content"])


# In[3]:


OUTPUT_PATH = "/shared/3/projects/benlitterer/localNews/NetworkMVP/dataWithEmbeddings.tsv"
MODEL_PATH = "/home/blitt/projects/localNews/models/sentEmbeddings/2.0-biModelAblation/finalModel/state_dict.tar"
DEVICE = 1


# In[4]:


#get only institutional national data 
toKeep = ["vox", "cbsnews", "usatoday", "buzzfeednews", "businessinsider", "cbssports", "foxnews", "cnn", "cnbc", "huffingtonpost", "washingtonpost", "msnbc", "yahoonews", "thenewyorktimes", "fivethirtyeight", "forbes", "abcnews", "huffpost", "nytimes", "newyorkpost", "dailymail", "vanityfair"]

#Keep row if it is local news or if it is national news in one of the above categories 
institutional = df[(df["national"] == False) | ((df["national"] == True) & (df["source"].isin(toKeep)))]

del df 

"""
USED IN PREVIOUS MVP VERSION
get may-september 2021 data 
institutionalSubset = institutional[(institutional["date"] >= "2021-05-01") & (institutional["date"] <= "2021-09-01")]
"""


# In[5]:


#sanity check 
print(max(institutional["date"]))
print(min(institutional["date"]))
print(institutional[institutional["national"] == False].shape)
print(institutional[institutional["national"] == True].shape)


# In[6]:


class BiModel(torch.nn.Module): 
    def __init__(self):
        super(BiModel,self).__init__()
        self.model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device).train()
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-4)
        
    def mean_pooling(self, token_embeddings, attention_mask): 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask): 
        
        input_ids_a = input_ids[0].to(device)
        input_ids_b = input_ids[1].to(device)
        attention_a = attention_mask[0].to(device)
        attention_b = attention_mask[1].to(device)
        
        #encode sentence and get mean pooled sentence representation 
        encoding1 = self.model(input_ids_a, attention_mask=attention_a)[0] #all token embeddings
        encoding2 = self.model(input_ids_b, attention_mask=attention_b)[0]
        
        meanPooled1 = self.mean_pooling(encoding1, attention_a)
        meanPooled2 = self.mean_pooling(encoding2, attention_b)
        
        pred = self.cos(meanPooled1, meanPooled2)
        return pred


# In[8]:


#load trainedModel 
device = torch.device("cuda:" + str(DEVICE) if torch.cuda.is_available() else "cpu")

trainedModel = BiModel()
trainedModel.load_state_dict(torch.load(MODEL_PATH), map_location="cpu")
#trainModel.to(device)


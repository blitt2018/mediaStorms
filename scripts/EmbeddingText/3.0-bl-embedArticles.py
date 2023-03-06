#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[8]:


#we can't have any na rows when calculating embeddings on content 
#NOTE: when merging back in we will need to run .dropna(columns=["headTail"])
#use nrows = some_number, to test
df = pd.read_csv("/shared/3/projects/newsDiffusion/data/interim/NEREmbedding/headTailMerged.tsv", sep="\t", usecols=["key", "headTail"])
df["headTail"] = df["headTail"].fillna("")
#/shared/3/projects/benlitterer/localNews/mergedNewsData/mergedNER.tsv

print("read dataframe") 

df.columns


# In[31]:


OUTPUT_PATH = "/shared/3/projects/newsDiffusion/data/interim/NEREmbedding/embeddingsKeys.tsv"

#the model that performed best with the 5 random seeds 
MODEL_PATH = "/shared/3/projects/newsDiffusion/models/2.0-biModelAblation/finalModel/92/state_dict.tar"
DEVICE = 0
BATCH_SIZE = 8


# In[19]:


"""
ACTUALLY DECIDED TO KEEP ALL DATA 
#get only institutional national data 
toKeep = ["vox", "cbsnews", "usatoday", "buzzfeednews", "businessinsider", "cbssports", "foxnews", "cnn", "cnbc", "huffingtonpost", "washingtonpost", "msnbc", "yahoonews", "thenewyorktimes", "fivethirtyeight", "forbes", "abcnews", "huffpost", "nytimes", "newyorkpost", "dailymail", "vanityfair"]

#Keep row if it is local news or if it is national news in one of the above categories 
institutionalDf = df[(df["national"] == False) | ((df["national"] == True) & (df["source"].isin(toKeep)))]

#del df 


USED IN PREVIOUS MVP VERSION
get may-september 2021 data 
institutionalSubset = institutional[(institutional["date"] >= "2021-05-01") & (institutional["date"] <= "2021-09-01")]
"""


# In[20]:


#sanity check 
print(len(df))


# In[25]:


class BiModel(torch.nn.Module): 
    def __init__(self):
        super(BiModel,self).__init__()
        self.model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device).train()
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-4)
        
    def mean_pooling(self, token_embeddings, attention_mask): 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    #NOTE: here we expect only one batch of input ids and attention masks 
    def encode(self, input_ids, attention_mask):
        encoding = self.model(input_ids.squeeze(1), attention_mask=attention_mask.squeeze(1))[0]
        meanPooled = self.mean_pooling(encoding, attention_mask.squeeze(1))
        return meanPooled 
    
    #NOTE: here we expect a list of two that we then unpack 
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


# In[28]:


#load trainedModel 
#device = torch.device("cuda:" + str(DEVICE) if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")
trainedModel = BiModel()
trainedModel.load_state_dict(torch.load(MODEL_PATH))

device = torch.device("cuda:" + str(DEVICE))
trainedModel = trainedModel.to(device)


# In[42]:


tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
print("loading dataset") 
dataset = Dataset.from_pandas(df[["key", "headTail"]])

print("tokenizing dataset") 
dataset = dataset.map(lambda x: tokenizer(x["headTail"], max_length=384, padding="max_length", truncation=True, return_tensors="pt"))

print("formatting dataset") 
dataset = dataset.remove_columns(["headTail"])
dataset.set_format(type="torch", columns=["key", "input_ids", "attention_mask"])

embeddings = []
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

print("EMBEDDING ARTICLES") 
with open(OUTPUT_PATH, "w") as outFile: 
    for i, batch in tqdm(enumerate(loader)): 
        #print(batch)
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        keys = batch["key"].tolist()
        
        encodingList = trainedModel.encode(ids, mask).detach().to("cpu").tolist()
    
        for j in range(len(keys)): 
            key = keys[j]
            encoding = encodingList[j]
            outFile.write(str(key) + "\t" + str(encoding) + "\n")
        

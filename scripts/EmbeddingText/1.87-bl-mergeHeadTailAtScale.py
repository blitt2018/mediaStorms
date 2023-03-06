#!/usr/bin/env python
# coding: utf-8

# In[67]:


import random
import os
import numpy as np
import torch
import json
from tqdm import tqdm
import pandas as pd
from transformers import RobertaTokenizer
import time
from transformers import AutoTokenizer, AutoModel


# In[59]:


df = pd.read_csv("/shared/3/projects/newsDiffusion/data/processed/newsData/fullMergedNELAdata.tsv", sep="\t")

df[["title", "content"]] = df[["title", "content"]].fillna("") 

#merge title into content 
df["content"] = df["title"] + df["content"] 

#NOTE: very important step, we are removing rows here 
#df = df.dropna(subset=["content"])

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')


# In[61]:

"""
# truncate text from either head or tail part
def trunc_text(text, trunc_pos, length):

    text_ids = tokenizer.encode(text)[1:-1]

    if trunc_pos == 'head':
        text_trunc_ids = text_ids[:length]
    elif trunc_pos == 'tail':
        text_trunc_ids = text_ids[-length:]

    text_trunc_tokens = tokenizer.decode(text_trunc_ids)
    #text_trunc_back_sent = ''.join([x.replace('_', ' ') for x in text_trunc_tokens])[:-1]

    return text_trunc_tokens
"""

# merged head and tail of text according to 
#head_length and tail_length 
def trunc_text(text, tokenizer, head_length, tail_length):
    text_ids = tokenizer.encode(text)[1:-1]
    #if we don't enough text that we need to take some
    #head and tail, then we just take the original text
    if len(text_ids) < head_length + tail_length: 
        return text
    
    #if we have extra text, we want to take some
    #from the beginning and some from the end 
    else: 
        head_trunc_ids = text_ids[:head_length]
        tail_trunc_ids = text_ids[-tail_length:]
        head_trunc_tokens = tokenizer.decode(head_trunc_ids)
        tail_trunc_tokens = tokenizer.decode(tail_trunc_ids)
        return head_trunc_tokens + tail_trunc_tokens 
                                                                                                

HEAD_COUNT = 288
TAIL_COUNT = 96


# In[66]:


len(df)


# In[64]:


#6 seconds for 1,000 
#1 minute for 10,000
print("starting tokenization")
st = time.time()
df["headTail"] = df["content"].apply(trunc_text, args=[tokenizer, HEAD_COUNT, TAIL_COUNT])
et = time.time()
print(et - st)


#NOTE: only uncomment if ready to write! always make sure we write to /shared 
#and not /home 
df.to_csv("/shared/3/projects/newsDiffusion/data/interim/NEREmbedding/headTailMerged.tsv", sep="\t")


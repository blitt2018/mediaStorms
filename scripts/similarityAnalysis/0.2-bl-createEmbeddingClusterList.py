#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import pandas as pd 
import re
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

SIM_CUTOFF = float(sys.argv[1])
PAIR_INFO_PATH = str(sys.argv[2])  
SIM_INFO_PATH = str(sys.argv[3]) 
OUT_PATH = str(sys.argv[4]) 
OUT_GRAPH_PATH = str(sys.argv[5]) 

#load pairwise information 
#unfortunately this takes a while since it includes the embeddings 
pairsDf = pd.read_pickle(PAIR_INFO_PATH)


# In[5]:


pairsDf = pairsDf.drop(columns=["embedding"])


# In[6]:


#load embeddings to be merged to the pairsDf 
simDf = pd.read_pickle(SIM_INFO_PATH)


# In[10]:


#merge embedings onto pairsDf 
#since they have the exact same length and ordering, 
#this can be done very simply 
pairsDf["similarity"] = simDf["similarity"]


# In[13]:


#keep only edges with similarity over the cutoff 
pairsDf = pairsDf[pairsDf["similarity"] >= SIM_CUTOFF].reset_index(drop=True)


# In[17]:


print(str(len(pairsDf)) + " pairs >=  " + str(SIM_CUTOFF)) 


# In[19]:


#this automatically gets rid of duplicates since parallel edges aren't 
#allowed and graph is undirected by default 
print("creating graph")
graph = nx.from_pandas_edgelist(pairsDf[["key1", "key2"]], "key1", "key2")

print("writing graph") 
nx.write_edgelist(graph, OUT_GRAPH_PATH)

print("generating components")
components = nx.connected_components(graph)
compList = [comp for comp in components]


# In[22]:


#put clustered data into long form 
clusters = pd.DataFrame({"cluster":compList}).reset_index()
clustDf = clusters.explode("cluster").rename(columns={"index":"clustNum", "cluster":"key"})


# In[26]:


clustDf.to_pickle(OUT_PATH)


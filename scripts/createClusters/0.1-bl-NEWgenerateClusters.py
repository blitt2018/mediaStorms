#!/usr/bin/env python
# coding: utf-8

# ## Sandbox for getting article pair similarity after filtering by named entities 

# In[26]:


import pandas as pd
import numpy as np
from ast import literal_eval
import re
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance 
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from tqdm import tqdm 
import networkx as nx
import pickle


# In[27]:


#NOTE: very important, which entity categories to keep 
#article showing all entity types below
# https://www.kaggle.com/code/curiousprogrammer/entity-extraction-and-classification-using-spacy
TO_KEEP = ["org","event", "person", "work_of_art", "product"]
CLUSTER_CUTOFF = [2, 60000]
#for testing 

INVERTED_ENT_PATH = "/shared/3/projects/newsDiffusion/data/interim/NEREmbedding/invertedEntityIndex.pkl"
EMBEDS_PATH = "/shared/3/projects/newsDiffusion/data/processed/articleEmbeddings/embeddings.pkl"

CLEANED_DF_PATH = "/shared/3/projects/newsDiffusion/data/processed/newsData/fullDataWithNERCleaned.tsv"


#this is the df with our inverted index in it
invertedDf = pd.read_pickle(INVERTED_ENT_PATH)
invertedDf.sort_values("numArticles", ascending=False).head()

print(f"len before filtering: {len(invertedDf)}")
invertedDf = invertedDf[(invertedDf["numArticles"] >= CLUSTER_CUTOFF[0]) & (invertedDf["numArticles"] <= CLUSTER_CUTOFF[1])]
print(f"len after filtering: {len(invertedDf)}")

max(invertedDf["numArticles"])


#a dictionary so we can get the embeddings we need quickly 
#embeddingsDict = embeddingsDf.set_index("key").to_dict(orient="index")
embedsFile = open(EMBEDS_PATH, "rb")
embeddingsDict = pickle.load(embedsFile)


#get a list of the keys that correspond to each named entity 
#sort so that smaller clusters will be processed first :) 
keyOptions = list(invertedDf.sort_values("numArticles", ascending=False)["key"])


from numpy import dot
import math

#compared = {i:{} for i in range(0, 6000000)}

simList = []
lKeyList = []
rKeyList = []

#for each list of article keys associated with entities 
for i, entGroup in enumerate(tqdm(keyOptions)): 
    
    #within each list of article keys, consider the unique pairs 
    #and get their cosine similarities 
    #for i in range(0, len(entGroup)): 
    myMat = np.matrix([embeddingsDict[key]["embedding"] for key in entGroup])
    pairSims = cosine_similarity(myMat).flatten()
    entGroup = np.array(entGroup)
    
    greaterThan = np.where(pairSims > .85)
    simList.append(pairSims[greaterThan])
    
    #get the equivalent row and column indices to what we have in the flattened array 
    left = [math.floor(index / len(entGroup)) for index in greaterThan[0]]
    right = [index % len(entGroup) for index in greaterThan[0]]
    
    #get the keys corresponding to the elements we selected from the list 
    lKeyList.append(entGroup[left])
    rKeyList.append(entGroup[right])
    

# In[36]:


#concatenate all of the lists of np arrays we made 
simCat = np.concatenate(simList)
lKeyCat = np.concatenate(lKeyList)
rKeyCat = np.concatenate(rKeyList)

pairsDf = pd.DataFrame({"lKey":lKeyCat, "rKey":rKeyCat,"simScore":simCat})

from scipy.spatial.distance import cosine

#quick sanity check
randIndex = 56566740
randRow = pairsDf.loc[randIndex]

1 - cosine(embeddingsDict[randRow["lKey"]]["embedding"], embeddingsDict[randRow["rKey"]]["embedding"])

#same thing!
randRow["simScore"]



# In[38]:


pairsUnique = pairsDf.drop_duplicates(subset=["lKey", "rKey"])


# In[18]:


artsDf = pd.read_csv(CLEANED_DF_PATH,usecols=["key", "date"], sep="\t")


# In[19]:


artsDf["date"] = pd.to_datetime(artsDf["date"])


# In[20]:


dateDict = artsDf.set_index("key").to_dict(orient="index")


# In[23]:


import math 
import numpy as np


#TODO: this needs to be absolute value!!
lKeys = pairsDf["lKey"]
rKeys = pairsDf["rKey"]

dateIndices = []
for i in tqdm(range(len(pairsDf))): 
    lDate = dateDict[lKeys[i]]["date"]
    rDate = dateDict[rKeys[i]]["date"]
    
    dateDist = (rDate - lDate).days
    
    #so we can have Sunday - Sunday but not Sunday - Monday 
    if np.abs(dateDist) < 7: 
        dateIndices.append(i)


#the remaining pairs 
datePairs = pairsDf.iloc[dateIndices]



BASE_PATH = "/shared/3/projects/newsDiffusion/data/interim/NEREmbedding/"
#OUT_PATHS = ["embeddingClusterDf_2_3000_83.tsv", "embeddingClusterDf_2_3000_85.tsv", "embeddingClusterDf_2_3000_87.tsv", "embeddingClusterDf_2_3000_9.tsv"]
OUT_PATHS = ["embeddingClusterDf_60000_85.tsv", "embeddingClusterDf_60000_90.tsv"]
CUTOFFS = [.85, .90]

for i, cutoff in enumerate(tqdm(CUTOFFS)): 
    outPath = BASE_PATH + OUT_PATHS[i]
    overThresh = datePairs[datePairs["simScore"] >= cutoff]
    graph = nx.from_pandas_edgelist(overThresh[["lKey", "rKey"]], "lKey", "rKey")

    components = nx.connected_components(graph)
    compList = [comp for comp in components]

    clusters = pd.DataFrame({"cluster":compList}) #.reset_index()

    #we can remove clusters of size one 
    clusters["clustSize"] = clusters["cluster"].apply(lambda x: len(list(x)))

    print(f'first 10:\n{sorted(clusters["clustSize"], reverse=True)[:10]}')
    
    clusters = clusters[clusters["clustSize"] > 1]

    clusters["clustNum"] = list(range(0, len(clusters)))

    clustDf = clusters.explode("cluster").rename(columns={"index":"clustNum", "cluster":"key"})

    clustSizes = pd.DataFrame(clustDf["clustNum"].value_counts()).reset_index()

    clustSizes.value_counts()
    
    clustDf.to_csv(outPath, sep="\t")


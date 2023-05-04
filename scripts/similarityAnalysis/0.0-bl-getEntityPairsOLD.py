#!/usr/bin/env python
# coding: utf-8

# ## Sandbox for getting article pair similarity after filtering by named entities 

# In[4]:


import pandas as pd
import numpy as np
from ast import literal_eval
import re
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance 
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns


# In[38]:


#NOTE: very important, which entity categories to keep 
TO_KEEP = ["org","event", "person", "work_of_art", "product"]
CLUSTER_CUTOFF = [5, 1000]
#for testing 
#NROWS = 10000
OUT_PATH = "/shared/3/projects/newsDiffusion/data/interim/NEREmbedding/entityPairs2020.pkl"


# In[6]:


#load in main data source 
#we don't want to use "content", because it takes up a lot of space and
#we have already embedded the content. Can always merge back in later so long as we 
#keep the "key" column
LOAD_COLS = list(pd.read_csv("/shared/3/projects/newsDiffusion/data/processed/newsData/fullDataWithNER.tsv", nrows = 1, sep="\t").columns)
LOAD_COLS.remove("content")


# ## this might take a minute 

# In[19]:


#load in main data source 
print("loading news data")
df = pd.read_csv("/shared/3/projects/newsDiffusion/data/processed/newsData/fullDataWithNER.tsv", sep="\t", usecols = LOAD_COLS)

df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year

#filter so we only use 2020 where we have Reuters 
df = df[df["year"] == 2020]

#get length of new rows 
print(str(len(df)) + " rows in 2020 df")

#load in Embeddings, which haven't been merged yet
#we merge them in this step because they are very large and don't
#want to write them to disk again if we can help it
print("loading embeddings")
embeddingsDf = pd.read_csv("/shared/3/projects/newsDiffusion/data/interim/NEREmbedding/embeddingsKeys.tsv", sep="\t", names=["key", "embedding"], converters={"embedding":lambda x: np.array(x.strip("[]").split(","), dtype=float)})

print("merging embeddings")
df = pd.merge(df, embeddingsDf, how="left", on="key").dropna(subset=["key", "embedding"])
print(str(len(df)) + " rows after merging, dropping na keys, embeddings")


# In[31]:


print("date range: ")
print(max(pd.to_datetime(df["date"])))
print(min(pd.to_datetime(df["date"])))


# In[45]:


leanDf = df[["key", "NamedEntities", "embedding"]]

print("parsing")

def cleanList(inList): 
    return [str(re.sub("[^a-zA-Z0-9 ]", "", item).lower()) for item in inList]

def parseList(inStr): 
    split = inStr.split("\'), (\'")
    return [cleanList(item.split("', '")) for item in split]

#parse topics from string to actual list of tuples 
leanDf["NamedEntities"] = leanDf["NamedEntities"].apply(parseList)

print("parsed")

#test out idea for creating reverse mapping 
#how many na vals do we have in "NamedEntities"? 
print(str(sum(leanDf["NamedEntities"].isna())) + " NA values in Named Entities column")
print("Filling with '' instead")
leanDf["NamedEntities"] = leanDf["NamedEntities"].fillna("")


# Note: we see below that we have things like "date: week" as named entities. This must be addressed somewhere 

# In[46]:


#bring each tuple into its own row 
print("exploding #1")
leanDf = leanDf.explode("NamedEntities")

#bring each tuple entry into its own column 
#split ent_type, entity pairs to columns 
print("splitting entity, type")
leanDf[["ent_type","entity"]] = pd.DataFrame(leanDf["NamedEntities"].tolist(), index=leanDf.index)

#remove occurences where we double count an entity for the same article 
leanDf = leanDf.drop_duplicates(subset=["key", "ent_type", "entity"])

print("filtering by entity type, grouping")
#keep only the entity types that may be interesting 
leanDf = leanDf[leanDf["ent_type"].isin(TO_KEEP)]

#group articles by their named entities  
groupedDf = leanDf[["ent_type", "entity", "key", "embedding"]].groupby(by=["ent_type", "entity"]).agg(list)

print(str(len(groupedDf)) + " rows in entity-grouped df")


# In[47]:


groupedDf["numArticles"] = groupedDf["key"].apply(len)

print("filtering clusters not between 5, 1000")
#only keep named entity clusters that are of a particular size 
groupedDf = groupedDf[(groupedDf["numArticles"] >= CLUSTER_CUTOFF[0]) & (groupedDf["numArticles"] <= CLUSTER_CUTOFF[1])]

print("down to " + str(groupedDf.shape[0]) + " clusters ") 


#now that we have clusters, we want to get embeddings 
#first, we need to unravel the clusters so that we have pairwise rows


# In[51]:


#define functions for getting pairs from a list 
def getPairwise(inList): 
    outList = []
    inLen = len(inList)
    for i in range(0, inLen):
        for j in range(i+1, inLen): 
            outList.append([inList[i], inList[j]])
    return outList
    
def getPairwiseSim(inList):
    outList = []
    inLen = len(inList)
    for i in range(0, inLen):
        for j in range(i+1, inLen):
            sim = 1-distance.cosine(inList[i], inList[j])
            outList.append(sim)
            
    return outList


# # NOTE: this step uses lots of memory 
# # 1% when using 300,000 rows of original data 

# In[64]:


#def getPairwise(inList): 
#    return [(item1, item2) for item2 in inList for item1 in inList]

print("HIGH MEMORY: getting pairs")
#we only needed the articleNum column for filtering 
groupedDf = groupedDf[["key", "embedding"]]

#get all of the pairs for a given list of articles and a given list of embeddings 
groupedDf["embedding"] = groupedDf["embedding"].apply(getPairwise)
groupedDf["key"] = groupedDf["key"].apply(getPairwise)


# In[56]:


import gc
del leanDf 
del df 
gc.collect()


# In[58]:


#give each 
print("HIGH MEMORY: exploding pairs")
pairDf = groupedDf.apply(pd.Series.explode)
print(pairDf.shape)


# In[59]:


del groupedDf
gc.collect()


# In[65]:


print("making key columns")
pairDf[["key1", "key2"]] = pd.DataFrame(pairDf["key"].to_list(), index=pairDf.index)
pairDf = pairDf.drop(columns=["key"]).reset_index()


# ### NOTE: 
# this may be a line to consider if we are interested in knowing how many clusters a pair is in 

# In[61]:


#pairDf.groupby(by=["key1", "key2"]).agg({"ent_type":list, "entity":list, "embedding":lambda x: x.iloc[0]})


# ### NOTE: 
# this is for memory preservation, we need to also drop duplicates that are simply the reverse direction of eachother. This may happen automatically when we create an undirected graph from these 

# Maybe try a duplicate removal solution from here: https://stackoverflow.com/questions/44792969/pandas-drop-duplicates-based-on-subset-where-order-doesnt-matter

# In[67]:


print("dropping duplicates")
pairDf = pairDf.drop_duplicates(subset=["key1", "key2"])
print("final length of " + str(len(pairDf)) + " rows")


# In[63]:


pairDf.to_pickle(OUT_PATH)


# ### NOTE: 
# This cell (below) will be put in another file. want to simply write out pairwise dataframe and then reload to avoid any memory overhead we may have. 

# In[85]:


"""
from tqdm.notebook import tqdm 
from multiprocessing import Pool 

def getCos(inList): 
    return 1 - distance.cosine(inList[0], inList[1])

def getCosSeries(inSeries): 
    return inSeries.apply(getCos)

#tqdm.pandas()
#exploded.head(1000000)["embedding"].progress_map(getCos)

testDf = grouped
testEmbeddings = testDf["embedding"]

with Pool(12) as pool: 
    splitList = np.array_split(testEmbeddings, 10)
    similarityArrs = list(tqdm(pool.imap(getCosSeries, splitList), total=10))
    similarity = pd.concat(similarityArrs)

testDf["similarity"] = similarity
"""


# In[ ]:





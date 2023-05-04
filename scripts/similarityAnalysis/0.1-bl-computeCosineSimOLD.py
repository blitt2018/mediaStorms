#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
from tqdm.notebook import tqdm 
from multiprocessing import Pool 
from scipy.spatial import distance 
import numpy as np


# ## First step is to load very large pickle file 
# This file contains ~1.3 billion rows 

# In[ ]:


OUT_PATH = "/shared/3/projects/newsDiffusion/data/interim/NEREmbedding/embeddingPairSimilarity2020.pkl"


# In[2]:


#load pairs that were generated from entity clustering 
pairDf = pd.read_pickle("/shared/3/projects/newsDiffusion/data/interim/NEREmbedding/entityPairs2020.pkl")

#hopefully this reduces memory..? but not sure 
pairDf = pairDf[["embedding"]]
# In[49]:



# In[11]:


def getCos(inList): 
    return 1 - distance.cosine(inList[0], inList[1])

def getCosSeries(inSeries): 
    return inSeries.apply(getCos)

#tqdm.pandas()
#exploded.head(1000000)["embedding"].progress_map(getCos)

embeddings = pairDf["embedding"]

print("starting similarity calculations")
with Pool(4) as pool: 
    print("splitting embedding list")
    splitList = np.array_split(embeddings, 10)
    
    print("getting similarity")
    similarityArrs = list(tqdm(pool.imap(getCosSeries, splitList), total=10))
    
    print("concatenating chunks together")
    similarity = pd.concat(similarityArrs)

print("adding column to dataframe")
pairDf = pairDf.drop(columns=["embedding"])
pairDf["similarity"] = similarity


# In[16]:


pairDf.to_pickle(OUT_PATH)


"""
when embedding the articles, we just wrote them to a file. Unfortunately, it is super super slow to load this file. 
Load in the embeddings, then pickle them. Not a great solution, but works for now. 
"""
import pandas as pd
import numpy as np
from ast import literal_eval
import re
import matplotlib.pyplot as plt
import pickle

#read in embeddings. takes 35ish minutes 
embeddingsDf = pd.read_csv("/shared/3/projects/newsDiffusion/data/interim/NEREmbedding/embeddingsKeys.tsv", sep="\t", names=["key", "embedding"], converters={"embedding":lambda x: np.array(x.strip("[]").split(","), dtype=float)})

#a dictionary so we can get the embeddings we need quickly 
embeddingsDict = embeddingsDf.set_index("key").to_dict(orient="index")

#pickle the embeddings dataframe so we don't have to reload (very redundant but eh)
EMBEDS_FILE = "/shared/3/projects/newsDiffusion/data/processed/articleEmbeddings/embeddings.pkl"
with open(EMBEDS_FILE, "wb") as handle: 
    pickle.dump(embeddingsDict, handle)
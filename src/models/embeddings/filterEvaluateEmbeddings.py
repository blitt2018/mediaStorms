import pandas as pd
import numpy as np
import csv
import re
from scipy.spatial import distance 
from tqdm import tqdm 
from multiprocessing import Pool 

#Output path for the dataframe after filtering and getting cosine similarity of filtered pairs 
outputPath = "/shared/3/projects/benlitterer/localNews/data/interim/articlePairsCosineSim.tsv"

#read in data 
df = pd.read_csv("/shared/3/projects/benlitterer/localNews/NetworkMVP/dataWithEmbeddings.tsv", sep="\t", converters={"embedding":lambda x: np.array(x.strip("[]").split(), dtype=float)})

print("Input data shape: " + str(df.shape))

df = df[["key", "topics", "embedding"]]

print("parsing")

def cleanList(inList): 
    return [str(re.sub("[^a-zA-Z0-9 ]", "", item).lower()) for item in inList]

def parseList(inStr): 
    split = inStr.split("\'), (\'")
    return [cleanList(item.split("', '")) for item in split]

#parse topics from string to actual list of tuples 
df["topics"] = df["topics"].apply(parseList)

print("parsed")

#test out idea for creating reverse mapping 
df = df.dropna(subset=["topics"])

#bring each tuple into its own row 
df = df.explode("topics")

#bring each tuple entry into its own column 
#split ent_type, entity pairs to columns 
df[["ent_type","entity"]] = pd.DataFrame(df["topics"].tolist(), index=df.index)

print("formatted") 

#keep only the entity types that may be interesting 
toKeep = ["org","event", "person", "work_of_art", "product"]
df = df[df["ent_type"].isin(toKeep)]

#the data grouped into entity clusters 
grouped = df[["embedding", "ent_type", "entity", "key"]].groupby(by=["ent_type", "entity"]).agg(list)

#get the cluster length 
grouped["articleNum"] = grouped["key"].apply(len)

#filter out named entities that are too common, since they likely aren't that meaningful/specific 
groupedLean = grouped[(grouped["articleNum"] > 5) & (grouped["articleNum"] < 1000)]

print("Shape after grouping: " + str(groupedLean.shape))

def getPairwise(inList): 
    return [(item1, item2) for item2 in inList for item1 in inList]

#we only needed the articleNum column for filtering 
df = groupedLean[["key", "embedding"]]

#get all of the pairs for a given list of articles and a given list of embeddings 
df["embedding"] = df["embedding"].apply(getPairwise)
df["key"] = df["key"].apply(getPairwise)

#"explode" the pairs into a seperate row (this is when we have a TON of data)
exploded = df.apply(pd.Series.explode).drop_duplicates(subset=["key"])

print("Exploded size: " + str(exploded.shape))

def getCos(inList): 
    return 1 - distance.cosine(inList[0], inList[1])

def getCosSeries(inSeries): 
    return inSeries.apply(getCos)

embeddings = exploded["embedding"]

with Pool(12) as pool: 
    splitList = np.array_split(embeddings, 10)
    similarityArrs = list(tqdm(pool.imap(getCosSeries, splitList), total=10))
    similarity = pd.concat(similarityArrs)

exploded["similarity"] = similarity

print("writing")
exploded[["key", "similarity"]].to_csv(outputPath, sep="\t", quoting=csv.QUOTE_NONNUMERIC)
print("written")

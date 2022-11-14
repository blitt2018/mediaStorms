#import what we need 
import pandas as pd 
import re
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import csv

OUTPATH = "/shared/3/projects/benlitterer/localNews/data/interim/SingleNE_85_clustered.tsv"
SIM_CUTOFF = .85

#read in article pair data with cosine similarity. Takes a little while since many many rows 
df = pd.read_csv("/shared/3/projects/benlitterer/localNews/data/interim/articlePairsCosineSim.tsv", sep="\t")
print("Shape of df for articles with one named entity in common: ")
print(df.shape)

#keep only edges with similarity over the cutoff 
df = df[df["similarity"] >= SIM_CUTOFF].reset_index(drop=True)
print("Shape of dataframe keeping only edges over similarity cutoff: ")
print(df.shape)

def cleanStr(inStr): 
    #we need to keep in mind that the qoutes could be single or double.. 
    return re.sub("(\([\'\"])|([\'\"]\))", "", inStr)

def parseTuple(inStr): 
    #split the tuple on qoutes/comma combination 
    #quotes could be single or double though 
    splitStr = re.split("[\'\"], [\'\"]", inStr)
    return [cleanStr(subStr) for subStr in splitStr] 

print("Parsing strings into tuples")
#parse the key column (now a string) into a tuple, which is its true type 
df["key"] = df["key"].apply(parseTuple)
df["len"] = df["key"].apply(lambda x: len(x))

#number of cases parsed incorrectly 
print("Number of strings parsed to tuples incorrectly: ")
print(df[df["len"] != 2].shape[0])

print("Seperating tuples into two columns")
df[["key1", "key2"]] = pd.DataFrame(df["key"].tolist(), index = df.index)

print("Creating graph")
graph = nx.from_pandas_edgelist(df[["key1", "key2"]], "key1", "key2")

print("Generating components")
components = nx.connected_components(graph)
compList = [comp for comp in components]

print("Number of componenets:")
print(len(compList))

#put clustered data into long form 
clusters = pd.DataFrame({"cluster":compList}).reset_index()
clustDf = clusters.explode("cluster").rename(columns={"index":"clustNum", "cluster":"key"})

print("Reading in data with demographics and full text")
ogDf = pd.read_csv("/shared/3/projects/benlitterer/localNews/NetworkMVP/dataWithEmbeddings.tsv", sep="\t")

print("Merging cluster data with demographic and text data")
#merge clusters into the original data using the key column 
merged = pd.merge(ogDf, clustDf, how="right", on="key")

print("Filtering out articles with the same exact content and source.")
print("Removes paywall articles that don't have real text + duplicates that have wrong title.")
#we don't want reprints, and in some cases the title is different but the content 
#is the same (it's just paywall text or something), so we remove that. 
merged = merged.drop_duplicates(subset=["source", "content"]).rename(columns={"topics":"namedEntities"})
merged.to_csv(OUTPATH, sep="\t",  quoting=csv.QUOTE_NONNUMERIC)

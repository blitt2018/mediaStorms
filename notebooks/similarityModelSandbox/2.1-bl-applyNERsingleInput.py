import pandas as pd
import spacy
import numpy as np
from collections import Counter
import csv
from multiprocessing import Pool
import sys

inFile = "/shared/3/projects/benlitterer/localNews/mergedNewsData/dataSplits/" + sys.argv[1]
newsDf = pd.read_csv(inFile, sep="\t")

#TODO: remove this 
inDf = newsDf
contentList = list(inDf["content"].astype(str))

nlp = spacy.load("en_core_web_md")
docs = nlp.pipe(contentList, batch_size=1, n_process=1)

NERList = []
for doc in docs:
    outList = [(ent.label_, ent.text) for ent in doc.ents]
    NERList.append(outList)

inDf["topics"] = NERList

inDf.to_csv(sys.argv[2] + sys.argv[1].split("/")[-1] + "topics.tsv" , sep="\t",  quoting=csv.QUOTE_NONNUMERIC)

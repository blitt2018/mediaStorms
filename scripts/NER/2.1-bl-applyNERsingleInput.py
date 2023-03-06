import pandas as pd
import spacy
import numpy as np
from collections import Counter
import csv
from multiprocessing import Pool
import sys
from tqdm import tqdm 

inFile = "/shared/3/projects/newsDiffusion/data/interim/NEREmbedding/NERSplits/" + sys.argv[1]
print(inFile) 
newsDf = pd.read_csv(inFile, sep="\t")


#TODO: remove this 
inDf = newsDf
contentList = list(inDf["content"].astype(str))
keys = inDf["key"] 

nlp = spacy.load("en_core_web_md")
docs = nlp.pipe(contentList, batch_size=1, n_process=1)

outHandle = sys.argv[2] + sys.argv[1] + "topics.tsv"
print(outHandle) 
outFile = open(outHandle, "w") 

#NERList = []
for i, doc in enumerate(docs):
    if i == 1000: 
        print(i) 
    if i % 3000 == 0: 
        print(i) 
    currKey = keys[i] 
    outList = [(ent.label_, ent.text) for ent in doc.ents]
    #NERList.append(outList)
    outFile.write(str(currKey) + "\t" + str(outList) + "\n")

outFile.close()

#inDf["topics"] = NERList
#inDf.to_csv(sys.argv[2] + sys.argv[1].split("/")[-1] + "topics.tsv" , sep="\t",  quoting=csv.QUOTE_NONNUMERIC)

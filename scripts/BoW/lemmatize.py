import pandas as pd
import numpy as np
import spacy
import tqdm as tqdm

IN_PATH = "/shared/3/projects/newsDiffusion/data/processed/fullDataWithClustNums.tsv"
LEMMAS_OUTFILE = "/shared/3/projects/newsDiffusion/data/interim/topicModelling/cleaning/keysLemmas.tsv"

#MVP path merged = pd.read_csv("/shared/3/projects/benlitterer/localNews/data/interim/SingleNE_85_clustered.tsv", sep="\t")
merged = pd.read_csv(IN_PATH, sep="\t")
merged = merged.dropna(subset=["clustNum"])

merged["date"] = pd.to_datetime(merged["date"])

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
lemmatizer = nlp.get_pipe("lemmatizer")

with open(LEMMAS_OUTFILE, "w") as outfile:
    keys = list(merged["key"])
    for i, doc in enumerate(tqdm.tqdm(nlp.pipe(merged["content"]))):
        key = keys[i]
        docLemmas = "\"" + " ".join([token.lemma_ for token in doc if token.is_space == False]) + "\"\n"
        outfile.write(str(key) + "\t" + docLemmas)



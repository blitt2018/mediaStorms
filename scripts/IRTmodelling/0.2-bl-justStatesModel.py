#!/usr/bin/env python
# coding: utf-8

# create simple model draft to get outlet embeddings 

import pandas as pd
import numpy as np
import torch 
from torchmetrics.functional.classification import f1_score
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
import matplotlib.pyplot as plt
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm 
import random
import wandb 


GROUP_NAME="testJustStates" 
TRAIN_BATCH_SIZE=200
VALID_BATCH_SIZE=200
EPOCHS=1
LR = .0001
K_FOLDS = 3

#number of times we want validation to run 
VALID_COUNT = 12 


# In[35]:


# load in the news data of interest 
df = pd.read_csv("/shared/3/projects/newsDiffusion/data/processed/fullDataWithClustNums.tsv", sep="\t") 


# In[36]:


PICKLE_PATH = "/shared/3/projects/newsDiffusion/data/processed/IRTmodel/storyEmbeddingsMean.pkl"
storyEmbeddings = pd.read_pickle(PICKLE_PATH)


# In[37]:


#we give a story cluster number and get back the average embedding for that story cluster 
storyDict = storyEmbeddings.set_index("clustNum")[["storyMean"]].to_dict()["storyMean"] 

#we want to get a list of all possible story clusters that an outlet can cover
allClusts = storyEmbeddings["clustNum"].tolist()

#keep only the articles that we have embeddings for, since we removed some clusters above
outletStoryDf = df.loc[df["clustNum"].isin(allClusts), ["source", "clustNum"]]

#now we have each outlet and stories it covered 
outletStoryDf = outletStoryDf.drop_duplicates()


# In[38]:


clusteredStories = outletStoryDf.groupby("source").agg(set)
clusteredStories["covered"] = 1

notCoveredSamples = [] 
i = 0 
for source, currStories in tqdm(clusteredStories.iterrows()): 
    # we get the stories not covered by this outlet 
    # simply all stories minus the stories this outlet did cover 
    currStories = currStories["clustNum"]
    notCovered = set(allClusts) - currStories
    
    #take 1 times as many negative examples as positive 
    sample = random.sample(list(notCovered), 1 * len(currStories))
    notCoveredSamples.append((source, sample)) 

#create dataframe from samples of not covered stories 
notCoveredDf = pd.DataFrame(notCoveredSamples, columns=["source", "clustNum"])
notCoveredDf["covered"] = 0

#get covered/non-covered stories in long form 
clusteredStories = clusteredStories.reset_index().explode("clustNum")
notCoveredDf = notCoveredDf.explode("clustNum") 

#merge both covered and not covered training examples 
#a long form dataframe that gives us outlet, story cluster num, covered or not
allCoverage = pd.concat([notCoveredDf.reset_index(drop=True), clusteredStories.reset_index(drop=True)],axis=0) 

# mix up the rows so that we have equal number of pos/neg training examples 
# we reset index so we can troubleshoot cross val splits later on
allCoverage = allCoverage.sample(frac = 1).reset_index(drop=True)


# In[40]:


#get outlet level information 
outletDf = df[["source", "state"]].drop_duplicates().fillna("National")


# In[41]:


allCoverage = pd.merge(allCoverage, outletDf[["source", "state"]], on="source", how="left") 


# ### beginning of code for model training 

deviceNum = 2
device = torch.device("cuda:" + str(deviceNum) if torch.cuda.is_available() else "cpu")

class BasicModel(nn.Module):

    def __init__(self, numEmbeddings, embeddingLen, storyDict):
        super(BasicModel, self).__init__()
        #in our case, we will have STATE embeddings instead of 
        #outlet embeddings 
        self.embeddings = nn.Embedding(numEmbeddings, embeddingLen)
        self.storyDict = storyDict
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        self.Dropout = nn.Dropout()
        
        #we do the length of our embeddings (states) in this case 
        #plus the length of our story embeddings which is 768 
        self.linear1 = nn.Linear(embeddingLen + 768, 200)
        self.linear2 = nn.Linear(200, 1) 
    
    #input will be the indices of the embeddings 
    def forward(self, embedIds, storyVecIds):
        #these are the outlet embeddings NOT the story embeddings 
        stateEmbeds = self.embeddings(embedIds) #.view((1, -1))
        storyVecs = torch.tensor([self.storyDict[int(clustNum)] for clustNum in storyVecIds], dtype=torch.float32).to(device)
        inTens = torch.concat((stateEmbeds, storyVecs), dim=1).to(device)
              
        out = self.ReLU(self.Dropout(self.linear1(inTens)))
        out = self.linear2(out)
        probs = self.Sigmoid(out)
        return probs
        


# In[72]:


#TOTAL_ROWS = 10000
trainDf = allCoverage #.head(10000) 
dataset = Dataset.from_pandas(trainDf)


#create dictionary that gives a unique index for each state 
states = allCoverage["state"].unique()
stateDict = {states[i]:i for i in range(0, len(states))}


def validate(validLoader): 
    #validation loop 
    allPreds = []
    allGts = []
    for batch in validLoader: 
        stateLookups = torch.tensor([stateDict[state] for state in batch["state"]]).to(device)
        preds = model(stateLookups, batch["clustNum"].to(device))
        gts = torch.unsqueeze(batch["covered"], dim=1).to(device) 
        allPreds += preds.detach().squeeze().cpu().tolist()
        allGts += gts.detach().squeeze().cpu().tolist()
    return f1_score(torch.tensor(allPreds), torch.tensor(allGts)) 

loss_func = torch.nn.BCELoss()

# testing out cross validation

validTups = []
trainTups = []

totalRows = len(trainDf) 
trainExamples = totalRows * ((K_FOLDS - 1)/ K_FOLDS) 
kfold = KFold(n_splits=K_FOLDS, shuffle=True)
config = {
    "lr":LR,
    "batchSize":TRAIN_BATCH_SIZE,
    "numFolds":K_FOLDS, 
    "totalExamples":totalRows,
    "trainExamples":trainExamples, 
    "loss":"Binary Cross Entropy"
}

#we also want to calculate how frequently we should be running on the validation set
validMultiple = int((trainExamples / TRAIN_BATCH_SIZE) / VALID_COUNT ) 


for fold, (trainIds, validIds) in enumerate(kfold.split(trainDf)):
    
    wandb.init(dir="/shared/3/projects/newsDiffusion/models/IRTModelling/",reinit=True, config=config,group=GROUP_NAME)
    model = BasicModel(len(states) , 200, storyDict) 
    model.to(device)
    model.train()
    
    print(f"fold: {fold}")
    
    trainDataset = Dataset.from_pandas(trainDf.iloc[trainIds,])
    validDataset = Dataset.from_pandas(trainDf.iloc[validIds,])
    
    trainLoader = DataLoader(trainDataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    validLoader = DataLoader(validDataset, batch_size=VALID_BATCH_SIZE, shuffle=False)
    
    # and setup a warmup for the first ~10% steps
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    total_steps = int((len(trainDataset) * EPOCHS) / TRAIN_BATCH_SIZE)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps - warmup_steps)
    
    subLossList = []
    i = 0 
    for batch in tqdm(trainLoader): 
        model.train()
        optim.zero_grad()
        
        #get the outlet ids from the outlet names fed into lookup dictionary  
        stateLookups = torch.tensor([stateDict[state] for state in batch["state"]]).to(device)
        preds = model(stateLookups, batch["clustNum"].to(device))

        #get ground truth labels from the batch 
        gts = torch.unsqueeze(batch["covered"], dim=1).type("torch.FloatTensor").to(device) 

        loss = loss_func(preds, gts)
        loss.backward()
        optim.step()
        scheduler.step()
        subLossList.append(loss.detach().item())
        if i % validMultiple == 0:
            model.eval()
            trainLoss = np.mean(subLossList) 
            validF1 = validate(validLoader)
            
            #add to dataframe 
            validTups.append((fold, i, validF1))
            trainTups.append((fold, i, trainLoss))
            
            #log to weights and biases 
            wandb.log({"trainLoss":trainLoss, "validF1":validF1}) 
            subLossList = []
            model.train()
        i += 1


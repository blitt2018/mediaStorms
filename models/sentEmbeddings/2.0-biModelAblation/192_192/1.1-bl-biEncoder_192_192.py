#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm.auto import tqdm
import torch 
import transformers
from transformers import PreTrainedTokenizer
from transformers import RobertaTokenizer, PreTrainedTokenizer, DistilBertTokenizer, DistilBertModel, RobertaModel
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, util
from datasets import Dataset
import pandas as pd
from transformers.optimization import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt 
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from torch.nn import CosineEmbeddingLoss
import random
from torch.nn import CosineEmbeddingLoss
from torch import nn
import time

deviceNum = 1 
device = torch.device("cuda:" + str(deviceNum) if torch.cuda.is_available() else "cpu")

GRAD_ACC = 6
EPOCHS = 2
FOLDS = 5
SEED = 85
BATCH_SIZE = 5

#set seeds 
torch.manual_seed(85)
random.seed(85)



# In[3]:


#df = pd.read_csv("/shared/3/projects/benlitterer/localNews/NetworkMVP/translatedCleaned.tsv", sep="\t")
df = pd.read_csv("/shared/3/projects/newsDiffusion/data/processed/translated_192_192.tsv", sep="\t")

#put ground truth values into a list 
df["ground_truth"] = df['Overall']

#get only the columns we need 
#TODO: do we need "pair_id"? 
#leanDf = df[["ground_truth",  'text1', 'text2', 'title1', 'title2', 'url1_lang', 'url2_lang']].dropna()
#for when using merged text
leanDf = df[["ground_truth",  'text1Merged', 'text2Merged', 'url1_lang', 'url2_lang']].dropna()

#rescale data from (0, 4): (0, 1)
leanDf["ground_truth"] = 1 - ((leanDf["ground_truth"] - 1) / 3)

#reset index so it is contiguous set of numbers 
leanDf = leanDf.reset_index(drop=True)



# In[4]:


class BiModel(nn.Module): 
    def __init__(self):
        super(BiModel,self).__init__()
        self.model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device).train()
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-4)
        
    def mean_pooling(self, token_embeddings, attention_mask): 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask): 
        
        input_ids_a = input_ids[0].to(device)
        input_ids_b = input_ids[1].to(device)
        attention_a = attention_mask[0].to(device)
        attention_b = attention_mask[1].to(device)
        
        #encode sentence and get mean pooled sentence representation 
        encoding1 = self.model(input_ids_a, attention_mask=attention_a)[0] #all token embeddings
        encoding2 = self.model(input_ids_b, attention_mask=attention_b)[0]
        
        meanPooled1 = self.mean_pooling(encoding1, attention_a)
        meanPooled2 = self.mean_pooling(encoding2, attention_b)
        
        pred = self.cos(meanPooled1, meanPooled2)
        return pred


# In[5]:


first = [1, 2, 3]
second = [4, 5, 6]
first += second
print(first)


# In[6]:


def validateBi(model, validLoader, loss_func):
    model.eval()
   
    preds = []
    gts = []
    
    for batch in tqdm(validLoader): 
        
        input_ids = [batch["text1Merged_input_ids"], batch["text2Merged_input_ids"]]
        attention_masks = [batch["text1Merged_attention_mask"], batch["text2Merged_attention_mask"]]
        pred = model(input_ids, attention_masks)
        gt = batch["ground_truth"].to(device)
        
        preds += list(pred.detach().cpu().tolist())
        gts += list(gt.detach().cpu().tolist())
    corr = np.corrcoef(preds, gts)[1,0]
    print(corr)
    model.train()
    return corr


# In[7]:


def trainBi(trainDataset, validDataset): 
    model = BiModel().to(device)
    
    # we would initialize everything first
    optim = torch.optim.Adam(model.parameters(), lr=2e-6)
    
    # and setup a warmup for the first ~10% steps
    total_steps = int(len(trainDataset) / BATCH_SIZE)*EPOCHS
    warmup_steps = int(0.1 * total_steps)
    #TODO: change warmup steps back after 
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps - warmup_steps)

    loss_func = torch.nn.MSELoss(reduction="mean")

    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
    validLoader = torch.utils.data.DataLoader(validDataset, batch_size=2, shuffle=False)

    corrList = []    
    for epoch in range(EPOCHS):
        print("EPOCH: " + str(epoch))
        validateBi(model, validLoader, loss_func)
        
        model.train()  # make sure model is in training mode

        #DEBUGGING
        #prevParam = list(model.parameters())[0].clone()
        for batch in tqdm(trainLoader):
            optim.zero_grad()
            
            input_ids = [batch["text1Merged_input_ids"], batch["text2Merged_input_ids"]]
            attention_masks = [batch["text1Merged_attention_mask"], batch["text2Merged_attention_mask"]]
            pred = model(input_ids, attention_masks)
            
            gt = batch["ground_truth"].to(device)
            loss = loss_func(pred, gt)
            
            # using loss, calculate gradients and then optimize
            loss.backward()
            optim.step()
            scheduler.step()
            
            
            #DEBUGGING
            """
            print(loss)
            print(scheduler.get_last_lr())
            
            #see if model params have changed
            param = list(model.parameters())[0].clone()
            print(torch.equal(param.data, prevParam.data))
            """
    print("final validation")
    corrList.append(validateBi(model, validLoader, loss_func))
    return corrList

# In[8]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=FOLDS, shuffle=True)


# In[9]:


metrics = []
transformers.logging.set_verbosity_error()
biTokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')

crossTokenizer = RobertaTokenizer.from_pretrained('roberta-base')

#NOTE: THIS LINE IS ONLY FOR QUICK TRAINING CHECK 
#leanDf = leanDf.iloc[:100, :]

#we only want to sample validation data from the pairs that are both english 
enDf = leanDf[(leanDf["url1_lang"] == "en") & (leanDf["url2_lang"] == "en")]

print("Total df len: " +  str(len(leanDf)))
print("English df len: " +  str(len(enDf)))

biCorrs = []
st = time.time()

#we create splits based on the position (not the actual index) of rows in enDf
#the idea is to get a split of the english dataset to set aside and then 
#grab everything else in the en + translated dataset to train on 
for i, (train_index, valid_index) in enumerate(kf.split(enDf)): 
    
    #grab the rows in enDf corresponding to the positions of our split 
    validDf = enDf.iloc[valid_index]
    
    #now get the actual indicies that have been selected
    #and subtract the indices in trainDf away from those 
    remainingIndices = list(set(leanDf.index) - set(validDf.index))
    trainDf = leanDf.loc[remainingIndices]
    validDf = validDf.reset_index(drop=True)
    
    print("###### " + str(i).upper() + " ######")
    print("Train df len: " + str(len(trainDf)))
    print("Valid df len: " + str(len(validDf)))
    
    
    trainDataset = Dataset.from_pandas(trainDf)
    validDataset = Dataset.from_pandas(validDf)
    
    all_cols = ["ground_truth"]
    #NOTE: here we use the merged text
    for part in ["text1Merged", "text2Merged"]: 
        #tokenizes each row of the dataset and gives us back tuple of lists 
        trainDataset = trainDataset.map(lambda x: biTokenizer(x[part], max_length=384, padding="max_length", truncation=True))
        validDataset = validDataset.map(lambda x: biTokenizer(x[part], max_length=384, padding="max_length", truncation=True))
        
        for col in ['input_ids', 'attention_mask']: 
            trainDataset = trainDataset.rename_column(col, part+'_'+col)
            validDataset = validDataset.rename_column(col, part+'_'+col)
            all_cols.append(part+'_'+col)
            
    trainDataset.set_format(type='torch', columns=all_cols)
    validDataset.set_format(type='torch', columns=all_cols)
    
    biCorrs.append(trainBi(trainDataset, validDataset))


et = time.time()
elapsed = et - st

#send corrList to pickled object for later analysis 
import pickle
RESULTS_PATH = "/home/blitt/projects/localNews/models/sentEmbeddings/2.0-biModelAblation/192_192"  

#write to an output folder 
with open(RESULTS_PATH + "/outputData.pkl", "wb") as f:
    pickle.dump(biCorrs, f)

with open(RESULTS_PATH + "/time.pkl", "wb") as f:
    pickle.dump(elapsed, f)


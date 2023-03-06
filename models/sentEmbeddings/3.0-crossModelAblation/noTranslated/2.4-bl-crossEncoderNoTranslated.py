#!/usr/bin/env python
# coding: utf-8

# ## News Article Similarity Modelling
# - Cross encoding 
# - Translated data 
# - Using Title 

# In[5]:


from tqdm.auto import tqdm
import torch 
import random
from torch import nn
from transformers import RobertaTokenizer, PreTrainedTokenizer, DistilBertTokenizer, DistilBertModel, RobertaModel
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses
from datasets import Dataset
import pandas as pd
from transformers.optimization import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from torch.nn import CosineEmbeddingLoss
import transformers
import pickle 
import time
#Build up to SBERT model 


# In[6]:


DEVICE_NUM = 0 
BATCH_SIZE = 5
EPOCHS = 3
SEED = 85
FOLDS = 5
RDROP_WEIGHT = .1
FORWARD_WEIGHT = (1 - RDROP_WEIGHT) / 2

device = torch.device("cuda:" + str(DEVICE_NUM) if torch.cuda.is_available() else "cpu")

RESULTS_PATH = "/home/blitt/projects/localNews/models/sentEmbeddings/3.0-crossModelAblation/noTranslated"


# In[7]:


#set seeds 
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


# In[8]:


#just needed to switch out datasets for this ablation 
df = pd.read_csv("/shared/3/projects/newsDiffusion/data/processed/en_200_56.tsv", sep="\t")

#put ground truth values into a list 
df["ground_truth"] = df['Overall']

#get only the columns we need 
#for when using merged text
leanDf = df[["ground_truth",  'text1Merged', 'text2Merged', 'url1_lang', 'url2_lang']]

#rescale data from (0, 4): (0, 1)
leanDf["ground_truth"] = 1 - ((leanDf["ground_truth"] - 1) / 3)

#reset index so it is contiguous set of numbers 
leanDf = leanDf.reset_index(drop=True)

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
     #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# In[7]:


class Model(nn.Module): 
    def __init__(self):
        super(Model,self).__init__()
        self.model = RobertaModel.from_pretrained('roberta-base')
        self.l1 = nn.Linear(768, 256).to(device)
        self.l2 = nn.Linear(256, 1)
        self.GELU = nn.GELU()
        self.loss_func = torch.nn.MSELoss(reduction="mean")
        
    def mean_pooling(self, token_embeddings, attention_mask): 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask): 
        
        #encode sentence and get mean pooled sentence representation 
        encoding1 = self.model(input_ids, attention_mask=attention_mask)[0]  #all token embeddings
        meanPooled1 = self.mean_pooling(encoding1, attention_mask)
       
        pred1 = self.l2(self.GELU(self.l1(meanPooled1)))
        
        encoding2 = self.model(input_ids, attention_mask=attention_mask)[0]  #all token embeddings
        meanPooled2 = self.mean_pooling(encoding2, attention_mask)
        
        pred2 = self.l2(self.GELU(self.l1(meanPooled2)))
        
        
        return [pred1, pred2]


# In[8]:


def validation(model, validLoader, loss_func): 
    model.eval()
    lossList = []
    predList = []
    GT = []

    i = True 
    for batch in validLoader: 

        # prepare batches and more all to the active device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch['ground_truth'].to(device).unsqueeze(1)

        #send batch info through model 
        pred1, pred2 = model(input_ids, attention_mask)
        #pred1, pred2 = pred1.unsqueeze(0), pred2.unsqueeze(0)
        
        #get loss relating to label prediction 
        loss1 = loss_func(label, pred1) * FORWARD_WEIGHT 
        loss2 = loss_func(label, pred2) * FORWARD_WEIGHT
        loss_r = loss_func(pred1, pred2) * RDROP_WEIGHT
        loss = (loss1 + loss2 + loss_r)
        
        #get output metrics 
        lossList.append(loss1.detach().cpu().item())
        predList.append(float(pred1.detach().cpu()))
        GT.append(float(label.detach().cpu()))
        
        del loss1
        del loss2
        del loss_r
        del loss
        del pred1
        del pred2
        del label 
    #print(vGT)
    return [lossList, predList, GT]

        


# In[9]:


#set up relevant variables 
def train(trainDataset, validDataset): 
    torch.cuda.empty_cache()
    #get loaders 
    trainLoader = torch.utils.data.DataLoader(
        trainDataset, batch_size=BATCH_SIZE, shuffle=True
    )
    validLoader = torch.utils.data.DataLoader(
        validDataset, batch_size=1, shuffle=True
    )
    
    trainLen = len(trainDataset)

    #load the model 
    model = Model().to(device)

    #TODO: double check on if reduction="mean" is the right move here...
    #could cosine similarity also work..? I think that is between the two predicted vectors though.. 
    loss_func = torch.nn.MSELoss(reduction="mean")

    # we would initialize everything first
    optim = torch.optim.Adam(model.parameters(), lr=2e-5)

    #set up scheduler
    # and setup a warmup for the first ~10% steps
    total_steps = int((trainLen*EPOCHS) / BATCH_SIZE)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps - warmup_steps)
    
    
    #now run training loop 
    lossList = []
    validMetrics = []
    subLossList = []
    # increase from 1 epoch if need be 
    for epoch in range(EPOCHS):
        torch.cuda.empty_cache()
        model.train()  # make sure model is in training mode

        # initialize the dataloader loop with tqdm (tqdm == progress bar)
        loop = tqdm(trainLoader, leave=True)

        validMetrics.append(validation(model, validLoader, loss_func))
        model.train()

        for i, batch in enumerate(loop): 
            # zero all gradients on each new step
            optim.zero_grad()

            # prepare batches and more all to the active device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch['ground_truth'].to(device).unsqueeze(1)

            #send batch info through model 
            pred1, pred2 = model(input_ids, attention_mask)
            #pred1, pred2 = pred1.unsqueeze(0), pred2.unsqueeze(0)
            
            #get loss for label prediction, rdrop 
            loss1 = loss_func(label, pred1) * FORWARD_WEIGHT 
            loss2 = loss_func(label, pred2) * FORWARD_WEIGHT
            loss_r = loss_func(pred1, pred2) * RDROP_WEIGHT
            loss = (loss1 + loss2 + loss_r)

            # using loss, calculate gradients and then optimize
            loss.backward()
            optim.step()

            #get mean loss over last 20 batches 
            if i % 20 == 0: 
                lossList.append(np.mean(subLossList))
                subLossList = []
                pass

            subLossList.append(float(loss.detach().item()))
            

            # update learning rate scheduler
            scheduler.step()

            # update the TDQM progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
            del loss1
            del loss2
            del loss_r
            del loss
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    validMetrics.append(validation(model, validLoader, loss_func))
    return validMetrics 
    del model
    del trainLoader
    del validLoader


# In[10]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=FOLDS, shuffle=True)


# In[11]:


#time how long it takes 
st = time.time()

metrics = []
transformers.logging.set_verbosity_error()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

#JUST FOR TESTING 
#leanDf = leanDf[:300]

#we only want to sample validation data from the pairs that are both english 
enDf = leanDf[(leanDf["url1_lang"] == "en") & (leanDf["url2_lang"] == "en")]

print("Total df len: " +  str(len(leanDf)))
print("English df len: " +  str(len(enDf)))
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
    print("###### " + str(i).upper() + " ######")
    print("Train df len: " + str(len(trainDf)))
    print("Valid df len: " + str(len(validDf)))
    
    #get data loaded in properly 
    trainDataset = Dataset.from_pandas(trainDf)
    validDataset = Dataset.from_pandas(validDf)
    
    
    trainDataset = trainDataset.map(lambda x: tokenizer(x["text1Merged"], x["text2Merged"], max_length=512, padding="max_length", truncation=True))
    validDataset = validDataset.map(lambda x: tokenizer(x["text1Merged"], x["text2Merged"], max_length=512, padding="max_length", truncation=True))

    #only need the input information 
    trainDataset = trainDataset.remove_columns(["text1Merged", "text2Merged", "__index_level_0__"])
    validDataset = validDataset.remove_columns(["text1Merged", "text2Merged", "__index_level_0__"])

    # convert dataset features to PyTorch tensors
    validDataset.set_format(type='torch', columns=["ground_truth", "input_ids", "attention_mask"])
    trainDataset.set_format(type='torch', columns=["ground_truth", "input_ids", "attention_mask"])

    validMetrics = train(trainDataset, validDataset)
    metrics.append(validMetrics)
    
    del trainDataset
    del validDataset
    

et = time.time()
elapsed = et - st
print("ELAPSED TIME")
print(elapsed)


# In[ ]:


#write to an output folder 
import pickle 
with open(RESULTS_PATH + "/outputData.pkl", "wb") as f: 
    pickle.dump(metrics, f)
    
with open(RESULTS_PATH + "/time.pkl", "wb") as f: 
    pickle.dump(elapsed, f)


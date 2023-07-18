#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
#Build up to SBERT model 


# In[3]:


deviceNum = 3
device = torch.device("cuda:" + str(deviceNum) if torch.cuda.is_available() else "cpu")

GRAD_ACC = 6
EPOCHS = 2
FOLDS = 5
SEED = 85
BATCH_SIZE = 5

#set seeds 
torch.manual_seed(85)
random.seed(85)

MODEL_OUTPUT_PATH = "/home/blitt/projects/localNews/models/sentEmbeddings/2.0-biModelAblation/finalModel/state_dict.tar"


# In[5]:


#df = pd.read_csv("/shared/3/projects/benlitterer/localNews/NetworkMVP/translatedCleaned.tsv", sep="\t")
df = pd.read_csv("/home/blitt/projects/localNews/data/processed/translated_288_96.tsv", sep="\t")

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



# In[6]:


len(leanDf)


# In[8]:


len(leanDf[(leanDf["url1_lang"] == "en") & (leanDf["url1_lang"] == "en")])


# In[9]:


1787-1738


# In[8]:


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


# In[13]:


def trainBi(trainDataset): 
    model = BiModel().to(device)
    
    # we would initialize everything first
    optim = torch.optim.Adam(model.parameters(), lr=2e-6)
    
    # and setup a warmup for the first ~10% steps
    total_steps = int(len(trainDataset) / BATCH_SIZE)*EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps - warmup_steps)

    loss_func = torch.nn.MSELoss(reduction="mean")

    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(EPOCHS):
        print("EPOCH: " + str(epoch))
        
        model.train()  # make sure model is in training mode

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
            
    return model 


# In[14]:


transformers.logging.set_verbosity_error()
biTokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')

print("Train df len: " +  str(len(leanDf)))

biCorrs = []

trainDataset = Dataset.from_pandas(leanDf)

all_cols = ["ground_truth"]
#NOTE: here we use the merged text
for part in ["text1Merged", "text2Merged"]: 
    #tokenizes each row of the dataset and gives us back tuple of lists 
    trainDataset = trainDataset.map(lambda x: biTokenizer(x[part], max_length=384, padding="max_length", truncation=True))

    for col in ['input_ids', 'attention_mask']: 
        trainDataset = trainDataset.rename_column(col, part+'_'+col)
        all_cols.append(part+'_'+col)

trainDataset.set_format(type='torch', columns=all_cols)
trainedModel = trainBi(trainDataset)


# In[21]:


torch.save(trainedModel.state_dict(), MODEL_OUTPUT_PATH)


# In[22]:


#load trainedModel 
trainedModel = BiModel()
trainedModel.load_state_dict(torch.load(MODEL_OUTPUT_PATH))



# In[ ]:





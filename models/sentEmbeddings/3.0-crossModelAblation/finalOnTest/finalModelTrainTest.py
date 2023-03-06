#!/usr/bin/env python

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


# In[30]:


DEVICE_NUM = 0

GRAD_ACC = 6
EPOCHS = 3
BATCH_SIZE = 5
RDROP_WEIGHT = .1
FORWARD_WEIGHT = (1 - RDROP_WEIGHT) / 2

device = torch.device("cuda:" + str(DEVICE_NUM) if torch.cuda.is_available() else "cpu")


#df = pd.read_csv("/shared/3/projects/benlitterer/localNews/NetworkMVP/translatedCleaned.tsv", sep="\t")
df = pd.read_csv("/shared/3/projects/newsDiffusion/data/processed/translated_200_56.tsv", sep="\t") 

#put ground truth values into a list 
df["ground_truth"] = df['Overall']

#get only the columns we need 
leanDf = df[["ground_truth",  'text1Merged', 'text2Merged', 'url1_lang', 'url2_lang']]

#rescale data from (0, 4): (0, 1)
leanDf["ground_truth"] = 1 - ((leanDf["ground_truth"] - 1) / 3)

#reset index so it is contiguous set of numbers 
leanDf = leanDf.reset_index(drop=True)

#get the test data loaded in 
#this is the test data that has already had the 
#title concatenated and the head + tail merged
#TODO: this test set doesn't currently exist so we need to make it
testDf = pd.read_csv("/shared/3/projects/newsDiffusion/data/processed/enTest_200_56.tsv", sep="\t")

testDf["ground_truth"] = testDf["Overall"]

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
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(.25)
        self.loss_func = torch.nn.MSELoss(reduction="mean")
        
    def mean_pooling(self, token_embeddings, attention_mask): 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask): 
        
        #encode sentence and get mean pooled sentence representation 
        encoding1 = self.model(input_ids, attention_mask=attention_mask)[0]  #all token embeddings
        meanPooled1 = self.mean_pooling(encoding1, attention_mask)
       
        pred1 = self.l2(self.dropout(self.ReLU(self.l1(meanPooled1))))
        
        encoding2 = self.model(input_ids, attention_mask=attention_mask)[0]  #all token embeddings
        meanPooled2 = self.mean_pooling(encoding2, attention_mask)
        
        pred2 = self.l2(self.dropout(self.ReLU(self.l1(meanPooled2))))
        
        
        return [pred1, pred2]

#set up relevant variables 
def train(trainDataset): 
    torch.cuda.empty_cache()
    #get loaders 
    trainLoader = torch.utils.data.DataLoader(
        trainDataset, batch_size=BATCH_SIZE, shuffle=True
    )
    
    trainLen = len(trainDataset)

    #load the model 
    model = Model().to(device)

    loss_func = torch.nn.MSELoss(reduction="mean")

    # we would initialize everything first
    optim = torch.optim.Adam(model.parameters(), lr=2e-5)

    #set up scheduler
    # and setup a warmup for the first ~10% steps
    total_steps = int((trainLen*EPOCHS) / BATCH_SIZE)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps - warmup_steps)
    
    
    # increase from 1 epoch if need be 
    for epoch in range(EPOCHS):
        torch.cuda.empty_cache()
        model.train()  # make sure model is in training mode

        # initialize the dataloader loop with tqdm (tqdm == progress bar)
        loop = tqdm(trainLoader, leave=True)

        for i, batch in enumerate(loop): 
            # zero all gradients on each new step
            optim.zero_grad()

            # prepare batches and more all to the active device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch['ground_truth'].to(device).unsqueeze(1)

            #send batch info through model 
            pred1, pred2 = model(input_ids, attention_mask)
        
            #get loss for label prediction, rdrop 
            loss1 = loss_func(label, pred1) * FORWARD_WEIGHT 
            loss2 = loss_func(label, pred2) * FORWARD_WEIGHT
            loss_r = loss_func(pred1, pred2) * RDROP_WEIGHT
            loss = (loss1 + loss2 + loss_r)

            # using loss, calculate gradients and then optimize
            loss.backward()
            optim.step()

            # update learning rate scheduler
            scheduler.step()

            # update the TDQM progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
            del loss1
            del loss2
            del loss_r
            del loss

    del trainLoader
    return model 

#tokenizer 
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

#load in our test data 
testDataset = Dataset.from_pandas(testDf[["text1Merged", "text2Merged"]])
    
testDataset = testDataset.map(lambda x: tokenizer(x["text1Merged"], x["text2Merged"], max_length=512, padding="max_length", truncation=True))

#only need the input information 
testDataset = testDataset.remove_columns(["text1Merged", "text2Merged"])

# convert dataset features to PyTorch tensors
testDataset.set_format(type='torch', columns=["input_ids", "attention_mask"])

def testModel(trainedModel, testDataset): 
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=False)
    print(len(testLoader))
    simList = []
    for i, batch in tqdm(enumerate(testLoader)): 
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sim = trainedModel(input_ids, attention_mask)[0]
        print((BATCH_SIZE*i) +1) 
        #if we just so happen to get a batch size of one at the end  
        if (BATCH_SIZE*i) +1 == len(testDataset) :  
            simList += sim.detach().cpu().tolist()[0]
        else: 
            simList += sim.squeeze().detach().cpu().tolist()
    
    print(simList)
    testDf["sims"] = simList
    testDf["scaledSims"] = (3*(1-testDf["sims"])) + 1
    
    corrMat = np.corrcoef(testDf["ground_truth"], testDf["scaledSims"])
    corr = corrMat[0, 1]
    print(corr)
    return [corr, testDf, corrMat]

#write to an output folder 
RESULTS_PATH = "/shared/3/projects/newsDiffusion/models/3.0-crossModelAblation/finalOnTest/"

seedList = [85, 92, 200, 135, 60]
finalCorrs = {}

for seed in seedList:
    
    #set seeds 
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed) 

    #get tokenizer. This is done in the loop so we have random ordering
    transformers.logging.set_verbosity_error()
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    print("Train df len: " +  str(len(leanDf)))

    trainDataset = Dataset.from_pandas(leanDf[["text1Merged", "text2Merged", "ground_truth"]])
    
    trainDataset = trainDataset.map(lambda x: tokenizer(x["text1Merged"], x["text2Merged"], max_length=512, padding="max_length", truncation=True))

    #only need the input information 
    trainDataset = trainDataset.remove_columns(["text1Merged", "text2Merged"])
    
    # convert dataset features to PyTorch tensors
    trainDataset.set_format(type='torch', columns=["ground_truth", "input_ids", "attention_mask"])
    
    trainedModel = train(trainDataset)
    corr, testDf, corrMat = testModel(trainedModel, testDataset)
    finalCorrs[seed] = corr

    #write output
    testDf.to_csv(RESULTS_PATH + str(seed) + "testDf.tsv", "\t")

    #save this trained model. We will use the best one in the pipeline 
    torch.save(trainedModel.state_dict(), RESULTS_PATH + str(seed) + "/state_dict.tar")
    
    #just for memory purposes 
    del trainedModel 
    del trainDataset


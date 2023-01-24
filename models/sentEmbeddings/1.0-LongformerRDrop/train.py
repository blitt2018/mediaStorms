from tqdm.auto import tqdm
import torch 
import random
from torch import nn
import json 
from transformers import LongformerConfig, LongformerModel, PreTrainedTokenizer, LongformerTokenizer
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
import seaborn as sns
import sys
#Build up to SBERT model 

jsonPath = sys.argv[1]
with open(jsonPath, "r") as f:
    my_dict = json.load(f)
GPU = int(my_dict["GPU"]) 
EPOCHS = int(my_dict["EPOCHS"])
BATCH_SIZE = int(my_dict["BATCH_SIZE"])
DROPOUT = float(my_dict["DROPOUT"])
REG_ALPHA = float(my_dict["REG_ALPHA"])
NUM_TOKENS = int(my_dict["NUM_TOKENS"])
LANGUAGES = str(my_dict["LANGUAGES"])

deviceNum = GPU
device = torch.device("cuda:" + str(deviceNum) if torch.cuda.is_available() else "cpu")

def check_mem():
    torch.cuda.empty_cache()
    a = torch.cuda.memory_allocated(deviceNum)/1024/1024/1024
    r = torch.cuda.memory_reserved(deviceNum)/1024/1024/1024
    print("torch.cuda.memory_allocated: %fGB"%a)
    print("torch.cuda.memory_reserved: %fGB"%r)
    print("torch.cuda.memory_free: %fGB"%(r-a))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(deviceNum)/1024/1024/1024))
check_mem()

df = pd.read_csv("/shared/3/projects/benlitterer/localNews/NetworkMVP/translatedCleaned.tsv", sep="\t")
#do language selection/cutoff if need be 
if LANGUAGES != "justEn":
    print("TODO: figure out language selection for config") 
#df = df.loc[(df["url1_lang"] == "en") & (df["url2_lang"] == "en")]

#TODO: figure out how to do selection here from config as well...
groundTruths = ["Overall"]
features = ['text1', 'text2', 'title1', 'title2', 'url1_lang', 'url2_lang']
toSelect = groundTruths + features

#get only the columns we need
#TODO: do we need "pair_id"?
leanDf = df[toSelect].dropna()

#rescale data from (0, 4): (0, 1)
for colName in groundTruths:
    leanDf[colName] = 1 - ((leanDf[colName] - 1) / 3)

#reset index so it is contiguous set of numbers
leanDf = leanDf.reset_index(drop=True)

#now combine title and text together
#first add ". " to title
leanDf["title1"] = leanDf["title1"].apply(lambda x: x + ". ")
leanDf["title2"] = leanDf["title2"].apply(lambda x: x + ". ")

leanDf["text1"] = leanDf["title1"] + leanDf["text1"]
leanDf["text2"] = leanDf["title2"] + leanDf["text2"]


#we only want to sample validation data from the pairs that are both english
enDf = leanDf[(leanDf["url1_lang"] == "en") & (leanDf["url2_lang"] == "en")]
validProp = .1
validCount = int(validProp * len(enDf))
print(validCount)
validIndices = random.sample(list(enDf.index), validCount)

#get dataframe with indices of only the original english pairs
validDf = enDf.loc[validIndices]

#train data should be all rows that aren't in the validation set
#here we are taking a set difference and then indexing what remains
trainDf = leanDf.loc[set(leanDf.index) - set(validIndices)]

#get data loaded in properly
trainDataset = Dataset.from_pandas(trainDf)
validDataset = Dataset.from_pandas(validDf)

#get tokenizer 
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

#tokenize
from transformers.utils import logging
logging.set_verbosity_error()
trainDataset = trainDataset.map(lambda x: tokenizer(x["text1"], x["text2"], max_length=1024, padding="max_length", truncation=True))
validDataset = validDataset.map(lambda x: tokenizer(x["text1"], x["text2"], max_length=1024, padding="max_length", truncation=True))

#only need the input information 
trainDataset = trainDataset.remove_columns(["text1", "text2", "__index_level_0__"])
validDataset = validDataset.remove_columns(["text1", "text2", "__index_level_0__"]
)


# convert dataset features to PyTorch tensors
formatColumns = groundTruths + ["input_ids", "attention_mask"]
validDataset.set_format(type='torch', columns=formatColumns)
trainDataset.set_format(type='torch', columns=formatColumns)

# initialize the dataloader
trainLoader = torch.utils.data.DataLoader(
    trainDataset, batch_size=BATCH_SIZE, shuffle=True
)
validLoader = torch.utils.data.DataLoader(
    validDataset, batch_size=1, shuffle=True
)

#import model 
from Model import Model 

#load in the model 
model = Model(device, DROPOUT).to(device)

#TODO: double check on if reduction="mean" is the right move here...
#could cosine similarity also work..? I think that is between the two predicted vectors though..
loss_func = torch.nn.MSELoss(reduction="mean")

trainLen = len(trainDataset)

# we would initialize everything first
optim = torch.optim.Adam(model.parameters(), lr=5e-6)

# and setup a warmup for the first ~10% steps
total_steps = int((trainLen*EPOCHS) / BATCH_SIZE)
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps - warmup_steps)

#set up fancy weighted loss function 
"""
Get the loss across multiple different objectives.
Since overall is most important it gets more weight.
"""
def getWeightedLoss(predTens, gtTens):
    #try getting rid of Tone and Style
    LOSS_WEIGHTS = [1]
    loss = 0.0
    for i in range(len(LOSS_WEIGHTS)):

        #get ground truth value associated with this column name
        currGT = gtTens[:, :, i]

        #TODO: figure out how to index properly here
        pred = predTens[:, :, i]

        """
        print("pred")
        print(pred)
        print(pred.shape)
        print("GT")
        print(currGT)
        print(currGT.shape)
        """
        #get loss
        loss += (loss_func(pred, currGT) * LOSS_WEIGHTS[i])
    return loss

#validation loop
def validation():
    model.eval()
    lossList = []
    pred = []
    GT = []

    i = True
    for batch in validLoader:

        # prepare batches and more all to the active device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        #label = batch['ground_truth'].to(device).unsqueeze(1)

        #send batch info through model
        pred1, pred2 = model(input_ids, attention_mask)
        pred1 = pred1.unsqueeze(0)
        pred2 = pred2.unsqueeze(0)
        #print(pred1)
        gts = torch.stack([batch[colName] for colName in groundTruths], 0).to(device).T.unsqueeze(0)
        #return gts, pred1

        #get wegihted loss relating to label prediction
        loss1 = getWeightedLoss(gts, pred1)
        loss2 = getWeightedLoss(gts, pred2)
        loss_b = .5*(loss1 + loss2)

        #get loss relating to invariance to dropout
        #NOTE:
        loss_r = getWeightedLoss(pred1, pred2)

        #combine losses with alpha hyperparam
        loss = REG_ALPHA*loss_r + (1-REG_ALPHA)*loss_b
        lossList.append(loss.item())

        #careful about dimensions...
        #we will definitely have 3 dimensions here, if they sum to 3 then
        #that means every dimension is one
        if sum(pred1.size()) != 3:
            pred.append([float(item) for item in list(pred1.squeeze())])
            GT.append([float(item) for item in list(gts.squeeze())])
        else:
            pred.append([float(item) for item in [pred1.squeeze()]])
            GT.append([float(item) for item in [gts.squeeze()]])

        if not (len(lossList) == len(pred) == len(pred)):
            print("lens not equal")
    #print(vGT)
    return [lossList, pred, GT]

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
trainDict = {}
lossList = []
validMetrics = []
trainMetrics = []
subLossList = []
# increase from 1 epoch if need be
for epoch in range(EPOCHS):

    model.train()  # make sure model is in training mode

    # initialize the dataloader loop with tqdm (tqdm == progress bar)
    loop = tqdm(trainLoader, leave=True)

    print("starting validation")
    validMetrics.append(validation())
    #validTester = validation()
    print("finishing validation")


    model.train()

    for i, batch in enumerate(loop):
        # zero all gradients on each new step
        optim.zero_grad()

        # prepare batches and more all to the active device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch["attention_mask"].to(device)


        #send batch info through model
        pred1, pred2 = model(input_ids, attention_mask)
        pred1 = pred1.unsqueeze(0)
        pred2 = pred2.unsqueeze(0)

        gts = torch.stack([batch[colName] for colName in groundTruths], 0).T.to(device).unsqueeze(0)


        #get loss relating to label prediction
        loss1 = getWeightedLoss(gts, pred1)
        loss2 = getWeightedLoss(gts, pred2)
        loss_b = .5*(loss1 + loss2)

        #get loss relating to invariance to dropout
        loss_r = getWeightedLoss(pred1, pred2)

        #combine losses with alpha hyperparam
        loss = REG_ALPHA*loss_r + (1-REG_ALPHA)*loss_b

        # using loss, calculate gradients and then optimize
        loss.backward()
        optim.step()

        #get mean loss over last 20 batches
        if i % 20 == 0 and i > 0:
            #print(subLossList)
            lossList.append(np.mean(subLossList))
            subLossList = []

        subLossList.append(float(loss.item()))

        # update learning rate scheduler
        scheduler.step()

        # update the TDQM progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
validMetrics.append(validation())

#evaluation of results!
outPath = sys.argv[2].rstrip("/") 

lossSmoothing = 20
lossIndex = [(i * lossSmoothing)/len(loop) for i in range(len(lossList))]
lossDf = pd.DataFrame({"lossIndex":lossIndex, "lossList":lossList}) 
lossDf.to_pickle(outPath + "/lossDf.pickle") 

validArr = np.array(validMetrics)

outDfList = []
iterList = []
corrList = []
#go through each validation step

for i in range(validArr.shape[0]):
    print(i)
    subDf = pd.DataFrame(validArr[i].T)
    subDf.columns = ["loss", "pred", "true"]

    predCols = ["pred" + item for item in groundTruths]
    gtCols = ["gt" + item for item in groundTruths]


    subDf[predCols] = pd.DataFrame(subDf["pred"].tolist(), index=subDf.index)
    subDf[gtCols] = pd.DataFrame(subDf["true"].tolist(), index=subDf.index)

    corrScores = []
    for colName in groundTruths:
        corr = np.corrcoef(subDf["pred" + colName], subDf["gt" + colName])[1, 0]
        corrScores.append(corr)
    corrList.append(corrScores)

corrDf = pd.DataFrame(corrList, columns=groundTruths)
corrDf.to_pickle(outPath + "/corrDf.pickle") 





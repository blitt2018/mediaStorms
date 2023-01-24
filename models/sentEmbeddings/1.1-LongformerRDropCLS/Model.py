from tqdm.auto import tqdm
import torch
import random
from torch import nn
from transformers import LongformerConfig, LongformerModel, PreTrainedTokenizer, LongformerTokenizer
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses
from datasets import Dataset
import pandas as pd
from transformers.optimization import get_linear_schedule_with_warmup
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from torch.nn import CosineEmbeddingLoss
import transformers
import json 

class Model(nn.Module): 
    def __init__(self, device, DROPOUT):
        super(Model,self).__init__()
        #test getting the longformer model going 
        self.model = LongformerModel.from_pretrained('allenai/longformer-base-4096',output_hidden_states = True)
        self.ReLU = nn.ReLU()
        self.GELU = nn.GELU
        self.dropout = nn.Dropout(DROPOUT)
        self.l1 = nn.Linear(768, 512).to(device)
        self.l2 = nn.Linear(512, 250).to(device)
        self.l3 = nn.Linear(250, 1).to(device)
        self.loss_func = torch.nn.MSELoss(reduction="mean")
        
    def mean_pooling(self, token_embeddings, attention_mask): 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask): 
    
    
        #encode sentence and get mean pooled sentence representation 
        
        #referring here to: 
        #https://huggingface.co/docs/transformers/v4.24.0/en/model_doc/longformer#transformers.LongformerModel.forward
        #note that the first index of model output should give us a good cls token 
        #there is an additional linear layer here as well 
        encoding = self.model(input_ids, attention_mask=attention_mask)  #all token embeddings
        #Debugging: print(encoding.squeeze().shape)
        encoding = encoding["last_hidden_state"][:,1,:] 

        #NOTE: Since dropout is random we simply send data through twice 
        #to get two predictions that have some noise 
        out = self.l1(encoding)
        out = self.ReLU(out)
       
        out = self.l2(out)
        out = self.ReLU(out)
        out = self.dropout(out)
        
        pred1 = self.l3(out)
        
        #the first index of the model output should give us a good cls token 
        #note that there is an additional linear layer here as well 
        encoding = self.model(input_ids, attention_mask=attention_mask) #all token embeddings
        encoding = encoding["last_hidden_state"][:,1,:] 

        #NOTE: Since dropout is random we simply send data through twice 
        #to get two predictions that have some noise 
        out = self.l1(encoding)
        out = self.ReLU(out)
        
        out = self.l2(out)
        out = self.ReLU(out)
        out = self.dropout(out)
        
        pred2 = self.l3(out)
        return pred1, pred2

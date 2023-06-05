
#environment variables 
export ENTITY_LIST="person,norp,fac,org,gpe,loc,product,event,work_of_art,law"

#cutoffs for the number of articles associated with a named entity before that 
#named entity is no longer considered in filtering process 
export MIN_ENTITY_GROUP_SIZE="2" 
export MAX_ENTITY_GROUP_SIZE="1000"
export CLEANED_IN_PATH="/shared/3/projects/newsDiffusion/data/processed/newsData/fullDataWithNERCleaned.tsv"
export EMBEDDINGS_IN_PATH="/shared/3/projects/newsDiffusion/data/interim/NEREmbedding/embeddingsKeys.tsv" 
export PLOT_OUTPUT="/home/blitt/projects/localNews/reports/figures/TEST_RUN/clusterSizeHist.png"

export PAIR_OUT_PATH="/shared/3/projects/newsDiffusion/data/interim/NEREmbedding/entityPairs2020.pkl" 
export SIM_PAIR_OUT_PATH="/shared/3/projects/newsDiffusion/data/interim/NEREmbedding/embeddingPairSimilarity2020_2_1000_8.pkl" 

export SIM_CUTOFF=".8"
export CLUSTER_LABEL_OUT_PATH="/shared/3/projects/newsDiffusion/data/interim/NEREmbedding/embeddingClusterList2020_2_1000_8.pkl"
export GRAPH_EDGELIST_OUT_PATH="/shared/3/projects/newsDiffusion/data/interim/NEREmbedding/graph2020_2_1000_8.pkl"

#get candidate pairs using named entity inverted index  
python3 -u 0.0-bl-getEntityPairs.py $ENTITY_LIST $MIN_ENTITY_GROUP_SIZE $MAX_ENTITY_GROUP_SIZE $CLEANED_IN_PATH $EMBEDDINGS_IN_PATH $PAIR_OUT_PATH $PLOT_OUTPUT  

#compute pairwise cosine similarity from list of candidate pairs 
python3 -u 0.1-bl-computeCosineSim.py $PAIR_OUT_PATH $SIM_PAIR_OUT_PATH  

#apply similarity threshold, get connected components and assign unique cluster id to each clust 
python3 -u 0.2-bl-createEmbeddingClusterList.py $SIM_CUTOFF $PAIR_OUT_PATH $SIM_PAIR_OUT_PATH $CLUSTER_LABEL_OUT_PATH $GRAPH_EDGELIST_OUT_PATH 

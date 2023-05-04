BASE_PATH="/shared/3/projects/newsDiffusion/data/interim/topicModelling"
/shared/0/resources/mallet/mallet-2.0.8/bin/mallet train-topics --input $BASE_PATH/formatted_text_2020.mallet \
      --num-topics 40 --num-threads 50 --output-state $BASE_PATH/topic-state.gz --output-doc-topics $BASE_PATH/doc_topics.txt --output-topic-keys $BASE_PATH/topic_keys.txt 

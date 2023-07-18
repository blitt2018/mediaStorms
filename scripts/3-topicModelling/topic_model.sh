BASE_PATH="/shared/3/projects/newsDiffusion/data/interim/topicModelling"
/shared/0/resources/mallet/mallet-2.0.8/bin/mallet train-topics --input $BASE_PATH/formatted_text_full_time_series.mallet \
      --num-topics 40 --num-threads 50 --output-state $BASE_PATH/40/topic-state.gz --output-doc-topics $BASE_PATH/40/doc_topics.txt --output-topic-keys $BASE_PATH/40/topic_keys.txt 

BASE_PATH="/shared/3/projects/newsDiffusion/data/interim/topicModelling" 
/shared/0/resources/mallet/mallet-2.0.8/bin/mallet import-file --input $BASE_PATH/raw_text_2020.txt --keep-sequence --remove-stopwords TRUE --output $BASE_PATH/formatted_text_2020.mallet

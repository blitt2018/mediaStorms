#run all ablations for the bi-encoder model 
BASE_PATH="/home/blitt/projects/localNews/scripts/1-similarityModelling"

python3 $BASE_PATH/1.2-bl-biEncoderHead.py
python3 $BASE_PATH/1.1-bl-biEncoder_192_192.py
python3 $BASE_PATH/1.0-bl-biEncoder_288_96.py
python3 $BASE_PATH/1.3-bl-biEncoder_en.py

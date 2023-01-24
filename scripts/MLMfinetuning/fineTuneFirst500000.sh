#just a wrapper for calling a script 
export CUDA_VISIBLE_DEVICES=2
python run_mlm.py \
    --model_name_or_path roberta-base \
    --train_file /shared/3/projects/benlitterer/localNews/fineTuningIntermediate/first500000title_text.txt \
    --per_device_train_batch_size 5 \
    --per_device_eval_batch_size 5 \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --overwrite_output_dir \
    --validation_split_percentage 1 \
    --output_dir /home/blitt/projects/localNews/models/sentEmbeddings/2.0-RoBERTaFinetuned500000 

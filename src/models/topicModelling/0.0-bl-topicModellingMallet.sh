#for 30 topics 
/shared/0/resources/mallet/mallet-2.0.8/bin/mallet import-file --input malletInputQuarter.tsv --output /shared/3/projects/benlitterer/localNews/malletInputQuarter.mallet --keep-sequence

/shared/0/resources/mallet/mallet-2.0.8/bin/mallet train-topics --input malletInputQuarter.mallet --num-topics 30 --num-iterations 1000 --optimize-interval 10 --output-doc-topics /shared/3/projects/benlitterer/localNews/firstMallet/30topics/docTopics.txt --output-topic-keys /shared/3/projects/benlitterer/localNews/firstMallet/30topics/topicKeys.txt

mkdir firstMallet/50topics 

/shared/0/resources/mallet/mallet-2.0.8/bin/mallet train-topics --input malletInputQuarter.mallet --num-topics 50 --num-iterations 1500 --optimize-interval 10 --output-doc-topics /shared/3/projects/benlitterer/localNews/firstMallet/50topics/docTopics.txt --output-topic-keys /shared/3/projects/benlitterer/localNews/firstMallet/50topics/topicKeys.txt

mkdir firstMallet/70topics 

/shared/0/resources/mallet/mallet-2.0.8/bin/mallet train-topics --input malletInputQuarter.mallet --num-topics 70 --num-iterations 1500 --optimize-interval 10 --output-doc-topics /shared/3/projects/benlitterer/localNews/firstMallet/70topics/docTopics.txt --output-topic-keys /shared/3/projects/benlitterer/localNews/firstMallet/70topics/topicKeys.txt

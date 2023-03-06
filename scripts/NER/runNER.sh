#seq 0 159 | xargs -P 8 -I % sh -c 'python3 .py splitNews% NEROutput/'
ls /shared/3/projects/newsDiffusion/data/interim/NEREmbedding/NERSplits/ | xargs -i -n 1 -P 160 python3 2.1-bl-applyNERsingleInput.py {} /shared/3/projects/newsDiffusion/data/interim/NEREmbedding/NERSplitsComplete/  

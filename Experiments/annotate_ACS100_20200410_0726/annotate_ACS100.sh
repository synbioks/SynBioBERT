#!/bin/bash
#$ -cwd
#$ -j y -o /home/rodriguezne2/development/logs/bert_ner/synbiobert.log
#$ -m beas
#$ -M rodriguezne2@mymail.vcu.edu
#$ -l mem_free=15G,ram_free=15G,gpu=1,hostname=!b0[123456789]*&!b10*
# -l hostname=!b0[123456789]*&!b10*
#$ -pe smp 1
#$ -V
#$ -q g.q

#export PATH="/home/rodriguezne2/anaconda3/bin:$PATH"

source ~/envs/mtt_env/bin/activate

echo $PATH
export PYTHONPATH=${PYTHONPATH}:.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/NVIDIA/cuda-9.0/lib64/
python -V
CUDA_VISIBLE_DEVICES=0 python $1

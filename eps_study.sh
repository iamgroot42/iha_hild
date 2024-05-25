#!/bin/bash

DATASET=$1
MODEL=$2

# Purchase100-S needs special treatment - model0 is weird
for i in {1..10}
do
    python mib/attack.py --dataset $DATASET --model_arch $MODEL --num_points -1 --attack ProperTheoryRef --target_model_index $i --damping_eps 2e-1
    python mib/attack.py --dataset $DATASET --model_arch $MODEL --num_points -1 --attack ProperTheoryRef --target_model_index $i --damping_eps 1e-2
    python mib/attack.py --dataset $DATASET --model_arch $MODEL --num_points -1 --attack ProperTheoryRef --target_model_index $i --damping_eps 1e-1
    python mib/attack.py --dataset $DATASET --model_arch $MODEL --num_points -1 --attack ProperTheoryRef --target_model_index $i --damping_eps 2e-1 --low_rank
    python mib/attack.py --dataset $DATASET --model_arch $MODEL --num_points -1 --attack ProperTheoryRef --target_model_index $i --damping_eps 1e-2 --low_rank
    python mib/attack.py --dataset $DATASET --model_arch $MODEL --num_points -1 --attack ProperTheoryRef --target_model_index $i --damping_eps 1e-1 --low_rank
done

#!/bin/bash

for i in {0..9}
do
    # Run the attack
    python mib/attack.py --dataset purchase100 --model_arch mlp2 --num_points -1 --attack LiRAOnline --target_model_index $i
done
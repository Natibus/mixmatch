#!/usr/bin/env bash
# Create semi-supervised subsets ( you can use different seeds like in the readme )

#for seed in 1 2 3; do
for size in 10 50 100 250 500 1000 4000; do
    if [ -f $ML_DATA/${DATA_SET}-unlabeled.tfrecord ]; then
       CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=3 --size=${size} ${ML_DATA}/SSL/${DATA_SET} \
        ${ML_DATA}/${DATA_SET}-train.tfrecord $ML_DATA/${DATA_SET}-unlabeled.tfrecord
    else
       CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=3 --size=${size} ${ML_DATA}/SSL/${DATA_SET} \
        ${ML_DATA}/${DATA_SET}-train.tfrecord
    fi
    wait
done
#done
#!/usr/bin/env bash
usage="$(basename "$0") [-h] [-d dataset] [-n nb_samples] -- Script used to prepare datasets for mixmatch training

where:
    -h  show this help text
    -d  set the dataset name (cifar10 or mnist)
    "

while getopts 'h?d:n:' option; do
    case "${option}" in
        h) echo "$usage"
           exit
           ;;
        d) DATA_SET=${OPTARG};;
    esac
done

# Create semi-supervised subsets ( you can use different seeds like in the readme )

# The code below is based on mixmatch's README
export PYTHONPATH=${PYTHONPATH}:.
export ML_DATA='Data'

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
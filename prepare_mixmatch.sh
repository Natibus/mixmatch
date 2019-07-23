#!/usr/bin/env bash
# argument parsing in bash : https://www.lifewire.com/pass-arguments-to-bash-script-2200571
while getopts d:n: option
do
case "${option}"
in
d) DATA_SET=${OPTARG};;
n) NB_SAMPLES=${OPTARG};;
esac
done

#the code below is based on mixmatch's README
export PYTHONPATH=${PYTHONPATH}:.
export ML_DATA='Data'

# Download datasets
if [[ -z "$NB_SAMPLES" ]] ; then
    CUDA_VISIBLE_DEVICES= ./scripts/create_datasets.py --dataset_name=${DATA_SET}
else
    CUDA_VISIBLE_DEVICES= ./scripts/create_datasets.py --dataset_name=${DATA_SET} --nb_samples=${NB_SAMPLES}
fi

# Create semi-supervised subsets
 for size in 10 50 100 250 500 1000 4000; do
    CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=3 --size=${size} ${ML_DATA}/SSL/${DATA_SET} ${ML_DATA}/${DATA_SET}-train.tfrecord
    wait
done

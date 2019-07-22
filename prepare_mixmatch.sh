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

if [[ -z "$NB_SAMPLES" ]] ; then
    CUDA_VISIBLE_DEVICES= ./scripts/create_datasets.py --dataset_name=${DATA_SET}
else
    CUDA_VISIBLE_DEVICES= ./scripts/create_datasets.py --dataset_name=${DATA_SET} --nb_samples=${NB_SAMPLES}
fi

# Download datasets
#cp $ML_DATA/svhn-test.tfrecord $ML_DATA/svhn_noextra-test.tfrecord

# Create semi-supervised subsets
# for seed in 1 2 3 4 5; do
#      for size in 250 500 1000 2000 4000; do
 for size in 100 250 500 1000 3000; do
#         CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL/svhn $ML_DATA/svhn-train.tfrecord $ML_DATA/svhn-extra.tfrecord &
#         CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL/svhn_noextra $ML_DATA/svhn-train.tfrecord &
#         CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL/cifar10 $ML_DATA/cifar10-train.tfrecord &
    CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=3 --size=${size} ${ML_DATA}/SSL/${DATA_SET} ${ML_DATA}/${DATA_SET}-train.tfrecord
# done
     # CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=10000 $ML_DATA/SSL/cifar100 $ML_DATA/cifar100-train.tfrecord &
     # CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=1000 $ML_DATA/SSL/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord &
    wait
done
# CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=1 --size=5000 $ML_DATA/SSL/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord
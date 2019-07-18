#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:.
export ML_DATA="Data"
# Download datasets
CUDA_VISIBLE_DEVICES= ./scripts/create_datasets.py mnist
cp $ML_DATA/svhn-test.tfrecord $ML_DATA/svhn_noextra-test.tfrecord

# Create semi-supervised subsets
 for seed in 1 2 3 4 5; do
#      for size in 250 500 1000 2000 4000; do
     for size in 250 500; do
#         CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL/svhn $ML_DATA/svhn-train.tfrecord $ML_DATA/svhn-extra.tfrecord &
#         CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL/svhn_noextra $ML_DATA/svhn-train.tfrecord &
#         CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL/cifar10 $ML_DATA/cifar10-train.tfrecord &
         CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL/mnist $ML_DATA/mnist-train.tfrecord &
     done
     # CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=10000 $ML_DATA/SSL/cifar100 $ML_DATA/cifar100-train.tfrecord &
     # CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=1000 $ML_DATA/SSL/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord &
     wait
 done
# CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=1 --size=5000 $ML_DATA/SSL/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord
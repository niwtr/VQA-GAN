#!/bin/bash
SETTING="$1"
GPU="$2"
PYTHON=$(which python)

if [ "$SETTING" = "damsm_l2" ]; then
    echo "Starting training DAMSM (level 2) on the CLEVR 128 data set ."
    cd code/
    $PYTHON pretrain_DAMSM_l2.py --cfg cfg/DAMSM/clevr_l2.yml --gpu "$GPU"
    cd ../
elif [ "$SETTING" = "damsm_l1" ]; then
    echo "Starting training DAMSM (level 1) on the CLEVR 128 data set ."
    cd code/
    $PYTHON pretrain_DAMSM_l1.py --cfg cfg/DAMSM/clevr_l1.yml --gpu "$GPU"
    cd ../
elif [ "$SETTING" = "vqagan" ]; then
    echo "Starting training VQA-GAN on the CLEVR 128 data set."
    cd code/
    $PYTHON main.py --cfg cfg/clevr128_train.yml --gpu "$GPU"
    cd ../
elif [ "$SETTING" = "vqagan_l1" ]; then
    echo "Starting training VQA-GAN on the CLEVR 128 data set using L1 QA encoder."
    cd code/
    $PYTHON main_l1.py --cfg cfg/l1_clevr128_train.yml --gpu "$GPU"
    cd ../
elif [ "$SETTING" = "test_vqagan" ]; then
    echo "[PROFILE] Starting training on the CLEVR 128 data set."
    cd code/
    CUDA_LAUNCH_BLOCKING=1 kernprof -l main.py --cfg cfg/clevr128_train.yml --gpu "$GPU" 
    cd ../
else
    echo "Bad argument."
fi

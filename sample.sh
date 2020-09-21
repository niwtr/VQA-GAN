#!/bin/bash
SETTING="$1"
GPU="$2"
PYTHON=$(which python)

if [ "$SETTING" = "vqagan" ]; then
    echo "Sampling from the CLEVR data set."
    echo "Going to CLEVR folder."
    cd code/
    $PYTHON main.py --cfg cfg/clevr128_eval.yml --gpu "$GPU" --manualSeed 0
    cd ../
elif [ "$SETTING" = "vqagan_l1" ]; then
    echo "Sampling from the CLEVR data set."
    echo 
    cd code/
    $PYTHON main_l1.py --cfg cfg/l1_clevr128_eval.yml --gpu "$GPU" --manualSeed 0
    cd ../
else
    echo "Bad argument."
fi

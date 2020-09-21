input_image_dir=$1
feature_name=$2
gpu_id=$3

data_dir=/home/ffx/dataset/clevr128/test

export CUDA_VISIBLE_DEVICES=${gpu_id}
which python
python -u scripts/extract_features.py \
    --input_image_dir ${input_image_dir} \
    --output_h5_file ${data_dir}/${feature_name} --model_stage 2 || exit -1

cd ../tbd-nets
python -u vqa_evaluate.py ${feature_name} ${gpu_id} ${data_dir} || exit -1

# VQA-GAN: Image Synthesis from Locally Related Texts

This repository contains the PyTorch implementation for paper[ *Image Synthesis from Locally Related Texts*](https://dl.acm.org/doi/abs/10.1145/3372278.3390684) by Tianrui Niu, Fangxiang Feng, Lingxuan Li and Xiaojie Wang. 

We have currently released our CLEVR version of VQA-GAN only, because we consider the CLEVR version better highlights our motivation. This version can be used to reproduce the experiments on CLEVR dataset. For the MS-COCO dataset, we used a much bigger network that is impractical to train on normal GPUs. In our setting, training a MS-COCO version of VQA-GAN takes ~20GB memory on Nvidia TITAN RTX under batchsize 20 and image resolution of 128x128. Contact the author if you need more information. 

## Dependencies 

### Hardware Requirements

You need a decent GPU with proper memory space. We have trained the model under batch size 64 on Nvidia TITAN RTX. We recommend a minimum GPU memory size of ~11GB (GTX 1080Ti series or higher) so you can train the network with reasonable batch size. 

### Software Requirements

The code works under Python 2.7 with PyTorch 0.4.1 or PyTorch 1.1.0. We have not tested the code under other platforms. See [requirements.txt](requirements.txt) for other requirements.

## Deployment

### Download Preprocessed Data and Pretrained Models

Download raw QA data from [here](https://drive.google.com/file/d/1hJjvdWEZNeQLl_mSSIBpT9LpT4sK1fRv/view?usp=sharing) (~6.2MB) and decompress it to `qa_data` dir.

```bash
mkdir qa_data && cd qa_data
# put the downloaded file `clevr128_QA_01hop.tar.gz` here.
tar xf clevr128_QA_01hop.tar.xz
cd ..
```

Download images, scenes and program files from [here](https://drive.google.com/file/d/15V_x7TmvWkhm5uTt_594SPHe-UZoxSdL/view?usp=sharing) (~566MB), decompress it to `data` dir.

```bash
mkdir data && cd data
# put the downloaded file `clevr128_fix01hop.tar.gz` here.
tar xf clevr128_fix01hop.tar.xz
cd ..
```

Download pretrained DAMSM RNN encoders and TBD-Nets from [here](https://drive.google.com/file/d/1wCykklFpL12L0UK73v35sWs07L8FBSMV/view?usp=sharing) (~598MB) , decompress it to model dir. 

```bash
# put the downloaded file `models.tar.gz` in the root directory of this project.
tar xf models.tar.gz
```

### [Optional] Download Pretrained VQA-GAN models

You can grab our pretrained VQA-GAN models from [here](https://drive.google.com/file/d/18YjK-_qKNnIwmTcB4eMBaQktjf2shLOF/view?usp=sharing) (~2.45GB).

```bash
# put the downloaded file `pretrained.tar.gz` in the root directory of this project.
tar xf pretrained.tar.gz
```

### [Optional] Generate Data By Yourself

You can also generate your own data if you like, by following the instructions below. Note that the data generation code are adopted from [clevr-iep](https://github.com/facebookresearch/clevr-iep) and they require a different license. 

> NOTE: You need another Python enviroment to run data generation code. Check out [clevr-dataset-gen](https://github.com/facebookresearch/clevr-dataset-gen) and [clevr-iep](https://github.com/facebookresearch/clevr-iep) for additional hints.

Generate images and scenes.

```bash
cd data_gen/image_generation
mkdir -p ../data_output
blender --background --python render_images.py -- --num_images 25000 --output_image_dir ../data_output/train/images --output_scene_dir ../data_output/train/scenes --output_scene_file ../data_output/train/scenes.json
blender --background --python render_images.py -- --num_images 10000 --output_image_dir ../data_output/test/images --output_scene_dir ../data_output/test/scenes --output_scene_file ../data_output/test/scenes.json
```

Generate 1-hop questions.

```bash
cd data_gen/question_generation
# Train set. 
python generate_questions.py --instances_per_template 1 --templates_per_image 4 --input_scene_file ../data_output/train/scenes.json --output_questions_file ../data_output/train/questions_1hop.json --template_dir onehop_templates

# Test set.
python generate_questions.py --instances_per_template 1 --templates_per_image 4 --input_scene_file ../data_output/test/scenes.json --output_questions_file ../data_output/test/questions_1hop.json --template_dir onehop_templates
```

Generate 1-hop moformat scene. 

```bash
cd data_gen/question_generation
python question2moformat.py --scene_path ../data_output/test/scenes.json --question_path ../data_output/test/questions_1hop.json --output_scene_dir ../data_output/test/moformat_scenes_1hop
```

Generate 01-hop moformat scene.

```bash
cd data_gen/question_generation
python generate_0hop_question.py
```

Generate 0-hop questions and pack up 0-hop and 1-hop questions. 

```bash
cd data_gen/question_generation

python generate_questions_json.py --input_scene_0hop_dir ../data_output/train/moformat_scenes_0hop --input_question_1hop_file ../data_output/train/questions_1hop.json --output_question_0hop_file ../data_output/train/questions_0hop.json --output_question_01hop_file ../data_output/train/questions_01hop.json
```

Extract image features.

```\
cd data_gen/clevr_iep
CUDA_VISIBLE_DEVICES=3 python scripts/extract_features.py --input_image_dir ../data_output/test/images --output_h5_file ../output/data_output/image_features.h5 --batch_size 64
```

Preprocess questions.

```bash
cd data_gen/clevr_iep
python scripts/preprocess_questions.py --input_questions_json ../data_output/train/questions_01hop.json --output_h5_file ../data_output/train/questions_01hop.h5 --input_vocab_json ../../models/tbd-nets/data/vocab.json

python scripts/preprocess_questions.py --input_questions_json ../data_output/test/questions_01hop.json --output_h5_file ../data_output/test/questions_01hop.h5 --input_vocab_json ../../models/tbd-nets/data/vocab.json
```

## Training

Make sure you have finished the **Deployment** section before continuing. You may modify the YAML files in `code/cfg/` directory to adapt to your needs. 

Basic training command: 

```bash
chmod +x train.sh
./train.sh [SETTING] [GPUID]
```

For example, to train a VQA-GAN with pretrained RNN and TBD-Nets on GPU 0:

```bash
# Maybe modify code/cfg/clevr128_train.yml.
./train.sh vqagan 0
```

To train a VQA-GAN/L1-RNN with pretrained 1-level RNN and TBD-Nets on GPU 1:

```bash
# Maybe modify code/cfg/l1_clevr128_train.yml.
./train.sh vqagan_l1 1
```

[Optional] Train a DAMSM RNN Encoder by yourself:

```bash
# Maybe modify code/cfg/DAMSM/clevr_l2.yml.
./train.sh damsm_l2 0 # 2-level RNN encoder.
# Maybe modify code/cfg/DAMSM/clevr_l1.yml.
./train.sh damsm_l1 0 # 1-level RNN encoder.
```

## Sampling

We provide two sampling methods:

* `normal` : Generate single images for the whole (test, by default) dataset. This is useful for quantitative evaluation, e.g., FID evaluation.
* `visualization`: Generate multiple images of a batch in grid form and make visualizations of attention maps. This is useful for qualitative evaluations. 

The sampling method is set default to `normal`. To modify that, change the `SAMPLING_TYPE` in the config files, e.g. `code/cfg/clevr128_eval.yml`.

Basic training command: 

```bash
chmod +x sample.sh
./sample.sh [SETTING] [GPUID]
```

For example: 

```bash
# Maybe modify code/cfg/clevr128_eval.yml
./sample.sh vqagan 0 # sample VQA-GAN model on GPU 0.
# Maybe modify code/cfg/l1_clevr128_eval.yml
./sample.sh vqagan_l1 1 # sample VQA-GAN/L1 model on GPU 1.
```

The genereted samples are stored in the same directory of model checkpoints, specified in `code/cfg/[l1_]clevr128_eval.yml`.

## Acknowledges

* Code for VQA-GAN is adapted from [Multiple Objects GAN](https://github.com/tohinz/multiple-objects-gan).
* Code for training DAMSM Encoder is adapted from [AttnGAN](https://github.com/taoxugit/AttnGAN).
* Code for training EVQAL is adapted from [TBD-Nets](https://github.com/davidmascharka/tbd-nets).
* Code for generating CLEVR QAs is adapted from [clevr-dataset-gen](https://github.com/facebookresearch/clevr-dataset-gen).
* Code for generating CLEVR programs is adapted from [clevr-iep](https://github.com/facebookresearch/clevr-iep).

## Citing

Please consider citing our paper if you find our work useful:

```latex
@inproceedings{niuvqagan,
author = {Niu, Tianrui and Feng, Fangxiang and Li, Lingxuan and Wang, Xiaojie},
title = {Image Synthesis from Locally Related Texts},
year = {2020},
doi = {10.1145/3372278.3390684},
series = {ICMR '20}
}
```


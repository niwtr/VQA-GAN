from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from clevr_dataset import ClevrDataset
from trainer import condGANTrainer as trainer

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from shutil import copyfile
from miscc.utils import mkdir_p
from miscc.utils import tarball_directory

import torch
import torchvision.transforms as transforms

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird_attn2.yml', type=str)
    # parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    parser.add_argument('--resume', dest='resume', type=str, default='')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def gen_example(wordtoix, algo):
    '''generate images from example sentences'''
    from nltk.tokenize import RegexpTokenizer
    filepath = '%s/example_filenames.txt' % (cfg.DATA_DIR)
    data_dic = {}
    with open(filepath, "r") as f:
        filenames = f.read().decode('utf8').split('\n')
        for name in filenames:
            if len(name) == 0:
                continue
            filepath = '%s/%s.txt' % (cfg.DATA_DIR, name)
            with open(filepath, "r") as f:
                print('Load from:', name)
                sentences = f.read().decode('utf8').split('\n')
                # a list of indices for a sentence
                captions = []
                cap_lens = []
                for sent in sentences:
                    if len(sent) == 0:
                        continue
                    sent = sent.replace("\ufffd\ufffd", " ")
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(sent.lower())
                    if len(tokens) == 0:
                        print('sent', sent)
                        continue

                    rev = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0 and t in wordtoix:
                            rev.append(wordtoix[t])
                    captions.append(rev)
                    cap_lens.append(len(rev))
            max_len = np.max(cap_lens)

            sorted_indices = np.argsort(cap_lens)[::-1]
            cap_lens = np.asarray(cap_lens)
            cap_lens = cap_lens[sorted_indices]
            cap_array = np.zeros((len(captions), max_len), dtype='int64')
            for i in range(len(captions)):
                idx = sorted_indices[i]
                cap = captions[idx]
                c_len = len(cap)
                cap_array[i, :c_len] = cap
            key = name[(name.rfind('/') + 1):]
            data_dic[key] = [cap_array, cap_lens, sorted_indices]
    algo.gen_example(data_dic)


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print('Setting seed to: %d.' % args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    if args.resume == "":
        resume = False
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        output_dir = '../output/%s_%s_%s' % \
            (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
    else:
        assert os.path.isdir(args.resume)
        resume = True
        output_dir = args.resume

    split_dir, bshuffle = 'train', True
    eval = False
    if not cfg.TRAIN.FLAG:
        split_dir = cfg.SAMPLING_SPLIT
        eval = True

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))

    if 'clevr' in cfg.DATASET_NAME:
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize)),
        ])
        dataset = ClevrDataset(cfg.DATA_DIR, split_dir,
                               base_size=cfg.TREE.BASE_SIZE,
                               transform=image_transform,
                               load_programs = cfg.TRAIN.EVQAL.B_EVQAL,
                               )
    else:
        raise RuntimeError("Unsupported dataset name: " + str(cfg.DATASET_NAME))
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))
    dataloader_sampling = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=False, shuffle=False, num_workers=int(cfg.WORKERS))

    # Define models and go to train/evaluate
    algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword, resume)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        if not resume:
            tarball_directory(os.path.join(output_dir, 'code_shapshot.tar.gz'), '.')
        algo.train()
    elif cfg.SAMPLING_TYPE == 'normal':
        algo.sample_full_split(dataloader_sampling, split_dir)
    else:
        assert cfg.SAMPLING_TYPE in ('normal', 'visualization')
        algo.sample_samples(dataloader_sampling, split_dir)

    end_t = time.time()

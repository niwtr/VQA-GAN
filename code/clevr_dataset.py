from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from collections import defaultdict as ddict
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import glob

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
import random as pyrandom
import json
import h5py as h5
from tbd.utils import map_ans
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
    

shape_dict = {
    "cube": 0,
    "cylinder": 1,
    "sphere": 2
}

color_dict  = {
    "gray": 0,
    "red": 1,
    "blue": 2,
    "green": 3,
    "brown": 4,
    "purple": 5,
    "cyan": 6,
    "yellow": 7
}

material_dict = {
    'metal': 0,
    'rubber': 1
}
g_ixtoword = None
g_wordtoix = None

def compute_transformation_matrix(bbox):
    x, y = bbox[:, 0], bbox[:, 1]
    w, h = bbox[:, 2], bbox[:, 3]

    scale_x = w
    scale_y = h

    t_x = 2 * ((x + 0.5 * w) - 0.5)
    t_y = 2 * ((y + 0.5 * h) - 0.5)

    zeros = torch.FloatTensor(bbox.shape[0],1).fill_(0)

    transformation_matrix = torch.cat([scale_x.unsqueeze(-1), zeros, t_x.unsqueeze(-1),
                                       zeros, scale_y.unsqueeze(-1), t_y.unsqueeze(-1)], 1).view(-1, 2, 3)

    return transformation_matrix


def compute_transformation_matrix_inverse(bbox):
    x, y = bbox[:, 0], bbox[:, 1]
    w, h = bbox[:, 2], bbox[:, 3]

    scale_x = 1.0 / w
    scale_y = 1.0 / h

    t_x = 2 * scale_x * (0.5 - (x + 0.5 * w))
    t_y = 2 * scale_y * (0.5 - (y + 0.5 * h))

    zeros = torch.FloatTensor(bbox.shape[0],1).fill_(0)

    transformation_matrix = torch.cat([scale_x.unsqueeze(-1), zeros, t_x.unsqueeze(-1),
                                       zeros, scale_y.unsqueeze(-1), t_y.unsqueeze(-1)], 1).view(-1, 2, 3)

    return transformation_matrix

def prepare_data(data):
    imgs, captions, captions_lens, class_ids, bbox, label, trans_matrix, keys, prog = data

    class_ids = None
    caption_nums = (captions_lens > 0).sum(1)
    # sort QAs by number of QA in a decreasing order
    sorted_cap_nums, sorted_cap_indices = torch.sort(caption_nums, 0, True)
    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    if 'programs' in prog.keys():
        prog['programs'] = prog['programs'][sorted_cap_indices].cuda()
        prog['answers'] = map_ans(prog['answers'])[sorted_cap_indices].cuda()

    captions = captions[sorted_cap_indices].squeeze()
    sorted_cap_lens = captions_lens[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    label = label[sorted_cap_indices]
    trans_matrix[0] = trans_matrix[0][sorted_cap_indices]
    trans_matrix[1] = trans_matrix[1][sorted_cap_indices]
    bbox = bbox[sorted_cap_indices]

    # for ii in range(16):
    #     captions[ii][16:] = 0
    #     sorted_cap_lens[ii][16:] = 0
    # chosen_ix = 4
    # newix = 16
    # for nth, i in enumerate(captions[chosen_ix]):
    #     print(nth, ' '.join([g_ixtoword[j.tolist()] for j in i]))
    # print(sorted_cap_lens[chosen_ix])
    # exit(0)
    # captions[chosen_ix][newix:] = 0
    # sorted_cap_lens[chosen_ix][newix:] = 0
    # new_qas = [
    #     'there is a block in front of the tiny yellow metal thing what is its color brown',
    #     'there is a block in front of the tiny yellow metal thing what is its color brown',
    #     # 'the cylinder that is on the right side of the big brown metallic cylinder is what color red'
    # ]
    # for newqa in new_qas:
    #     newqa = [g_wordtoix[w] for w in newqa.split(' ')]
    #     newqa_len = len(newqa)
    #     newqa = newqa + [0] * (25 - newqa_len)
    #     newqa = torch.LongTensor(newqa)
    #     captions[chosen_ix][newix] = newqa
    #     sorted_cap_lens[chosen_ix][newix] = newqa_len
    #     newix += 1

    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)
    return [real_imgs, 
            captions, sorted_cap_lens,
            class_ids,
            bbox, 
            label,
            trans_matrix,
            keys,
            prog]

def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size

    if transform is not None:
        img = transform(img)

    ret = []
    for i in range(cfg.TREE.BRANCH_NUM):
        # print(imsize[i])
        if i < (cfg.TREE.BRANCH_NUM - 1):
            re_img = transforms.Resize(imsize[i])(img)
        else:
            re_img = img
        ret.append(normalize(re_img))
    return ret


class ClevrDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 base_size=64, # 299
                 transform=None, target_transform=None, load_programs = False):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        self.max_objects = 4
        if cfg.DATASET_NAME == 'clevr': 
            split_dir = self.split_dir = os.path.join(data_dir, split + '_512')
        elif cfg.DATASET_NAME == 'clevr128':
            split_dir = self.split_dir = os.path.join(data_dir, split)

        self.img_dir = os.path.join(self.split_dir, "images")
        self.scene_dir = os.path.join(self.split_dir, cfg.SCENE_DIR_NAME)
        self.filenames = self.load_filenames()
        self.load_qa_data(os.path.join(cfg.QA_DIR, split, 'questions_answers.json'))
        self.number_example = len(self.filenames)
        self.programp = load_programs
        if load_programs:
            self.load_programs()

    def get_transformation_matrices(self, bbox):
        bbox = torch.from_numpy(bbox)
        bbox = bbox.view(-1, 4)
        transf_matrices_inv = compute_transformation_matrix_inverse(bbox)
        transf_matrices_inv = transf_matrices_inv.view(self.max_objects, 2, 3)
        transf_matrices = compute_transformation_matrix(bbox)
        transf_matrices = transf_matrices.view(self.max_objects, 2, 3)
        return transf_matrices, transf_matrices_inv
      
    def label_one_hot(self, label, dim):
        labels = torch.from_numpy(label)
        labels = labels.long()
        # remove -1 to enable one-hot converting
        labels[labels < 0] = dim-1
        label_one_hot = torch.FloatTensor(labels.shape[0], dim).fill_(0)
        label_one_hot = label_one_hot.scatter_(1, labels, 1).float()
        return label_one_hot
    
    def load_programs(self):
        print('Loading programs ...')
        f = h5.File(os.path.join(self.split_dir, cfg.TRAIN.EVQAL.PROGRAM_FILE_NAME), 'r')
        # f = h5.File(os.path.join(cfg.TRAIN.EVQAL.PROGRAM_FILE_NAME), 'r')
        programs = f['programs']
        answers = f['answers']
        questions = f['questions']
        idx = f['image_idxs']
        iid2pid = ddict(list)
        for pix, iix in enumerate(idx):
            iid2pid[iix].append(pix)
        self.programs_index_table = iid2pid
        self.programs = torch.LongTensor(np.asarray(programs))
        self.program_questions = torch.LongTensor(np.asarray(questions))
        self.program_answers = torch.LongTensor(np.asarray(answers))
        print('Loading OK.')
        return self

    # def metamorph(self, morph_type = 'standard'):
    #     assert morph_type in ('standard', 'extract', 'extract_image')
    #     self.meta = morph_type
    #     return self
        
    def load_qa_data(self, qa_data_path):
        global g_ixtoword, g_wordtoix
        all_qas = json.load(open(qa_data_path, 'r'))
        all_iixs = all_qas['image_indices']
        self.image_indices = all_iixs
        all_qas_raw = all_qas['raw_QAs']
        all_qa_labels = all_qas['QAs']
        
        # build image_index to index lookup table 
        iix2inds = ddict(list)
        for ind, ix in enumerate(all_iixs):
            iix2inds[ix].append(ind)
        self.imgid2qaids = iix2inds
        self.QA_labels = all_qa_labels
        self.QA_raw = all_qas_raw
        vocab = ['<end>'] + all_qas['vocab']
        self.QA_vocab = vocab
        self.n_words = len(self.QA_vocab)
        self.ixtoword = { ix: wd for (ix, wd) in enumerate(vocab) }
        self.wordtoix = { wd: ix for (ix, wd) in enumerate(vocab) }
        assert self.ixtoword[0] == '<end>'
        assert self.wordtoix['<end>'] == 0

        g_ixtoword = self.ixtoword
        g_wordtoix = self.wordtoix
        return self


    ''' load filenames of a certain JSON file. '''
    def load_filenames(self):
        filenames = [filename for filename in glob.glob(self.scene_dir + '/*.json')]
        print('Load scenes from: %s (%d)' % (self.scene_dir, len(filenames)))
        return filenames

    
    def get_QA(self, qaix):
        # we do not need <end> token.
        qa = np.asarray(self.QA_labels[qaix]).astype('int64')
        if (qa == 0).sum() > 0:
            print('Error: do not need END (0) token', qa)
        nwords = len(qa)
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = nwords
        if nwords <= cfg.TEXT.WORDS_NUM:
            x[:nwords, 0] = qa
        else:
            raise NotImplementedError('This operation is not implemented!')
            ix = list(np.arange(nwords))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = qa[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len
    
    def pad_zeros_batch(self, x, ndim = 10):
        if ndim < x.size(0):
            raise RuntimeError('Ndim must > x.size(0).')
        elif ndim == x.size(0):
            return x
        else:
            sz = list(x.size())
            sz[0] = ndim - sz[0]
            sz = tuple(sz)
            pad = torch.zeros(sz, dtype=x.dtype)
            return torch.cat((x, pad), 0)
        
    def __getitem__main(self, index):
        key = self.filenames[index]
        data_dir = self.data_dir
        anno = json.load(open(key, 'r'))
        img_name = self.img_dir + '/' + anno['image_filename']
        image_ix = anno["image_index"]
        
        qa_ixs = self.imgid2qaids[image_ix]
        if qa_ixs == [ ]:
            raise NotImplementedError('this feature is currently not implemented!')
            return self.__getitem__(index + 1)
        

        qas, qa_lens = [], []
        for qa_ix in qa_ixs:
            _qa, _qa_len = self.get_QA(qa_ix)
            qas.append(torch.LongTensor(_qa).view(1, -1))
            qa_lens.append(_qa_len)
            
        qas = self.pad_zeros_batch(torch.cat(qas, 0), cfg.TEXT.MAX_QA_NUM)
        qa_lens = self.pad_zeros_batch(torch.LongTensor(qa_lens), cfg.TEXT.MAX_QA_NUM)
        nqa = len(qa_ixs)
        imgs = get_imgs(img_name, self.imsize,
                        None, self.transform, normalize=self.norm)

        # load bbox #
        bbox = np.zeros((self.max_objects, 4), dtype=np.float32)
        bbox[:] = -1.0
        for idx in range(len(anno["objects"])):
            bbox[idx, :] = anno["objects"][idx]["bbox"]

        if cfg.DATASET_NAME == 'clevr':
            basesize = 64
        elif cfg.DATASET_NAME == 'clevr128':
            basesize = 128
        else:
            raise NotImplementedError('You know it.')
        bbox = bbox / float(basesize)
        tmatrix = self.get_transformation_matrices(bbox)

        # load labels #
        label_shape = np.zeros(self.max_objects)
        label_shape[:] = -1
        label_color = np.zeros(self.max_objects)
        label_color[:] = -1
        label_material = np.zeros(self.max_objects)
        label_material[:] = -1
        for idx in range(len(anno["objects"])):
            label_shape[idx] = shape_dict[anno["objects"][idx]["shape"]]
            label_color[idx] = color_dict[anno["objects"][idx]["color"]]
            label_material[idx] = material_dict[anno["objects"][idx]["material"]]
      
        label_shape = self.label_one_hot(np.expand_dims(label_shape, 1), 4)
        label_color = self.label_one_hot(np.expand_dims(label_color, 1), 9)
        label_material= self.label_one_hot(np.expand_dims(label_material, 1), 3)
        # label = torch.cat((label_shape, label_color, label_material), 1)
        label = label_shape

        if self.programp:
            pixs = self.programs_index_table[image_ix]
            if pixs == [ ]:
                raise RuntimeError('Panic!')
                return self.__getitem__(index + 1)
            pid = pyrandom.sample(pixs, 1)
            prog = self.programs[pid].squeeze()
            prog_asr = self.program_answers[pid].squeeze()
            prog_ret = {
                'programs': prog, 
                'answers': prog_asr
            }
       
        if self.programp: 
            return imgs, qas, qa_lens, [], bbox, label, tmatrix, key, prog_ret
        else:
            return imgs, qas, qa_lens, [], bbox, label, tmatrix, key, {}
        

    def __getitem__(self, index):
        return self.__getitem__main(index)
    def __len__(self):
        return len(self.filenames)


    # def __len__main(self):
    #     return len(self.filenames)
    # def __len__extract(self):
    #     return len(self.QA_labels)
    # def __len__extract_image(self):
    #     return len(self.filenames)
    # def __getitem__extract(self, index):
    #     qa_ix = index 
    #     qa, qa_len = self.get_QA(qa_ix)
    #     return [], qa, qa_len, [], []

    # def __getitem__extract_image(self, index):
    #     key = self.filenames[index]
    #     data_dir = self.data_dir
    #     anno = json.load(open(key, 'r'))
    #     img_name = self.img_dir + '/' + anno['image_filename']
    #     imgs = get_imgs(img_name, self.imsize,
    #                     None, self.transform, normalize=self.norm)
    #     return imgs, [], [], [], key
    
    # def __getitem__(self, index):
    #     return {
    #         'standard' : self.__getitem__main,
    #         'extract': self.__getitem__extract,
    #         'extract_image': self.__getitem__extract_image
    #     }[self.meta](index)
    
    # def __len__(self):
    #     return {
    #         'standard' : self.__len__main,
    #         'extract': self.__len__extract,
    #         'extract_image': self.__len__extract_image,
    #     }[self.meta]()



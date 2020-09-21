from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from PIL import Image

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2, render_attn_to_html
from miscc.utils import weights_init, load_params, copy_G_params
from miscc.utils import save_pure_img_results, save_batch_images
from miscc.utils import AverageMeter
from model import G_DCGAN, G_NET
from model import RNN_ENCODER, CNN_ENCODER, HigherLevelRNN, Level2RNNEncodeMagic
from clevr_dataset import prepare_data as prepare_data_clevr
import itertools
from miscc.losses import words_loss
from miscc.losses import discriminator_loss, generator_loss, KL_loss, VQA_loss
import os
import time
import numpy as np
import sys
import glob

# -- tbd-nets
from tbd.resnet_encoder import load_resnet_image_encoder, extract_image_feats
from tbd.module_net import load_tbd_net as load_vqa_net
from tbd.utils import load_vocab as load_program_vocab
from tbd.utils import map_ans


def make_fake_captions(num_caps):
    batch_size = num_caps.size(0)
    caps = torch.zeros(batch_size, cfg.TEXT.MAX_QA_NUM, dtype = torch.int64)
    ref = torch.arange(0, cfg.TEXT.MAX_QA_NUM).view(1, -1).repeat(batch_size, 1).cuda()
    targ = num_caps.view(-1, 1).repeat(1, cfg.TEXT.MAX_QA_NUM)
    caps[ref < targ] = 1
    return caps, {1: 'DUMMY'}

# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, n_words, ixtoword, resume):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.resume = resume

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)
        self.prepare_data = prepare_data_clevr

        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

    def build_models(self):
        # ###################encoders######################################## #
        if cfg.TRAIN.NET_E == '':
            print('Error: no pretrained text-image encoders')
            return

        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()

        text_encoder = HigherLevelRNN(ninput = cfg.TEXT.EMBEDDING_DIM, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(cfg.TRAIN.NET_E, map_location= lambda s,l:s)
        text_encoder.load_state_dict(state_dict)

        text_encoder_L = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        L_path = cfg.TRAIN.NET_E.replace('text_encoder', 'text_encoder_L')
        state_dict = torch.load(L_path, map_location=lambda s, l: s)
        text_encoder_L.load_state_dict(state_dict)
        for p in  itertools.chain(text_encoder.parameters(),\
                                  text_encoder_L.parameters()):
            p.requires_grad = False
        print('Loaded text encoder: %s' % cfg.TRAIN.NET_E)
        print('Loaded low level text encoder: %s' % L_path)
        text_encoder.eval()
        text_encoder_L.eval()

        # #######################generator and discriminators############## #
        netsD = []
        from model import D_NET64, D_NET128, D_NET256
        netG = G_NET()
        if cfg.TREE.BRANCH_NUM > 0:
            netsD.append(D_NET64())
        if cfg.TREE.BRANCH_NUM > 1:
            netsD.append(D_NET128())
        if cfg.TREE.BRANCH_NUM > 2:
            netsD.append(D_NET256())

        netG.apply(weights_init)
        # print(netG)
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
            # print(netsD[i])
        print('# of netsD', len(netsD))
        epoch = 0

        if self.resume:
            checkpoint_list = sorted([ckpt for ckpt in glob.glob(self.model_dir + "/" + '*.pth')])
            latest_checkpoint = checkpoint_list[-1]
            state_dict = torch.load(latest_checkpoint, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict["netG"])
            for i in range(len(netsD)):
                netsD[i].load_state_dict(state_dict["netD"][i])
            epoch = int(latest_checkpoint[-8:-4]) + 1
            print("Resuming training from checkpoint {} at epoch {}.".format(latest_checkpoint, epoch))

        #
        if cfg.TRAIN.NET_G != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            epoch = state_dict['epoch'] + 1
            netG.load_state_dict(state_dict['netG'])
            for i in range(len(netsD)):
                netsD[i].load_state_dict(state_dict['netD'][i])

            # netG.load_state_dict(state_dict)
            # print('Load G from: ', cfg.TRAIN.NET_G)
            # istart = cfg.TRAIN.NET_G.rfind('_') + 1
            # iend = cfg.TRAIN.NET_G.rfind('.')
            # epoch = cfg.TRAIN.NET_G[istart:iend]
            # epoch = int(epoch) + 1
            # if cfg.TRAIN.B_NET_D:
            #     Gname = cfg.TRAIN.NET_G
            #     for i in range(len(netsD)):
            #         s_tmp = Gname[:Gname.rfind('/')]
            #         Dname = '%s/netD%d.pth' % (s_tmp, i)
            #         print('Load D from: ', Dname)
            #         state_dict = \
            #             torch.load(Dname, map_location=lambda storage, loc: storage)
            #         netsD[i].load_state_dict(state_dict)
        # ########################################################### #
        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            text_encoder_L = text_encoder_L.cuda()
            image_encoder = image_encoder.cuda()
            netG.cuda()
            for i in range(len(netsD)):
                netsD[i].cuda()
        return [(text_encoder, text_encoder_L), image_encoder, netG, netsD, epoch]

    def define_optimizers(self, netG, netsD):
        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(netsD[i].parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))

        if self.resume:
            checkpoint_list = sorted([ckpt for ckpt in glob.glob(self.model_dir + "/" + '*.pth')])
            latest_checkpoint = checkpoint_list[-1]
            state_dict = torch.load(latest_checkpoint, map_location=lambda storage, loc: storage)
            optimizerG.load_state_dict(state_dict["optimG"])

            for i in range(len(netsD)):
                optimizersD[i].load_state_dict(state_dict["optimD"][i])

        return optimizerG, optimizersD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, netsD, optimG, optimsD, epoch, max_to_keep=5):
        netDs_state_dicts = []
        optimDs_state_dicts = []
        for i in range(len(netsD)):
            netD = netsD[i]
            optimD = optimsD[i]
            netDs_state_dicts.append(netD.state_dict())
            optimDs_state_dicts.append(optimD.state_dict())

        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        checkpoint = {
            'epoch': epoch,
            'netG': netG.state_dict(),
            'optimG': optimG.state_dict(),
            'netD': netDs_state_dicts,
            'optimD': optimDs_state_dicts}
        torch.save(checkpoint, "{}/checkpoint_{:04}.pth".format(self.model_dir, epoch))
        print('Save G/D models')

        load_params(netG, backup_para)

        if max_to_keep is not None and max_to_keep > 0:
            checkpoint_list = sorted([ckpt for ckpt in glob.glob(self.model_dir + "/" + '*.pth')])
            while len(checkpoint_list) > max_to_keep:
                os.remove(checkpoint_list[0])
                checkpoint_list = checkpoint_list[1:]

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    def save_img_samples(self, real_img, netG, noise, sent_emb, words_embs, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations, transf_matrices_inv, label_one_hot, name='current', num_visualize = 8):
       
        qa_nums = (cap_lens > 0).sum(1)
        real_captions = captions
        captions, _ = make_fake_captions(qa_nums) # fake caption.

        # Save images
        # fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
        inputs = (noise, sent_emb, words_embs, mask, transf_matrices_inv, label_one_hot)
        fake_imgs, attention_maps, _, _ = nn.parallel.data_parallel(netG, inputs, self.gpus)

        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)

            html_doc = render_attn_to_html([real_img[i+1].detach().cpu(), lr_img, img],
                                           real_captions, self.ixtoword,
                                           attn_maps, att_sze, None,
                                           info=['Real Images', 'LR Fake Images', 'Fake Images'],
                                           nvis=64, v_sigma=8)

            with open('%s/G_attn_%d_l%d.html' % (self.image_dir, gen_iterations, i), 'w') as html_f:
                html_f.write(str(html_doc))

        for i in range(cfg.TREE.BRANCH_NUM):
            save_pure_img_results(real_img[i].detach().cpu(),
                                  fake_imgs[i].detach().cpu(),
                                  gen_iterations, self.image_dir,
                                  token='level%d' % i)

        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, qa_nums,
                                    None, self.batch_size)
        # FIXME currently the `render_attn_to_html` supports only the last level.
        # please implement multiple level rendering.
        html_doc = render_attn_to_html([real_img[i].detach().cpu(),
                                        fake_imgs[i].detach().cpu(),],
                                       real_captions, self.ixtoword,
                                       att_maps, att_sze, None,
                            info=['Real Images', 'Fake Images'])
        with open('%s/damsm_attn_%d.html' % (self.image_dir, gen_iterations), 'w') as html_f:
            html_f.write(str(html_doc))





    def save_img_results(self, real_img, netG, noise, sent_emb, words_embs, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations, transf_matrices_inv, label_one_hot, name='current', num_visualize = 8):
       
        qa_nums = (cap_lens > 0).sum(1)
        real_captions = captions
        captions, _ = make_fake_captions(qa_nums) # fake caption.

        # Save images
        # fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
        inputs = (noise, sent_emb, words_embs, mask, transf_matrices_inv, label_one_hot)
        fake_imgs, attention_maps, _, _ = nn.parallel.data_parallel(netG, inputs, self.gpus)
        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = \
                build_super_images(img, captions, self.ixtoword,
                                   attn_maps, att_sze, lr_imgs=lr_img, nvis = num_visualize)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png'\
                    % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        for i in range(cfg.TREE.BRANCH_NUM):
            save_pure_img_results(real_img[i].detach().cpu(),
                                  fake_imgs[i].detach().cpu(),
                                  gen_iterations, self.image_dir,
                                  token='level%d' % i)

        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, qa_nums,
                                    None, self.batch_size)
        img_set, _ = build_super_images(fake_imgs[i].detach().cpu(),
                                        captions, self.ixtoword, att_maps, att_sze, nvis = num_visualize)
        # FIXME currently the `render_attn_to_html` supports only the last level.
        # please implement multiple level rendering.
        html_doc = render_attn_to_html([real_img[i].detach().cpu(),
                                        fake_imgs[i].detach().cpu(),],
                                       real_captions, self.ixtoword,
                                       att_maps, att_sze, None,
                            info=['Real Images', 'Fake Images'])
        with open('%s/damsm_attn_%d.html' % (self.image_dir, gen_iterations), 'w') as html_f:
            html_f.write(str(html_doc))

        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png'\
                % (self.image_dir, name, gen_iterations)
            im.save(fullpath)

    def train(self):
        text_encoder, image_encoder, netG, netsD, start_epoch = self.build_models()
        H_rnn_model, L_rnn_model = text_encoder
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD)

        if cfg.TRAIN.EVQAL.B_EVQAL:
            netVQA_E = load_resnet_image_encoder(model_stage = 2)
            netVQA = load_vqa_net(
                cfg.TRAIN.EVQAL.NET,
                load_program_vocab(cfg.TRAIN.EVQAL.PROGRAM_VOCAB_FILE),
                feature_dim=(512, 28, 28)
            )
        else:
            netVQA_E = netVQA = None

        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            am_vqa_loss = AverageMeter('VQA Loss')
            data_iter = iter(self.data_loader)
            step = 0
            while step < self.num_batches:
                # reset requires_grad to be trainable for all Ds
                # self.set_requires_grad_value(netsD, True)

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()
                imgs, captions, cap_lens, class_ids, bbox, label_one_hot, transformation_matrices, keys, prog = self.prepare_data(data)
                class_ids = None
                batch_size = captions.size(0)

                transf_matrices = transformation_matrices[0].detach()
                transf_matrices_inv = transformation_matrices[1].detach()
                
                per_qa_embs, avg_qa_embs, qa_nums =\
                    Level2RNNEncodeMagic(captions, cap_lens, L_rnn_model, H_rnn_model)
                per_qa_embs, avg_qa_embs = (per_qa_embs.detach(), avg_qa_embs.detach())

                _nmaxqa = cfg.TEXT.MAX_QA_NUM
                mask = torch.ones(batch_size, _nmaxqa, dtype = torch.uint8).cuda()
                _ref = torch.arange(0, _nmaxqa).view(1, -1).repeat(batch_size, 1).cuda()
                _targ = qa_nums.view(-1, 1).repeat(1, _nmaxqa)
                mask[_ref < _targ] = 0
                num_words = per_qa_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                inputs = (noise, avg_qa_embs, per_qa_embs, mask, transf_matrices_inv, label_one_hot)
                fake_imgs, _, mu, logvar = nn.parallel.data_parallel(netG, inputs, self.gpus)

                #######################################################
                # (3) Update D network
                ######################################################
                errD_total = 0
                D_logs = ''
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    if i == 0: # NOTE only the first level Discriminator is modified.
                        errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                                  avg_qa_embs, real_labels, fake_labels, self.gpus,
                                                  local_labels=label_one_hot, transf_matrices=transf_matrices,
                                                  transf_matrices_inv=transf_matrices_inv)
                    else:
                        errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                                  avg_qa_embs, real_labels, fake_labels, self.gpus)

                    # backward and update parameters
                    errD.backward()
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs += 'errD%d: %.2f ' % (i, errD.item())

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                step += 1
                gen_iterations += 1

                # do not need to compute gradient for Ds
                # self.set_requires_grad_value(netsD, False)
                netG.zero_grad()
                errG_total, G_logs = \
                    generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                                   per_qa_embs, avg_qa_embs, match_labels, qa_nums, class_ids, self.gpus,
                                   local_labels=label_one_hot, transf_matrices=transf_matrices,
                                   transf_matrices_inv=transf_matrices_inv)
                
                if cfg.GAN.B_CA_NET:
                    kl_loss = KL_loss(mu, logvar)
                else:
                    kl_loss = torch.FloatTensor([0.]).squeeze().cuda()

                errG_total += kl_loss
                G_logs += 'kl_loss: %.2f ' % kl_loss.item()
 
                if cfg.TRAIN.EVQAL.B_EVQAL:
                    fake_img_fvqa = extract_image_feats(fake_imgs[-1], netVQA_E, self.gpus)
                    errVQA = VQA_loss(netVQA, fake_img_fvqa,
                                      prog['programs'], 
                                      prog['answers'],
                                      self.gpus)
                else:
                    errVQA = torch.FloatTensor([0.]).squeeze().cuda()
                G_logs += 'VQA_loss: %.2f ' % errVQA.data.item()
                beta = cfg.TRAIN.EVQAL.BETA
                errG_total += (errVQA * beta)

                # backward and update parameters
                errG_total.backward()
                optimizerG.step()
                
                am_vqa_loss.update(errVQA.cpu().item())
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                # save images
                if gen_iterations % 100 == 0:
                    print(D_logs + '\n' + G_logs)
                if gen_iterations % 500 == 0: # FIXME original: 1000
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    self.save_img_results(imgs, netG, fixed_noise, avg_qa_embs,
                                          per_qa_embs, mask, image_encoder,
                                          captions, cap_lens, epoch,  transf_matrices_inv,
                                          label_one_hot, name='average')
                    load_params(netG, backup_para)
            end_t = time.time()

            print('''[%d/%d][%d]
                  Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.item(), errG_total.item(),
                     end_t - start_t))
            if cfg.TRAIN.EVQAL.B_EVQAL:
                print('Avg. VQA Loss of this epoch: %s' % str(am_vqa_loss))
            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netsD, optimizerG, optimizersD, epoch)

        self.save_model(netG, avg_param_G, netsD, optimizerG, optimizersD, epoch)

    def save_singleimages(self, images, filenames, save_dir,
                          split_dir, sentenceID=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
            # range from [-1, 1] to [0, 1]
            # img = (images[i] + 1.0) / 2
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)
         
        
    def sample_samples(self, data_loader, split = 'train'):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for models is not found!')
        else:
            if split == 'test':
                split = 'valid'
        assert split != 'test'
        
        text_encoder, image_encoder, netG, netsD, start_epoch = self.build_models()
        H_rnn_model, L_rnn_model = text_encoder
        H_rnn_model.eval();L_rnn_model.eval()
        image_encoder.eval()
        netG.eval(); del netsD
        save_dir = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.find('.pth')] + "_samples_" + split + '_quality_analysis'
        print("saving to:", save_dir)
        mkdir_p(save_dir)

        nz = cfg.GAN.Z_DIM
        batch_size = self.batch_size
        noise = torch.FloatTensor(batch_size, nz)
        if cfg.CUDA:
            noise = noise.cuda()

        imsize = 64
        nsamples = 10

        for i, data in enumerate(data_loader, 0):
            with torch.no_grad():
                imgs, captions, cap_lens, class_ids, bbox, label_one_hot, transformation_matrices, keys, progs = self.prepare_data(data)
                keys = map(lambda x:os.path.basename(x[:-5])+'.png', keys)
                class_ids = None
                batch_size = captions.size(0)
                transf_matrices = transformation_matrices[0]
                transf_matrices_inv = transformation_matrices[1]

                per_qa_embs, avg_qa_embs, qa_nums =\
                    Level2RNNEncodeMagic(captions, cap_lens, L_rnn_model, H_rnn_model)
                per_qa_embs, avg_qa_embs = (per_qa_embs.detach(), avg_qa_embs.detach())

                _nmaxqa = cfg.TEXT.MAX_QA_NUM
                mask = torch.ones(batch_size, _nmaxqa, dtype = torch.uint8).cuda()
                _ref = torch.arange(0, _nmaxqa).view(1, -1).repeat(batch_size, 1).cuda()
                _targ = qa_nums.view(-1, 1).repeat(1, _nmaxqa)
                mask[_ref < _targ] = 0
                num_words = per_qa_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                # noise_feed = noise[:batch_size]
                # inputs = (noise_feed, avg_qa_embs, per_qa_embs, mask, transf_matrices_inv, label_one_hot)
                # fake_imgs, _, mu, logvar = nn.parallel.data_parallel(netG, inputs, self.gpus)
                self.image_dir = os.path.join(save_dir, 'sample%d' % i)
                mkdir_p(self.image_dir)
                self.save_img_samples(imgs, netG, noise, avg_qa_embs,
                                        per_qa_embs, mask, image_encoder,
                                        captions, cap_lens, 999,  transf_matrices_inv,
                                        label_one_hot, name='sample', num_visualize= 64)
                if i > nsamples:
                    exit(0)
                else:
                    print(i)
        
    def sample_full_split(self, data_loader, split = 'train'):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for models is not found!')
        else:
            if split == 'test':
                split = 'valid'
        assert split != 'test'
        
        text_encoder, image_encoder, netG, netsD, start_epoch = self.build_models()
        H_rnn_model, L_rnn_model = text_encoder
        H_rnn_model.eval();L_rnn_model.eval()
        image_encoder.eval()
        netG.eval(); del netsD


        extract_level = 1
        save_dir = cfg.TRAIN.NET_G[: cfg.TRAIN.NET_G.find('.pth')] + "_samples_" + split + '_level%d' % extract_level
        print("saving to:", save_dir)
        mkdir_p(save_dir)

        nz = cfg.GAN.Z_DIM
        batch_size = self.batch_size
        noise = torch.FloatTensor(batch_size, nz)
        if cfg.CUDA:
            noise = noise.cuda()

        imsize = 64
        for i, data in enumerate(data_loader, 0):
            with torch.no_grad():
                imgs, captions, cap_lens, class_ids, bbox, label_one_hot, transformation_matrices, keys, progs = self.prepare_data(data)
                keys = map(lambda x:os.path.basename(x[:-5])+'.png', keys)
                class_ids = None
                batch_size = captions.size(0)
                transf_matrices = transformation_matrices[0]
                transf_matrices_inv = transformation_matrices[1]

                per_qa_embs, avg_qa_embs, qa_nums =\
                    Level2RNNEncodeMagic(captions, cap_lens, L_rnn_model, H_rnn_model)
                per_qa_embs, avg_qa_embs = (per_qa_embs.detach(), avg_qa_embs.detach())

                _nmaxqa = cfg.TEXT.MAX_QA_NUM
                mask = torch.ones(batch_size, _nmaxqa, dtype = torch.uint8).cuda()
                _ref = torch.arange(0, _nmaxqa).view(1, -1).repeat(batch_size, 1).cuda()
                _targ = qa_nums.view(-1, 1).repeat(1, _nmaxqa)
                mask[_ref < _targ] = 0
                num_words = per_qa_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                noise_feed = noise[:batch_size]
                inputs = (noise_feed, avg_qa_embs, per_qa_embs, mask, transf_matrices_inv, label_one_hot)
                fake_imgs, _, mu, logvar = nn.parallel.data_parallel(netG, inputs, self.gpus)
                save_batch_images(fake_imgs[extract_level], keys, save_dir)



    def sampling(self, split_dir, num_samples=30000):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for models is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            # Build and load the generator
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            netG.apply(weights_init)
            netG.cuda()
            netG.eval()
            #
            text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            batch_size = self.batch_size
            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, nz))
            noise = noise.cuda()

            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            netG.load_state_dict(state_dict["netG"])
            print('Load G from: ', model_dir)

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfind('.pth')]
            save_dir = '%s/%s' % (s_tmp, split_dir)
            mkdir_p(save_dir)

            cnt = 0

            for _ in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
                for step, data in enumerate(self.data_loader, 0):
                    cnt += batch_size
                    if step % 10000 == 0:
                        print('step: ', step)
                    if step >= num_samples:
                        break

                    imgs, captions, cap_lens, class_ids, keys, transformation_matrices, label_one_hot = self.prepare_data(data)
                    transf_matrices_inv = transformation_matrices[1]

                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                    mask = (captions == 0)
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    inputs = (noise, sent_emb, words_embs, mask, transf_matrices_inv, label_one_hot)
                    with torch.no_grad():
                        fake_imgs, _, mu, logvar = nn.parallel.data_parallel(netG, inputs, self.gpus)
                    for j in range(batch_size):
                        s_tmp = '%s/single/%s' % (save_dir, keys[j])
                        folder = s_tmp[:s_tmp.rfind('/')]
                        if not os.path.isdir(folder):
                            print('Make a new folder: ', folder)
                            mkdir_p(folder)
                        k = -1
                        # for k in range(len(fake_imgs)):
                        im = fake_imgs[k][j].data.cpu().numpy()
                        # [-1, 1] --> [0, 255]
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        im = np.transpose(im, (1, 2, 0))
                        im = Image.fromarray(im)
                        fullpath = '%s_s%d.png' % (s_tmp, k)
                        im.save(fullpath)

    def sample(self, split_dir, num_samples=25, draw_bbox=False):
        from PIL import Image, ImageDraw, ImageFont
        import cPickle as pickle
        import torchvision
        import torchvision.utils as vutils

        if cfg.TRAIN.NET_G == '':
            print('Error: the path for model NET_G is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            # Build and load the generator
            text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            batch_size = cfg.TRAIN.BATCH_SIZE
            nz = cfg.GAN.Z_DIM

            model_dir = cfg.TRAIN.NET_G
            state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            netG = G_NET()
            print('Load G from: ', model_dir)
            netG.apply(weights_init)

            netG.load_state_dict(state_dict["netG"])
            netG.cuda()
            netG.eval()

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfind('.pth')]
            save_dir = '%s_%s' % (s_tmp, split_dir)
            mkdir_p(save_dir)
            #######################################
            noise = Variable(torch.FloatTensor(9, nz))

            imsize = 256

            for step, data in enumerate(self.data_loader, 0):
                if step >= num_samples:
                    break

                imgs, captions, cap_lens, class_ids, keys, transformation_matrices, label_one_hot, bbox = \
                    self.prepare_data(data, eval=True)
                transf_matrices_inv = transformation_matrices[1][0].unsqueeze(0)
                label_one_hot = label_one_hot[0].unsqueeze(0)

                img = imgs[-1][0]
                val_image = img.view(1, 3, imsize, imsize)

                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs[0].unsqueeze(0).detach(), sent_emb[0].unsqueeze(0).detach()
                words_embs = words_embs.repeat(9, 1, 1)
                sent_emb = sent_emb.repeat(9, 1)
                mask = (captions == 0)
                mask = mask[0].unsqueeze(0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]
                mask = mask.repeat(9, 1)
                transf_matrices_inv = transf_matrices_inv.repeat(9, 1, 1, 1)
                label_one_hot = label_one_hot.repeat(9, 1, 1)

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                inputs = (noise, sent_emb, words_embs, mask, transf_matrices_inv, label_one_hot)
                with torch.no_grad():
                    fake_imgs, _, mu, logvar = nn.parallel.data_parallel(netG, inputs, self.gpus)

                data_img = torch.FloatTensor(10, 3, imsize, imsize).fill_(0)
                data_img[0] = val_image
                data_img[1:10] = fake_imgs[-1]

                if draw_bbox:
                    for idx in range(3):
                        x, y, w, h = tuple([int(imsize*x) for x in bbox[0, idx]])
                        w = imsize-1 if w > imsize-1 else w
                        h = imsize-1 if h > imsize-1 else h
                        if x <= -1:
                            break
                        data_img[:10, :, y, x:x + w] = 1
                        data_img[:10, :, y:y + h, x] = 1
                        data_img[:10, :, y+h, x:x + w] = 1
                        data_img[:10, :, y:y + h, x + w] = 1

                # get caption
                cap = captions[0].data.cpu().numpy()
                sentence = ""
                for j in range(len(cap)):
                    if cap[j] == 0:
                        break
                    word = self.ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')
                    sentence += word + " "
                sentence = sentence[:-1]
                vutils.save_image(data_img, '{}/{}_{}.png'.format(save_dir, sentence, step), normalize=True, nrow=10)
            print("Saved {} files to {}".format(step, save_dir))

    def gen_example(self, data_dic):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for models is not found!')
        else:
            # Build and load the generator
            text_encoder = \
                RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            # the path to save generated images
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict["netG"])
            print('Load G from: ', model_dir)
            netG.cuda()
            netG.eval()
            for key in data_dic:
                save_dir = '%s/%s' % (s_tmp, key)
                mkdir_p(save_dir)
                captions, cap_lens, sorted_indices = data_dic[key]

                batch_size = captions.shape[0]
                nz = cfg.GAN.Z_DIM
                captions = Variable(torch.from_numpy(captions), volatile=True)
                cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

                captions = captions.cuda()
                cap_lens = cap_lens.cuda()
                for i in range(1):  # 16
                    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
                    noise = noise.cuda()
                    #######################################################
                    # (1) Extract text embeddings
                    ######################################################
                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    mask = (captions == 0)
                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    with torch.no_grad():
                        fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
                    # G attention
                    cap_lens_np = cap_lens.cpu().data.numpy()
                    for j in range(batch_size):
                        save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                        for k in range(len(fake_imgs)):
                            im = fake_imgs[k][j].data.cpu().numpy()
                            im = (im + 1.0) * 127.5
                            im = im.astype(np.uint8)
                            # print('im', im.shape)
                            im = np.transpose(im, (1, 2, 0))
                            # print('im', im.shape)
                            im = Image.fromarray(im)
                            fullpath = '%s_g%d.png' % (save_name, k)
                            im.save(fullpath)

                        for k in range(len(attention_maps)):
                            if len(fake_imgs) > 1:
                                im = fake_imgs[k + 1].detach().cpu()
                            else:
                                im = fake_imgs[0].detach().cpu()
                            attn_maps = attention_maps[k]
                            att_sze = attn_maps.size(2)
                            img_set, sentences = \
                                build_super_images2(im[j].unsqueeze(0),
                                                    captions[j].unsqueeze(0),
                                                    [cap_lens_np[j]], self.ixtoword,
                                                    [attn_maps[j]], att_sze)
                            if img_set is not None:
                                im = Image.fromarray(img_set)
                                fullpath = '%s_a%d.png' % (save_name, k)
                                im.save(fullpath)

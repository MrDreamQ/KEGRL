# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import json
import sklearn.metrics
# import matplotlib
# matplotlib.use('Agg')
import random
import gc
from collections import defaultdict

from operator import add

import dgl

# BERT_ENCODER = 'bert'
# CHEMICAL_TYPE = 'Chemical'
# GENE_TYPE = 'Gene'
# DISEASE_TYPE = 'Disease'
# DEBUG_MODE = "DEBUG"
# RUNNING_MODE = 'RUN'

MAX_NODE_NUM = 512

IGNORE_INDEX = -100
is_transformer = False

DEBUG_DOC_NO = 60

GRAPH_LR = 5E-4

from utils import torch_utils

def isNaN(num):
    return num != num

class Node:
    def __init__(self, id, v_id, v_no, sent_id, pos_start, pos_end):
        self.id = id
        self.v_id = v_id
        self.v_no = v_no
        self.sent_id = sent_id
        self.pos_start = pos_start
        self.pos_end = pos_end

class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0
    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1
    def get(self):
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total
    def clear(self):
        self.correct = 0
        self.total = 0

class Config(object):
    def __init__(self, args):
        self.sgraph = args.sgraph
        self.dgraph = args.dgraph
        self.dataset = args.dataset
        self.ada_lamb = args.ada_lamb
        self.dataset_name = 'DocRED' if self.dataset != 'dwie' else 'DWIE'

        self.opt = args
        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()
        self.data_path = args.data_path
        self.use_bag = False
        self.use_gpu = True
        self.is_training = True
        self.max_length = 512  if self.dataset != 'dwie' else 1800
        self.pos_num = 2 * self.max_length
        self.entity_num = self.max_length

        self.relation_num = 97 if self.dataset != 'dwie' else 66
        self.ner_vocab_len = 13 

        self.max_sent_len = 200 if self.dataset != 'dwie' else 100
        self.max_entity_num = 100 if self.dataset != 'dwie' else 100
        self.max_sent_num = 30 if self.dataset != 'dwie' else 220
        self.max_node_num = 200 if self.dataset != 'dwie' else 300
        self.max_node_per_sent = 40 if self.dataset != 'dwie' else 20

        self.rnn_hidden = args.hidden_dim # hidden emb dim
        self.coref_size = args.coref_dim # coref emb dim
        self.entity_type_size = args.pos_dim # entity emb dim
        self.max_epoch = args.num_epoch
        self.opt_method = 'Adam'
        self.optimizer = None

        self.checkpoint_dir = './checkpoint'
        self.fig_result_dir = './fig_result'
        self.test_epoch = 1
        self.pretrain_model = None

        self.word_size = 100
        self.epoch_range = None
        self.dropout_rate = args.dropout_rate  # for sequence
        self.keep_prob = 0.8  # for lstm

        self.period = 50
        self.batch_size = args.batch_size
        self.h_t_limit = 1800 if self.dataset != 'dwie' else 8000
        self.e_num = 42 if self.dataset != 'dwie' else 88

        self.test_batch_size = self.batch_size
        self.test_relation_limit = 1800 if self.dataset != 'dwie' else 4900
        self.char_limit = 16
        self.sent_limit = 25
        self.dis2idx = np.zeros((self.max_length), dtype='int64')
        self.dis2idx[1] = 1
        self.dis2idx[2:] = 2
        self.dis2idx[4:] = 3
        self.dis2idx[8:] = 4
        self.dis2idx[16:] = 5
        self.dis2idx[32:] = 6
        self.dis2idx[64:] = 7
        self.dis2idx[128:] = 8
        self.dis2idx[256:] = 9
        self.dis_size = 20

        self.train_prefix = args.train_prefix
        self.test_prefix = args.test_prefix

        self.lr = args.lr
        self.decay_epoch = args.decay_epoch

        self.lr_decay = args.lr_decay
        if not os.path.exists("log"):
            os.mkdir("log")

        self.softmax = nn.Softmax(dim=-1)

        self.dropout_emb = args.dropout_emb
        self.dropout_rnn = args.dropout_rnn
        self.dropout_gcn = args.dropout_gcn

        self.max_grad_norm = args.max_grad_norm # gradient clipping
        self.grad_accu_step = args.grad_accu_step
        self.optim = args.optim

        self.use_struct_att = args.use_struct_att

        self.use_reasoning_block = args.use_reasoning_block
        self.reasoner_layer_first = args.reasoner_layer_first
        self.reasoner_layer_second = args.reasoner_layer_second

        self.evaluate_epoch = args.evaluate_epoch
        self.finetune_emb = args.finetune_emb

        self.input_theta = -1

    def set_data_path(self, data_path):
        self.data_path = data_path
    def set_max_length(self, max_length):
        self.max_length = max_length
        self.pos_num = 2 * self.max_length
    def set_num_classes(self, num_classes):
        self.num_classes = num_classes
    def set_window_size(self, window_size):
        self.window_size = window_size
    def set_word_size(self, word_size):
        self.word_size = word_size
    def set_max_epoch(self, max_epoch):
        self.max_epoch = max_epoch
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    def set_opt_method(self, opt_method):
        self.opt_method = opt_method
    def set_drop_prob(self, drop_prob):
        self.drop_prob = drop_prob
    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
    def set_test_epoch(self, test_epoch):
        self.test_epoch = test_epoch
    def set_pretrain_model(self, pretrain_model):
        self.pretrain_model = pretrain_model
    def set_is_training(self, is_training):
        self.is_training = is_training
    def set_use_bag(self, use_bag):
        self.use_bag = use_bag
    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
    def set_epoch_range(self, epoch_range):
        self.epoch_range = epoch_range

    def load_train_data(self):
        print("Reading training data...")
        prefix = self.train_prefix

        print ('train', prefix)

        self.data_train_word = np.load(os.path.join(self.data_path, prefix + '_word.npy'))

        # elmo_ids = batch_to_ids(batch_words).cuda()
        self.data_train_pos = np.load(os.path.join(self.data_path, prefix+'_pos.npy'))
        self.data_train_ner = np.load(os.path.join(self.data_path, prefix+'_ner.npy')) # word_embedding
        self.data_train_char = np.load(os.path.join(self.data_path, prefix+'_char.npy'))
        self.data_train_seg = np.load(os.path.join(self.data_path, prefix+'_seg.npy'))
        self.data_train_node_position = np.load(os.path.join(self.data_path, prefix+'_node_position.npy'))

        self.data_train_node_position_sent = np.load(os.path.join(self.data_path, prefix+'_node_position_sent.npy'))

        self.data_train_node_sent_num = np.load(os.path.join(self.data_path, prefix+'_node_sent_num.npy'))

        self.data_train_node_num = np.load(os.path.join(self.data_path, prefix+'_node_num.npy'))
        self.data_train_entity_position = np.load(os.path.join(self.data_path, prefix+'_entity_position.npy'))
        self.train_file = json.load(open(os.path.join(self.data_path, prefix+'.json')))

        self.data_train_sdp_position = np.load(os.path.join(self.data_path, prefix + '_sdp_position.npy'))
        self.data_train_sdp_num = np.load(os.path.join(self.data_path, prefix+'_sdp_num.npy'))

        self.train_len = ins_num = self.data_train_word.shape[0]

        assert(self.train_len==len(self.train_file))
        print("Finish reading, total reading {} train documetns".format(self.train_len))

        self.train_order = list(range(ins_num))
        self.train_batches = ins_num // self.batch_size
        if ins_num % self.batch_size != 0:
            self.train_batches += 1

    def load_test_data(self):
        print("Reading testing data...")

        self.data_word_vec = np.load(os.path.join(f'../dataset/{self.dataset_name}', 'vec.npy'))
        
        self.rel2id = json.load(open(os.path.join(f'../dataset/{self.dataset_name}', 'rel2id.json')))
        self.id2rel = {v: k for k,v in self.rel2id.items()}
        self.ner2id = json.load(open(os.path.join(f'../dataset/{self.dataset_name}', 'ner2id.json')))

        prefix = self.test_prefix

        print (prefix)
        self.is_test = ('dev_test' == prefix)

        self.data_test_word = np.load(os.path.join(self.data_path, prefix + '_word.npy'))
        self.data_test_pos = np.load(os.path.join(self.data_path, prefix+'_pos.npy'))
        self.data_test_ner = np.load(os.path.join(self.data_path, prefix+'_ner.npy'))
        self.data_test_char = np.load(os.path.join(self.data_path, prefix+'_char.npy'))

        self.data_test_node_position = np.load(os.path.join(self.data_path, prefix+'_node_position.npy'))

        self.data_test_node_position_sent = np.load(os.path.join(self.data_path, prefix+'_node_position_sent.npy'))
        #self.data_test_adj = np.load(os.path.join(self.data_path, prefix+'_adj.npy'))

        self.data_test_node_sent_num = np.load(os.path.join(self.data_path, prefix+'_node_sent_num.npy'))

        self.data_test_node_num = np.load(os.path.join(self.data_path, prefix+'_node_num.npy'))
        self.data_test_entity_position = np.load(os.path.join(self.data_path, prefix+'_entity_position.npy'))
        self.test_file = json.load(open(os.path.join(self.data_path, prefix+'.json')))
        self.data_test_seg = np.load(os.path.join(self.data_path, prefix+'_seg.npy'))
        self.test_len = self.data_test_word.shape[0]

        self.data_test_sdp_position = np.load(os.path.join(self.data_path, prefix + '_sdp_position.npy'))
        self.data_test_sdp_num = np.load(os.path.join(self.data_path, prefix+'_sdp_num.npy'))

        assert(self.test_len==len(self.test_file))

        print("Finish reading, total reading {} test documetns".format(self.test_len))

        self.test_batches = self.data_test_word.shape[0] // self.test_batch_size
        if self.data_test_word.shape[0] % self.test_batch_size != 0:
            self.test_batches += 1

        self.test_order = list(range(self.test_len))
        self.test_order.sort(key=lambda x: np.sum(self.data_test_word[x] > 0), reverse=True)
        
        dataset2edgedir = {'docred': 'DocRED', 're': 'Re-DocRED', 'dwie': 'DWIE'}
        train_rely_weight = json.load(open(os.path.join(f'../dataset/{dataset2edgedir[self.dataset]}', 'train_rely_weight.json')))
        self.fq_hlr = np.array(train_rely_weight['p_hlr'])
        self.fq_tlr = np.array(train_rely_weight['p_tlr'])
        self.fq_rlh = np.array(train_rely_weight['p_rlh'])
        self.fq_rlt = np.array(train_rely_weight['p_rlt'])

    def get_train_batch(self):

        random.shuffle(self.train_order)
        context_idxs = torch.LongTensor(self.batch_size, self.max_length).cuda()
        context_pos = torch.LongTensor(self.batch_size, self.max_length).cuda()
        h_mapping = torch.Tensor(self.batch_size, self.h_t_limit, self.max_length).cuda()
        t_mapping = torch.Tensor(self.batch_size, self.h_t_limit, self.max_length).cuda()
        e_mapping = torch.Tensor(self.batch_size, self.e_num, self.max_length).cuda()
        relation_multi_label = torch.Tensor(self.batch_size, self.h_t_limit, self.relation_num).cuda()
        relation_mask = torch.Tensor(self.batch_size, self.h_t_limit).cuda()

        pos_idx = torch.LongTensor(self.batch_size, self.max_length).cuda()

        context_ner = torch.LongTensor(self.batch_size, self.max_length).cuda()
        context_char_idxs = torch.LongTensor(self.batch_size, self.max_length, self.char_limit).cuda()

        relation_label = torch.LongTensor(self.batch_size, self.h_t_limit).cuda()

        ht_pair_pos = torch.LongTensor(self.batch_size, self.h_t_limit).cuda()

        context_seg = torch.LongTensor(self.batch_size, self.max_length).cuda()

        node_position_sent = torch.zeros(self.batch_size, self.max_sent_num, self.max_node_per_sent, self.max_sent_len).float()

        #cgnn_adj = torch.zeros(self.batch_size, 5, self.max_length,self.max_length).float() # 5 indicate the rational type in GCGNN
        node_position =  torch.zeros(self.batch_size, self.max_node_num, self.max_length).float().cuda()

        sdp_position = torch.zeros(self.batch_size, self.max_entity_num, self.max_length).float().cuda()
        sdp_num = torch.zeros(self.batch_size, 1).long().cuda()

        node_sent_num =  torch.zeros(self.batch_size, self.max_sent_num).float().cuda()

        entity_position =  torch.zeros(self.batch_size, self.max_entity_num, self.max_length).float().cuda()
        node_num = torch.zeros(self.batch_size, 1).long().cuda()

        for b in range(self.train_batches):

            entity_num = []
            sentence_num = []
            sentence_len = []
            node_num_per_sent = []

            start_id = b * self.batch_size
            cur_bsz = min(self.batch_size, self.train_len - start_id)
            cur_batch = list(self.train_order[start_id: start_id + cur_bsz])
            cur_batch.sort(key=lambda x: np.sum(self.data_train_word[x]>0) , reverse = True)

            for mapping in [h_mapping, t_mapping, e_mapping]:
                mapping.zero_()

            for mapping in [relation_multi_label, relation_mask, pos_idx]:
                mapping.zero_()

            ht_pair_pos.zero_()

            relation_label.fill_(IGNORE_INDEX)

            max_h_t_cnt = 1

            sdp_nums = []

            hts = []
            hts_len = []
            graph_big_ls, graph_ht_ls, graph_th_ls = [], [], []
            for i, index in enumerate(cur_batch):
                tmp_hts = []

                context_idxs[i].copy_(torch.from_numpy(self.data_train_word[index, :]))
                context_pos[i].copy_(torch.from_numpy(self.data_train_pos[index, :])) #???
                context_char_idxs[i].copy_(torch.from_numpy(self.data_train_char[index, :]))
                context_ner[i].copy_(torch.from_numpy(self.data_train_ner[index, :]))
                context_seg[i].copy_(torch.from_numpy(self.data_train_seg[index, :]))

                ins = self.train_file[index]
                labels = ins['labels']
                idx2label = defaultdict(list)

                for label in labels:
                    idx2label[(label['h'], label['t'])].append(int(label['r']))

                node_position[i].copy_(torch.from_numpy(self.data_train_node_position[index]))

                node_position_sent[i].copy_(torch.from_numpy(self.data_train_node_position_sent[index]))

                node_sent_num[i].copy_(torch.from_numpy(self.data_train_node_sent_num[index]))

                node_num[i].copy_(torch.from_numpy(self.data_train_node_num[index]))
                entity_position[i].copy_(torch.from_numpy(self.data_train_entity_position[index]))

                entity_num.append(len(ins['vertexSet']))
                sentence_num.append(len(ins['sents']))
                sentence_len.append(max([len(sent) for sent in ins['sents']])) # max sent len of a document
                node_num_per_sent.append(max(node_sent_num[i].tolist()))

                sdp_position[i].copy_(torch.from_numpy(self.data_train_sdp_position[index]))
                sdp_num[i].copy_(torch.from_numpy(self.data_train_sdp_num[index]))

                sdp_no_trucation = sdp_num[i].item()
                if sdp_no_trucation > self.max_entity_num:
                    sdp_no_trucation = self.max_entity_num
                sdp_nums.append(sdp_no_trucation)

                for j in range(self.max_length):
                    if self.data_train_word[index, j]==0:
                        break
                    pos_idx[i, j] = j+1

                train_tripe = list(idx2label.keys())
                for j, (h_idx, t_idx) in enumerate(train_tripe):
                    if h_idx == t_idx:
                        continue
                    hts.append([h_idx, t_idx])
                    tmp_hts.append([h_idx, t_idx])

                    hlist = ins['vertexSet'][h_idx]
                    tlist = ins['vertexSet'][t_idx]

                    for h in hlist:
                        h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])

                    for t in tlist:
                        t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])

                    label = idx2label[(h_idx, t_idx)]

                    delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]

                    if abs(delta_dis) >= self.max_length: # for gda
                       continue

                    if delta_dis < 0:
                        ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                    else:
                        ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])

                    for r in label:
                        relation_multi_label[i, j, r] = 1

                    relation_mask[i, j] = 1
                    rt = np.random.randint(len(label))
                    relation_label[i, j] = label[rt]

                lower_bound = len(ins['na_triple'])

                for j, (h_idx, t_idx) in enumerate(ins['na_triple'][:lower_bound], len(train_tripe)):
                    if h_idx == t_idx:
                        continue
                    hts.append([h_idx, t_idx])
                    tmp_hts.append([h_idx, t_idx])
                    
                    hlist = ins['vertexSet'][h_idx]
                    tlist = ins['vertexSet'][t_idx]


                    for h in hlist:
                        h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])

                    for t in tlist:
                        t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])

                    relation_multi_label[i, j, 0] = 1
                    relation_label[i, j] = 0
                    relation_mask[i, j] = 1
                    delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]

                    if abs(delta_dis) >= self.max_length:#for gda
                       continue

                    if delta_dis < 0:
                        ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                    else:
                        ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])

                max_h_t_cnt = max(max_h_t_cnt, len(train_tripe) + lower_bound)

                for e_idx, e_list in enumerate(ins['vertexSet']):
                    for e in e_list:
                        e_mapping[i, e_idx, e['pos'][0]:e['pos'][1]] = 1.0 / len(e_list) / (e['pos'][1] - e['pos'][0])
                hts_len.append(len(tmp_hts))
                if self.sgraph or self.dgraph:
                    graphs = self.create_graph(ins)
                    graph_big_ls.append(graphs['graph_big'])
                    graph_ht_ls.append(graphs['graph_ht'])
                    graph_th_ls.append(graphs['graph_th'])

            input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max()) # max length of a document

            entity_mention_num = list(map(add, entity_num, node_num[:cur_bsz].squeeze(1).tolist()))
            max_sdp_num = max(sdp_nums)
            all_node_num = list(map(add, sdp_nums , entity_mention_num))

            max_entity_num = max(entity_num)
            max_sentence_num = max(sentence_num)
            b_max_mention_num = int(node_num[:cur_bsz].max()) #- max_entity_num - max_sentence_num
            all_node_num = torch.LongTensor(all_node_num)

            yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
                   'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   'relation_label': relation_label[:cur_bsz, :max_h_t_cnt].contiguous(),
                   'input_lengths' : input_lengths,
                   'pos_idx': pos_idx[:cur_bsz, :max_c_len].contiguous(),
                   'relation_multi_label': relation_multi_label[:cur_bsz, :max_h_t_cnt],
                   'relation_mask': relation_mask[:cur_bsz, :max_h_t_cnt],
                   'context_ner': context_ner[:cur_bsz, :max_c_len].contiguous(),
                   'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt],
                   'context_seg': context_seg[:cur_bsz, :max_c_len].contiguous(),
                   'node_position': node_position[:cur_bsz, :b_max_mention_num, :max_c_len].contiguous(),
                   'node_sent_num': node_sent_num[:cur_bsz, :max_sentence_num].contiguous(),
                   'entity_position': entity_position[:cur_bsz, :max_entity_num, :max_c_len].contiguous(),
                   'all_node_num':all_node_num,
                   'entity_num': entity_num,
                   'sent_num': sentence_num,
                   'sdp_position': sdp_position[:cur_bsz, :max_sdp_num, :max_c_len].contiguous(),
                   'sdp_num': sdp_nums,

                    'e_mapping': e_mapping[:cur_bsz, :max_entity_num, :max_c_len],
                    'hts': hts,
                    'hts_len': hts_len,
				   'graph_big': dgl.batch(graph_big_ls).to('cuda') if self.sgraph else None,
				   'graph_ht': dgl.batch(graph_ht_ls).to('cuda') if self.dgraph else None,
				   'graph_th': dgl.batch(graph_th_ls).to('cuda') if self.dgraph else None,
                   }

    def get_test_batch(self):
        context_idxs = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
        context_pos = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
        h_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_length).cuda()
        t_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_length).cuda()
        e_mapping = torch.Tensor(self.batch_size, self.e_num, self.max_length).cuda()
        context_ner = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
        context_char_idxs = torch.LongTensor(self.test_batch_size, self.max_length, self.char_limit).cuda()
        relation_mask = torch.Tensor(self.test_batch_size, self.h_t_limit).cuda()
        ht_pair_pos = torch.LongTensor(self.test_batch_size, self.h_t_limit).cuda()
        context_seg = torch.LongTensor(self.batch_size, self.max_length).cuda()

        node_position_sent =  torch.zeros(self.batch_size, self.max_sent_num, self.max_node_per_sent, self.max_sent_len).float()

        node_position =  torch.zeros(self.batch_size, self.max_node_num, self.max_length).float().cuda()
        entity_position =  torch.zeros(self.batch_size, self.max_entity_num, self.max_length).float().cuda()
        node_num = torch.zeros(self.batch_size, 1).long().cuda()

        node_sent_num =  torch.zeros(self.batch_size, self.max_sent_num).float().cuda()

        sdp_position = torch.zeros(self.batch_size, self.max_entity_num, self.max_length).float().cuda()
        sdp_num = torch.zeros(self.batch_size, 1).long().cuda()

        for b in range(self.test_batches):

            entity_num = []
            sentence_num = []
            sentence_len = []
            node_num_per_sent = []

            start_id = b * self.test_batch_size
            cur_bsz = min(self.test_batch_size, self.test_len - start_id)
            cur_batch = list(self.test_order[start_id : start_id + cur_bsz])

            for mapping in [h_mapping, t_mapping, relation_mask, e_mapping]:
                mapping.zero_()

            ht_pair_pos.zero_()

            max_h_t_cnt = 1

            cur_batch.sort(key=lambda x: np.sum(self.data_test_word[x]>0) , reverse = True)

            labels = []

            L_vertex = []
            titles = []
            indexes = []
            sdp_nums = []

            vertexSets = []

            hts = []
            hts_len = []
            graph_big_ls, graph_ht_ls, graph_th_ls = [], [], []
            for i, index in enumerate(cur_batch):
                context_idxs[i].copy_(torch.from_numpy(self.data_test_word[index, :]))
                context_pos[i].copy_(torch.from_numpy(self.data_test_pos[index, :]))
                context_char_idxs[i].copy_(torch.from_numpy(self.data_test_char[index, :]))
                context_ner[i].copy_(torch.from_numpy(self.data_test_ner[index, :]))
                context_seg[i].copy_(torch.from_numpy(self.data_test_seg[index, :]))

                idx2label = defaultdict(list)
                ins = self.test_file[index]

                for label in ins['labels']:
                    idx2label[(label['h'], label['t'])].append(label['r'])

                node_position[i].copy_(torch.from_numpy(self.data_test_node_position[index]))
                node_position_sent[i].copy_(torch.from_numpy(self.data_test_node_position_sent[index]))

                node_sent_num[i].copy_(torch.from_numpy(self.data_test_node_sent_num[index]))

                node_num[i].copy_(torch.from_numpy(self.data_test_node_num[index]))
                entity_position[i].copy_(torch.from_numpy(self.data_test_entity_position[index]))
                entity_num.append(len(ins['vertexSet']))
                sentence_num.append(len(ins['sents']))
                sentence_len.append(max([len(sent) for sent in ins['sents']])) # max sent len of a document
                node_num_per_sent.append(max(node_sent_num[i].tolist()))

                sdp_position[i].copy_(torch.from_numpy(self.data_test_sdp_position[index]))
                sdp_num[i].copy_(torch.from_numpy(self.data_test_sdp_num[index]))

                sdp_no_trucation = sdp_num[i].item()
                if sdp_no_trucation > self.max_entity_num:
                    sdp_no_trucation = self.max_entity_num
                sdp_nums.append(sdp_no_trucation)

                L = len(ins['vertexSet'])
                titles.append(ins['title'])

                vertexSets.append(ins['vertexSet'])

                j = 0
                for h_idx in range(L):
                    for t_idx in range(L):
                        if h_idx != t_idx:
                            hlist = ins['vertexSet'][h_idx]
                            tlist = ins['vertexSet'][t_idx]

                            for h in hlist:
                                h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])
                            for t in tlist:
                                t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])

                            relation_mask[i, j] = 1

                            delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]

                            if delta_dis < 0:
                                ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                            else:
                                ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])
                            j += 1

                            hts.append([h_idx, t_idx])
                hts_len.append(j)

                max_h_t_cnt = max(max_h_t_cnt, j)
                label_set = {}
                for label in ins['labels']:
                    label_set[(label['h'], label['t'], label['r'])] = label['in'+self.train_prefix]

                labels.append(label_set)

                L_vertex.append(L)
                indexes.append(index)

                for e_idx, e_list in enumerate(ins['vertexSet']):
                    for e in e_list:
                        if self.dataset != 'dwie':
                            e_mapping[i, e_idx, e['pos'][0]:e['pos'][1]] = 1.0 / len(e_list) / (e['pos'][1] - e['pos'][0])
                        elif self.dataset == 'dwie': 
                            e_mapping[i, e_idx, e['absolute_pos'][0]:e['absolute_pos'][1]] = 1.0 / len(e_list) / (e['absolute_pos'][1] - e['absolute_pos'][0])

                if self.sgraph or self.dgraph:
                    graphs = self.create_graph(ins)
                    graph_big_ls.append(graphs['graph_big'])
                    graph_ht_ls.append(graphs['graph_ht'])
                    graph_th_ls.append(graphs['graph_th'])

            input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())

            entity_mention_num = list(map(add, entity_num, node_num[:cur_bsz].squeeze(1).tolist()))
            max_sdp_num = max(sdp_nums)
            all_node_num = list(map(add, sdp_nums , entity_mention_num))

            max_entity_num = max(entity_num)
            max_sentence_num = max(sentence_num)
            b_max_mention_num = int(node_num[:cur_bsz].max()) #- max_entity_num - max_sentence_num
            all_node_num = torch.LongTensor(all_node_num)

            yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
                   'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   'context_seg': context_seg[:cur_bsz, :max_c_len].contiguous(),
                   'labels': labels,
                   'L_vertex': L_vertex,
                   'input_lengths': input_lengths,
                   'context_ner': context_ner[:cur_bsz, :max_c_len].contiguous(),
                   'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'relation_mask': relation_mask[:cur_bsz, :max_h_t_cnt],
                   'titles': titles,
                   'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt],
                   'node_position': node_position[:cur_bsz, :b_max_mention_num, :max_c_len].contiguous(),
                   'entity_position': entity_position[:cur_bsz, :max_entity_num, :max_c_len].contiguous(),
                   'node_sent_num': node_sent_num[:cur_bsz, :max_sentence_num].contiguous(),
                   'indexes': indexes,
                   'all_node_num': all_node_num,
                   'entity_num': entity_num,
                   'sent_num': sentence_num,
                   'sdp_position': sdp_position[:cur_bsz, :max_sdp_num, :max_c_len].contiguous(),
                   'sdp_num': sdp_nums,
                   'vertexsets': vertexSets,

                    'e_mapping': e_mapping[:cur_bsz, :max_entity_num, :max_c_len],
                    'hts': hts,
                    'hts_len': hts_len,
				   'graph_big': dgl.batch(graph_big_ls).to('cuda') if self.sgraph else None,
				   'graph_ht': dgl.batch(graph_ht_ls).to('cuda') if self.dgraph else None,
				   'graph_th': dgl.batch(graph_th_ls).to('cuda') if self.dgraph else None,
                   }

    def train(self, model_pattern, model_name):

        ori_model = model_pattern(config = self)
        if self.pretrain_model != None:
            ori_model.load_state_dict(torch.load(self.pretrain_model))
        ori_model.cuda()

        graph_layer = ["sgnn", "dgnn", "ra_enc"]
        graph_params = [p for n, p in ori_model.named_parameters() if any(nd in n for nd in graph_layer)]
        other_params = [p for n, p in ori_model.named_parameters() if p.requires_grad and not any(nd in n for nd in graph_layer)]

        parameters = [
                        {"params": other_params, "lr": self.lr}, 
                        {"params": graph_params, "lr": GRAPH_LR},
                      ]
        optimizer = torch_utils.get_optimizer(self.optim, parameters, self.lr)
        print(optimizer, flush=True)

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, self.lr_decay)

        model = nn.DataParallel(ori_model)

        BCE = nn.BCEWithLogitsLoss(reduction='none')

        self.checkpoint_dir = os.path.join(self.checkpoint_dir, str(model_name).split('.pth')[0])
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        best_auc = 0.0
        best_f1 = 0.0
        best_epoch = 0

        model.train()

        global_step = 0
        total_loss = 0
        start_time = time.time()

        def logging(s, print_=True, log_=False):
            if print_:
                print(s, flush=True)
            if log_:
                with open(os.path.join(os.path.join("log", model_name)), 'a+') as f_log:
                    f_log.write(s + '\n')

        dev_score_list = []
        f1 = 0
        dev_score_list.append(f1)
        for epoch in range(self.max_epoch):
            gc.collect()
            self.acc_NA.clear()
            self.acc_not_NA.clear()
            self.acc_total.clear()
            print("epoch:{}, Learning rate:{}".format(epoch,optimizer.param_groups[0]['lr']))

            epoch_start_time = time.time()

            for no, data in enumerate(self.get_train_batch()):
                context_idxs = data['context_idxs']
                context_pos = data['context_pos']
                h_mapping = data['h_mapping']
                t_mapping = data['t_mapping']
                relation_label = data['relation_label']
                input_lengths =  data['input_lengths']
                relation_multi_label = data['relation_multi_label']
                relation_mask = data['relation_mask']
                context_ner = data['context_ner']
                context_char_idxs = data['context_char_idxs']
                ht_pair_pos = data['ht_pair_pos']
                context_seg = data['context_seg']

                dis_h_2_t = ht_pair_pos+10
                dis_t_2_h = -ht_pair_pos+10

                # torch.cuda.empty_cache()

                context_idxs = context_idxs.cuda()
                context_pos = context_pos.cuda()
                context_ner = context_ner.cuda()
                #context_char_idxs = context_char_idxs.cuda()
                #input_lengths = input_lengths.cuda()
                h_mapping = h_mapping.cuda()
                t_mapping = t_mapping.cuda()
                relation_mask = relation_mask.cuda()
                dis_h_2_t = dis_h_2_t.cuda()
                dis_t_2_h = dis_t_2_h.cuda()

                node_position = data['node_position'].cuda()
                entity_position = data['entity_position'].cuda()
                node_sent_num = data['node_sent_num'].cuda()
                all_node_num = data['all_node_num'].cuda()
                entity_num = torch.Tensor(data['entity_num']).cuda()
                #sent_num = torch.Tensor(data['sent_num']).cuda()

                sdp_pos = data['sdp_position'].cuda()
                sdp_num = torch.Tensor(data['sdp_num']).cuda()

                e_mapping = data['e_mapping'].cuda()
                hts = data['hts']
                hts_len = data['hts_len']
                graph_big = data['graph_big']
                graph_ht = data['graph_ht']
                graph_th = data['graph_th']

                predict_re = model(context_idxs, context_pos, context_ner, h_mapping, 
                                   t_mapping, relation_mask, dis_h_2_t, dis_t_2_h, context_seg,
                                   node_position, entity_position, node_sent_num,
                                   all_node_num, entity_num, sdp_pos, sdp_num, 
                                   e_mapping=e_mapping, hts=hts, hts_len=hts_len, graph_big=graph_big, graph_ht=graph_ht, graph_th=graph_th)

                relation_multi_label = relation_multi_label.cuda()

                loss = torch.sum(BCE(predict_re, relation_multi_label)*relation_mask.unsqueeze(2)) / torch.sum(relation_mask)

                output = torch.argmax(predict_re, dim=-1)
                output = output.data.cpu().numpy()

                optimizer.zero_grad()
                if model.module.ada_loss is not None:
                    loss = loss + model.module.ada_loss * self.ada_lamb
                loss.backward()

                if no % self.grad_accu_step == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    model.zero_grad()

                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()

                relation_label = relation_label.data.cpu().numpy()

                for i in range(output.shape[0]):
                    for j in range(output.shape[1]):
                        label = relation_label[i][j]
                        if label<0:
                            break
                        if label == 0:
                            self.acc_NA.add(output[i][j] == label)
                        else:
                            self.acc_not_NA.add(output[i][j] == label)

                        self.acc_total.add(output[i][j] == label)

                global_step += 1
                total_loss += loss.item()

                if global_step % self.period == 0:
                    cur_loss = total_loss / self.period
                    elapsed = time.time() - start_time
                    logging('| epoch {:2d} | step {:4d} |  ms/b {:5.2f} | train loss {} | not NA acc: {:4.2f}  | tot acc: {:4.2f} '.format(epoch, global_step, elapsed * 1000 / self.period, cur_loss, self.acc_not_NA.get(), self.acc_total.get()))
                    total_loss = 0
                    start_time = time.time()

            if epoch % self.evaluate_epoch == 0 and (epoch + 1) >= 50:
                logging('-' * 89)
                eval_start_time = time.time()
                model.eval()

                f1, f1_ig, auc, pr_x, pr_y = self.test(model, model_name)

                model.train()
                logging('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
                
                logging('-' * 89)

                if f1 > best_f1:
                    best_f1 = f1
                    best_auc = auc
                    best_epoch = epoch
                    path = os.path.join(self.checkpoint_dir, model_name)
                    torch.save(ori_model.state_dict(), path)
                    logging("best f1 is: {}, epoch is: {}, save path is: {}".format(best_f1, best_epoch, path))

            if epoch > self.decay_epoch:  # and epoch < self.evaluate_epoch:# and epoch < self.evaluate_epoch:
                if self.optim == 'sgd' and f1 < dev_score_list[-1]:
                    self.lr *= self.lr_decay
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = self.lr

                if self.optim == 'adam' and optimizer.param_groups[0]['lr'] > 1e-4: #epoch < 30:# and f1 < dev_score_list[-1]:
                    scheduler.step()

            dev_score_list.append(f1)
            print("train time for epoch {}: {}".format(epoch,time.time()-epoch_start_time))

        print("Finish training")
        print("Best epoch = {} | F1 {}, auc = {}, input_theta = {}".format(best_epoch, best_f1, best_auc, self.input_theta))
        print("Storing best result...")
        print("Finish storing")

    def test(self, model, model_name, output=False, input_theta=-1):
        data_idx = 0
        eval_start_time = time.time()
        test_result_ignore = []
        total_recall_ignore = 0

        test_result = []
        total_recall = 0
        top1_acc = have_label = 0

        def logging(s, print_=True, log_=False):
            if print_:
                print(s, flush=True)
            if log_:
                with open(os.path.join(os.path.join("log", model_name)), 'a+') as f_log:
                    f_log.write(s + '\n')

        each_total = [0] * (self.relation_num-1)
        each_pred = [0] * (self.relation_num-1)
        each_correct = [0] * (self.relation_num-1)
        each_correct_intrain = [0] * (self.relation_num-1)

        for i, data in enumerate(self.get_test_batch()):
            with torch.no_grad():
                context_idxs = data['context_idxs']
                context_pos = data['context_pos']
                h_mapping = data['h_mapping']
                t_mapping = data['t_mapping']
                labels = data['labels']
                L_vertex = data['L_vertex']
                #input_lengths =  data['input_lengths']
                context_ner = data['context_ner']
                #context_char_idxs = data['context_char_idxs']
                relation_mask = data['relation_mask']
                ht_pair_pos = data['ht_pair_pos']

                titles = data['titles']
                indexes = data['indexes']

                context_seg = data['context_seg']

                dis_h_2_t = ht_pair_pos+10
                dis_t_2_h = -ht_pair_pos+10

                node_position = data['node_position'].cuda()
                entity_position = data['entity_position'].cuda()
                #node_position_sent = data['node_position_sent']#.cuda()
                node_sent_num = data['node_sent_num'].cuda()
                all_node_num = data['all_node_num'].cuda()
                entity_num = torch.Tensor(data['entity_num']).cuda()
                #sent_num = torch.Tensor(data['sent_num']).cuda()
                sdp_pos = data['sdp_position'].cuda()
                sdp_num = torch.Tensor(data['sdp_num']).cuda()

                e_mapping = data['e_mapping']
                hts = data['hts']
                hts_len = data['hts_len']
                graph_big = data['graph_big']
                graph_ht = data['graph_ht']
                graph_th = data['graph_th']

                predict_re = model(context_idxs, context_pos, context_ner,
                                   h_mapping, t_mapping, relation_mask, dis_h_2_t, dis_t_2_h, context_seg,
                                   node_position, entity_position, node_sent_num,
                                   all_node_num, entity_num, sdp_pos, sdp_num, 
                                   e_mapping=e_mapping, hts=hts, hts_len=hts_len, graph_big=graph_big, graph_ht=graph_ht, graph_th=graph_th)
                                   

            predict_re = torch.sigmoid(predict_re)

            predict_re = predict_re.data.cpu().numpy()

            for i in range(len(labels)):
                label = labels[i]
                index = indexes[i]

                for tmp_label in label:
                    each_total[tmp_label[-1] - 1] += 1

                total_recall += len(label)
                for l in label.values():
                    if not l:
                        total_recall_ignore += 1

                L = L_vertex[i] # the number of entities in each instance.
                j = 0

                for h_idx in range(L):
                    for t_idx in range(L):
                       if h_idx != t_idx:

                            pre_r = np.argmax(predict_re[i, j])
                            if (h_idx, t_idx, pre_r) in label:
                                top1_acc += 1

                            flag = False

                            for r in range(1, self.relation_num):
                                intrain = False

                                if (h_idx, t_idx, r) in label:
                                    flag = True
                                    if label[(h_idx, t_idx, r)]==True:
                                        intrain = True
                                # if not intrain:
                                #     test_result_ignore.append( ((h_idx, t_idx, r) in label, float(predict_re[i,j,r]),  titles[i], self.id2rel[r], index, h_idx, t_idx, r, np.argmax(predict_re[i,j])))

                                test_result.append( ((h_idx, t_idx, r) in label, float(predict_re[i,j,r]), intrain,  titles[i], self.id2rel[r], index, h_idx, t_idx, r) )

                            if flag:
                                have_label += 1

                            j += 1

            data_idx += 1

            if data_idx % self.period == 0:
                print('| step {:3d} | time: {:5.2f}'.format(data_idx // self.period, (time.time() - eval_start_time)))
                eval_start_time = time.time()

        test_result.sort(key = lambda x: x[1], reverse=True)

        print ('total_recall', total_recall)

        pr_x = []
        pr_y = []
        correct = 0
        w = 0

        if total_recall == 0:
            total_recall = 1  # for test

        for i, item in enumerate(test_result):
            correct += item[0]
            pr_y.append(float(correct) / (i + 1)) # precision
            pr_x.append(float(correct) / total_recall) # recall
            if item[1] > input_theta:
                w = i

        if self.is_test:
            results = []
            for wi in range(w):
                item = test_result[wi]
                result = {
                    'title': item[3], 
                    'h_idx': item[6], 
                    't_idx': item[7], 
                    'r': item[4]
                }
                results.append(result)
            file_name = os.path.dirname(model_name) + os.sep + 'result.json'
            json.dump(results, open(file_name, 'w'))
            print(f"Generate result at {file_name}")

        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max()
        f1_pos = f1_arr.argmax()
        theta = test_result[f1_pos][1]

        if input_theta==-1:
            w = f1_pos
            input_theta = theta
            self.input_theta = input_theta

        auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
        
        if not self.is_test:
            logging('ALL   : Theta {:3.4f} | F1 {:3.4f} | Precision {:3.4f} | Recall {:3.4f} | AUC {:3.4f} '.format(theta, f1, pr_x[f1_pos], pr_y[f1_pos], auc))
        else:
            logging('ma_f1{:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(f1, input_theta, f1_arr[w], pr_x[w], pr_y[w], auc))

        #logging("precision {}, recall {}".format(pr_y, pr_x))

        if output:
            # output = [x[-4:] for x in test_result[:w+1]]
            output = [{'index': x[-4], 'h_idx': x[-3], 't_idx': x[-2], 'r_idx': x[-1], 'r': x[-5], 'title': x[-6]} for x in test_result[:w+1]]
            json.dump(output, open(self.test_prefix + "_index.json", "w"))

        input_theta = theta
        pr_x = []
        pr_y = []
        correct = correct_in_train = 0
        w = 0
        
        each_pred = [0] * (self.relation_num-1)
        each_correct = [0] * (self.relation_num-1)
        each_correct_intrain = [0] * (self.relation_num-1)

        for i, item in enumerate(test_result):
            tmp_r_idx = item[-1] - 1

            correct += item[0]
            if item[0] & item[2]:
                correct_in_train += 1
            if correct_in_train==correct:
                p = 0
            else:
                p = float(correct - correct_in_train) / (i + 1 - correct_in_train)
            pr_y.append(p)
            pr_x.append(float(correct) / total_recall)
            if item[1] > input_theta:
                w = i

            if item[1] > input_theta:
                each_pred[tmp_r_idx] += 1
                if item[0]:
                    each_correct[tmp_r_idx] += 1
                if item[0] & item[2]:
                    each_correct_intrain[tmp_r_idx] += 1

        p_each_rel = np.array(each_correct) / (np.array(each_pred) + 1e-10)
        r_each_rel = np.array(each_correct) / (np.array(each_total) + 1e-10)
        f1_each_rel = 2 * p_each_rel * r_each_rel / (p_each_rel + r_each_rel + 1e-10)

        p_each_rel_ign = (np.array(each_correct) - np.array(each_correct_intrain)) / \
            (np.array(each_pred) - np.array(each_correct_intrain) + 1e-10)
        f1_each_rel_ign = 2 * p_each_rel_ign * r_each_rel / (p_each_rel_ign + r_each_rel + 1e-10)

        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        ign_f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        ign_f1 = ign_f1_arr.max()

        ign_f1_pos = ign_f1_arr.argmax()
        theta = test_result[ign_f1_pos][1]

        ign_auc = sklearn.metrics.auc(x = pr_x, y = pr_y)

        logging('Ignore ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(ign_f1, theta, ign_f1_arr[w], ign_auc), log_=False)
        logging('avg_f1 {:3.4f} | ign_avg_f1 {:3.4f}'.format(f1_each_rel.mean(), f1_each_rel_ign.mean()), log_=False)
  
        return f1, ign_f1, auc, pr_x, pr_y


    def testall(self, model_pattern, model_name, input_theta):#, ignore_input_theta):
        model = model_pattern(config = self)

        model.load_state_dict(torch.load(os.path.join(model_name)))
        model.cuda()
        model.eval()
        f1, f1_ig, auc, pr_x, pr_y = self.test(model, model_name,  True, input_theta)


    def create_graph(self, sample):
        g_data = defaultdict(list)
        g_ht_data, g_th_data = defaultdict(list), defaultdict(list)
        entity_ls = sample['vertexSet']
        e_type_id = torch.tensor([self.ner2id[e[0]['type']] - 1 for e in entity_ls])	# 实体结点类型id
        # 实体节点与关系节点连线边的权重
        w_rlh = torch.zeros(len(entity_ls) * (len(self.rel2id) - 1))
        w_rlt = torch.zeros(len(entity_ls) * (len(self.rel2id) - 1))
        w_hlr = torch.zeros(len(entity_ls) * (len(self.rel2id) - 1))
        w_tlr = torch.zeros(len(entity_ls) * (len(self.rel2id) - 1))

        edge_id = 0
        for entity_id in range(len(entity_ls)):
            for r_type in range(len(self.rel2id) - 1):
                e_type = self.ner2id[entity_ls[entity_id][0]['type']] - 1
                # Double Graph Network
                g_ht_data[('n_e', 'e_rlh', 'n_rel')].append((entity_id, r_type))
                g_ht_data[('n_rel', 'e_tlr', 'n_e')].append((r_type, entity_id))
                g_th_data[('n_e', 'e_rlt', 'n_rel')].append((entity_id, r_type))
                g_th_data[('n_rel', 'e_hlr', 'n_e')].append((r_type, entity_id))
                # Single Big Graph Network
                g_data[('n_eh', 'e_rlh', 'n_rel')].append((entity_id, r_type))
                g_data[('n_et', 'e_rlt', 'n_rel')].append((entity_id, r_type))
                g_data[('n_rel', 'e_hlr', 'n_eh')].append((r_type, entity_id))
                g_data[('n_rel', 'e_tlr', 'n_et')].append((r_type, entity_id))
                # 统计权重
                w_hlr[edge_id] = self.fq_hlr[r_type, e_type]
                w_tlr[edge_id] = self.fq_tlr[r_type, e_type]
                w_rlh[edge_id] = self.fq_rlh[e_type, r_type]
                w_rlt[edge_id] = self.fq_rlt[e_type, r_type]
                
                edge_id += 1
        graph_big = dgl.heterograph(g_data)
        graph_big.nodes['n_eh'].data['type_id'] = e_type_id
        graph_big.nodes['n_et'].data['type_id'] = e_type_id
        graph_big.edges['e_rlh'].data['w'] = w_rlh
        graph_big.edges['e_rlt'].data['w'] = w_rlt
        graph_big.edges['e_hlr'].data['w'] = w_hlr
        graph_big.edges['e_tlr'].data['w'] = w_tlr

        graph_ht = dgl.heterograph(g_ht_data)
        graph_ht.nodes['n_e'].data['type_id'] = e_type_id
        graph_ht.edges['e_rlh'].data['w'] = w_rlh
        graph_ht.edges['e_tlr'].data['w'] = w_tlr

        graph_th = dgl.heterograph(g_th_data)
        graph_th.nodes['n_e'].data['type_id'] = e_type_id
        graph_th.edges['e_rlt'].data['w'] = w_rlt
        graph_th.edges['e_hlr'].data['w'] = w_hlr

        return {
            "graph_big": graph_big,
            "graph_ht": graph_ht,
            "graph_th": graph_th
        }

def cal_long_tail_metrics(f1_each_rel, f1_each_rel_ign):
    long_tail_500 = [15, 16, 21, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 39, 41, 42, 43,
                    45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                    63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]
    long_tail_200 = [58, 62, 63, 65, 66, 69, 70, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
                     83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]
    long_tail_100 = [84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]
    
    long_tail_metric = {
        "macro_f1": f1_each_rel.mean() * 100, "macro_f1_ign": f1_each_rel_ign.mean() * 100,
        "macro_f1_@500": f1_each_rel[long_tail_500].mean() * 100, "macro_f1_@500_ign": f1_each_rel_ign[long_tail_500].mean() * 100,
        "macro_f1_@200": f1_each_rel[long_tail_200].mean() * 100, "macro_f1_@200_ign": f1_each_rel_ign[long_tail_200].mean() * 100, 
        "macro_f1_@100": f1_each_rel[long_tail_100].mean() * 100, "macro_f1_@100_ign": f1_each_rel_ign[long_tail_100].mean() * 100, 
    }
    
    print(long_tail_metric, flush=True)
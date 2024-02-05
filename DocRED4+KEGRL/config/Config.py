# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import dgl
import os
import time
import json
import sklearn.metrics
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import random
from collections import defaultdict
from .g_config import cfg

rel2id = json.load(open(f'../dataset/{cfg.DATASET}/rel2id.json'))
ner2id = json.load(open(f'../dataset/{cfg.DATASET}/ner2id.json'))

IGNORE_INDEX = -100
is_transformer = False

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
		self.args = args
		self.dataset = cfg.DATASET
		self.acc_NA = Accuracy()
		self.acc_not_NA = Accuracy()
		self.acc_total = Accuracy()
		dataset2path = {'DocRED': 'prepro_data', 'Re-DocRED': 'prepro_re', 'DWIE': 'prepro_dwie'}
		self.data_path = dataset2path[cfg.DATASET]
		self.max_length = 512 if cfg.DATASET != 'DWIE' else 1800
		self.relation_num = 97 if cfg.DATASET != 'DWIE' else 66
		self.ner_size = 7 if cfg.DATASET != 'DWIE' else 19
		self.pos_num = 2 * self.max_length
		self.entity_num = self.max_length
		self.use_bag = False
		self.use_gpu = True
		self.is_training = True
		self.coref_size = 20
		self.entity_type_size = 20
		self.max_epoch = 200
		self.opt_method = 'Adam'
		self.optimizer = None

		out_dir = os.path.join('output', os.path.basename(args.config).split('.')[0], 
								time.strftime("%Y%m%d_%H%M%S", time.localtime()))
		self.checkpoint_dir = os.path.join(out_dir, 'checkpoint')
		self.log_dir = os.path.join(out_dir, 'logs')

		self.test_epoch = 5
		self.pretrain_model = None
		self.word_size = 100
		self.epoch_range = None
		self.cnn_drop_prob = 0.5  # for cnn
		self.keep_prob = 0.8  # for lstm

		self.period = 50

		self.batch_size = cfg.BATCH_SIZE
		self.test_batch_size = self.batch_size
		self.e_num = 42 if cfg.DATASET != 'DWIE' else 88
		self.h_t_limit = 1800 if cfg.DATASET != 'DWIE' else 7700
		self.test_relation_limit = 1800 if cfg.DATASET != 'DWIE' else 8000
		self.char_limit = 16
		self.sent_limit = 25
  
		self.dis2idx = np.zeros((2 * self.max_length), dtype='int64')
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
  
		self.args = args

		self.ada_loss_w = cfg.ADA_LOSS_LAMB
		train_rely_weight = json.load(open(f'../dataset/{cfg.DATASET}/train_rely_weight.json'))
		self.fq_hlr = np.array(train_rely_weight['p_hlr'])
		self.fq_tlr = np.array(train_rely_weight['p_tlr'])
		self.fq_rlh = np.array(train_rely_weight['p_rlh'])
		self.fq_rlt = np.array(train_rely_weight['p_rlt'])
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

		self.data_train_word = np.load(os.path.join(self.data_path, prefix+'_word.npy'))
		self.data_train_pos = np.load(os.path.join(self.data_path, prefix+'_pos.npy'))
		self.data_train_ner = np.load(os.path.join(self.data_path, prefix+'_ner.npy'))
		self.train_file = json.load(open(os.path.join(self.data_path, prefix+'.json')))

		print("Finish reading")

		self.train_len = ins_num = self.data_train_word.shape[0]
		assert(self.train_len==len(self.train_file))

		self.train_order = list(range(ins_num))
		self.train_batches = ins_num // self.batch_size
		if ins_num % self.batch_size != 0:
			self.train_batches += 1
    
	def load_test_data(self):
		print("Reading testing data...")

		self.data_word_vec = np.load(os.path.join('../dataset', cfg.DATASET, 'vec.npy'))
		self.rel2id = json.load(open(os.path.join('../dataset', cfg.DATASET, 'rel2id.json')))
		self.ner2id = json.load(open(os.path.join('../dataset', cfg.DATASET, 'ner2id.json')))
		self.id2rel = {v: k for k, v in self.rel2id.items()}

		prefix = self.test_prefix
		print(prefix)
		print(os.path.join(self.data_path, prefix + '.json'))
		self.is_test = ('dev_test' == prefix)
		self.data_test_word = np.load(os.path.join(self.data_path, prefix + '_word.npy'))
		self.data_test_pos = np.load(os.path.join(self.data_path, prefix + '_pos.npy'))
		self.data_test_ner = np.load(os.path.join(self.data_path, prefix + '_ner.npy'))
		self.test_file = json.load(open(os.path.join(self.data_path, prefix + '.json')))

		self.test_len = self.data_test_word.shape[0]
		assert (self.test_len == len(self.test_file))
  
		print("Finish reading")

		self.test_batches = self.data_test_word.shape[0] // self.test_batch_size
		if self.data_test_word.shape[0] % self.test_batch_size != 0:
			self.test_batches += 1

		self.test_order = list(range(self.test_len))
		self.test_order.sort(key=lambda x: np.sum(self.data_test_word[x] > 0), reverse=True)


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
		relation_label = torch.LongTensor(self.batch_size, self.h_t_limit).cuda()
		ht_pair_pos = torch.LongTensor(self.batch_size, self.h_t_limit).cuda()
  
		for b in range(self.train_batches):
			start_id = b * self.batch_size
			cur_bsz = min(self.batch_size, self.train_len - start_id)
			cur_batch = list(self.train_order[start_id: start_id + cur_bsz])
			cur_batch.sort(key=lambda x: np.sum(self.data_train_word[x] > 0), reverse=True)

			for mapping in [h_mapping, t_mapping, e_mapping]:
				mapping.zero_()
			for mapping in [relation_multi_label, relation_mask, pos_idx]:
				mapping.zero_()
			ht_pair_pos.zero_()
			relation_label.fill_(IGNORE_INDEX)
			max_h_t_cnt = 1
   
			entities_num = []
			hts = []
			hts_len = []
			graph_big_ls, graph_ht_ls, graph_th_ls = [], [], []

			for i, index in enumerate(cur_batch):
				tmp_hts = []
				context_idxs[i].copy_(torch.from_numpy(self.data_train_word[index, :]))
				context_pos[i].copy_(torch.from_numpy(self.data_train_pos[index, :]))
				context_ner[i].copy_(torch.from_numpy(self.data_train_ner[index, :]))
    
				for j in range(self.max_length):
					if self.data_train_word[index, j] == 0:
						break
					pos_idx[i, j] = j + 1

				ins = self.train_file[index]
				labels = ins['labels']
				idx2label = defaultdict(list)

				for label in labels:
					idx2label[(label['h'], label['t'])].append(label['r'])
     
				train_triple = list(idx2label.keys())
				for j, (h_idx, t_idx) in enumerate(train_triple):
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
				for j, (h_idx, t_idx) in enumerate(ins['na_triple'][:lower_bound], len(train_triple)):
					if h_idx == t_idx:
						continue
					hlist = ins['vertexSet'][h_idx]
					tlist = ins['vertexSet'][t_idx]
					for h in hlist:
						if h['pos'][0] >= 1800:
							print(hlist)
						if h['pos'][1] >= 1800:
							print(hlist)
						h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])
					for t in tlist:
						t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])
      
					relation_multi_label[i, j, 0] = 1
					relation_label[i, j] = 0
					relation_mask[i, j] = 1
					delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]
					if delta_dis < 0:
						ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
					else:
						ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])
					hts.append([h_idx, t_idx])
					tmp_hts.append([h_idx, t_idx])
     
				max_h_t_cnt = max(max_h_t_cnt, len(tmp_hts))
    
				for e_idx, e_list in enumerate(ins['vertexSet']):
					for e in e_list:
						e_mapping[i, e_idx, e['pos'][0]:e['pos'][1]] = 1.0 / len(e_list) / (e['pos'][1] - e['pos'][0])

				entities_num.append(len(ins['vertexSet']))
				hts_len.append(len(tmp_hts))
    
				if cfg.SGRAPH.LAYERS_NUM > 0 or cfg.DGRAPH.LAYERS_NUM > 0 :
					graphs = self.create_graph(ins)
					graph_big_ls.append(graphs['graph_big'])
					graph_ht_ls.append(graphs['graph_ht'])
					graph_th_ls.append(graphs['graph_th'])
    
			input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
			max_c_len = int(input_lengths.max())
			max_e_num = int(np.array(entities_num).max())
   
			yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
				   'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
				   'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
				   't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
				   'relation_label': relation_label[:cur_bsz, :max_h_t_cnt].contiguous(),
				   'input_lengths': input_lengths,
				   'pos_idx': pos_idx[:cur_bsz, :max_c_len].contiguous(),
				   'relation_multi_label': relation_multi_label[:cur_bsz, :max_h_t_cnt],
				   'relation_mask': relation_mask[:cur_bsz, :max_h_t_cnt],
				   'context_ner': context_ner[:cur_bsz, :max_c_len].contiguous(),

				   'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt],
				   'e_mapping': e_mapping[:cur_bsz, :max_e_num, :max_c_len],
				   'entites_num': entities_num,
				   'hts': hts,
				   'hts_len': hts_len,
				   'graph_big': dgl.batch(graph_big_ls).to('cuda') if cfg.SGRAPH.LAYERS_NUM > 0 else None,
				   'graph_ht': dgl.batch(graph_ht_ls).to('cuda') if cfg.DGRAPH.LAYERS_NUM > 0 else None,
				   'graph_th': dgl.batch(graph_th_ls).to('cuda') if cfg.DGRAPH.LAYERS_NUM > 0 else None,
				   }
   
	def get_test_batch(self):
		context_idxs = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
		context_pos = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
		h_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_length).cuda()
		t_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_length).cuda()
		e_mapping = torch.Tensor(self.test_batch_size, self.e_num, self.max_length).cuda()
		context_ner = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
		relation_mask = torch.Tensor(self.test_batch_size, self.h_t_limit).cuda()
		ht_pair_pos = torch.LongTensor(self.test_batch_size, self.h_t_limit).cuda()
  
		for b in range(self.test_batches):
			start_id = b * self.test_batch_size
			cur_bsz = min(self.test_batch_size, self.test_len - start_id)
			cur_batch = list(self.test_order[start_id: start_id + cur_bsz])
			cur_batch.sort(key=lambda x: np.sum(self.data_test_word[x] > 0), reverse=True)
			for mapping in [h_mapping, t_mapping, relation_mask, e_mapping]:
				mapping.zero_()
			ht_pair_pos.zero_()
			max_h_t_cnt = 1
			labels = []
			L_vertex = []
			titles = []
			indexes = []
   
			entities_num = []
			hts = []
			hts_len = []
			graph_big_ls, graph_ht_ls, graph_th_ls = [], [], []

			for i, index in enumerate(cur_batch):
				context_idxs[i].copy_(torch.from_numpy(self.data_test_word[index, :]))
				context_pos[i].copy_(torch.from_numpy(self.data_test_pos[index, :]))
				context_ner[i].copy_(torch.from_numpy(self.data_test_ner[index, :]))

				idx2label = defaultdict(list)
				ins = self.test_file[index]

				for label in ins['labels']:
					idx2label[(label['h'], label['t'])].append(label['r'])

				L = len(ins['vertexSet'])
				titles.append(ins['title'])

				j = 0
				for h_idx in range(L):
					for t_idx in range(L):
						if h_idx != t_idx:
							hlist = ins['vertexSet'][h_idx]
							tlist = ins['vertexSet'][t_idx]
							hts.append([h_idx, t_idx])

							for h in hlist:
								h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (
										h['pos'][1] - h['pos'][0])
							for t in tlist:
								t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (
										t['pos'][1] - t['pos'][0])

							relation_mask[i, j] = 1

							delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]
							if delta_dis < 0:
								ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
							else:
								ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])
							j += 1

				max_h_t_cnt = max(max_h_t_cnt, j)
				label_set = {}
				for label in ins['labels']:
					label_set[(label['h'], label['t'], label['r'])] = label['in' + self.train_prefix]

				labels.append(label_set)

				L_vertex.append(L)
				indexes.append(index)
    
				for e_idx, e_list in enumerate(ins['vertexSet']):
					for e in e_list:
						if self.dataset != 'DWIE':
							e_mapping[i, e_idx, e['pos'][0]:e['pos'][1]] = 1.0 / len(e_list) / (e['pos'][1] - e['pos'][0])
						else: 
							e_mapping[i, e_idx, e['absolute_pos'][0]:e['absolute_pos'][1]] = 1.0 / len(e_list) / (e['absolute_pos'][1] - e['absolute_pos'][0])
				hts_len.append(j)
				entities_num.append(len(ins['vertexSet']))
				if cfg.SGRAPH.LAYERS_NUM or cfg.DGRAPH.LAYERS_NUM:
					graphs = self.create_graph(ins)
					graph_big_ls.append(graphs['graph_big'])
					graph_ht_ls.append(graphs['graph_ht'])
					graph_th_ls.append(graphs['graph_th'])
    
			input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
			max_c_len = int(input_lengths.max())
			max_e_num = int(np.array(entities_num).max())
   
			yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
				   'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
				   'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
				   't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
				   'labels': labels,
				   'L_vertex': L_vertex,
				   'input_lengths': input_lengths,
				   'context_ner': context_ner[:cur_bsz, :max_c_len].contiguous(),
				   'relation_mask': relation_mask[:cur_bsz, :max_h_t_cnt],
				   'titles': titles,
				   'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt],
				   'indexes': indexes,
       
				   'e_mapping': e_mapping[:cur_bsz, :max_e_num, :max_c_len],
				   'entites_num': entities_num,
				   'hts': hts,
				   'hts_len': hts_len,
				   'graph_big': dgl.batch(graph_big_ls).to('cuda') if cfg.SGRAPH.LAYERS_NUM > 0 else None,
				   'graph_ht': dgl.batch(graph_ht_ls).to('cuda') if cfg.DGRAPH.LAYERS_NUM > 0 else None,
				   'graph_th': dgl.batch(graph_th_ls).to('cuda') if cfg.DGRAPH.LAYERS_NUM > 0 else None,
				   }

	def train(self, model_pattern, model_name):

		ori_model = model_pattern(config=self)
		if self.pretrain_model != None:
			ori_model.load_state_dict(torch.load(self.pretrain_model))

		ori_model.cuda()
		# model = nn.DataParallel(ori_model)
		model = ori_model
        
		new_layer = ["extractor", "bili", "classifier",  "projection"]
		graph_layer = ["sgnn", "dgnn"]
		optimizer_grouped_parameters = [
			{"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in new_layer + graph_layer)], },
			{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in graph_layer)], "lr": cfg.GRAPH_LR},
			{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": cfg.CLASSIFIER_LR},
		]
		optimizer = optim.Adam(optimizer_grouped_parameters, lr=cfg.LR, eps=cfg.ADAM_EPSILON)
		BCE = nn.BCEWithLogitsLoss(reduction='none')
  
		best_auc = 0.0
		best_f1 = 0.0
		best_epoch = 0
		global_step = 0
		total_loss = 0
		model.train()
  
		def logging(s, print_=True, log_=False):
			if print_:
				print(s, flush=True)
			if log_:
				with open(os.path.join(os.path.join("logs", model_name)), 'a+') as f_log:
					f_log.write(s + '\n')
     
		for epoch in range(self.max_epoch):
			self.acc_NA.clear()
			self.acc_not_NA.clear()
			self.acc_total.clear()
			start_time = time.time()
			for step, data in enumerate(self.get_train_batch()):
				context_idxs = data['context_idxs']
				context_pos = data['context_pos']
				h_mapping = data['h_mapping']
				t_mapping = data['t_mapping']
				relation_label = data['relation_label']
				input_lengths = data['input_lengths']
				relation_multi_label = data['relation_multi_label']
				relation_mask = data['relation_mask']
				context_ner = data['context_ner']
				ht_pair_pos = data['ht_pair_pos']
				dis_h_2_t = ht_pair_pos + 10
				dis_t_2_h = -ht_pair_pos + 10    

				e_mapping = data['e_mapping']
				hts = data['hts']
				hts_len = data['hts_len']
				entites_num = data['entites_num']
				graph_big = data['graph_big']
				graph_ht = data['graph_ht']
				graph_th = data['graph_th']

				predict_re = model(context_idxs, context_pos, context_ner, input_lengths, h_mapping, t_mapping, relation_mask, dis_h_2_t, dis_t_2_h, 
           							e_mapping=e_mapping, entites_num=entites_num, hts=hts, hts_len=hts_len, graph_big=graph_big, graph_ht=graph_ht, graph_th=graph_th)
				loss = torch.sum(BCE(predict_re, relation_multi_label) * relation_mask.unsqueeze(2)) / (self.relation_num * torch.sum(relation_mask))

				if cfg.SGRAPH.LAYERS_NUM > 0 or cfg.DGRAPH.LAYERS_NUM > 0:
					loss = loss + model.ada_loss * self.ada_loss_w

				optimizer.zero_grad()
				loss = loss / cfg.ACCUMULATION_STEPS
				loss.backward()
				if step % cfg.ACCUMULATION_STEPS == 0:
					if cfg.MAX_GRAD_NORM > 0:
						torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
					optimizer.step()
					model.zero_grad()
					optimizer.zero_grad()

				relation_label = relation_label.data.cpu().numpy()
				output = torch.argmax(predict_re, dim=-1)
				output = output.data.cpu().numpy()
				for i in range(output.shape[0]):
					for j in range(output.shape[1]):
						label = relation_label[i][j]
						if label < 0:
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
					logging('| epoch {:2d} | step {:4d} |  s {:5.2f} | train loss {:5.3f} | NA acc: {:4.2f} | not NA acc: {:4.2f}  | tot acc: {:4.2f} '.format(
							epoch, global_step, elapsed, total_loss, self.acc_NA.get(),
							self.acc_not_NA.get(), self.acc_total.get()))
					total_loss = 0
					# start_time = time.time()

			if (epoch+1) % self.test_epoch == 0 and epoch > 50:
				logging('-' * 89)
				eval_start_time = time.time()
				model.eval()
				f1, auc, *_ = self.test(model, model_name)
				model.train()
				logging('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
				logging('-' * 89)

				if f1 > best_f1:
					best_f1 = f1
					best_auc = auc
					best_epoch = epoch
					if not os.path.exists(self.checkpoint_dir):
						os.makedirs(self.checkpoint_dir)
					path = os.path.join(self.checkpoint_dir, model_name)
					torch.save(ori_model.state_dict(), path)
					print('save as %s' % path)
     
		logging("Finish training")
		logging("Best epoch = %d | auc = %f | input_theta = %f" % (best_epoch, best_auc, self.input_theta))
		logging("Storing best result...")
		logging("Finish storing")
  
	def test(self, model, model_name, output=False, input_theta=-1):
		data_idx = 0
		eval_start_time = time.time()
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
  
		for data in self.get_test_batch():
			with torch.no_grad():
				context_idxs = data['context_idxs']
				context_pos = data['context_pos']
				h_mapping = data['h_mapping']
				t_mapping = data['t_mapping']
				labels = data['labels']
				L_vertex = data['L_vertex']
				input_lengths = data['input_lengths']
				context_ner = data['context_ner']
				relation_mask = data['relation_mask']
				ht_pair_pos = data['ht_pair_pos']
				titles = data['titles']
				indexes = data['indexes']
				dis_h_2_t = ht_pair_pos + 10
				dis_t_2_h = -ht_pair_pos + 10
    
				e_mapping = data['e_mapping']
				hts = data['hts']
				hts_len = data['hts_len']
				entites_num = data['entites_num']
				graph_big = data['graph_big']
				graph_ht = data['graph_ht']
				graph_th = data['graph_th']

				predict_re = model(context_idxs, context_pos, context_ner, input_lengths, h_mapping, t_mapping, relation_mask, dis_h_2_t, dis_t_2_h,
           							e_mapping=e_mapping, entites_num=entites_num, hts=hts, hts_len=hts_len, graph_big=graph_big, graph_ht=graph_ht, graph_th=graph_th)
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

				L = L_vertex[i]
				j = 0
				for h_idx in range(L):
					for t_idx in range(L):
						if h_idx != t_idx:
							r = np.argmax(predict_re[i, j])
							if (h_idx, t_idx, r) in label:
								top1_acc += 1
							flag = False
							for r in range(1, self.relation_num):
								intrain = False
								if (h_idx, t_idx, r) in label:
									flag = True
									if label[(h_idx, t_idx, r)] == True:
										intrain = True
								test_result.append(((h_idx, t_idx, r) in label, float(predict_re[i, j, r]), intrain,
													titles[i], self.id2rel[r], index, h_idx, t_idx, r))
							if flag:
								have_label += 1
							j += 1
			data_idx += 1
			if data_idx % self.period == 0:
				print('| step {:3d} | time: {:5.2f}'.format(data_idx // self.period, (time.time() - eval_start_time)))
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
			pr_y.append(float(correct) / (i + 1))			# 精确率 正确数/预测数
			pr_x.append(float(correct) / total_recall)		# 召回率 正确数/总数
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
			logging('ALL  : Theta {:3.4f} | F1 {:3.4f} | AUC {:3.4f}'.format(theta, f1, auc), log_=False)
		else:
			logging('ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(f1, input_theta, f1_arr[w], auc), log_=False)
   
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
		ign_auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
  
		logging('Ignore ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(ign_f1, input_theta, ign_f1_arr[w], ign_auc), log_=False)
		logging('avg_f1 {:3.4f} | ign_avg_f1 {:3.4f}'.format(f1_each_rel.mean(), f1_each_rel_ign.mean()), log_=False)

		return f1, auc, pr_x, pr_y, ign_f1, ign_auc, f1_each_rel.mean(), f1_each_rel_ign.mean()

	def testall(self, model_pattern, model_name, input_theta):
		model = model_pattern(config=self)
		print(os.path.join(model_name))
		model.load_state_dict(torch.load(os.path.join(model_name)))
		model.cuda()
		model.eval()
		f1, auc, *_ = self.test(model, model_name, False, input_theta)
  
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
		# 实体节点与关系节点连线边的标签，1为实体与关系之间连线成立
		l_rlh = torch.zeros(len(entity_ls) * (len(self.rel2id) - 1))
		l_rlt = torch.zeros(len(entity_ls) * (len(self.rel2id) - 1))
		l_hlr = torch.zeros(len(entity_ls) * (len(self.rel2id) - 1))
		l_tlr = torch.zeros(len(entity_ls) * (len(self.rel2id) - 1))
		
		hr_set, tr_set = set(), set()
		for tmp_label in sample['labels']:
			hr_set.add((tmp_label['h'], tmp_label['r'] - 1))
			tr_set.add((tmp_label['t'], tmp_label['r'] - 1))

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
				# 实体与关系边的标签
				if (entity_id, r_type) in hr_set:
					l_hlr[edge_id] = 1
					l_rlh[edge_id] = 1
				if (entity_id, r_type) in tr_set:
					l_tlr[edge_id] = 1
					l_rlt[edge_id] = 1
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

import torch
import torch.nn as nn

from .graph_modules import RaEncoder, GraphNetWork, DoubleGraphNetWork
from config.g_config import cfg

class CNN3(nn.Module):
	def __init__(self, config):
		super(CNN3, self).__init__()
		self.config = config
		self.word_emb = nn.Embedding(config.data_word_vec.shape[0], config.data_word_vec.shape[1])
		self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))
		self.word_emb.weight.requires_grad = False
		self.coref_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)
		self.ner_emb = nn.Embedding(config.ner_size, config.entity_type_size, padding_idx=0)
  
		input_size = config.data_word_vec.shape[1] + config.coref_size + config.entity_type_size #+ char_hidden
		self.in_channels = input_size
		self.out_channels = 200
  
		self.kernel_size = 3
		self.stride = 1
		self.padding = int((self.kernel_size - 1) / 2)

		self.cnn_1 = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
		self.cnn_2 = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
		self.cnn_3 = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
		self.max_pooling = nn.MaxPool1d(self.kernel_size, stride=self.stride, padding=self.padding)
		self.relu = nn.ReLU()

		self.dropout = nn.Dropout(config.cnn_drop_prob)

		self.bili = torch.nn.Bilinear(self.out_channels+config.dis_size, self.out_channels+config.dis_size, config.relation_num)
		self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)
  
		if cfg.SGRAPH.LAYERS_NUM > 0 or cfg.DGRAPH.LAYERS_NUM > 0:
			self.ra_enc = RaEncoder(config.relation_num - 1, self.out_channels, self.out_channels, layers_num=cfg.RA_LAYERS)
   
		if cfg.SGRAPH.LAYERS_NUM:
			self.sgnn = GraphNetWork(self.out_channels, self.out_channels,
                                    hidden_dim=self.out_channels, 
                                    output_dim=self.out_channels,
                                    rel_names=['e_rlh', 'e_rlt', 'e_hlr', 'e_tlr'],
                                    num_layers=cfg.SGRAPH.LAYERS_NUM)
   
		if cfg.DGRAPH.LAYERS_NUM:
			self.dgnn = DoubleGraphNetWork(self.out_channels, self.out_channels,
                                    hidden_dim=self.out_channels, 
                                    output_dim=self.out_channels,
                                    rel_names_ht=['e_rlh', 'e_tlr'], 
                                    rel_names_th=['e_rlt', 'e_hlr'],
                                    num_layers=cfg.DGRAPH.LAYERS_NUM)

		self.ada_loss = None
   
	def forward(self, context_idxs, pos, context_ner, context_lens, h_mapping, t_mapping, relation_mask, dis_h_2_t, dis_t_2_h,
				e_mapping=None, entites_num=None, hts=None, hts_len=None, graph_big=None, graph_ht=None, graph_th=None
				):
		sent = self.word_emb(context_idxs)
		sent_pos = self.coref_embed(pos)
		sent_ner = self.ner_emb(context_ner)
		sent = torch.cat([sent, sent_pos, sent_ner], dim=-1)
		sent = sent.permute(0, 2, 1)
  
  		# batch * embedding_size * max_len
		x = self.cnn_1(sent)
		x = self.max_pooling(x)
		x = self.relu(x)
		x = self.dropout(x)

		x = self.cnn_2(x)
		x = self.max_pooling(x)
		x = self.relu(x)
		x = self.dropout(x)

		x = self.cnn_3(x)
		x = self.max_pooling(x)
		x = self.relu(x)
		x = self.dropout(x)

		context_output = x.permute(0, 2, 1)
  
		if cfg.SGRAPH.LAYERS_NUM == 0 and cfg.DGRAPH.LAYERS_NUM == 0:
			start_re_output = torch.matmul(h_mapping, context_output)
			end_re_output = torch.matmul(t_mapping, context_output)

		if cfg.SGRAPH.LAYERS_NUM > 0 or cfg.DGRAPH.LAYERS_NUM > 0:
			rel_feat = self.ra_enc(context_output)
			entities_emb = torch.matmul(e_mapping, context_output)

		if cfg.SGRAPH.LAYERS_NUM > 0:
			hs, ts, self.ada_loss = self.sgnn(graph_big, rel_feat, batch_entities(entities_emb, entites_num))
			hs = unbatch_entities(hs, entites_num) 
			ts = unbatch_entities(ts, entites_num)
			start_re_output, end_re_output = get_ht(hs, ts, entites_num, hts, hts_len)
   
		if cfg.DGRAPH.LAYERS_NUM > 0:
			hs, ts, self.ada_loss = self.dgnn(graph_ht, graph_th, rel_feat, batch_entities(entities_emb, entites_num))
			hs = unbatch_entities(hs, entites_num) 
			ts = unbatch_entities(ts, entites_num)
			hs, ts = get_ht(hs, ts, entites_num, hts, hts_len)
			start_re_output, end_re_output = get_ht(hs, ts, entites_num, hts, hts_len)
   
		s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
		t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)

		predict_re = self.bili(s_rep, t_rep)

		return predict_re


def batch_entities(entities, e_nums):
	"""
 		将多个batch的实体表示拼接成(n, d)维的向量
	"""
	entities = [entities[i, :tmp_len] for i, tmp_len in enumerate(e_nums)]
	return torch.cat(entities, dim=0)


def unbatch_entities(entities, e_nums):
	"""
 		将实体表示分批转为(b, ne, d)维度的向量
	"""
	max_len = torch.tensor(e_nums).max().item()
	e_nums = torch.tensor([0] + e_nums).cumsum(0)
	entities_ls = []
	for i in range(len(e_nums) - 1):
		tmp_e = entities[e_nums[i]:e_nums[i+1]]
		padding_size = [max_len - tmp_e.size(0), tmp_e.size(-1)]
		entities_ls.append(
			torch.cat([tmp_e, torch.zeros(*padding_size).to(tmp_e)], dim=0) 
		)
	return torch.stack(entities_ls)

def get_ht(he_emb, te_emb, entities_num, hts, hts_len):
	'''
	he_emb: [bs, ne, hidden_dim]
	te_emb: [bs, ne, hidden_dim]
	'''
	assert he_emb.size(0) == te_emb.size(0)
	max_hts_len = torch.tensor(hts_len).max().item()
	hts_len = torch.tensor([0] + hts_len).cumsum(0)
	hts = [hts[hts_len[i] : hts_len[i+1]] for i in range(len(hts_len) - 1)]
	h_ls, t_ls = [], []
	for bi in range(he_emb.size(0)):
		tmp_hts = torch.tensor(hts[bi]).to(he_emb.device)
		assert tmp_hts.max() == entities_num[bi] - 1
		tmp_h = he_emb[bi][tmp_hts[:, 0]]
		tmp_t = te_emb[bi][tmp_hts[:, 1]]
		padding_size = [max_hts_len - tmp_h.size(0), tmp_h.size(1)]
		tmp_h = torch.cat([tmp_h, torch.zeros(*padding_size).to(tmp_h)], dim=0)
		tmp_t = torch.cat([tmp_t, torch.zeros(*padding_size).to(tmp_t)], dim=0)
		h_ls.append(tmp_h)
		t_ls.append(tmp_t)
	h_emb = torch.stack(h_ls)
	t_emb = torch.stack(t_ls)
	return h_emb, t_emb
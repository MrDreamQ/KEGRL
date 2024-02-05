import torch
import torch.nn as nn
from torch.nn.utils import rnn
import torch.nn.functional as F
from torch.autograd import Variable

from .graph_modules import RaEncoder, GraphNetWork, DoubleGraphNetWork
from config.g_config import cfg

class LSTM(nn.Module):
	def __init__(self, config):
		super(LSTM, self).__init__()
		self.config = config
		word_vec_size = config.data_word_vec.shape[0]
		self.word_emb = nn.Embedding(word_vec_size, config.data_word_vec.shape[1])
		self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))
		self.word_emb.weight.requires_grad = False  
		self.coref_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)
		self.ner_emb = nn.Embedding(config.ner_size, config.entity_type_size, padding_idx=0)
		hidden_size = 128
		self.hidden_dim = cfg.HIDDEN_DIM 

		input_size = config.data_word_vec.shape[1] + config.coref_size + config.entity_type_size #+ char_hidden

		self.rnn = EncoderLSTM(input_size, hidden_size, 1, True, False, 1 - config.keep_prob, False)
		self.linear_re = nn.Linear(hidden_size, hidden_size)  # *4 for 2layer
		self.bili = torch.nn.Bilinear(hidden_size+config.dis_size, hidden_size+config.dis_size, config.relation_num)
		self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)

		if cfg.SGRAPH.LAYERS_NUM > 0 or cfg.DGRAPH.LAYERS_NUM > 0:
			self.ra_enc = RaEncoder(config.relation_num - 1, hidden_size, hidden_size, layers_num=cfg.RA_LAYERS)
   
		if cfg.SGRAPH.LAYERS_NUM:
			self.sgnn = GraphNetWork(hidden_size, hidden_size,
                                    hidden_dim=self.hidden_dim, 
                                    output_dim=hidden_size,
                                    rel_names=['e_rlh', 'e_rlt', 'e_hlr', 'e_tlr'],
                                    num_layers=cfg.SGRAPH.LAYERS_NUM)
   
		if cfg.DGRAPH.LAYERS_NUM:
			self.dgnn = DoubleGraphNetWork(hidden_size, hidden_size,
                                    hidden_dim=self.hidden_dim, 
                                    output_dim=hidden_size,
                                    rel_names_ht=['e_rlh', 'e_tlr'], 
                                    rel_names_th=['e_rlt', 'e_hlr'],
                                    num_layers=cfg.DGRAPH.LAYERS_NUM)

		self.ada_loss = None

	def forward(self, context_idxs, pos, context_ner, context_lens, h_mapping, t_mapping, relation_mask, dis_h_2_t, dis_t_2_h,
				e_mapping=None, entites_num=None, hts=None, hts_len=None, graph_big=None, graph_ht=None, graph_th=None
				):
		sent = torch.cat([self.word_emb(context_idxs) , self.coref_embed(pos), self.ner_emb(context_ner)], dim=-1)
		context_output = self.rnn(sent, context_lens)
		context_output = torch.relu(self.linear_re(context_output))
     
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

class EncoderLSTM(nn.Module):
	def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
		super().__init__()
		self.rnns = []
		for i in range(nlayers):
			if i == 0:
				input_size_ = input_size
				output_size_ = num_units
			else:
				input_size_ = num_units if not bidir else num_units * 2
				output_size_ = num_units
			self.rnns.append(nn.LSTM(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
		self.rnns = nn.ModuleList(self.rnns)

		self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
		self.init_c = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])

		self.dropout = LockedDropout(dropout)
		self.concat = concat
		self.nlayers = nlayers
		self.return_last = return_last

		# self.reset_parameters()

	def reset_parameters(self):
		for rnn in self.rnns:
			for name, p in rnn.named_parameters():
				if 'weight' in name:
					p.data.normal_(std=0.1)
				else:
					p.data.zero_()

	def get_init(self, bsz, i):
		return self.init_hidden[i].expand(-1, bsz, -1).contiguous(), self.init_c[i].expand(-1, bsz, -1).contiguous()

	def forward(self, input, input_lengths=None):
		bsz, slen = input.size(0), input.size(1)
		output = input
		outputs = []
		if input_lengths is not None:
			lens = input_lengths.data.cpu().numpy()

		for i in range(self.nlayers):
			hidden, c = self.get_init(bsz, i)

			output = self.dropout(output)
			if input_lengths is not None:
				output = rnn.pack_padded_sequence(output, lens, batch_first=True, enforce_sorted=False)

			output, hidden = self.rnns[i](output, (hidden, c))


			if input_lengths is not None:
				output, _ = rnn.pad_packed_sequence(output, batch_first=True)
				if output.size(1) < slen: # used for parallel
					padding = Variable(output.data.new(1, 1, 1).zero_())
					output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
			if self.return_last:
				outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
			else:
				outputs.append(output)
		if self.concat:
			return torch.cat(outputs, dim=2)
		return outputs[-1]


class LockedDropout(nn.Module):
	def __init__(self, dropout):
		super().__init__()
		self.dropout = dropout

	def forward(self, x):
		dropout = self.dropout
		if not self.training:
			return x
		m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
		mask = Variable(m.div_(1 - dropout), requires_grad=False)
		mask = mask.expand_as(x)
		return mask * x

class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        att = att - 1e30 * (1 - mask[:, None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input * output_one, output_two * output_one], dim=-1)
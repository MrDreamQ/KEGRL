import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from models.encoder import Encoder
from models.attention import SelfAttention
from models.reasoner import DynamicReasoner
from models.reasoner import StructInduction

from models.graph_modules import RaEncoder, GraphNetWork, DoubleGraphNetWork

SGRAPH_HIDDEN_DIM = 512
DGRAPH_HIDDEN_DIM = 512

class LSR(nn.Module):
    def __init__(self, config):
        super(LSR, self).__init__()
        self.config = config
        
        self.finetune_emb = config.finetune_emb

        self.word_emb = nn.Embedding(config.data_word_vec.shape[0], config.data_word_vec.shape[1])
        self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))
        if not self.finetune_emb:
            self.word_emb.weight.requires_grad = False

        self.ner_emb = nn.Embedding(13 if config.dataset == 'docred' else 20, config.entity_type_size, padding_idx=0)

        self.coref_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)

        hidden_size = config.rnn_hidden
        input_size = config.data_word_vec.shape[1] + config.coref_size + config.entity_type_size #+ char_hidden
        
        self.linear_re = nn.Linear(hidden_size * 2,  hidden_size)

        self.linear_sent = nn.Linear(hidden_size * 2,  hidden_size)

        self.bili = torch.nn.Bilinear(hidden_size, hidden_size, hidden_size)

        self.self_att = SelfAttention(hidden_size)

        self.bili = torch.nn.Bilinear(hidden_size+config.dis_size,  hidden_size+config.dis_size, hidden_size)
        self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)

        self.linear_output = nn.Linear(2 * hidden_size, config.relation_num)

        self.relu = nn.ReLU()

        self.dropout_rate = nn.Dropout(config.dropout_rate)

        self.rnn_sent = Encoder(input_size, hidden_size, config.dropout_emb, config.dropout_rate)
        self.hidden_size = hidden_size

        self.use_struct_att = config.use_struct_att
        if  self.use_struct_att == True:
            self.structInduction = StructInduction(hidden_size // 2, hidden_size, True)

        self.dropout_gcn = nn.Dropout(config.dropout_gcn)
        self.reasoner_layer_first = config.reasoner_layer_first
        self.reasoner_layer_second = config.reasoner_layer_second
        self.use_reasoning_block = config.use_reasoning_block
        if self.use_reasoning_block:
            self.reasoner = nn.ModuleList()
            self.reasoner.append(DynamicReasoner(hidden_size, self.reasoner_layer_first, self.dropout_gcn))
            self.reasoner.append(DynamicReasoner(hidden_size, self.reasoner_layer_second, self.dropout_gcn))

        self.rnn_output_size = 2 * hidden_size
        
        if  config.sgraph or config.dgraph:
            self.ra_enc = RaEncoder(config.relation_num - 1, self.rnn_output_size, self.rnn_output_size, layers_num=1)

        if config.sgraph:
            print(f"SGRAPH HIDDEN DIM: {SGRAPH_HIDDEN_DIM}")
            self.sgnn = GraphNetWork(self.rnn_output_size, self.rnn_output_size, 
                                    hidden_dim=SGRAPH_HIDDEN_DIM, output_dim=self.rnn_output_size,
                                    rel_names=['e_rlh', 'e_rlt', 'e_hlr', 'e_tlr'],
                                    num_layers=config.sgraph, )
        
        if config.dgraph:
            print(f"DGRAPH HIDDEN DIM: {DGRAPH_HIDDEN_DIM}")
            self.dgnn = DoubleGraphNetWork(self.rnn_output_size, self.rnn_output_size, 
                                    hidden_dim=DGRAPH_HIDDEN_DIM,
                                    output_dim=self.rnn_output_size,
                                    rel_names_ht=['e_rlh', 'e_tlr'], 
                                    rel_names_th=['e_rlt', 'e_hlr'],
                                    num_layers=config.dgraph,)

        if self.config.sgraph or self.config.dgraph:
            self.entity_ensemble = nn.Linear(2 * self.rnn_output_size, self.rnn_output_size)
            self.head_proj = nn.Linear(hidden_size * 2, hidden_size)
            self.tail_proj = nn.Linear(hidden_size * 2, hidden_size)
            self.kegrl_proj = nn.Linear(hidden_size, hidden_size)
            self.lsr_proj = nn.Linear(hidden_size, hidden_size)
            
        self.ada_loss = None

    def doc_encoder(self, input_sent, context_seg):
        """
        :param sent: sent emb
        :param context_seg: segmentation mask for sentences in a document
        :return:
        """
        batch_size = context_seg.shape[0]
        docs_emb = [] # sentence embedding
        docs_len = []
        sents_emb = []

        for batch_no in range(batch_size):
            sent_list = []
            sent_lens = []
            sent_index = ((context_seg[batch_no] == 1).nonzero()).squeeze(-1).tolist() # array of start point for sentences in a document
            pre_index = 0
            for i, index in enumerate(sent_index):
                if i != 0:
                    if i == 1:
                        sent_list.append(input_sent[batch_no][pre_index:index+1])
                        sent_lens.append(index - pre_index + 1)
                    else:
                        sent_list.append(input_sent[batch_no][pre_index+1:index+1])
                        sent_lens.append(index - pre_index)
                pre_index = index

            sents = pad_sequence(sent_list).permute(1,0,2)
            sent_lens_t = torch.LongTensor(sent_lens).cuda()
            docs_len.append(sent_lens)
            sents_output, sent_emb = self.rnn_sent(sents, sent_lens_t) # sentence embeddings for a document.

            doc_emb = None
            for i, (sen_len, emb) in enumerate(zip(sent_lens, sents_output)):
                if i == 0:
                    doc_emb = emb[:sen_len]
                else:
                    doc_emb = torch.cat([doc_emb, emb[:sen_len]], dim = 0)

            docs_emb.append(doc_emb)
            sents_emb.append(sent_emb.squeeze(1))

        docs_emb = pad_sequence(docs_emb).permute(1,0,2) # B * # sentence * Dimention
        sents_emb = pad_sequence(sents_emb).permute(1,0,2)

        return docs_emb, sents_emb


    def forward(self, context_idxs, pos, context_ner, h_mapping, t_mapping,
                relation_mask, dis_h_2_t, dis_t_2_h, context_seg, mention_node_position, entity_position,
                mention_node_sent_num, all_node_num, entity_num_list, sdp_pos, sdp_num_list, 
                e_mapping=None, hts=None, hts_len=None, graph_big=None, graph_ht=None, graph_th=None):
        '''===========STEP1: Encode the document============='''
        sent_emb = torch.cat([self.word_emb(context_idxs), self.coref_embed(pos), self.ner_emb(context_ner)], dim=-1)
        docs_rep, sents_rep = self.doc_encoder(sent_emb, context_seg)  # doc_rep: [b, max_doc_len, 2 * hidden_dim]
        max_doc_len = docs_rep.shape[1]

        '''===========Insert KEGRL Layer============='''
        if self.config.sgraph or self.config.dgraph:
            entities_emb = torch.matmul(e_mapping, docs_rep)
            rel_feat = self.ra_enc(docs_rep)
            
        if self.config.sgraph:
            hs, ts, self.ada_loss = self.sgnn(graph_big, rel_feat, batch_entities(entities_emb, entity_num_list.long().tolist()))
            hs = unbatch_entities(hs, entity_num_list.long().tolist()) 
            ts = unbatch_entities(ts, entity_num_list.long().tolist())
        
        if self.config.dgraph:
            hs, ts, self.ada_loss = self.dgnn(graph_ht, graph_th, rel_feat, batch_entities(entities_emb, entity_num_list.long().tolist()))
            hs = unbatch_entities(hs, entity_num_list.long().tolist()) 
            ts = unbatch_entities(ts, entity_num_list.long().tolist())
        
        if self.config.sgraph or self.config.dgraph and self.config.dataset != 're':
            e_mapping = e_mapping.permute(0, 2, 1)
            max_entity_num = int(entity_num_list.max())
            entity_rep = self.entity_ensemble(torch.cat([hs, ts], dim=-1))
            output = torch.bmm(e_mapping[:, :max_doc_len, :max_entity_num], entity_rep[:, :max_entity_num])
            docs_rep = torch.add(docs_rep, output)
        elif self.config.sgraph or self.config.dgraph and self.config.dataset == 're':
            hs, ts = get_ht(hs, ts, entity_num_list.long().tolist(), hts, hts_len)
            hs = self.head_proj(hs)
            ts = self.tail_proj(ts)
        
        context_output = self.dropout_rate(torch.relu(self.linear_re(docs_rep)))

        '''===========STEP2: Extract all node reps of a document graph============='''
        '''extract Mention node representations'''
        mention_num_list = torch.sum(mention_node_sent_num, dim=1).long().tolist()  # 每句的提及之和    node_sent_num
        max_mention_num = max(mention_num_list)
        mentions_rep = torch.bmm(mention_node_position[:, :max_mention_num, :max_doc_len], context_output) # mentions rep   node_position
        '''extract MDP(meta dependency paths) node representations'''
        sdp_num_list = sdp_num_list.long().tolist()
        max_sdp_num = max(sdp_num_list)
        sdp_rep = torch.bmm(sdp_pos[:,:max_sdp_num, :max_doc_len], context_output)
        '''extract Entity node representations'''
        entity_rep = torch.bmm(entity_position[:,:,:max_doc_len], context_output)
        '''concatenate all nodes of an instance'''
        gcn_inputs = []
        all_node_num_batch = []
        for batch_no, (m_n, e_n, s_n) in enumerate(zip(mention_num_list, entity_num_list.long().tolist(), sdp_num_list)):
            m_rep = mentions_rep[batch_no][:m_n]
            e_rep = entity_rep[batch_no][:e_n]
            s_rep = sdp_rep[batch_no][:s_n]
            gcn_inputs.append(torch.cat((m_rep, e_rep, s_rep),dim=0))   # cat mention node, entity node, mdp node 
            node_num = m_n + e_n + s_n
            all_node_num_batch.append(node_num)

        gcn_inputs = pad_sequence(gcn_inputs).permute(1, 0, 2)
        output = gcn_inputs

        '''===========STEP3: Induce the Latent Structure============='''
        if self.use_reasoning_block:
            for i in range(len(self.reasoner)):
                output = self.reasoner[i](output)

        elif self.use_struct_att:
            gcn_inputs, _ = self.structInduction(gcn_inputs)
            max_all_node_num = torch.max(all_node_num).item()
            assert (gcn_inputs.shape[1] == max_all_node_num)

        mention_node_position = mention_node_position.permute(0, 2, 1)
        output = torch.bmm(mention_node_position[:, :max_doc_len, :max_mention_num], output[:, :max_mention_num])
        context_output = torch.add(context_output, output)

        if self.config.agraph or self.config.dagraph and self.config.dataset == 're':
            start_re_output = self.lsr_proj(start_re_output) + self.kegrl_proj(hs)
            end_re_output = self.lsr_proj(end_re_output) + self.kegrl_proj(ts)

        start_re_output = torch.matmul(h_mapping[:, :, :max_doc_len], context_output) # aggregation
        end_re_output = torch.matmul(t_mapping[:, :, :max_doc_len], context_output) # aggregation

        s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
        t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)

        re_rep = self.dropout_rate(self.relu(self.bili(s_rep, t_rep)))

        re_rep = self.self_att(re_rep, re_rep, relation_mask)
        return self.linear_output(re_rep)


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
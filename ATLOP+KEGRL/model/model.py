import torch
import torch.nn as nn
from opt_einsum import contract

from model.losses import ATLoss
from model.long_seq import process_long_input
from model.graph_modules import RaEncoder, GraphNetWork, DoubleGraphNetWork, BiLinear
from config import cfg

class DocREModel(nn.Module):
    def __init__(self, config, model, emb_size=768, block_size=64, num_labels=-1):
        super().__init__()
        self.config = config
        self.model = model
        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()
        tmp_input_dim = max((1 + 
                             int(cfg.SGRAPH.LAYERS_NUM > 0) + 
                             int(cfg.DGRAPH.LAYERS_NUM > 0)), 
                            2) * config.hidden_size
        self.head_extractor = nn.Linear(tmp_input_dim, emb_size)
        self.tail_extractor = nn.Linear(tmp_input_dim, emb_size)
        self.bilinear = BiLinear(emb_size, block_size, config.num_labels)
        
        self.hidden_dim = cfg.HIDDEN_DIM
        if  cfg.SGRAPH.LAYERS_NUM > 0 or cfg.DGRAPH.LAYERS_NUM > 0:
            self.ra_enc = RaEncoder(config.num_labels - 1, emb_size, self.config.hidden_size, layers_num=cfg.RA_LAYERS)
        else:
            self.ra_enc = None
            
        if cfg.SGRAPH.LAYERS_NUM:
            self.sgnn = GraphNetWork(config.hidden_size, config.hidden_size,
                                     hidden_dim=self.hidden_dim, output_dim=config.hidden_size, 
                                     rel_names=['e_rlh', 'e_rlt', 'e_hlr', 'e_tlr'], 
                                     num_layers=cfg.SGRAPH.LAYERS_NUM)
        else:
            self.sgnn = None
        
        if cfg.DGRAPH.LAYERS_NUM:
            self.dgnn = DoubleGraphNetWork(config.hidden_size, config.hidden_size,
                                    hidden_dim=self.hidden_dim, output_dim=config.hidden_size, 
                                    rel_names_ht=['e_rlh', 'e_tlr'], rel_names_th=['e_rlt', 'e_hlr'],
                                    num_layers=cfg.DGRAPH.LAYERS_NUM)
        else:
            self.dgnn = None
        
        self.ada_loss = None


    def forward(self, 
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                graph_big=None, 
                graph_ht=None, 
                graph_th=None, 
                end_pos=None,
                ):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts, all_entity_embs, entity_nums, entity_mask = \
            self.get_hrt(sequence_output, attention, entity_pos, hts, end_pos)
            
        if cfg.SGRAPH.LAYERS_NUM > 0 or cfg.DGRAPH.LAYERS_NUM > 0:
            hs_ls, ts_ls = [], []
        else:
            hs_ls, ts_ls = [hs, ], [ts, ]
            
        if cfg.SGRAPH.LAYERS_NUM > 0 or cfg.DGRAPH.LAYERS_NUM > 0:
            rel_feat = self.ra_enc(sequence_output)
        
        if cfg.SGRAPH.LAYERS_NUM > 0:
            hs, ts, self.ada_loss = self.sgnn(graph_big, rel_feat, all_entity_embs)
            sg_hs, sg_ts = self.select_ht(hs, ts, hts, entity_nums)
            hs_ls.append(sg_hs)
            ts_ls.append(sg_ts)
            
        if cfg.DGRAPH.LAYERS_NUM > 0:
            hs, ts, self.ada_loss = self.dgnn(graph_ht, graph_th, rel_feat, all_entity_embs)
            dg_hs, dg_ts = self.select_ht(hs, ts, hts, entity_nums)
            hs_ls.append(dg_hs)
            ts_ls.append(dg_ts)
        
        hs_ls.append(rs)
        ts_ls.append(rs)
        hs = torch.cat(hs_ls, dim=1)
        ts = torch.cat(ts_ls, dim=1)
        hs = torch.tanh(self.head_extractor(hs))
        ts = torch.tanh(self.tail_extractor(ts))
        logits = self.bilinear(hs, ts)
        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)
        
        if (labels is not None):
            assert self.training
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss_re = self.loss_fnt(logits.float(), labels.float(), cfg.DATA.DATASET == 'DWIE', entity_mask)
            loss = loss_re if self.ada_loss is None else loss_re + self.ada_loss * cfg.ADA_LOSS_LAMB
            
            return loss
        
        return output

    def encode(self, input_ids, attention_mask):
        if self.config.transformer_type == "bert":
            start_tokens = [self.config.cls_token_id]
            end_tokens = [self.config.sep_token_id]
        elif self.config.transformer_type == "roberta":
            start_tokens = [self.config.cls_token_id]
            end_tokens = [self.config.sep_token_id, self.config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts, end_pos):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        seq_length = 1400
        random_sample_offset = end_pos - seq_length if end_pos is not None else 0
        c_start = random_sample_offset
        c_end = end_pos if end_pos is not None else c
        
        hss, tss, rss = [], [], []
        all_entity_embs, entity_num = [], []
        entity_mask = []
        for i in range(len(entity_pos)):    # batch
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]: # entity
                if len(e) > 1:  # have more than 1 mention
                    e_emb, e_att = [], []
                    for start, end in e:    # mention
                        if start + offset < c_end and start + offset >= c_start:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset - random_sample_offset])
                            e_att.append(attention[i, :, start + offset - random_sample_offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                        entity_mask.append(True)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                        entity_mask.append(False)
                else:      # have 1 mention
                    start, end = e[0]
                    if start + offset < c_end and start + offset >= c_start:
                        e_emb = sequence_output[i, start + offset - random_sample_offset]
                        e_att = attention[i, :, start + offset - random_sample_offset]
                        entity_mask.append(True)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                        entity_mask.append(False)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            all_entity_embs.append(entity_embs)
            entity_num.append(len(entity_embs))

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        entity_num = torch.tensor(entity_num)
        all_entity_embs = torch.cat(all_entity_embs, dim=0)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss, all_entity_embs, entity_num, entity_mask
    
    def select_ht(self, input_h, input_t, hts, entity_nums):
        entity_nums = torch.cat([torch.tensor([0]), 
                                 torch.tensor(entity_nums).cumsum(dim=-1)], 
                                 dim=0)
        hs, ts = [], []
        for i, ht_i in enumerate(hts):
            ht_i = torch.LongTensor(ht_i).to(input_h.device)
            tmp_h = input_h[entity_nums[i]:entity_nums[i+1]]
            tmp_t = input_t[entity_nums[i]:entity_nums[i+1]]
            tmp_hs = torch.index_select(tmp_h, 0, ht_i[:, 0])
            tmp_ts = torch.index_select(tmp_t, 0, ht_i[:, 1])
            hs.append(tmp_hs)
            ts.append(tmp_ts)
        return torch.cat(hs, dim=0), torch.cat(ts, dim=0)

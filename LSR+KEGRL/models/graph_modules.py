import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn

ABLATION = ''
print(f"ABLATION: {ABLATION}", flush=True)

class RaEncoder(nn.Module):
    def __init__(self, num_classes, input_size, lm_size, layers_num):
        super().__init__()
        self.num_classes = num_classes
        self.layers_num = layers_num
        self.input_size = input_size
        self.lm_size = lm_size
        self.relation_embed = nn.Embedding(num_classes, lm_size)
        if layers_num > 0:
            self.q_proj = nn.Linear(lm_size, lm_size)
            self.k_proj = nn.Linear(lm_size, lm_size)
            self.v_proj = nn.Linear(lm_size, lm_size)
        else:
            self.q_proj = self.k_proj = self.v_proj = None

    def forward(self, seq_embed):
        n, _, _ = seq_embed.shape
        temp_idx = torch.arange(self.num_classes).to(seq_embed.device)
        rel_emb = self.relation_embed(temp_idx)
        if self.layers_num == 0:
            return torch.relu(rel_emb.unsqueeze(0).expand(n, -1, -1))
        else:
            scaling = float(self.lm_size) ** -0.5
            rel_q = self.q_proj(rel_emb) * scaling
            seq_k = self.k_proj(seq_embed)
            seq_v = self.v_proj(seq_embed)
            rel_attn = torch.bmm(rel_q.unsqueeze(0).expand(n, -1, -1), 
                                seq_k.permute(0, 2, 1))
            rel_attn = F.softmax(rel_attn, dim=-1)  # TODO render attention and token 
            rel_feat = torch.bmm(rel_attn, seq_v)
            return torch.relu(rel_feat + rel_emb.unsqueeze(0).expand(n, -1, -1))


class BiLinear(nn.Module):
    def __init__(self, emb_size, block_size, out_size):
        super().__init__()
        self.emb_size = emb_size
        self.block_size = block_size
        self.fc = nn.Linear(emb_size * block_size, out_size)

    def forward(self, x, y):
        b1 = x.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = y.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        return self.fc(bl)


class GCNLayer(nn.Module):
    def __init__(self, rel_names, input_dim, output_dim, bias=True, self_loop=True):
        super().__init__()
        self.bias = bias
        self.self_loop = self_loop
        self.rel_names = rel_names
        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(input_dim, output_dim, norm='right', weight=False, bias=False)
            for rel in rel_names
        })
        self.weight = nn.Parameter(torch.Tensor(len(rel_names), input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(output_dim))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def forward(self, graphs, input_feat_dict, mod_kwargs=None):
        if mod_kwargs is None:
            mod_kwargs = {r_name: {} for r_name in self.rel_names}
        with graphs.local_scope():
            graphs = dgl.edge_type_subgraph(graphs, self.rel_names)
            for i in range(len(self.rel_names)):
                mod_kwargs[self.rel_names[i]]['weight'] = self.weight[i]
            ret = self.conv(graphs, input_feat_dict, mod_kwargs=mod_kwargs)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(input_feat_dict[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)
        
        return {ntype: _apply(ntype, h) for ntype, h in ret.items()}


class GraphNetWork(nn.Module):
    def __init__(self, rel_indim, entity_indim, rel_names, hidden_dim=1024, output_dim=128, self_loop=True, num_layers=2, adaptive=True):
        super().__init__()
        self.ada_loss_w = 0.05
        self.self_loop = self_loop
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rel_names = rel_names
        self.num_layers = num_layers
        self.adaptive = adaptive
        self.rel_proj = nn.Sequential(nn.Linear(rel_indim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Dropout(0.1),
                                      nn.LayerNorm(hidden_dim)) 
        self.entity_proj = nn.Sequential(nn.Linear(entity_indim, 2 * hidden_dim),
                                         nn.ReLU(),
                                         nn.Dropout(0.1),
                                         nn.LayerNorm(2 * hidden_dim))
        self.conv = nn.ModuleList([GCNLayer(rel_names, hidden_dim, hidden_dim, 
                                            bias=True, self_loop=True) 
                                    for i in range(num_layers)])
        self.h_out = nn.Linear(hidden_dim * (num_layers + 1), self.output_dim)
        self.t_out = nn.Linear(hidden_dim * (num_layers + 1), self.output_dim)
        if adaptive:
            self.weight_proj = nn.ModuleDict({key: nn.ModuleList([nn.Sequential(nn.Linear(2 * hidden_dim, 128),
                                                                                    nn.ReLU(),
                                                                                    nn.Linear(128, 1),
                                                                                    nn.Sigmoid())
                                                                    for _ in range(num_layers)])
                                              for key in rel_names})
        else:
            self.weight_proj = None

    def forward(self, graphs, rel_embeds, entity_embeds):
        rel_embeds = self.rel_proj(rel_embeds)
        entity_embeds = self.entity_proj(entity_embeds)
        h_embed, t_embed = torch.split(entity_embeds, [self.hidden_dim] * 2, dim=-1)
        feat_dict = {'n_eh': h_embed, 
                     'n_et': t_embed, 
                     'n_rel': rel_embeds.view(-1, rel_embeds.size(-1))}
        feat_ls = [feat_dict, ]
        d_weight_ls = []
        for layer_id, conv in enumerate(self.conv):
            with graphs.local_scope():
                if self.adaptive:
                    d_weight = self.cal_weight(layer_id, feat_dict, graphs)
                    d_weight_ls.append(d_weight)
                    wdict = {tmp_r: {'edge_weight': graphs.edges[tmp_r].data['w'] + d_weight[tmp_r].squeeze(dim=1)}
                             for tmp_r in self.rel_names}  
                    entity_rel_names = ['e_rlh', 'e_rlt']
                    relation_rel_names = ['e_hlr', 'e_tlr']
                    if ABLATION == 'entity':
                        for e in entity_rel_names:
                            wdict[e]['edge_weight'] = torch.ones_like(wdict[e]['edge_weight']) / 96. + d_weight[e].squeeze(dim=1)
                    elif ABLATION == 'relation':
                        for r in relation_rel_names:
                            wdict[r]['edge_weight'] = torch.ones_like(wdict[r]['edge_weight']) / 6. + d_weight[r].squeeze(dim=1)
                    elif ABLATION == 'all':
                        for e in entity_rel_names:
                            wdict[e]['edge_weight'] = torch.ones_like(wdict[e]['edge_weight']) / 96. + d_weight[e].squeeze(dim=1)
                        for r in relation_rel_names:
                            wdict[r]['edge_weight'] = torch.ones_like(wdict[r]['edge_weight']) / 6. + d_weight[r].squeeze(dim=1)
                else:
                    wdict = {tmp_r: {'edge_weight': graphs.edges[tmp_r].data['w']}
                             for tmp_r in self.rel_names}
                    
                feat_dict = conv(graphs, feat_dict, mod_kwargs=wdict)
                feat_ls.append(feat_dict)
        n_eh = self.h_out(torch.cat([tmp_feat['n_eh'] for tmp_feat in feat_ls], dim=-1))
        n_et = self.t_out(torch.cat([tmp_feat['n_et'] for tmp_feat in feat_ls], dim=-1))
        if self.adaptive:
            ada_loss = self.cal_ada_weight_loss(d_weight_ls) * self.ada_loss_w
        else:
            ada_loss = None
        return n_eh, n_et, ada_loss
    
    def cal_ada_weight_loss(self, d_weight_ls):
        loss = torch.Tensor([0]).to(d_weight_ls[0][self.rel_names[0]])
        for d_weight in d_weight_ls:
            for r_name in self.rel_names:
                loss = loss + (d_weight[r_name].squeeze(1).square().sum(dim=0) / d_weight[r_name].shape[0]) ** 0.5
        loss = loss / len(self.rel_names) / len(d_weight_ls)
        return loss
    
    def cal_weight(self, layer_id, feat_dict, graphs) -> dict:
        def edge_func(etype):
            return lambda edges: {'h': self.weight_proj[etype][layer_id](torch.cat((edges.src['h'], edges.dst['h']), dim=-1))}
        ret_dict = {}
        with graphs.local_scope():
            graphs.nodes['n_eh'].data['h'] = feat_dict['n_eh']
            graphs.nodes['n_et'].data['h'] = feat_dict['n_et']
            graphs.nodes['n_rel'].data['h'] = feat_dict['n_rel']
            for tmp_type in graphs.etypes:
                graphs.apply_edges(edge_func(tmp_type), etype=tmp_type)
                ret_dict[tmp_type] = graphs.edges[tmp_type].data['h']
        return ret_dict


class MetaGraphNetWork(nn.Module):
    def __init__(self, rel_indim, entity_indim, rel_names, hidden_dim, self_loop=True, num_layers=2, adaptive=True):
        super().__init__()
        self.ada_loss_w = 0.05
        self.adaptive = adaptive
        self.self_loop = self_loop
        self.hidden_dim = hidden_dim
        self.rel_names = rel_names
        self.num_layers = num_layers
        self.rel_proj = nn.Sequential(nn.Linear(rel_indim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Dropout(0.1),
                                      nn.LayerNorm(hidden_dim))
        self.entity_proj = nn.Sequential(nn.Linear(entity_indim, hidden_dim),
                                         nn.ReLU(),
                                         nn.Dropout(0.1),
                                         nn.LayerNorm(hidden_dim))
        self.conv = nn.ModuleList([GCNLayer(rel_names, hidden_dim, hidden_dim, 
                                            bias=True, self_loop=True) 
                                    for i in range(num_layers)])
        if adaptive:
            self.weight_proj = nn.ModuleDict({key: nn.ModuleList([nn.Sequential(nn.Linear(2 * hidden_dim, 128),
                                                                                nn.ReLU(),
                                                                                nn.Linear(128, 1),
                                                                                nn.Sigmoid())
                                                                for _ in range(num_layers)])
                                            for key in rel_names})
        else:
            self.weight_proj = None

    def forward(self, graphs, rel_embeds, entity_embeds):
        rel_embeds = self.rel_proj(rel_embeds)
        e_embeds = self.entity_proj(entity_embeds)
        feat_dict = {'n_e': e_embeds, 
                     'n_rel': rel_embeds.view(-1, rel_embeds.size(-1))}
        feat_ls = [feat_dict, ]
        d_weight_ls = []
        for layer_id, conv in enumerate(self.conv):
            with graphs.local_scope():
                if self.adaptive:
                    d_weight = self.cal_weight(layer_id, feat_dict, graphs)
                    d_weight_ls.append(d_weight)
                    wdict = {tmp_r: {'edge_weight': graphs.edges[tmp_r].data['w'] + d_weight[tmp_r].squeeze(dim=1)}
                             for tmp_r in self.rel_names}
                    
                    entity_rel_names = ['e_rlh', 'e_rlt']
                    relation_rel_names = ['e_hlr', 'e_tlr']
                    if ABLATION == 'entity':
                        for e in entity_rel_names:
                            if e in d_weight.keys():
                                wdict[e]['edge_weight'] = torch.ones_like(wdict[e]['edge_weight']) / 96. + d_weight[e].squeeze(dim=1)
                    elif ABLATION == 'relation':
                        for r in relation_rel_names:
                            if r in d_weight.keys():
                                wdict[r]['edge_weight'] = torch.ones_like(wdict[r]['edge_weight']) / 6. + d_weight[r].squeeze(dim=1)
                    elif ABLATION == 'all':
                        for e in entity_rel_names:
                            if e in d_weight.keys():
                                wdict[e]['edge_weight'] = torch.ones_like(wdict[e]['edge_weight']) / 96. + d_weight[e].squeeze(dim=1)
                        for r in relation_rel_names:
                            if r in d_weight.keys():
                                wdict[r]['edge_weight'] = torch.ones_like(wdict[r]['edge_weight']) / 6. + d_weight[r].squeeze(dim=1)
                else:
                    wdict = {tmp_r: {'edge_weight': graphs.edges[tmp_r].data['w']}
                             for tmp_r in self.rel_names}
                
                feat_dict = conv(graphs, feat_dict, mod_kwargs=wdict)
                feat_ls.append(feat_dict)
        if self.adaptive:
            ada_loss = self.cal_ada_weight_loss(d_weight_ls) * self.ada_loss_w
        else:
            ada_loss = None
        return feat_ls, ada_loss
    
    def cal_ada_weight_loss(self, d_weight_ls):
        loss = torch.Tensor([0]).to(d_weight_ls[0][self.rel_names[0]])
        for d_weight in d_weight_ls:
            for r_name in self.rel_names:
                loss = loss + (d_weight[r_name].squeeze(1).square().sum(dim=0) / d_weight[r_name].shape[0]) ** 0.5
        loss = loss / len(self.rel_names) / len(d_weight_ls)
        return loss

    def cal_weight(self, layer_id, feat_dict, graphs) -> dict:
        def edge_func(etype):
            return lambda edges: {'h': self.weight_proj[etype][layer_id](torch.cat((edges.src['h'], edges.dst['h']), dim=-1))}
        ret_dict = {}
        with graphs.local_scope():
            graphs.nodes['n_e'].data['h'] = feat_dict['n_e']
            graphs.nodes['n_rel'].data['h'] = feat_dict['n_rel']
            for tmp_type in graphs.etypes:
                graphs.apply_edges(edge_func(tmp_type), etype=tmp_type)
                ret_dict[tmp_type] = graphs.edges[tmp_type].data['h']
        return ret_dict


class DoubleGraphNetWork(nn.Module):
    def __init__(self, rel_indim, entity_indim, rel_names_ht, rel_names_th, hidden_dim=1024, output_dim=128, self_loop=True, num_layers=2, adaptive=True):
        super().__init__()
        self.meta_graph_ht = MetaGraphNetWork(rel_indim, entity_indim, rel_names_ht, hidden_dim // 2, self_loop, num_layers, adaptive)
        self.meta_graph_th = MetaGraphNetWork(rel_indim, entity_indim, rel_names_th, hidden_dim // 2, self_loop, num_layers, adaptive)
        self.h_out = nn.Linear(hidden_dim * (num_layers + 1), output_dim)
        self.t_out = nn.Linear(hidden_dim * (num_layers + 1), output_dim)

    def forward(self, graph_ht, graph_th, rel_embeds, entity_embeds):
        ht_feats_ls, ht_ada_loss = self.meta_graph_ht(graph_ht, rel_embeds, entity_embeds)
        th_feats_ls, th_ada_loss = self.meta_graph_th(graph_th, rel_embeds, entity_embeds)
        ret = []
        for tmp_ht_feat, tmp_th_feat in zip(ht_feats_ls, th_feats_ls):
            tmp_dict = {}
            for tmp_key in tmp_ht_feat.keys():
                tmp_dict[tmp_key] = torch.cat([tmp_ht_feat[tmp_key], tmp_th_feat[tmp_key]], dim=1)
            ret.append(tmp_dict)
        ne_feat = torch.cat([tmp_feat['n_e'] for tmp_feat in ret], dim=-1)
        n_eh = self.h_out(ne_feat)
        n_et = self.t_out(ne_feat)
        if ht_ada_loss is not None and th_ada_loss is not None:
            ada_loss = (ht_ada_loss + th_ada_loss) / 2
        else:
            ada_loss = None
        return n_eh, n_et, ada_loss

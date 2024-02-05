from tqdm import tqdm
import json
import torch
import numpy as np
from collections import defaultdict
import dgl

from config import cfg

def docred_collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    graph_big = dgl.batch([f['graph_big'] for f in batch])
    graph_ht = dgl.batch([f['graph_ht'] for f in batch])
    graph_th = dgl.batch([f['graph_th'] for f in batch])
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    end_pos = None
    output = (input_ids, input_mask, labels, entity_pos, hts, 
              graph_big, graph_ht, graph_th, end_pos)
    
    return output


def create_graph(sample):
    docred_rel2id = json.load(open(f'../dataset/{cfg.DATA.DATASET}/rel2id.json', 'r'))
    docred_ner2id = json.load(open(f'../dataset/{cfg.DATA.DATASET}/ner2id.json', 'r'))
    train_rely_weight = json.load(open(f'../dataset/{cfg.DATA.DATASET}/train_rely_weight.json', 'r'))
    fq_hlr = np.array(train_rely_weight['p_hlr'])
    fq_tlr = np.array(train_rely_weight['p_tlr'])
    fq_rlh = np.array(train_rely_weight['p_rlh'])
    fq_rlt = np.array(train_rely_weight['p_rlt'])

    g_data = defaultdict(list)
    g_ht_data, g_th_data = defaultdict(list), defaultdict(list)
    entity_ls = sample['vertexSet']
    e_type_id = torch.tensor([docred_ner2id[e[0]['type']] - 1 for e in entity_ls])
    w_rlh = torch.zeros(len(entity_ls) * (len(docred_rel2id) - 1))
    w_rlt = torch.zeros(len(entity_ls) * (len(docred_rel2id) - 1))
    w_hlr = torch.zeros(len(entity_ls) * (len(docred_rel2id) - 1))
    w_tlr = torch.zeros(len(entity_ls) * (len(docred_rel2id) - 1))
    
    edge_id = 0
    for entity_id in range(len(entity_ls)):
        for r_type in range(len(docred_rel2id) - 1):
            e_type = docred_ner2id[entity_ls[entity_id][0]['type']] - 1

            g_ht_data[('n_e', 'e_rlh', 'n_rel')].append((entity_id, r_type))
            g_ht_data[('n_rel', 'e_tlr', 'n_e')].append((r_type, entity_id))

            g_th_data[('n_e', 'e_rlt', 'n_rel')].append((entity_id, r_type))
            g_th_data[('n_rel', 'e_hlr', 'n_e')].append((r_type, entity_id))
            
            g_data[('n_eh', 'e_rlh', 'n_rel')].append((entity_id, r_type))
            g_data[('n_et', 'e_rlt', 'n_rel')].append((entity_id, r_type))
            g_data[('n_rel', 'e_hlr', 'n_eh')].append((r_type, entity_id))
            g_data[('n_rel', 'e_tlr', 'n_et')].append((r_type, entity_id))

            w_hlr[edge_id] = fq_hlr[r_type, e_type]
            w_tlr[edge_id] = fq_tlr[r_type, e_type]
            w_rlh[edge_id] = fq_rlh[e_type, r_type]
            w_rlt[edge_id] = fq_rlt[e_type, r_type]
            
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
        "graph_th": graph_th,
    }


def read_docred(file_in, tokenizer, max_seq_length=1024):
    features = []
    if file_in == "":
        return None
    # read dataset from json file
    with open(file_in, "r") as fh:
        data = json.load(fh)
    print('reading %s' % file_in)
    for sample in tqdm(data, desc="Example"):
        feature = process_one_sample(sample, tokenizer, max_seq_length)
        features.append(feature)

    return features


def process_one_sample(sample, tokenizer, max_seq_length):
    docred_rel2id = json.load(open(f'../dataset/{cfg.DATA.DATASET}/rel2id.json', 'r'))

    graphs = create_graph(sample)
    graph_big = graphs['graph_big']
    graph_ht = graphs['graph_ht']
    graph_th = graphs['graph_th']

    entities = sample['vertexSet']
    entity_start, entity_end = [], []
    for entity in entities:
        for mention in entity:
            sent_id = mention["sent_id"]
            pos = mention["pos"]
            entity_start.append((sent_id, pos[0],))
            entity_end.append((sent_id, pos[1] - 1,))

    sents = []
    sent_map = []
    for i_s, sent in enumerate(sample['sents']):
        new_map = {}
        for i_t, token in enumerate(sent):
            tokens_wordpiece = tokenizer.tokenize(token)
            if (i_s, i_t) in entity_start:
                tokens_wordpiece = ["*"] + tokens_wordpiece
            if (i_s, i_t) in entity_end:
                tokens_wordpiece = tokens_wordpiece + ["*"]
            new_map[i_t] = len(sents)
            sents.extend(tokens_wordpiece)
        
        new_map[i_t + 1] = len(sents)
        
        sent_map.append(new_map)
    
    sent_starts = [x[0] for x in sent_map]
    sentid_mask = torch.zeros( (len(sents) + 2))
    for sent_id in range(len(sent_starts)):
        sentid_mask[sent_starts[sent_id]:] = (sent_id+1)
    
    train_triple = {}
    if "labels" in sample:
        for label in sample['labels']:
            #evidence = label['evidence']
            if label['r'] not in docred_rel2id.keys():
                continue
            r = int(docred_rel2id[label['r']])
            if (label['h'], label['t']) not in train_triple:
                train_triple[(label['h'], label['t'])] = [
                    {'relation': r}]
            else:
                train_triple[(label['h'], label['t'])].append(
                    {'relation': r})

    entity_pos = []
    for e in entities:
        entity_pos.append([])
        for m in e:
            start = sent_map[m["sent_id"]][m["pos"][0]]
            end = sent_map[m["sent_id"]][m["pos"][1]]
            entity_pos[-1].append((start, end,))
        
    relations, hts = [], []
    for h in range(len(entities)):
        for t in range(len(entities)):
            if (h, t) in train_triple.keys():
                relation = [0] * len(docred_rel2id)
                for mention in train_triple[h, t]:
                    relation[mention["relation"]] = 1
                relations.append(relation)
                hts.append([h,t])
            elif (h, t) not in train_triple.keys():
                relation = [1] + [0] * (len(docred_rel2id) - 1)
                relations.append(relation)
                hts.append([h, t])
    assert len(relations) == len(entities) * (len(entities))
    
    sents = sents[:max_seq_length - 2]
    sentid_mask = sentid_mask[:max_seq_length - 2]
    input_ids = tokenizer.convert_tokens_to_ids(sents)
    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

    if 'title' not in sample.keys():
        sample['title'] = ' '.join(sample['sents'][0])
        
    feature = {'input_ids': input_ids,
                'entity_pos': entity_pos,
                'labels': relations,
                'hts': hts,
                'title': sample['title'],
                'graph_big': graph_big,
                'graph_ht': graph_ht,
                'graph_th': graph_th,
                }

    return feature


import os
import json

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns


def cal_edge_weight(split, dataset):
    rel2id = json.load(open(os.path.join('dataset', dataset, 'rel2id.json'), 'r'))
    ner2id = json.load(open(os.path.join('dataset', dataset, 'ner2id.json'), 'r'))
    assert split in ['train', 'dev']
    data_dir = f"dataset/{dataset}"
    file_name = split + \
                ('_annotated' if split == 'train' else '') + \
                '.json'
    if dataset == 'Re-DocRED':
        file_name = split + '_revised.json'
    file_path = os.path.join(data_dir, file_name)
    with open(file_path, "r") as fh:
        data = json.load(fh)
    p_hlr, p_tlr = np.zeros((len(rel2id) - 1, len(ner2id) - 1)), \
                np.zeros((len(rel2id) - 1, len(ner2id) - 1))
    p_rlh, p_rlt = np.zeros((len(ner2id) - 1, len(rel2id) - 1)), \
                np.zeros((len(ner2id) - 1, len(rel2id) - 1))
    for sample in tqdm(data, desc="Example"):
        vertex_set = sample['vertexSet']
        for tmp_label in sample['labels']:
            tmp_h_type_id = ner2id[vertex_set[tmp_label['h']][0]['type']] - 1
            tmp_t_type_id = ner2id[vertex_set[tmp_label['t']][0]['type']] - 1
            tmp_rel_type_id = rel2id[tmp_label['r']] - 1
            p_hlr[tmp_rel_type_id, tmp_h_type_id] += 1
            p_tlr[tmp_rel_type_id, tmp_t_type_id] += 1
            p_rlh[tmp_h_type_id, tmp_rel_type_id] += 1
            p_rlt[tmp_t_type_id, tmp_rel_type_id] += 1
    to_prob = lambda x : x / (np.expand_dims(x.sum(-1), axis=-1) + 1e-12)
    p_hlr = to_prob(p_hlr)
    p_tlr = to_prob(p_tlr)
    p_rlh = to_prob(p_rlh)
    p_rlt = to_prob(p_rlt)
    
    save_dict = {
        "p_hlr": p_hlr.tolist(),
        "p_tlr": p_tlr.tolist(),
        "p_rlh": p_rlh.tolist(),
        "p_rlt": p_rlt.tolist()
    }
    save_path = os.path.join("dataset", dataset, "%s_rely_weight.json" % (split,))
    save_path_dir = os.path.dirname(save_path)
    os.makedirs(save_path_dir, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(save_dict, f)


def show_edge_weight(split, dataset):
    rel2id = json.load(open(os.path.join('dataset', dataset, 'rel2id.json'), 'r'))
    if dataset in ['DocRED', 'Re-DocRED']:
        rel_info = json.load(open(os.path.join('dataset', dataset, 'rel_info.json'), 'r'))
        rel_info['Na'] = 'Na'
        tmp = {rel_info[k]: v for k, v in rel2id.items()}
        rel2id = tmp
    ner2id = json.load(open(os.path.join('dataset', dataset, 'ner2id.json'), 'r'))
    id2ner = {v: k for k, v in ner2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}
    def limit_relation_str(relation_str):
        length_cnt = 0
        rel_str = ''
        max_length = 15
        for str_seg in relation_str.split():
            length_cnt += len(str_seg) + 1
            if length_cnt > max_length:
                rel_str += '...'
                break
            rel_str += str_seg + ' '
        
        return rel_str
        
    def draw_heatmap(input_array: np.ndarray, title):
        if input_array.shape[0] > input_array.shape[1]:
            input_array = input_array.transpose(1, 0)
        plt.title(title).set_fontsize(fontsize=18)
        xtick_ls = [limit_relation_str(id2rel[i]) for i in range(1, len(id2rel))] if dataset == 'DocRED' else \
            [id2rel[i] for i in range(1, len(id2rel))]
        ytick_ls = [id2ner[i] for i in range(1, len(id2ner))]
        sns.heatmap(data=input_array, cmap='rainbow', vmax=1, vmin=0, center=0, annot=False, 
                    square=True, linewidths=0.1, cbar_kws={"shrink":0.7}, 
                    xticklabels=xtick_ls, yticklabels=ytick_ls,)
        plt.xlabel('relation type', fontsize=18)
        plt.ylabel('entity type', fontsize=18)

    assert split in ['train', 'dev']
    file_path = os.path.join("dataset", dataset, "%s_rely_weight.json" % (split,))
    with open(file_path, 'r') as f:
        load_dict = json.load(f)
    p_hlr = np.array(load_dict["p_hlr"])
    p_tlr = np.array(load_dict["p_tlr"])
    p_rlh = np.array(load_dict["p_rlh"])
    p_rlt = np.array(load_dict["p_rlt"])
    
    sns.set(font_scale=1.5)
    plt.figure()
    draw_heatmap(p_hlr, 'P(h|r)')
    plt.gcf().set_size_inches(36, 8) if dataset == 'DocRED' else plt.gcf().set_size_inches(36, 12)
    os.makedirs(f'dataset/{dataset}/edge_weigth_pic', exist_ok=True)
    plt.savefig(f'dataset/{dataset}/edge_weigth_pic/hlr.pdf', dpi=1000)
    plt.clf()
    
    fig = plt.figure()
    draw_heatmap(p_tlr, 'P(t|r)')
    plt.gcf().set_size_inches(36, 8) if dataset == 'DocRED' else plt.gcf().set_size_inches(36, 12)
    os.makedirs(f'dataset/{dataset}/edge_weigth_pic', exist_ok=True)
    plt.savefig(f'dataset/{dataset}/edge_weigth_pic/tlr.pdf', dpi=1000)
    plt.clf()
    
    fig = plt.figure()
    draw_heatmap(p_rlh, 'P(r|h)')
    plt.gcf().set_size_inches(36, 8) if dataset == 'DocRED' else plt.gcf().set_size_inches(36, 12)
    os.makedirs(f'dataset/{dataset}/edge_weigth_pic', exist_ok=True)
    plt.savefig(f'dataset/{dataset}/edge_weigth_pic/rlh.pdf', dpi=1000)
    plt.clf()
    
    fig = plt.figure()
    draw_heatmap(p_rlt, 'P(r|t)')
    plt.gcf().set_size_inches(36, 8) if dataset == 'DocRED' else plt.gcf().set_size_inches(36, 12)
    os.makedirs(f'dataset/{dataset}/edge_weigth_pic', exist_ok=True)
    plt.savefig(f'dataset/{dataset}/edge_weigth_pic/rlt.pdf', dpi=1000)
    plt.clf()


def cal_entropy():
    for dataset in ['DocRED', 'Re-DocRED', 'DWIE']:
        rel2id = json.load(open(f'dataset/{dataset}/rel2id.json'))
        ner2id = json.load(open(f'dataset/{dataset}/ner2id.json'))
        
        relation = np.zeros((len(rel2id)-1))
        entity_type_rlh = np.zeros((len(ner2id)-1))
        entity_type_rlt = np.zeros((len(ner2id)-1))
    
        train_set = json.load(open(f'dataset/{dataset}/train_annotated.json' if dataset != 'Re-DocRED' else f'dataset/{dataset}/train_revised.json'))
        for feature in train_set:
            vertexSet = feature['vertexSet']
            labels = feature['labels']
            for l in labels:
                h, t, r = vertexSet[l['h']][0], vertexSet[l['t']][0], l['r']
                entity_type_rlh[ner2id[h['type']]-1] += 1
                entity_type_rlt[ner2id[t['type']]-1] += 1
                relation[rel2id[r]-1] += 1
        
        train_rely_weight = json.load(open(f'dataset/{dataset}/train_rely_weight.json', 'r'))
        fq_hlr = np.array(train_rely_weight['p_hlr']) + 1E-20   # (r, e)
        fq_tlr = np.array(train_rely_weight['p_tlr']) + 1E-20
        fq_rlh = np.array(train_rely_weight['p_rlh']) + 1E-20   # (e, r)
        fq_rlt = np.array(train_rely_weight['p_rlt']) + 1E-20
        
        E_hlr = ((-np.sum(fq_hlr * np.log(fq_hlr), axis=1)) * relation).sum()
        E_tlr = ((-np.sum(fq_tlr * np.log(fq_tlr), axis=1)) * relation).sum()
        E_rlh = ((-np.sum(fq_rlh * np.log(fq_rlh), axis=1)) * entity_type_rlh).sum()
        E_rlt = ((-np.sum(fq_rlt * np.log(fq_rlt), axis=1)) * entity_type_rlt).sum()
        
        print('-'*40 + ' ' + dataset + ' ' + '-'*40)
        print(f'E_hlr: {E_hlr}, E_tlr: {E_tlr}, E_rlh: {E_rlh}, E_rlt: {E_rlt}')
        uniform_elr = np.ones_like(fq_hlr) / 6.
        E_uniform_elr = ((-np.sum(uniform_elr * np.log(uniform_elr), axis=1)) * relation).sum()
        uniform_rle = np.ones_like(fq_rlh) / 96.
        E_uniform_rle = (-np.sum(uniform_rle * np.log(uniform_rle), axis=1) * entity_type_rlh).sum()
        print(f'E_uniform_elr: {E_uniform_elr}, E_uniform_rle: {E_uniform_rle}')
    

if __name__ == '__main__':
    cal_edge_weight('train', 'DocRED')
    cal_edge_weight('train', 'Re-DocRED')
    cal_edge_weight('train', 'DWIE')
    # show_edge_weight('train', 'DocRED')
    # show_edge_weight('train', 'Re-DocRED')
    # show_edge_weight('train', 'DWIE')
    # cal_entropy()

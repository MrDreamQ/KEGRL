import os
import json
import argparse
import torch
import numpy as np
from transformers import AutoConfig, AutoModel, AutoTokenizer

from data import get_loader
from model import DocREModel
from config import merge_cfg_from_file, cfg
from utils.utils import show_confuse_mat

def to_official(preds, features):
    rel2id = json.load(open(f'../dataset/{cfg.DATA.DATASET}/rel2id.json', 'r'))
    id2rel = {value: key for key, value in rel2id.items()}
    h_idx, t_idx, title = [], [], []

    for i in range(len(features)):
        f = features[i]
        hts = f["hts"]
        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [f["title"] for ht in hts]

    res = []
    for i in range(preds.shape[0]):
        pred = preds[i]
        pred = np.nonzero(pred)[0].tolist()     # 这里感觉有点问题啊，不是0的就算有预测
        for p in pred:
            if p != 0:  # filter not NA
                res.append(
                    {
                        'title': title[i],
                        'h_idx': h_idx[i],
                        't_idx': t_idx[i],
                        'r': id2rel[p],
                    }
                )
    return res


def gen_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train"):]
    fact_file_name = os.path.join(
        truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train


def official_evaluate(tmp, path, train_file, dev_file):
    rel2id = json.load(open(f'../dataset/{cfg.DATA.DATASET}/rel2id.json', 'r'))
    id2rel = {value: key for key, value in rel2id.items()}

    truth_dir = os.path.join(path, 'ref')
    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)
    
    fact_in_train = gen_train_facts(
        os.path.join(path, train_file), truth_dir)
    truth = json.load(open(os.path.join(path, dev_file)))
    
    std = {}
    tot_evidences = 0
    titleset = set([])
    title2vectexSet = {}
    
    total_each_rel = [0] * (cfg.DATA.NUM_CLASSES-1)
    total_each_ans = [0] * (cfg.DATA.NUM_CLASSES-1)
    total_each_correct = [0] * (cfg.DATA.NUM_CLASSES-1)
    total_each_correct_intrain = [0] * (cfg.DATA.NUM_CLASSES-1)
    
    for i, x in enumerate(truth):
        if 'title' in x:
            title = x['title']
        else:
            title = str(i)
        titleset.add(title)
        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet
        
        if 'labels' in x:
            for label in x['labels']:
                if 'evidence' not in label:
                    label['evidence'] = '0'
                r = label['r']
                h_idx = label['h']
                t_idx = label['t']
                std[(title, r, h_idx, t_idx)] = set(label['evidence'])
                tot_evidences += len(label['evidence'])
                total_each_rel[rel2id[r]-1] += 1

    tot_relations = len(std)
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])

    correct_re = 0
    correct_evidence = 0
    pred_evi = 0
    correct_in_train_annotated = 0
    correct_in_train_distant = 0
    titleset2 = set([])
    
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        total_each_ans[rel2id[r]-1] += 1
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]
        if 'evidence' in x:
            evi = set(x['evidence'])
        else:
            evi = set([])
        pred_evi += len(evi)
        
        if (title, r, h_idx, t_idx) in std:
            total_each_correct[rel2id[r]-1] += 1
            correct_re += 1
            stdevi = std[(title, r, h_idx, t_idx)]
            correct_evidence += len(stdevi & evi)
            in_train_annotated = in_train_distant = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train:
                        in_train_annotated = True
            if in_train_annotated:
                correct_in_train_annotated += 1
                total_each_correct_intrain[rel2id[r]-1] += 1
            if in_train_distant:
                correct_in_train_distant += 1
        
    re_p = 1.0 * correct_re / (len(submission_answer) + 1e-10)    # 准确率
    re_r = 1.0 * correct_re / (tot_relations + 1e-10)             # 召回率
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)   # f1
    
    evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = 1.0 * correct_evidence / tot_evidences
    if evi_p + evi_r == 0:
        evi_f1 = 0
    else:
        evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)
        
    re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (
        len(submission_answer) - correct_in_train_annotated + 1e-5)
    re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / \
        (len(submission_answer) - correct_in_train_distant + 1e-5)
        
    if re_p_ignore_train_annotated + re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * \
            re_r / (re_p_ignore_train_annotated + re_r)

    if re_p_ignore_train + re_r == 0:
        re_f1_ignore_train = 0
    else:
        re_f1_ignore_train = 2.0 * re_p_ignore_train * \
            re_r / (re_p_ignore_train + re_r)
    
    p_each_rel = np.array(total_each_correct) / (np.array(total_each_ans) + 1e-10)
    r_each_rel = np.array(total_each_correct) / (np.array(total_each_rel) + 1e-10)
    f1_each_rel = 2 * p_each_rel * r_each_rel / (p_each_rel + r_each_rel + 1e-10)

    p_each_rel_ign = (np.array(total_each_correct) - np.array(total_each_correct_intrain)) / \
        (np.array(total_each_ans) - np.array(total_each_correct_intrain) + 1e-10)
    f1_each_rel_ign = 2 * p_each_rel_ign * r_each_rel / (p_each_rel_ign + r_each_rel + 1e-10)
    
    each_metric = {
        "f1_each_rel": f1_each_rel.tolist(),
        "f1_each_rel_ign": f1_each_rel_ign.tolist(),
        "f1_each_rel_avg": f1_each_rel.mean(),
        "f1_each_rel_ign_avg": f1_each_rel_ign.mean(),
    }
    
    return re_f1, re_f1_ignore_train_annotated,  re_p, re_r, each_metric

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

def evaluate(eval_dataloader, eval_feature, model):
    tag = 'dev'
    preds = []

    for batch in eval_dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].cuda(),  # token index
                  'attention_mask': batch[1].cuda(),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                    'graph_big': batch[5].to('cuda') if cfg.SGRAPH.LAYERS_NUM > 0 else None,
                    'graph_ht': batch[6].to('cuda') if cfg.DGRAPH.LAYERS_NUM > 0 else None,
                    'graph_th': batch[7].to('cuda') if cfg.DGRAPH.LAYERS_NUM > 0 else None,
                    'end_pos': batch[8],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)

        pred = pred.cpu().numpy()
        pred[np.isnan(pred)] = 0
        preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, eval_feature)  # preds = [{'title', 'h_idx', 't_idx', 'r'}]
    # if len(ans) > 0:
    if tag == 'dev':
        train_file = 'train_annotated.json' if cfg.DATA.DATASET != 'Re-DocRED' else 'train_revised.json'
        dev_file = 'dev.json' if cfg.DATA.DATASET != 'Re-DocRED' else 'dev_revised.json'
        best_f1, best_f1_ign, best_p, best_r, each_metric = official_evaluate(
            ans, f"../dataset/{cfg.DATA.DATASET}", train_file, dev_file)

    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
        tag + "_avg_F1": each_metric["f1_each_rel_avg"] * 100,
        tag + "_avg_F1_ign": each_metric["f1_each_rel_ign_avg"] * 100,
        tag + "_P": best_p * 100,
        tag + "_R": best_r * 100,
    }
    return best_f1, output, each_metric
    

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action='store', dest='config',
                        type=str, default='configs/baseline.yaml')
    parser.add_argument('--ckpt', action='store', dest='ckpt',
                        type=str, required=True, help='checkpoint dir')
    parser.add_argument('--do_confusion', action='store_true',
                        help="Whether to run confusion matrix on the dev set.")
    args = parser.parse_args()
    return args

def predict(test_dataloader, test_feature, model):
    tag = 'test'
    
    preds = []
    for batch in test_dataloader:
        model.eval()
        inputs = {'input_ids': batch[0].cuda(),  # token index
                  'attention_mask': batch[1].cuda(),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                    'graph_big': batch[5].to('cuda') if cfg.SGRAPH.LAYERS_NUM > 0 else None,
                    'graph_ht': batch[6].to('cuda') if cfg.DGRAPH.LAYERS_NUM > 0 else None,
                    'graph_th': batch[7].to('cuda') if cfg.DGRAPH.LAYERS_NUM > 0 else None,
                    'end_pos': batch[8],
                  }
        
        with torch.no_grad():
            pred, *_ = model(**inputs)

        pred = pred.cpu().numpy()
        pred[np.isnan(pred)] = 0
        preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, test_feature)  # preds = [{'title', 'h_idx', 't_idx', 'r'}]
    if cfg.DATA.DATASET == 'DocRED':
        output_eval_file = os.path.join(os.path.dirname(args.ckpt), "result.json")
        with open(output_eval_file, 'w') as f:
            json.dump(ans, f)
        print(f"Generate test result at {output_eval_file}")
    
    elif cfg.DATA.DATASET == 'DWIE':
        train_file = 'train_annotated.json'
        dev_file = 'test.json'
        best_f1, best_f1_ign, best_p, best_r, each_metric= official_evaluate(
            ans, f"../dataset/{cfg.DATA.DATASET}", train_file, dev_file)
        output = {
            tag + "_F1": best_f1 * 100,
            tag + "_F1_ign": best_f1_ign * 100,
            tag + "_avg_F1": each_metric["f1_each_rel_avg"] * 100,
            tag + "_avg_F1_ign": each_metric["f1_each_rel_ign_avg"] * 100,
        }
        print(output)
    
    else:
        train_file = 'train_revised.json'
        dev_file = 'test_revised.json'
        best_f1, best_f1_ign, best_p, best_r, each_metric = official_evaluate(
            ans, f"../dataset/{cfg.DATA.DATASET}", train_file, dev_file)
        output = {
            tag + "_F1": best_f1 * 100,
            tag + "_F1_ign": best_f1_ign * 100,
            tag + "_avg_F1": each_metric["f1_each_rel_avg"] * 100,
            tag + "_avg_F1_ign": each_metric["f1_each_rel_ign_avg"] * 100,
            tag + "_Precision": best_p * 100, 
            tag + "_Recall": best_r * 100, 
        }
        print(output)

if __name__ == '__main__':
    args = parse_args()
    merge_cfg_from_file(args.config)

    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.PLM_NAME)
    config = AutoConfig.from_pretrained(
        cfg.MODEL.PLM_NAME,
        num_labels=cfg.DATA.NUM_CLASSES,
    )
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    transformer_types = {'bert-base-cased': "bert", 'roberta-large': "roberta"}
    config.transformer_type = transformer_types[cfg.MODEL.PLM_NAME]
    if cfg.DATA.DATASET == 'DWIE':
        config.output_attentions = True
    model = AutoModel.from_pretrained(
        cfg.MODEL.PLM_NAME,
        from_tf=bool(".ckpt" in "roberta-large"),
        config=config,
    )
    model = DocREModel(config, model, num_labels=4).cuda()
    model.load_state_dict(torch.load(args.ckpt)["state_dict"])

    dataset_name = cfg.DATA.DATASET
    dev_loader, dev_features = get_loader(dataset_name, 'dev', tokenizer,
                                          max_seq_length=cfg.DATA.MAX_LENGTH,
                                          batch_size=8,
                                          shuffle=False)
    test_loader, test_features = get_loader(dataset_name, 'test', tokenizer,
                                            max_seq_length=cfg.DATA.MAX_LENGTH,
                                            batch_size=8,
                                            shuffle=False)

    dev_score, dev_output, each_metric = evaluate(dev_loader, dev_features, model)  # eval
    print(dev_output)
    predict(test_loader, test_features, model)  # test 

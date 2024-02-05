import os
import time
import argparse
import random
from copy import deepcopy

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from data import get_loader
from config import cfg, merge_cfg_from_file
from model import DocREModel
from test import evaluate

def set_seed(args):
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action='store', dest='config',
                        type=str, default='configs/baseline.yaml')
    args = parser.parse_args()
    return args


def get_lr(optimizer):
    lm_lr = optimizer.param_groups[0]['lr']
    classifier_lr = optimizer.param_groups[1]['lr']
    return lm_lr, classifier_lr


def statistics_embed(embedding):
    m = embedding.mean()
    std = torch.pow(torch.pow(embedding - m, 2).mean(), 0.5)
    print("m = %.4f, std = %.4f" % (m, std))    # roberta_large word_embedding: m = -0.0181, std = 0.1358


def train(args):
    dataset_name = cfg.DATA.DATASET
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.PLM_NAME)
    
    config = AutoConfig.from_pretrained(
        cfg.MODEL.PLM_NAME,
        num_labels=cfg.DATA.NUM_CLASSES,
    )
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.output_hidden_states = True
    transformer_types = {'bert-base-cased': "bert", 'roberta-large': "roberta"}
    config.transformer_type = transformer_types[cfg.MODEL.PLM_NAME]
    if dataset_name == 'DWIE':
        config.output_attentions = True
    
    model = AutoModel.from_pretrained(
        cfg.MODEL.PLM_NAME,
        from_tf=bool(".ckpt" in "roberta-large"),
        config=config,
    )
    model = DocREModel(config, model, num_labels=4).cuda()
    train_loader, _ = get_loader(dataset_name, 'train', tokenizer, 
                              max_seq_length=cfg.DATA.MAX_LENGTH, 
                              batch_size=cfg.BATCH_SIZE,
                              shuffle=True)
    dev_loader, dev_features = get_loader(dataset_name, 'dev', tokenizer, 
                            max_seq_length=cfg.DATA.MAX_LENGTH, 
                            batch_size=8 if cfg.DATA.DATASET != 'DWIE' else 1,
                            shuffle=False)
    # test_loader, test_features = get_loader(dataset_name, 'test', tokenizer, 
    #                          max_seq_length=cfg.DATA.MAX_LENGTH, 
    #                          batch_size=cfg.BATCH_SIZE,
    #                          shuffle=False)

    output_dir = os.path.join('outputs',
                             os.path.basename(args.config).split('.')[0],
                             time.strftime("%Y%m%d_%H%M%S", time.localtime()))
    
    new_layer = ["extractor", "bilinear", "classifier",  "projection"]
    graph_layer = ["sgnn", "dgnn"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer + graph_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": cfg.CLASSIFIER_LR},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in graph_layer)], "lr": cfg.GRAPH_LR},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.LR, eps=cfg.ADAM_EPSILON)
    print("CLASSIFIER_LR:{}, GRAPH_LR:{}, LR:{}".format(cfg.CLASSIFIER_LR, cfg.GRAPH_LR, cfg.LR), flush=True)
    print(optimizer, flush=True)
    
    total_steps = int(len(train_loader) * cfg.EPOCH // cfg.ACCUMULATION_STEPS)
    warmup_steps = int(total_steps * cfg.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_f1 = -1
    best_output = {}
    save_dict = {}
    
    for epoch in range(cfg.EPOCH):
        print(f"train epoch: {epoch}")
        bar = enumerate(train_loader)
        for step, batch in bar:
            model.train()
            inputs = {'input_ids': batch[0].cuda(),
                      'attention_mask': batch[1].cuda(),
                      'labels': batch[2],
                      'entity_pos': batch[3],
                      'hts': batch[4],
                      'graph_big': batch[5].to('cuda') if cfg.SGRAPH.LAYERS_NUM > 0 else None,
                      'graph_ht': batch[6].to('cuda') if cfg.DGRAPH.LAYERS_NUM > 0 else None,
                      'graph_th': batch[7].to('cuda') if cfg.DGRAPH.LAYERS_NUM > 0 else None,
                      'end_pos': batch[8],
                      }
            loss = model(**inputs)
            loss = loss / cfg.ACCUMULATION_STEPS
            loss.backward()
            if step % cfg.ACCUMULATION_STEPS == 0:
                if cfg.MAX_GRAD_NORM > 0:
                    # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), cfg.MAX_GRAD_NORM)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                optimizer.zero_grad()
        lm_lr, classifier_lr = get_lr(optimizer)
        print('Current epoch: {:d},  Current LM lr: {:.5e}, Current Classifier lr: {:.5e}'.format(epoch, lm_lr, classifier_lr), flush=True)
        print('Current Loss: {}'.format(loss * cfg.ACCUMULATION_STEPS))
        
        if epoch >= 5:
            torch.cuda.empty_cache()
            _, output, each_metric = evaluate(dev_loader, dev_features, model)
            if output['dev_F1'] > best_f1:
                best_f1 = output['dev_F1']
                best_output = output
                save_dict = {
                    "state_dict": deepcopy(model.state_dict()),
                    "epoch": epoch
                }
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, f'best.pth')
                torch.save(save_dict, save_path)
                print('save as %s' % save_path)
            
            print('\033[32m' + 'Current: ' + '\033[0m', output)
            print('\033[33m' + 'Best:    ' + '\033[0m', best_output)

if __name__ == '__main__':
    def print_config(config):
        info = "Running with the following configs:\n"
        for k,v in vars(config).items():
            info += "\t{} : {}\n".format(k, str(v))
        print("\n" + info + "\n", flush=True)
        return
    
    args = parse_args()
    args.seed = 3407
    set_seed(args)
    print_config(args)
    merge_cfg_from_file(args.config)
    print_config(cfg)
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    train(args)

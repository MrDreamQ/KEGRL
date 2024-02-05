import os
import json
import numpy as np
import pickle
import lmdb
from tqdm import tqdm
from torch.utils.data import Dataset

from data.dwie_prepro import process_one_sample
from config import cfg

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


class DWIEDataset(Dataset):
    def __init__(self, split, tokenizer, max_seq_length) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.file_path = os.path.join("../dataset/DWIE", 
                                      split + ('_annotated' if split == 'train' else '') + '.json')
        self.lmdb_path = os.path.join('../dataset/DWIE/lmdb_roberta' if cfg.MODEL.PLM_NAME == 'roberta-large' else '../dataset/DWIE/lmdb_bert', split)
        if not os.path.exists(self.lmdb_path):
            self._gen_lmdb()
        self.lmdb = lmdb.open(self.lmdb_path, max_readers=8, readonly=True, lock=False, readahead=False, meminit=False)
        self._index_set = None
        with self.lmdb.begin(write=False) as txn:
            self.num_sample = int(txn.get('num_sample'.encode('utf-8')).decode())
    
    def __len__(self):
        if self._index_set is not None:
            return len(self._index_set)
        return self.num_sample
    
    def __getitem__(self, index):
        if self._index_set is not None:
            index = self._index_set[index]
        with self.lmdb.begin(write=False) as txn:
            feature = pickle.loads(txn.get(str(index).encode('utf-8')))
        return feature
    
    def _gen_lmdb(self):
        print('generate lmdb %s ...' % self.lmdb_path)
        os.makedirs(self.lmdb_path)
        env = lmdb.open(self.lmdb_path, map_size=1099511627776)
        cnt = 0
        cache = {}
        with open(self.file_path, "r") as fh:
            data = json.load(fh)
        print('reading %s' % self.file_path)
        for sample in tqdm(data, desc="Example"):
            feature = process_one_sample(sample, self.tokenizer, self.max_seq_length, cnt)
            if feature is None:
                cnt += 1
                continue
            data_byte = pickle.dumps(feature)
            data_id = str(cnt).encode('utf-8')
            cache[data_id] = data_byte
            if cnt % 200 == 0:
                writeCache(env, cache)
                cache = {}
            cnt += 1
        cache['num_sample'.encode('utf-8')] = str(cnt).encode()
        writeCache(env, cache)
        print('save {} samples to {}'.format(cnt, self.lmdb_path))
        env.close()

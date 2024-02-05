import config
import models
import argparse
import random
from config import merge_cfg_from_file, cfg

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'BiLSTM', help = 'name of the model')
parser.add_argument('--save_name', type = str, required=True)
parser.add_argument('--train_prefix', type = str, default = 'dev_train')
parser.add_argument('--test_prefix', type = str, default = 'dev_dev')
parser.add_argument('--config', type = str, default='yamls/baseline.yaml')

args = parser.parse_args()
merge_cfg_from_file(args.config)

def print_config(config):
	info = "Running with the following configs:\n"
	for k,v in vars(config).items():
		info += "\t{} : {}\n".format(k, str(v))
	print("\n" + info + "\n", flush=True)
	return

def set_seed(args):
    import numpy as np
    import torch
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

args.seed = 0
set_seed(args)

print_config(cfg)
print_config(args)

model = {
	'CNN3': models.CNN3,
	'LSTM': models.LSTM,
	'BiLSTM': models.BiLSTM,
	'ContextAware': models.ContextAware,
}

con = config.Config(args)
con.set_max_epoch(cfg.EPOCH)
con.load_train_data()
con.load_test_data()

con.train(model[args.model_name], args.save_name)

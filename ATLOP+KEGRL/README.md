# ATLOP+KEGRL
KEGRL model based on [ATLOP](https://github.com/wzhouad/ATLOP).

## Pretrained Language Models
Download pretrained language models **bert-base-cased** and **roberta-large** from [Hugging Face](https://huggingface.co/models). Or it will cache automatically.

## Train
```bash
bash run.sh 
```

You can edit `config` option in `run.sh` to run other settings. 

Edit yaml file in `configs/` to modify hyperparameter. 

## Test
```bash
bash test.sh 
```

## Configs
```python
├── configs
│   ├── baseline.yaml # ATLOP baseline on DocRED
│   ├── baseline_dwie.yaml # ATLOP baseline on DWIE
│   ├── baseline_re.yaml # ATLOP baseline on Re-DocRED
│   ├── dgraph.yaml # ATLOP+DGRAPH on DocRED
│   ├── dgraph_dwie.yaml # ATLOP+DGRAPH on DWIE
│   ├── dgraph_re.yaml # ATLOP+DGRAPH on Re-DocRED
│   ├── sgraph.yaml # ATLOP+SGRAPH on DocRED
│   ├── sgraph_dwie.yaml # ATLOP+SGRAPH on DWIE
│   └── sgraph_re.yaml # ATLOP+SGRAPH on Re-DocRED
```

# Hyperparameters
The following is a description of the hyperparameters defined in `config.py`
```python
EPOCH # The number of epochs to train.
BATCH_SIZE # The batch size for training.
ACCUMULATION_STEPS # The number of steps to accumulate gradients.
MAX_GRAD_NORM # The maximum norm of gradients.
WARMUP_RATIO # The ratio of warmup steps.
LR # The learning rate for training.
GRAPH_LR # The learning rate for KEGRL module.
CLASSIFIER_LR # The learning rate for projection layer.
ADAM_EPSILON # The epsilon parameters for Adam optimizer.
HIDDEN_DIM # The dimension of KEGRL node features.
RA_LAYER # The number of RaEncoder.
ADA_LOSS_LAMB # The weight of ADA loss.
SGRAPH.LAYERS_NUM # The number of SGRAPH layers.
DGRAPH.LAYERS_NUM # The number of DGRAPH layers.
```
# DocRED4+KEGRL
KEGRL model based on [DocRED4](https://github.com/thunlp/DocRED).

## Data Preprocess
```bash
bash prepro.sh
```
It will generate `prepro_data`, `prepro_re`, `prepro_dwie` directory storing preprocessed data.

## Train
```bash
bash run.sh
```

Edit `cfg` and `model_name` options in `run.sh` to run other settings.

Edit yaml file in `yamls` directory to modify hyperparameter. 

## Test

```bash
bash test.sh 
```

## Configs
```python
├── yamls
│   ├── baseline.yaml # baseline on DocRED
│   ├── baseline_dwie.yaml # baseline on DWIE
│   ├── baseline_re.yaml # baseline on Re-DocRED
│   ├── dgraph.yaml # DocRED4+DGRAPH on DocRED
│   ├── dgraph_dwie.yaml # DocRED4+DGRAPH on DWIE
│   ├── dgraph_re.yaml # DocRED4+DGRAPH on Re-DocRED
│   ├── sgraph.yaml # DocRED4+SGRAPH on DocRED
│   ├── sgraph_dwie.yaml # DocRED4+SGRAPH on DWIE
│   └── sgraph_re.yaml # DocRED4+SGRAPH on Re-DocRED
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
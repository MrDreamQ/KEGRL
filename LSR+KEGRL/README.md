# LSR+KEGRL
KEGRL model based on [LSR](https://github.com/nanguoshun/LSR).

## Pretrained Language Models
Download pretrained language models **bert-base-uncased** from [Hugging Face](https://huggingface.co/models). Or it will cache automatically.

## Data Preprocess
```bash
bash prepro.sh
```
It will generate `prepro_glove`, `prepro_re_glove`, `prepro_dwie_glove`, `prepro_bert`, `prepro_re_bert`, `prepro_dwie_bert` directory storing preprocessed data.

## Train
```bash
bash run.sh # On DocRED dataset
bash run_re.sh # On Re-DocRED dataset
bash run_dwie.sh # On DWIE dataset
```

Edit `sgraph` and `dgraph` option in `run.sh` to run SGRAPH and DGRAPH settings.

## Test
```bash
bash test.sh # On DocRED dataset
bash test_re.sh # On Re-DocRED dataset
bash test_dwie.sh # On DWIE dataset
```